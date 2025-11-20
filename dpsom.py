"""
PyTorch replacement for the original TensorFlow/TensorFlow-Probability DPSOM implementation,
Variable names and overall workflow are preserved as closely as possible.
"""

import uuid
from datetime import date
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import os
import time
import random
import numpy.random as nprand

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from decay_scheduler import ExponentialDecayScheduler
from dpsom_model import DPSOM
from utils import cluster_purity


class DPSOM_Config:
    def __init__(self):
        self.num_epochs = 300
        self.batch_size = 300
        self.latent_dim = 100
        self.som_dim = [8, 8]
        self.learning_rate = 0.001
        self.learning_rate_pretrain = 0.001
        self.alpha = 10.0
        self.beta = 0.25
        self.gamma = 20
        self.theta = 1
        self.epochs_pretrain = 15
        self.decay_factor = 0.99
        self.decay_steps = 5000
        self.name = "dpsom"
        self.ex_name = f"{self.name}_{self.latent_dim}_{self.som_dim[0]}-{self.som_dim[1]}_{str(date.today())}_{uuid.uuid4().hex[:5]}"
        self.logdir = f"logs/{self.ex_name}"
        self.modelpath = f"models/{self.ex_name}/{self.ex_name}.ckpt"
        self.data_set = "MNIST"
        self.dropout = 0.4
        self.prior_var = 1
        self.prior = 0.5
        self.convolution = False
        self.use_saved_pretrain = False
        self.save_pretrain = False
        self.random_seed = 2020

config = DPSOM_Config()


def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_data_generator(data_train, data_val, labels_train, labels_val, data_test, labels_test):
    """
    Creates a data generator for the training that yields numpy arrays,
    preserving the original interface.
    """
    def batch_generator(mode="train", batch_size=300):
        assert mode in ["train", "val", "test"], "The mode should be in {train, val, test}."
        if mode == "train":
            images = data_train.copy()
            labels = labels_train.copy()
        elif mode == "val":
            images = data_val.copy()
            labels = labels_val.copy()
        else:
            images = data_test.copy()
            labels = labels_test.copy()

        while True:
            for i in range(len(images) // batch_size):
                yield images[i * batch_size:(i + 1) * batch_size], labels[i * batch_size:(i + 1) * batch_size], i

    return batch_generator


def _np_to_torch(images_np, device):
    # Expect original shape [B,28,28,1]; convert to [B,1,28,28] float32 in [0,1]
    x = torch.from_numpy(images_np).float().permute(0, 3, 1, 2).to(device)
    return x


def _ensure_dir_for_model(modelpath):
    d = os.path.dirname(modelpath)
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def _preprocess_dataset(data):
    """Normalizes dataset to [0, 1] by dividing by max per image and shaping."""
    data_flat = data.reshape(-1, 28 * 28)
    maxx = np.maximum(np.reshape(np.amax(data_flat, axis=-1), [-1, 1]), 1.0)
    return (data_flat / maxx).reshape(-1, 28, 28, 1)


def _run_pretraining_phase(model, train_gen, val_gen, optimizer, writers, step, device,
                           epochs, batch_size, som_dim, learning_rate_pretrain, pbar, num_batches):

    train_writer, test_writer = writers

    print("\n\nAutoencoder Pretraining...\n")

    model.train()
    a = np.zeros((batch_size, som_dim[0] * som_dim[1]), dtype=np.float32)

    for epoch in range(epochs):
        for i in range(num_batches):
            batch_data, _, _ = next(train_gen)
            x_batch = _np_to_torch(batch_data, device)
            model.set_p(torch.from_numpy(a).to(device))

            for g in optimizer.param_groups:
                g["lr"] = float(learning_rate_pretrain)

            optimizer.zero_grad()
            z = model(x_batch)
            loss_rec = model.loss_reconstruction_ze(x_batch)
            loss_rec.backward()
            optimizer.step()

            train_elbo = loss_rec.item()
            train_kl = model._kl_divergence_diag(model.mu, model.logvar).item()
            train_log_lik = train_elbo - model.prior * train_kl

            if i % 100 == 0:
                batch_val, _, _ = next(val_gen)
                x_val = _np_to_torch(batch_val, device)
                with torch.no_grad():
                    model(x_val)
                    val_elbo = model.loss_reconstruction_ze(x_val).item()
                    val_kl = model._kl_divergence_diag(model.mu, model.logvar).item()
                    val_log_lik = val_elbo - model.prior * val_kl

                test_writer.add_scalar("loss/loss_elbo", val_elbo, step)
                test_writer.add_scalar("loss_reconstruction_ze/loss_reconstruction_kl", val_kl, step)
                test_writer.add_scalar("loss_reconstruction_ze/loss_reconstruction_log_lik_loss", val_log_lik, step)

                train_writer.add_scalar("loss/loss_elbo", train_elbo, step)
                train_writer.add_scalar("loss_reconstruction_ze/loss_reconstruction_kl", train_kl, step)
                train_writer.add_scalar("loss_reconstruction_ze/loss_reconstruction_log_lik_loss", train_log_lik,
                                        step)

                pbar.set_postfix(epoch=epoch, train_loss=train_elbo, test_loss=val_elbo, refresh=False)
            step += 1
            pbar.update(1)
    return step


def _run_som_initialization_phase(model, train_gen, val_gen, optimizer, writers, step, device,
                                  som_dim, batch_size, num_batches, pbar):
    train_writer, test_writer = writers
    print("\n\nSOM initialization...\n")

    # 4 phases * 5 epochs = 20 epochs total
    for lr_phase in [0.9, 0.3, 0.1, 0.01]:
        for epoch in range(5):
            for i in range(num_batches):
                batch_data, _, _ = next(train_gen)
                x_batch = _np_to_torch(batch_data, device)

                model.set_p(torch.zeros(batch_size, som_dim[0] * som_dim[1], device=device))

                for g in optimizer.param_groups:
                    g["lr"] = float(lr_phase)

                optimizer.zero_grad()
                z = model(x_batch)
                loss_init, loss_som_s, loss_commit_s = model.loss_a(z)
                loss_init.backward()
                optimizer.step()

                if i % 100 == 0:
                    batch_val, _, _ = next(val_gen)
                    x_val = _np_to_torch(batch_val, device)
                    with torch.no_grad():
                        loss_elbo = model.loss_reconstruction_ze(x_batch)
                        loss_init, loss_som_s, loss_commit_s = model.loss_a(z)
                        train_elbo = loss_elbo.item()
                        train_kl = model._kl_divergence_diag(model.mu, model.logvar).item()
                        train_log_lik = train_elbo - model.prior * train_kl

                        z_val = model(x_val)
                        val_loss_elbo = model.loss_reconstruction_ze(x_val)
                        test_elbo = val_loss_elbo.item()
                        val_loss_init, val_loss_som_s, val_loss_commit_s = model.loss_a(z_val)
                        test_kl = model._kl_divergence_diag(model.mu, model.logvar).item()
                        test_log_lik = test_elbo - model.prior * test_kl

                    test_writer.add_scalar("loss/loss_elbo", test_elbo, step)
                    train_writer.add_scalar("loss/loss_elbo", train_elbo, step)
                    test_writer.add_scalar("loss_commit_s/loss_commit_s", val_loss_commit_s.item(), step)
                    test_writer.add_scalar("loss_som_s/loss_som_s", val_loss_som_s.item(), step)
                    train_writer.add_scalar("loss_commit_s/loss_commit_s", loss_commit_s.item(), step)
                    train_writer.add_scalar("loss_som_s/loss_som_s", loss_som_s.item(), step)
                    test_writer.add_scalar("loss_reconstruction_ze/loss_reconstruction_kl", train_kl, step)
                    test_writer.add_scalar("loss_reconstruction_ze/loss_reconstruction_log_lik_loss", train_log_lik,
                                           step)
                    train_writer.add_scalar("loss_reconstruction_ze/loss_reconstruction_kl", train_kl, step)
                    train_writer.add_scalar("loss_reconstruction_ze/loss_reconstruction_log_lik_loss", train_log_lik,
                                            step)

                    pbar.set_postfix(epoch=epoch, train_loss=loss_init.item(), test_loss=val_loss_init.item(),
                                     refresh=False)
                step += 1
                pbar.update(1)
    return step


def _compute_target_distribution(model, data_source, device, chunk_size=5000):
    model.eval()
    with torch.inference_mode():
        q_list = []
        for start in range(0, len(data_source), chunk_size):
            end = start + chunk_size
            x_chunk = _np_to_torch(data_source[start:end], device)
            q_list.append(model.q_p(x_chunk).cpu().numpy())

        q = np.concatenate(q_list, axis=0)
        return model.target_distribution(q)


def _run_main_training_phase(model, train_gen, val_gen, optimizer, scheduler, writers, step, device,
                             num_epochs, batch_size, data_train, data_val, modelpath, num_batches, pbar):
    train_writer, test_writer = writers
    print("\n\nTraining...\n")
    lratios, l2ratios, l3ratios = [], [], []
    test_losses = []

    for epoch in range(num_epochs):
        ppt = _compute_target_distribution(model, data_train, device)
        ppv = _compute_target_distribution(model, data_val, device)

        for i in range(num_batches):
            batch_data, _, ii = next(train_gen)
            x_batch = _np_to_torch(batch_data, device)

            p_batch = torch.from_numpy(ppt[ii * batch_size: (ii + 1) * batch_size]).float().to(device)
            model.set_p(p_batch)

            new_lr = scheduler.get_lr(step)
            for g in optimizer.param_groups:
                g["lr"] = new_lr

            optimizer.zero_grad()
            z = model(x_batch)
            loss_total, loss_elbo, loss_commit, loss_som = model.loss(x_batch, z)
            loss_total.backward()
            optimizer.step()

            train_loss = loss_total.item()
            train_elbo = loss_elbo.item()
            train_commit = loss_commit.item()
            train_som = loss_som.item()
            _, train_som_s, train_commit_s = model.loss_a(z)

            train_kl = model._kl_divergence_diag(model.mu, model.logvar).item()
            train_log_lik = train_elbo - model.prior * train_kl
            train_loss_commit = train_commit
            train_loss_som = train_som

            with torch.inference_mode():
                batch_val, _, ii_v = next(val_gen)
                x_val = _np_to_torch(batch_val, device)
                p_val = torch.from_numpy(ppv[ii_v * batch_size: (ii_v + 1) * batch_size]).float().to(device)
                model.set_p(p_val)

                z = model(x_val)
                val_loss, val_elbo, val_commit, val_som = model.loss(x_val, z)
                test_loss = val_loss.item()
                test_elbo_val = val_elbo.item()
                test_commit = val_commit.item()
                test_som = val_som.item()
                _, test_som_s, test_commit_s = model.loss_a(z)

                test_kl = model._kl_divergence_diag(model.mu, model.logvar).item()
                test_log_lik = test_elbo_val - model.prior * test_kl

                elbo_loss = test_elbo_val
                cah_loss = test_commit
                ssom_loss = test_som

            test_losses.append(test_loss)

            if i % 100 == 0:
                test_writer.add_scalar("loss/loss", test_loss, step)
                test_writer.add_scalar("loss/loss_elbo", test_elbo_val, step)
                test_writer.add_scalar("loss/loss_commit", test_commit, step)
                test_writer.add_scalar("loss/loss_som", test_som, step)
                test_writer.add_scalar("loss_commit_s/loss_commit_s", test_commit_s.item(), step)
                test_writer.add_scalar("loss_som_s/loss_som_s", test_som_s.item(), step)
                test_writer.add_scalar("loss_reconstruction_ze/loss_reconstruction_kl", test_kl, step)
                test_writer.add_scalar("loss_reconstruction_ze/loss_reconstruction_log_lik_loss", test_log_lik, step)

                train_writer.add_scalar("loss/loss", train_loss, step)
                train_writer.add_scalar("loss/loss_elbo", train_elbo, step)
                train_writer.add_scalar("loss/loss_commit", train_loss_commit, step)
                train_writer.add_scalar("loss/loss_som", train_loss_som, step)
                train_writer.add_scalar("loss_commit_s/loss_commit_s", train_commit_s.item(), step)
                train_writer.add_scalar("loss_som_s/loss_som_s", train_som_s.item(), step)
                train_writer.add_scalar("loss_reconstruction_ze/loss_reconstruction_kl", train_kl, step)
                train_writer.add_scalar("loss_reconstruction_ze/loss_reconstruction_log_lik_loss", train_log_lik, step)

                cah_ssom_ratio = (cah_loss / ssom_loss) if ssom_loss != 0 else float("inf")
                vae_cah_ratio = (elbo_loss / cah_loss) if cah_loss != 0 else float("inf")
                clust_vae_ratio = (elbo_loss / (ssom_loss + cah_loss)) if (ssom_loss + cah_loss) != 0 else float("inf")
                lratios.append(cah_ssom_ratio)
                l2ratios.append(vae_cah_ratio)
                l3ratios.append(clust_vae_ratio)

                test_s = np.mean(test_losses) if len(test_losses) > 0 else test_loss
                pbar.set_postfix(
                    epoch=epoch,
                    train_loss=loss_total.item(),
                    test_loss=test_s,
                    ssom=ssom_loss,
                    cah=cah_loss,
                    vae=elbo_loss,
                    cs_ratio=(np.mean(lratios) if len(lratios) > 0 else 0.0),
                    vc_ratio=(np.mean(l2ratios) if len(l2ratios) > 0 else 0.0),
                    cr_ratio=(np.mean(l3ratios) if len(l3ratios) > 0 else 0.0),
                    refresh=False
                )

            step += 1
            pbar.update(1)

        torch.save({"model": model.state_dict()}, modelpath)
        model.inc_epoch()

    return step


def train_model(
        model,
        data_train,
        data_val,
        generator,
        lr_val,
        num_epochs=config.num_epochs,
        batch_size=config.batch_size,
        logdir=config.logdir,
        ex_name=config.ex_name,
        modelpath=config.modelpath,
        learning_rate=config.learning_rate,
        epochs_pretrain=config.epochs_pretrain,
        som_dim=config.som_dim,
        use_saved_pretrain=config.use_saved_pretrain,
        learning_rate_pretrain=config.learning_rate_pretrain,
        save_pretrain=config.save_pretrain,
        decay_factor=config.decay_factor,
        decay_steps=config.decay_steps
):
    """
    Trains the DPSOM model using three-phase workflow.
    """
    device = _get_device()
    model = model.to(device)

    # Writers
    _ensure_dir_for_model(modelpath)
    train_writer = SummaryWriter(logdir + "/train")
    test_writer = SummaryWriter(logdir + "/test")
    writers = (train_writer, test_writer)

    # Optimizers: three heads to mirror original
    train_step_psom = optim.Adam(model.parameters(), lr=learning_rate)
    train_step_vae = optim.Adam(model.parameters(), lr=learning_rate_pretrain)
    train_step_som = optim.Adam(model.parameters(), lr=0.9)

    scheduler = ExponentialDecayScheduler(
        initial_lr=learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_factor,
        staircase=True
    )

    train_gen = generator("train", batch_size)
    val_gen = generator("val", batch_size)
    num_batches = len(data_train) // batch_size

    print("\n**********Starting job {}*********\n".format(ex_name))

    # SOM init: 4 phases * 5 epochs = 20 epochs
    total_epochs = num_epochs + epochs_pretrain + 20
    pbar = tqdm(total=total_epochs * num_batches)

    step = 0

    pretrain_modelpath = f"models/pretrain.ckpt"

    run_pretrain = True
    if use_saved_pretrain and os.path.exists(pretrain_modelpath):
        print("\n\nUsing Saved Pretraining...\n")
        try:
            ckpt = torch.load(pretrain_modelpath, map_location=device)
            model.load_state_dict(ckpt["model"], strict=True)

            skipped_steps = (epochs_pretrain + 20) * num_batches
            step += skipped_steps
            pbar.update(skipped_steps)
            run_pretrain = False
        except RuntimeError as e:
            print(f"\033[91mPretrained parameters mismatch ({e}).\033[0mnStarting pretraining...")

    if run_pretrain:
        step = _run_pretraining_phase(
            model, train_gen, val_gen, train_step_vae, writers, step, device,
            epochs_pretrain, batch_size, som_dim, learning_rate_pretrain, pbar, num_batches
        )

        step = _run_som_initialization_phase(
            model, train_gen, val_gen, train_step_som, writers, step, device,
            som_dim, batch_size, num_batches, pbar
        )
        if save_pretrain:
            torch.save({"model": model.state_dict()}, pretrain_modelpath)

    step = _run_main_training_phase(
        model, train_gen, val_gen, train_step_psom, scheduler, writers, step, device,
        num_epochs, batch_size, data_train, data_val, modelpath, num_batches, pbar
    )
    pbar.close()


def evaluate_model(
    model,
    data_test,
    generator,
    batch_size=config.batch_size,
    latent_dim=config.latent_dim,
    som_dim=config.som_dim,
    learning_rate=config.learning_rate,
    alpha=config.alpha,
    gamma=config.gamma,
    beta=config.beta,
    theta=config.theta,
    epochs_pretrain=config.epochs_pretrain,
    decay_factor=config.decay_factor,
    ex_name=config.ex_name,
    data_set=config.data_set,
    dropout=config.dropout,
    prior_var=config.prior_var,
    prior=config.prior,
    convolution=config.convolution
):
    """
    Evaluates the trained model (NMI, AMI, Purity), preserving output format and files.
    """
    device = _get_device()
    model = model.to(device)
    model.eval()

    test_k_all = []
    labels_val_all = []
    print("\n\nEvaluation...\n")

    val_gen = generator("test", batch_size)
    len_data_val = len(data_test)
    num_batches = len_data_val // batch_size

    with torch.no_grad():
        for _ in range(num_batches):
            batch_data, batch_labels, _ii = next(val_gen)
            labels_val_all.extend(batch_labels.tolist() if hasattr(batch_labels, "tolist") else list(batch_labels))
            x_val = _np_to_torch(batch_data, device)
            z = model(x_val)
            k_pred = model.k(z).cpu().numpy().tolist()
            test_k_all.extend(k_pred)

    test_nmi = metrics.normalized_mutual_info_score(np.array(labels_val_all), np.array(test_k_all), average_method="geometric")
    test_purity = cluster_purity(np.array(test_k_all), np.array(labels_val_all))
    test_ami = metrics.adjusted_mutual_info_score(np.array(test_k_all), np.array(labels_val_all))

    results = {"NMI": float(test_nmi), "Purity": float(test_purity), "AMI": float(test_ami)}

    # Same sentinel as original
    if np.abs(test_ami - 0.0) < 1e-4 and np.abs(test_nmi - 0.125) < 1e-4:
        return None

    # Write results to files, preserving original filenames
    if data_set == "fMNIST":
        fpath = "results_fMNIST_conv.txt" if convolution else "results_fMNIST.txt"
    else:
        fpath = "results_MNIST_conv.txt" if convolution else "results_MNIST.txt"

    with open(fpath, "a+") as f:
        f.write(
            "Epochs= %d, som_dim=[%d,%d], latent_dim= %d, batch_size= %d, learning_rate= %f, beta=%f, gamma=%f, "
            "theta=%f, alpha=%f, dropout=%f, decay_factor=%f, prior_var=%f, prior=%f, epochs_pretrain=%d"
            % (
                model.get_epoch(),  # epoch count not tracked here; keep placeholder to preserve format
                som_dim[0],
                som_dim[1],
                latent_dim,
                batch_size,
                learning_rate,
                beta,
                gamma,
                theta,
                alpha,
                dropout,
                decay_factor,
                prior_var,
                prior,
                epochs_pretrain,
            )
        )
        f.write(", RESULTS NMI: %f, AMI: %f, Purity: %f. Name: %r \n" % (results["NMI"], results["AMI"], results["Purity"], ex_name))

    return results


def main(config=config):
    """
    Builds the model, prepares data, trains, and evaluates; variable names and workflow preserved.
    """
    # Seeding
    random.seed(config.random_seed)
    nprand.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    start = time.time()
    if not os.path.exists("./models"):
        os.mkdir("./models")

    # Dimensions for MNIST-like data
    input_length = 28
    input_channels = 28

    # Build model
    model = DPSOM(
        latent_dim=config.latent_dim,
        som_dim=config.som_dim,
        learning_rate=config.learning_rate,
        alpha=config.alpha,
        decay_factor=config.decay_factor,
        input_length=input_length,
        input_channels=input_channels,
        beta=config.beta,
        theta=config.theta,
        gamma=config.gamma,
        convolution=config.convolution,
        dropout=config.dropout,
        prior_var=config.prior_var,
        prior=config.prior,
    )

    # Load data (replace tf.keras datasets with torchvision tensors and then numpy)
    from torchvision import datasets

    # Load data
    is_mnist = config.data_set == "MNIST"
    root = "./data" if is_mnist else "data/"

    if is_mnist:
        ds_cls = datasets.MNIST
    else:
        ds_cls = datasets.FashionMNIST

    train_ds = ds_cls(root=root, train=True, download=True)
    test_ds = ds_cls(root=root, train=False, download=True)

    data_total = train_ds.data.numpy().astype(np.float32)
    labels_total = train_ds.targets.numpy().astype(np.int64)
    data_test = test_ds.data.numpy().astype(np.float32)
    labels_test = test_ds.targets.numpy().astype(np.int64)

    # Normalization
    data_total = _preprocess_dataset(data_total)
    data_test = _preprocess_dataset(data_test)

    data_train, data_val, labels_train, labels_val = train_test_split(data_total, labels_total, test_size=0.15,
                                                                      random_state=42)

    data_generator = get_data_generator(data_train, data_val, labels_train, labels_val, data_test, labels_test)

    train_model(model, data_train, data_val, data_generator, lr_val=config.learning_rate)
    results = evaluate_model(model, data_test, data_generator)
    if results is not None:
        print(f"NMI: {results['NMI']}, AMI: {results['AMI']}, PUR: {results['Purity']}. Name: {config.ex_name}.")

    elapsed_time_fl = (time.time() - start)
    print("\n Time: {}".format(elapsed_time_fl))
    return results


if __name__ == "__main__":
    main()