# dpsom.py
"""
PyTorch replacement for the original TensorFlow/TensorFlow-Probability DPSOM implementation,
Variable names and overall workflow are preserved as closely as possible.
"""

import uuid
from datetime import date
from pathlib import Path
from tqdm import tqdm
import sacred
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import os
import time
import random
import csv
import numpy.random as nprand

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from decay_scheduler import ExponentialDecayScheduler
from dpsom_model import DPSOM
from utils import cluster_purity

# =========================
# TRAINING SCRIPT (PyTorch)
# =========================

ex = sacred.Experiment("hyperopt")
ex.observers.append(sacred.observers.FileStorageObserver.create("../sacred_runs"))
ex.captured_out_filter = sacred.utils.apply_backspaces_and_linefeeds


@ex.config
def ex_config():
    """
    Sacred configuration for the experiment.
    """
    num_epochs = 300
    batch_size = 300
    latent_dim = 100
    som_dim = [8, 8]
    learning_rate = 0.001
    learning_rate_pretrain = 0.001
    alpha = 7.5
    beta = 0.25
    gamma = 20
    theta = 1
    epochs_pretrain = 15
    decay_factor = 0.99
    decay_steps = 5000
    name = ex.get_experiment_info()["name"]
    ex_name = "{}_{}_{}-{}_{}_{}".format(name, latent_dim, som_dim[0], som_dim[1], str(date.today()), uuid.uuid4().hex[:5])
    logdir = "logs/{}".format(ex_name)
    modelpath = "models/{}/{}.ckpt".format(ex_name, ex_name)
    data_set = "MNIST"
    validation = False
    dropout = 0.4
    prior_var = 1
    prior = 0.5
    convolution = False
    val_epochs = False
    more_runs = False
    use_saved_pretrain = False
    save_pretrain = False
    random_seed = 2020
    exp_output = False  # Output to Google Cloud File System
    exp_path = "data/variational_psom/static_clustering_FMNIST/robustness"


def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@ex.capture
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


@ex.capture
def train_model(
    model,
    data_train,
    data_val,
    generator,
    lr_val,
    num_epochs,
    batch_size,
    logdir,
    ex_name,
    validation,
    val_epochs,
    modelpath,
    learning_rate,
    epochs_pretrain,
    som_dim,
    latent_dim,
    use_saved_pretrain,
    learning_rate_pretrain,
    save_pretrain,
    alpha,
    beta,
    gamma,
    theta,
    decay_factor,
    decay_steps
):
    """
    Trains the DPSOM model using PyTorch, preserving the original three-phase workflow.
    """
    device = _get_device()
    model = model.to(device)

    # Writers
    _ensure_dir_for_model(modelpath)
    train_writer = SummaryWriter(logdir + "/train")
    test_writer = SummaryWriter(logdir + "/test")

    # Optimizers: three heads to mirror original
    train_step_VARPSOM = optim.Adam(model.parameters(), lr=learning_rate)
    train_step_vae = optim.Adam(model.parameters(), lr=learning_rate_pretrain)
    train_step_som = optim.Adam(model.parameters(), lr=0.9)

    scheduler = ExponentialDecayScheduler(
        initial_lr=learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_factor,
        staircase=True
    )

    model._global_step = 0

    train_gen = generator("train", batch_size)
    val_gen = generator("val" if validation else "test", batch_size)

    len_data_train = len(data_train)
    len_data_val = len(data_val)
    num_batches = len_data_train // batch_size

    # Pretraining
    test_losses = []
    test_losses_mean = []

    print("\n**********Starting job {}*********\n".format(ex_name))
    pbar = tqdm(total=(num_epochs + epochs_pretrain + 40) * num_batches)

    # Pretraining
    if use_saved_pretrain and os.path.exists(modelpath):
        print("\n\nUsing Saved Pretraining...\n")
        ckpt = torch.load(modelpath, map_location=device)
        model.load_state_dict(ckpt["model"])
    else:
        print("\n\nAutoencoder Pretraining...\n")
        model.train()
        # 'p' placeholder initialized as zeros, only to keep API parity
        a = np.zeros((batch_size, som_dim[0] * som_dim[1]), dtype=np.float32)

        for epoch in range(epochs_pretrain):
            for i in range(num_batches):
                batch_data, _, _ = next(train_gen)
                x_batch = _np_to_torch(batch_data, device)
                model.set_p(torch.from_numpy(a).to(device))

                # Adjust pretrain LR
                for g in train_step_vae.param_groups:
                    g["lr"] = float(learning_rate_pretrain)

                train_step_vae.zero_grad()
                loss_rec = model.loss_reconstruction_ze(x_batch)
                loss_rec.backward()
                train_step_vae.step()

                if i % 100 == 0:
                    batch_val, _, _ = next(val_gen)
                    x_val = _np_to_torch(batch_val, device)
                    with torch.no_grad():
                        # Compute ELBO (reconstruction + KL)
                        val_elbo = model.loss_reconstruction_ze(x_val).item()
                        train_elbo = model.loss_reconstruction_ze(x_batch).item()

                        # Get mu and logvar to compute KL separately
                        mu_val, logvar_val = model._encode(x_val)
                        val_kl = model._kl_divergence_diag(mu_val, logvar_val).item()

                        mu_train, logvar_train = model._encode(x_batch)
                        train_kl = model._kl_divergence_diag(mu_train, logvar_train).item()

                        # Reconstruction log-likelihood only (ELBO - prior*KL)
                        val_log_lik = val_elbo - model.prior * val_kl
                        train_log_lik = train_elbo - model.prior * train_kl

                    step = model._global_step
                    # Log with TensorFlow naming convention
                    test_writer.add_scalar("loss/loss_elbo", val_elbo, step)
                    test_writer.add_scalar("loss_reconstruction_ze/loss_reconstruction_kl", val_kl, step)
                    test_writer.add_scalar("loss_reconstruction_ze/loss_reconstruction_log_lik_loss", val_log_lik, step)

                    train_writer.add_scalar("loss/loss_elbo", train_elbo, step)
                    train_writer.add_scalar("loss_reconstruction_ze/loss_reconstruction_kl", train_kl, step)
                    train_writer.add_scalar("loss_reconstruction_ze/loss_reconstruction_log_lik_loss", train_log_lik, step)

                    pbar.set_postfix(epoch=epoch, train_loss=train_elbo, test_loss=val_elbo, refresh=False)
                model._global_step += 1
                pbar.update(1)

        # SOM initialization: multiple passes with decreasing learning rates
        print("\n\nSOM initialization...\n")
        for lr_phase in [0.9, 0.3, 0.1, 0.01]:
            for epoch in range(5):
                for i in range(num_batches):
                    batch_data, _, _ = next(train_gen)
                    x_batch = _np_to_torch(batch_data, device)

                    # Set 'p' placeholder-like to zeros (not used in init losses)
                    model.set_p(torch.zeros(batch_size, som_dim[0] * som_dim[1], device=device))

                    for g in train_step_som.param_groups:
                        g["lr"] = float(lr_phase)

                    train_step_som.zero_grad()
                    loss_init = model.loss_a(x_batch)
                    loss_init.backward()
                    train_step_som.step()

                    if i % 100 == 0:
                        batch_val, _, _ = next(val_gen)
                        x_val = _np_to_torch(batch_val, device)
                        with torch.no_grad():
                            val_l = model.loss_a(x_val).item()
                            train_l = model.loss_a(x_batch).item()
                        step = model._global_step
                        test_writer.add_scalar("loss_a", val_l, step)
                        train_writer.add_scalar("loss_a", train_l, step)
                        pbar.set_postfix(epoch=epoch, train_loss=train_l, test_loss=val_l, refresh=False)
                    model._global_step += 1
                    pbar.update(1)

        if save_pretrain:
            torch.save({"model": model.state_dict()}, modelpath)

    # Main training
    print("\n\nTraining...\n")
    lratios, l2ratios, l3ratios = [], [], []

    for epoch in range(num_epochs):
        # Compute initial soft probabilities q over full train to build target distribution p_t
        model.eval()
        with torch.inference_mode():
            q_list = []
            for t in range(9):
                sl = slice(int(len(data_train) / 10) * t, int(len(data_train) / 10) * (t + 1))
                x_t = _np_to_torch(data_train[sl], device)
                q_t = model.q(x_t).cpu().numpy()
                q_list.append(q_t)
            sl = slice(int(len(data_train) / 10) * 9, len(data_train))
            x_t = _np_to_torch(data_train[sl], device)
            q_t = model.q(x_t).cpu().numpy()
            q_list.append(q_t)
            q = np.concatenate(q_list, axis=0)

            ppt = model.target_distribution(q)
            qv = model.q(_np_to_torch(data_val, device)).cpu().numpy()
            ppv = model.target_distribution(qv)

        for i in range(num_batches):
            batch_data, _, ii = next(train_gen)
            x_batch = _np_to_torch(batch_data, device)

            # Train set p
            p_batch = torch.from_numpy(ppt[ii * batch_size: (ii + 1) * batch_size]).float().to(device)
            model.set_p(p_batch)

            # Adjust train LR
            new_lr = scheduler.get_lr(model._global_step)
            for g in train_step_VARPSOM.param_groups:
                g["lr"] = new_lr

            train_step_VARPSOM.zero_grad()
            loss_total = model.loss(x_batch)
            loss_total.backward()
            train_step_VARPSOM.step()

            with torch.inference_mode():
                # Compute for training batch
                train_loss = loss_total.item()
                train_elbo = model.loss_reconstruction_ze(x_batch).item()
                train_commit = model.loss_commit(x_batch).item()
                train_som = model.loss_som(x_batch).item()
                train_commit_s = model.loss_commit_s(x_batch).item()
                train_som_s = model.loss_som_s(x_batch).item()

                mu_train, logvar_train = model._encode(x_batch)
                train_kl = model._kl_divergence_diag(mu_train, logvar_train).item()
                train_log_lik = train_elbo - model.prior * train_kl

                train_loss_commit = train_commit * model.gamma
                train_loss_som = train_som * model.beta

            # Validation loss and logging (safe, no side effects)
            batch_val, _, ii_v = next(val_gen)
            x_val = _np_to_torch(batch_val, device)
            p_val = torch.from_numpy(ppv[ii_v * batch_size: (ii_v + 1) * batch_size]).float().to(device)
            model.set_p(p_val)

            with torch.inference_mode():
                # Compute all component losses for validation
                test_loss = model.loss(x_val).item()
                test_elbo = model.loss_reconstruction_ze(x_val).item()
                test_commit = model.loss_commit(x_val).item()
                test_som = model.loss_som(x_val).item()
                test_commit_s = model.loss_commit_s(x_val).item()
                test_som_s = model.loss_som_s(x_val).item()

                # Get KL and log-likelihood separately
                mu_val, logvar_val = model._encode(x_val)
                test_kl = model._kl_divergence_diag(mu_val, logvar_val).item()
                test_log_lik = test_elbo - model.prior * test_kl

                # For progress display (weighted losses)
                elbo_loss = model.theta * test_elbo
                cah_loss = model.gamma * test_commit
                ssom_loss = model.beta * test_som

            test_losses.append(test_loss)

            if i % 100 == 0:
                step = model._global_step

                # Test writer - all TensorFlow scalars
                test_writer.add_scalar("loss/loss", test_loss, step)
                test_writer.add_scalar("loss/loss_elbo", elbo_loss, step)
                test_writer.add_scalar("loss/loss_commit", cah_loss, step)
                test_writer.add_scalar("loss/loss_som", ssom_loss, step)
                test_writer.add_scalar("loss_commit_s/loss_commit_s", test_commit_s, step)
                test_writer.add_scalar("loss_som_s/loss_som_s", test_som_s, step)
                test_writer.add_scalar("loss_reconstruction_ze/loss_reconstruction_kl", test_kl, step)
                test_writer.add_scalar("loss_reconstruction_ze/loss_reconstruction_log_lik_loss", test_log_lik, step)

                # Train writer - all TensorFlow scalars
                train_writer.add_scalar("loss/loss", train_loss, step)
                train_writer.add_scalar("loss/loss_elbo", train_elbo, step)
                train_writer.add_scalar("loss/loss_commit", train_loss_commit, step)
                train_writer.add_scalar("loss/loss_som", train_loss_som, step)
                train_writer.add_scalar("loss_commit_s/loss_commit_s", train_commit_s, step)
                train_writer.add_scalar("loss_som_s/loss_som_s", train_som_s, step)
                train_writer.add_scalar("loss_reconstruction_ze/loss_reconstruction_kl", train_kl, step)
                train_writer.add_scalar("loss_reconstruction_ze/loss_reconstruction_log_lik_loss", train_log_lik, step)

                # Ratios
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

            if i % 1000 == 0 and len(test_losses) > 0:
                test_losses_mean.append(np.mean(test_losses))
                test_losses = []

            model._global_step += 1
            pbar.update(1)

        # Save checkpoint per epoch
        torch.save({"model": model.state_dict()}, modelpath)

        # Optional evaluation every 10 epochs
        if val_epochs and ((epoch + 1) % 10 == 0):
            results = evaluate_model(model, generator, len_data_val)
            if results is None:
                return None

    # Final evaluation
    results = evaluate_model(model, generator, len_data_val)
    return results


@ex.capture
def evaluate_model(
    model,
    generator,
    len_data_val,
    batch_size,
    latent_dim,
    som_dim,
    learning_rate,
    alpha,
    gamma,
    beta,
    theta,
    epochs_pretrain,
    decay_factor,
    ex_name,
    data_set,
    validation,
    dropout,
    prior_var,
    prior,
    convolution,
):
    """
    Evaluates the trained model (NMI, AMI, Purity), preserving output format and files.
    """
    device = _get_device()
    model = model.to(device)
    model.eval()

    test_k_all = []
    labels_val_all = []
    print("Evaluation...")

    num_batches = len_data_val // batch_size
    val_gen = generator("val" if validation else "test", batch_size)

    with torch.no_grad():
        for _ in range(num_batches):
            batch_data, batch_labels, _ii = next(val_gen)
            labels_val_all.extend(batch_labels.tolist() if hasattr(batch_labels, "tolist") else list(batch_labels))
            x_val = _np_to_torch(batch_data, device)
            k_pred = model.k(x_val).cpu().numpy().tolist()
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
                0,  # epoch count not tracked here; keep placeholder to preserve format
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


@ex.automain
def main(
    latent_dim,
    som_dim,
    learning_rate,
    decay_factor,
    alpha,
    beta,
    gamma,
    theta,
    ex_name,
    more_runs,
    data_set,
    dropout,
    prior_var,
    convolution,
    prior,
    validation,
    epochs_pretrain,
    num_epochs,
    batch_size,
    random_seed,
    exp_output,
    exp_path,
):
    """
    Builds the model, prepares data, trains, and evaluates; variable names and workflow preserved.
    """
    # Seeding
    random.seed(random_seed)
    nprand.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    start = time.time()
    if not os.path.exists("../models"):
        os.mkdir("../models")

    # Dimensions for MNIST-like data
    input_length = 28
    input_channels = 28

    # Build model
    model = DPSOM(
        latent_dim=latent_dim,
        som_dim=som_dim,
        learning_rate=learning_rate,
        alpha=alpha,
        decay_factor=decay_factor,
        input_length=input_length,
        input_channels=input_channels,
        beta=beta,
        theta=theta,
        gamma=gamma,
        convolution=convolution,
        dropout=dropout,
        prior_var=prior_var,
        prior=prior,
    )

    # Load data (replace tf.keras datasets with torchvision tensors and then numpy)
    from torchvision import datasets

    if data_set == "MNIST":
        train_ds = datasets.MNIST(root="../data", train=True, download=True)
        test_ds = datasets.MNIST(root="../data", train=False, download=True)

        data_total = train_ds.data.numpy().astype(np.float32)  # [N,28,28]
        labels_total = train_ds.targets.numpy().astype(np.int64)
        data_test = test_ds.data.numpy().astype(np.float32)
        labels_test = test_ds.targets.numpy().astype(np.int64)
    else:
        train_ds = datasets.FashionMNIST(root="../data", train=True, download=True)
        test_ds = datasets.FashionMNIST(root="../data", train=False, download=True)

        data_total = train_ds.data.numpy().astype(np.float32)
        labels_total = train_ds.targets.numpy().astype(np.int64)
        data_test = test_ds.data.numpy().astype(np.float32)
        labels_test = test_ds.targets.numpy().astype(np.int64)

    # Match original per-sample normalization: divide by max pixel per image, then reshape to [N,28,28,1]
    data_total_flat = data_total.reshape(-1, 28 * 28)
    maxx = np.maximum(np.reshape(np.amax(data_total_flat, axis=-1), [-1, 1]), 1.0)
    data_total = (data_total_flat / maxx).reshape(-1, 28, 28, 1)

    data_test_flat = data_test.reshape(-1, 28 * 28)
    maxx_t = np.maximum(np.reshape(np.amax(data_test_flat, axis=-1), [-1, 1]), 1.0)
    data_test = (data_test_flat / maxx_t).reshape(-1, 28, 28, 1)

    data_train, data_val, labels_train, labels_val = train_test_split(
        data_total, labels_total, test_size=0.15, random_state=42
    )

    data_generator = get_data_generator(data_train, data_val, labels_train, labels_val, data_test, labels_test)
    if not validation:
        data_val = data_test

    # Training runs
    if more_runs:
        NMI = []
        PUR = []
        for _ in range(10):
            results = train_model(model, data_train, data_val, data_generator, lr_val=learning_rate)
            if results is None:
                continue
            NMI.append(results["NMI"])
            PUR.append(results["Purity"])

        NMI_mean = float(np.mean(NMI)) if len(NMI) > 0 else 0.0
        NMI_sd = float(np.std(NMI) / np.sqrt(max(len(NMI), 1)))
        PUR_mean = float(np.mean(PUR)) if len(PUR) > 0 else 0.0
        PUR_sd = float(np.std(PUR) / np.sqrt(max(len(PUR), 1)))

        print("\nRESULTS NMI: %f +- %f, PUR: %f +- %f. Name: %r.\n" % (NMI_mean, NMI_sd, PUR_mean, PUR_sd, ex_name))

        if data_set == "MNIST":
            f = open("evaluation_MNIST.txt", "a+")
        else:
            f = open("evaluation_fMNIST.txt", "a+")
        f.write(
            "som_dim=[%d,%d], latent_dim= %d, batch_size= %d, learning_rate= %f, theta= %f, "
            "dropout=%f, prior=%f, gamma=%f, beta%f, epochs_pretrain=%d, epochs= %d"
            % (
                som_dim[0],
                som_dim[1],
                latent_dim,
                batch_size,
                learning_rate,
                theta,
                dropout,
                prior,
                gamma,
                beta,
                epochs_pretrain,
                num_epochs,
            )
        )
        f.write(", RESULTS NMI: %f + %f, PUR: %f + %f. Name: %r \n" % (NMI_mean, NMI_sd, PUR_mean, PUR_sd, ex_name))
        f.close()
        results = {"NMI": NMI_mean, "Purity": PUR_mean, "AMI": 0.0}
    else:
        results = train_model(model, data_train, data_val, data_generator, lr_val=learning_rate)
        if results is not None:
            print("\n NMI: {}, AMI: {}, PUR: {}. Name: {}.\n".format(results["NMI"], results["AMI"], results["Purity"], ex_name))

        # Optional export
        if exp_output and results is not None:
            lock_path = os.path.join(
                exp_path,
                "exp_beta_{:.4f}_gamma_{:.4f}_bsize_{}_seed_{}_epochs_{}.LOCK".format(
                    beta, gamma, batch_size, random_seed, num_epochs
                ),
            )
            Path(lock_path).touch()
            out_path = os.path.join(
                exp_path, "exp_beta_{:.4f}_gamma_{:.4f}_bsize_{}_seed_{}_epochs_{}.tsv".format(beta, gamma, batch_size, random_seed, num_epochs)
            )
            with open(out_path, "w") as out_fp:
                csv_fp = csv.writer(out_fp, delimiter="\t")
                csv_fp.writerow(["DATASET", "NMI", "AMI", "Purity"])
                csv_fp.writerow([data_set, str(results["NMI"]), str(results["AMI"]), str(results["Purity"])])
            if os.path.exists(lock_path):
                os.remove(lock_path)

    elapsed_time_fl = (time.time() - start)
    print("\n Time: {}".format(elapsed_time_fl))
    return results
