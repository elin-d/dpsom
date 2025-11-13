import functools

import torch
import torch.nn as nn
import torch.nn.functional as F


def lazy_scope(function):
    """
    Preserves the original API surface by caching property-like calls.
    In PyTorch, this simply memorizes the method result on first access.
    Inspired by the original TF version. Variable names preserved.
    """
    attribute = "_cache_" + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


class DPSOM(nn.Module):
    """Class for the DPSOM model (PyTorch version)"""

    def __init__(
        self,
        latent_dim=100,
        som_dim=[8, 8],
        learning_rate=1e-4,
        decay_factor=0.99,
        decay_steps=1000,
        input_length=28,
        input_channels=28,
        alpha=10.0,
        beta=20.0,
        gamma=20.0,
        theta=1.0,
        dropout=0.5,
        prior_var=1,
        prior=0.5,
        convolution=False,
        toroidal=False
    ):
        """
        Initialization method for the DPSOM model object.
        All constructor arguments and attribute names preserved.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.som_dim = som_dim
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor
        self.decay_steps = decay_steps
        self.input_length = input_length
        self.input_channels = input_channels
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.theta = theta
        self.dropout = dropout
        self.prior_var = prior_var
        self.prior = prior
        self.convolution = convolution
        self.toroidal = toroidal

        # Prior parameters for KL (zero-mean, diagonal with prior_var)
        self._prior_loc = torch.zeros(self.latent_dim)
        self._prior_scale = torch.ones(self.latent_dim) * float(self.prior_var)

        # SOM embeddings: shape [H, W, latent_dim] as in TF
        H, W = self.som_dim[0], self.som_dim[1]
        self._embeddings = nn.Parameter(
            torch.empty(H, W, self.latent_dim)
        )
        # Truncated normal-like init (approx) to match tf.truncated_normal_initializer(stddev=0.05)
        with torch.no_grad():
            self._embeddings.normal_(mean=0.0, std=0.05)
            self._embeddings.clamp_(min=-0.1, max=0.1)

        # Encoder and decoder
        if not self.convolution:
            # Non-convolutional encoder
            self.enc_fc1 = nn.Linear(28*28, 500)
            self.enc_drop1 = nn.Dropout(dropout)
            self.enc_bn1 = nn.BatchNorm1d(500, eps=1e-3, momentum=0.01, affine=True, track_running_stats=True)

            self.enc_fc2 = nn.Linear(500, 500)
            self.enc_drop2 = nn.Dropout(dropout)
            self.enc_bn2 = nn.BatchNorm1d(500, eps=1e-3, momentum=0.01, affine=True, track_running_stats=True)

            self.enc_fc3 = nn.Linear(500, 2000)
            self.enc_drop3 = nn.Dropout(dropout)
            self.enc_bn3 = nn.BatchNorm1d(2000, eps=1e-3, momentum=0.01, affine=True, track_running_stats=True)

            self.enc_mu = nn.Linear(2000, latent_dim)
            self.enc_logvar = nn.Linear(2000, latent_dim)

            # decoder
            self.dec_fc4 = nn.Linear(latent_dim, 2000)
            self.dec_bn4 = nn.BatchNorm1d(2000, eps=1e-3, momentum=0.01, affine=True, track_running_stats=True)
            self.dec_fc3 = nn.Linear(2000, 500)
            self.dec_bn3 = nn.BatchNorm1d(500, eps=1e-3, momentum=0.01, affine=True, track_running_stats=True)
            self.dec_fc2 = nn.Linear(500, 500)
            self.dec_bn2 = nn.BatchNorm1d(500, eps=1e-3, momentum=0.01, affine=True, track_running_stats=True)
            self.dec_fc1 = nn.Linear(500, 28*28)
        else:
            # Convolutional encoder
            self.enc_conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
            self.enc_bn1 = nn.BatchNorm2d(32, eps=1e-3, momentum=0.01, affine=True, track_running_stats=True)
            self.enc_pool1 = nn.MaxPool2d(2)
            self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.enc_bn2 = nn.BatchNorm2d(64, eps=1e-3, momentum=0.01, affine=True, track_running_stats=True)
            self.enc_pool2 = nn.MaxPool2d(2)
            self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.enc_bn3 = nn.BatchNorm2d(128, eps=1e-3, momentum=0.01, affine=True, track_running_stats=True)
            self.enc_conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
            self.enc_bn4 = nn.BatchNorm2d(256, eps=1e-3, momentum=0.01, affine=True, track_running_stats=True)

            flat_size = 7 * 7 * 256
            self.enc_mu = nn.Linear(flat_size, self.latent_dim)
            self.enc_logvar = nn.Linear(flat_size, self.latent_dim)

            # Convolutional decoder (transpose-conv + upsampling)
            self.dec_fc = nn.Linear(self.latent_dim, flat_size)
            self.dec_deconv5 = nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1)
            self.dec_bn5 = nn.BatchNorm2d(128, eps=1e-3, momentum=0.01)
            self.dec_deconv6 = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
            self.dec_bn6 = nn.BatchNorm2d(64, eps=1e-3, momentum=0.01)
            self.dec_deconv7 = nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1)
            self.dec_bn7 = nn.BatchNorm2d(32, eps=1e-3, momentum=0.01)
            self.dec_conv_out = nn.Conv2d(32, 1, kernel_size=3, padding=1)
            self.dec_up = nn.Upsample(scale_factor=2, mode="nearest")

        # Training state and counters (preserve names)
        self._global_step = 0     # mirrors TF global_step

        # Placeholder-like attributes to preserve names used in the script
        self._p_tensor = None  # target distribution p (set externally each batch)
        self._z_placeholder = None  # kept as stub for API compatibility

        # Expose attributes names to mirror original file (lazy properties)
        self.input_dim
        self.embeddings
        self.global_step
        self.z_e
        self.sample_z_e
        self.z_dist_flat
        self.k
        self.z_q
        self.z_q_neighbors
        self.reconstruction_e
        self.loss_reconstruction_ze
        self.p
        self.q
        self.loss_commit
        self.loss_commit_s
        self.loss_som_s
        self.loss_som
        self.loss
        self.loss_a
        self.optimize  # will be used externally in the training script

    @lazy_scope
    def input_dim(self):
        return 28 * 28

    @lazy_scope
    def embeddings(self):
        return self._embeddings

    @lazy_scope
    def global_step(self):
        return self._global_step

    # ====== Encoder/Decoder helpers ======

    def _encode(self, x):
        """
        x: tensor [B, 28, 28, 1] or [B,1,28,28]
        Returns mu, logvar
        """
        if x.dim() == 4 and x.shape[1] == 1 and x.shape[2] == 28:
            x = x.permute(0, 2, 3, 1)  # [B,28,28,1] for uniform handling
        if not self.convolution:
            flat = x.view(x.size(0), -1)
            h = F.leaky_relu(self.enc_fc1(flat), 0.2)
            h = self.enc_drop1(h)
            h = self.enc_bn1(h)

            h = F.leaky_relu(self.enc_fc2(h), 0.2)
            h = self.enc_drop2(h)
            h = self.enc_bn2(h)

            h = F.leaky_relu(self.enc_fc3(h), 0.2)
            h = self.enc_drop3(h)
            h = self.enc_bn3(h)

            mu = self.enc_mu(h)
            logvar = self.enc_logvar(h)
        else:
            # Expect [B,1,28,28]
            if x.shape[-1] == 1 and x.shape[1] != 1:
                # [B,28,28,1] -> [B,1,28,28]
                x = x.permute(0, 3, 1, 2).contiguous()
            x = x.float() - 0.5
            h = F.leaky_relu(self.enc_conv1(x), negative_slope=0.2)
            h = self.enc_bn1(h)
            h = self.enc_pool1(h)
            h = F.leaky_relu(self.enc_conv2(h), negative_slope=0.2)
            h = self.enc_bn2(h)
            h = self.enc_pool2(h)
            h = F.leaky_relu(self.enc_conv3(h), negative_slope=0.2)
            h = self.enc_bn3(h)
            h = F.leaky_relu(self.enc_conv4(h), negative_slope=0.2)
            h = self.enc_bn4(h)
            h = h.reshape(h.shape[0], -1)
            mu = self.enc_mu(h)
            logvar = self.enc_logvar(h)
        return mu, logvar

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _decode_logits(self, z):
        if not self.convolution:
            h = F.leaky_relu(self.dec_fc4(z), negative_slope=0.2)
            h = self.dec_bn4(h)
            h = F.leaky_relu(self.dec_fc3(h), negative_slope=0.2)
            h = self.dec_bn3(h)
            h = F.leaky_relu(self.dec_fc2(h), negative_slope=0.2)
            h = self.dec_bn2(h)
            logits = self.dec_fc1(h)  # [B, 784]
            return logits
        else:
            h = self.dec_fc(z)
            h = h.view(z.shape[0], 256, 7, 7)
            h = F.leaky_relu(self.dec_deconv5(h), negative_slope=0.2)
            h = self.dec_bn5(h)
            h = F.leaky_relu(self.dec_deconv6(h), negative_slope=0.2)
            h = self.dec_bn6(h)
            h = self.dec_up(h)
            h = F.leaky_relu(self.dec_deconv7(h), negative_slope=0.2)
            h = self.dec_bn7(h)
            h = self.dec_up(h)
            logits_map = self.dec_conv_out(h)  # [B,1,28,28]
            logits = logits_map.view(z.shape[0], 28 * 28)
            return logits

    # ===== Properties and methods with original names =====

    @lazy_scope
    def z_e(self):
        # Cache struct for mu/logvar
        return {"mu": None, "logvar": None}

    def sample_z_e(self, x):
        """
        Sample from q(z|x) and store mu/logvar under the same name structure.
        """
        mu, logvar = self._encode(x)
        z = self._reparameterize(mu, logvar)
        self._cache_z_e = {"mu": mu, "logvar": logvar}
        return z  # [B, D]

    def z_dist_flat(self, x, stop_grad=False):
        """
        Compute squared distances between embeddings and latent samples.
        Returns [B, H*W].
        """
        z = self.sample_z_e(x)
        if stop_grad:
            z = z.detach()
        H, W = self.som_dim[0], self.som_dim[1]
        E = self.embeddings  # [H, W, D]
        z_exp = z[:, None, None, :]  # [B,1,1,D]
        diff = z_exp - E[None, :, :, :]  # [B,H,W,D]
        z_dist = (diff * diff).sum(dim=-1)  # [B,H,W]
        z_dist_flat = z_dist.view(z_dist.shape[0], H * W)  # [B, H*W]
        return z_dist_flat

    def z_dist_flat_ng(self, x):
        return self.z_dist_flat(x, stop_grad=True)

    def k(self, x):
        """
        Pick index of closest centroid for every embedding. Returns LongTensor [B]
        """
        dist_flat = self.z_dist_flat(x)  # [B, H*W]
        return torch.argmin(dist_flat, dim=-1)

    def z_q(self, x):
        """
        Aggregate the closest centroid for every embedding. Returns [B, D]

        For each sample in the batch, finds the nearest SOM centroid
        and returns its embedding vector.

        Convention:
            som_dim = [H, W] (H=rows, W=columns)
            Linear indexing: flat_idx = row * W + col
        """
        H, W = self.som_dim[0], self.som_dim[1]
        k = self.k(x)  # [B]
        k1 = k // W
        k2 = k % W
        # Gather the centroid embeddings [H, W, D]
        gathered = self.embeddings[k1, k2, :]  # [B, D]
        return gathered

    def z_q_neighbors(self, x):
        """
        Aggregate neighbor centroids in the SOM grid for each z_q.
        Returns [B, 5, D] = [center, up, down, right, left]
        """
        H, W = self.som_dim[0], self.som_dim[1]
        k = self.k(x)  # [B]
        k1 = k // W  # Row index
        k2 = k % W  # Column index
        if self.toroidal:
            k1_up = (k1 + 1) % H  # Wraps at bottom
            k1_down = (k1 - 1) % H  # Wraps at top
            k2_right = (k2 + 1) % W  # Wraps at left
            k2_left = (k2 - 1) % W  # Wraps at right
        else:
            k1_not_top = (k1 < (H - 1))
            k1_not_bottom = (k1 > 0)
            k2_not_right = (k2 < (W - 1))
            k2_not_left = (k2 > 0)

            k1_up = torch.where(k1_not_top, k1 + 1, torch.zeros_like(k1))
            k1_down = torch.where(k1_not_bottom, k1 - 1, torch.ones_like(k1) * (H - 1))
            k2_right = torch.where(k2_not_right, k2 + 1, torch.zeros_like(k2))
            k2_left = torch.where(k2_not_left, k2 - 1, torch.ones_like(k2) * (W - 1))

        z_center = self.embeddings[k1, k2, :]  # [B, D]
        z_up = self.embeddings[k1_up, k2, :]  # [B, D]
        z_down = self.embeddings[k1_down, k2, :]  # [B, D]
        z_right = self.embeddings[k1, k2_right, :]  # [B, D]
        z_left = self.embeddings[k1, k2_left, :]  # [B, D]

        return torch.stack([z_center, z_up, z_down, z_right, z_left], dim=1)  # [B, 5, D]

    def reconstruction_e(self, x):
        """
        Reconstruct input from encodings with Bernoulli likelihood over pixels.
        Returns logits [B, 784].
        """
        z = self.sample_z_e(x)
        logits = self._decode_logits(z)  # [B,784]
        return logits

    def _kl_divergence_diag(self, mu, logvar):
        """
        KL(q || p) with q ~ N(mu, diag(exp(logvar))), p ~ N(0, diag(prior_var))
        Per-sample KL, then mean over batch.
        """
        prior_scale = self._prior_scale.to(mu.device)
        prior_var = prior_scale.pow(2)

        var_q = torch.exp(logvar)
        kl = 0.5 * (
            (var_q / prior_var).sum(dim=1)
            + ((mu ** 2) / prior_var).sum(dim=1)
            - self.latent_dim
            + (prior_var.log().sum() - logvar.sum(dim=1))
        )
        return kl.mean()

    def loss_reconstruction_ze(self, x):
        """
        Compute ELBO: negative log-likelihood + prior * KL
        Bernoulli likelihood with logits; inputs expected in [0,1].
        """
        mu, logvar = self._encode(x)
        z = self._reparameterize(mu, logvar)
        logits = self._decode_logits(z)  # [B,784]
        x_flat = x.view(x.shape[0], -1).clamp(0.0, 1.0)

        loss_pix = F.binary_cross_entropy_with_logits(logits, x_flat, reduction="none")  # [B, 784]
        log_lik_loss = loss_pix.sum(dim=1).mean()  # scala

        kl_loss = self._kl_divergence_diag(mu, logvar)
        loss_rec = log_lik_loss + self.prior * kl_loss
        return loss_rec

    def q(self, x, detach_z=False):
        """
        Soft assignments between embeddings and centroids (Student's t-kernel).
        Returns [B, H*W].
        """
        if detach_z:
            dist_flat = self.z_dist_flat_ng(x)
        else:
            dist_flat = self.z_dist_flat(x)
        eps = torch.finfo(dist_flat.dtype).eps
        q = eps + 1.0 / torch.pow(1.0 + dist_flat / self.alpha, (self.alpha + 1.0) / 2.0)
        q = q / q.sum(dim=1, keepdim=True)
        return q

    @property
    def p(self):
        """
        Placeholder-like attribute for target distribution p.
        In PyTorch, this is set externally per batch (tensor [B, H*W]).
        """
        return self._p_tensor

    def set_p(self, p_tensor):
        """
        Helper to set the 'p' tensor externally to match TF placeholder usage.
        """
        self._p_tensor = p_tensor

    def loss_commit(self, x, eps: float = 1e-8):
        """
        Commitment loss: KL(P || Q) with per-sample sum, then batch mean.
        Epsilon is applied by clamping p and q before log to match TF behavior.
        """
        q = self.q(x)                          # [B, K], soft assignments
        p = self.p                              # [B, K], set via set_p() beforehand
        p = p.to(q.dtype).to(q.device)
        # Clamp BEFORE logs to avoid -inf and match TF clip_by_value placement
        p_safe = p.clamp_min(eps)
        q_safe = q.clamp_min(eps)
        # Per-sample KL(P||Q): sum over K, then mean over batch
        per_sample_kl = (p_safe * (p_safe.log() - q_safe.log())).sum(dim=1)
        return per_sample_kl.mean()

    def target_distribution(self, q_np):
        """
        Compute target distribution from soft assignments (on numpy, to match original usage).
        """
        p = q_np ** 2 / (q_np.sum(axis=0, keepdims=False))
        p = p / p.sum(axis=1, keepdims=True)
        return p

    def loss_som(self, x):
        """
        SOM neighbor regularization using detached q for neighbors.
        Uses row-major indexing with toroidal topology (modulo wrap).
        """
        H, W = self.som_dim[0], self.som_dim[1]
        k = torch.arange(H * W, device=self.embeddings.device, dtype=torch.long)  # [H*W]

        # Row-major indexing
        k1 = k // W
        k2 = k % W

        # Toroidal neighbor wrap (modulo)
        k1_up = (k1 + 1) % H
        k1_down = (k1 - 1) % H
        k2_right = (k2 + 1) % W
        k2_left = (k2 - 1) % W

        k_up = k1_up * W + k2
        k_down = k1_down * W + k2
        k_right = k1 * W + k2_right
        k_left = k1 * W + k2_left

        q_t = self.q(x, detach_z=True).transpose(0, 1)  # [H*W, B]
        q_up = q_t[k_up].transpose(0, 1)
        q_down = q_t[k_down].transpose(0, 1)
        q_right = q_t[k_right].transpose(0, 1)
        q_left = q_t[k_left].transpose(0, 1)
        q_neighbours = torch.stack([q_up, q_down, q_right, q_left], dim=2)  # [B, H*W, 4]

        eps = torch.finfo(q_neighbours.dtype).eps
        q_neighbours_log_sum = torch.sum(torch.log(q_neighbours + eps), dim=-1)  # [B, H*W]

        new_q = self.q(x)  # [B, H*W]
        q_n = q_neighbours_log_sum * new_q.detach()
        q_n = torch.sum(q_n, dim=-1)  # [B]
        qq = -torch.mean(q_n)
        return qq

    def loss(self, x):
        """
        Aggregate total loss.
        """
        return self.theta * self.loss_reconstruction_ze(x) + self.gamma * self.loss_commit(x) + self.beta * self.loss_som(x)

    def loss_commit_s(self, x):
        """
        Commitment loss for standard SOM initialization: ||z - z_q||^2
        """
        z = self.sample_z_e(x).detach()
        zq = self.z_q(x)
        return torch.mean((z - zq) ** 2)

    def loss_som_s(self, x):
        """
        SOM neighbor loss for standard SOM initialization: ||z - neighbors||^2
        """
        z = self.sample_z_e(x).detach().unsqueeze(1)  # [B,1,D]
        z_ngb = self.z_q_neighbors(x)  # [B,5,D]
        return torch.mean((z - z_ngb) ** 2)

    def loss_a(self, x):
        """
        Clustering loss of standard SOM used for initialization.
        """
        return self.loss_som_s(x) + self.loss_commit_s(x)

    @lazy_scope
    def optimize(self):
        """
        Placeholder to preserve original attribute name; in PyTorch the training
        script will construct optimizers and learning-rate scheduling.
        """
        return None

    # Utility to sync prior buffers to device
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        device = kwargs.get("device", args[0] if len(args) > 0 else None)
        if device is not None:
            self._prior_loc = self._prior_loc.to(device)
            self._prior_scale = self._prior_scale.to(device)
        return self

    def train(self, mode: bool = True):
        return super().train(mode)

    def eval(self):
        return super().eval()
