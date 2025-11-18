import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.register_buffer("_prior_loc", torch.zeros(self.latent_dim))
        self.register_buffer("_prior_scale", torch.ones(self.latent_dim) * float(self.prior_var))

        # SOM embeddings: shape [H, W, latent_dim] as in TF
        H, W = self.som_dim[0], self.som_dim[1]
        self._embeddings = nn.Parameter(torch.empty(H, W, self.latent_dim))

        # Truncated normal-like init (approx) to match tf.truncated_normal_initializer(stddev=0.05)
        with torch.no_grad():
            self._embeddings.normal_(mean=0.0, std=0.05)
            self._embeddings.clamp_(min=-0.1, max=0.1)

        # Encoder and decoder
        if not self.convolution:
            # Non-convolutional encoder
            self.enc_fc1 = nn.Linear(28 * 28, 500)
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

            # Decoder
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

        self._train_epoch = 0

        self._p_tensor = None  # target distribution p (set externally each batch)
        self.mu = None
        self.logvar = None

    def get_epoch(self):
        return self._train_epoch

    def inc_epoch(self):
        self._train_epoch += 1

    # ====== Encoder/Decoder helpers ======

    def _encode(self, x):
        """
        x: tensor [B, 28, 28, 1] or [B,1,28,28]
        Returns mu, logvar
        """
        if x.dim() == 4 and x.shape[1] == 1 and x.shape[2] == 28:
            x = x.permute(0, 2, 3, 1).contiguous()  # [B,28,28,1] for uniform handling
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

    def _decode(self, z):
        if not self.convolution:
            h = F.leaky_relu(self.dec_fc4(z), negative_slope=0.2)
            h = self.dec_bn4(h)
            h = F.leaky_relu(self.dec_fc3(h), negative_slope=0.2)
            h = self.dec_bn3(h)
            h = F.leaky_relu(self.dec_fc2(h), negative_slope=0.2)
            h = self.dec_bn2(h)
            logits_linear = self.dec_fc1(h)
            logits = F.leaky_relu(logits_linear, negative_slope=0.2)
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

    # ====== Latent sampling and distances ======

    def forward(self, x):
        mu, logvar = self._encode(x)
        eps = torch.randn_like(mu)
        z = mu + eps * torch.exp(0.5 * logvar)
        self.mu = mu
        self.logvar = logvar
        self.logits = self._decode(z)
        return z

    def z_dist_flat(self, z):
        """
        Compute squared distances between embeddings and latent samples.
        Returns [B, H*W].
        """
        H, W = self.som_dim[0], self.som_dim[1]
        E = self._embeddings  # [H, W, D]
        z_exp = z[:, None, None, :]  # [B,1,1,D]
        diff = z_exp - E[None, :, :, :]  # [B,H,W,D]
        z_dist = (diff * diff).sum(dim=-1)  # [B,H,W]
        z_dist_flat = z_dist.view(z_dist.shape[0], H * W)  # [B, H*W]
        return z_dist_flat


    def k(self, z):
        """
        Pick index of closest centroid for every embedding. Returns LongTensor [B]
        """
        dist_flat = self.z_dist_flat(z)  # [B, H*W]
        return torch.argmin(dist_flat, dim=-1)

    def z_q(self, z):
        """
        Aggregate closest centroid for every embedding. Returns [B, D]

        For each sample in the batch, finds the nearest SOM centroid
        and returns its embedding vector.

        Convention:
            som_dim = [H, W] (H=rows, W=columns)
            Linear indexing: flat_idx = row * W + col
        """
        H, W = self.som_dim[0], self.som_dim[1]
        k = self.k(z)  # [B]
        k1 = k // W
        k2 = k % W
        gathered = self._embeddings[k1, k2, :]  # [B, D]
        return gathered

    def som_neighbors(self, k):
        """
        SOM neighbors (up, down, right, left) for k.
        k: LongTensor [N]
        """
        H, W = self.som_dim[0], self.som_dim[1]

        k1 = k // W
        k2 = k % W

        if self.toroidal:
            k1_up = torch.remainder(k1 + 1, H)
            k1_down = torch.remainder(k1 - 1, H)
            k2_right = torch.remainder(k2 + 1, W)
            k2_left = torch.remainder(k2 - 1, W)
        else:
            k1_not_top = (k1 < (H - 1))
            k1_not_bottom = (k1 > 0)
            k2_not_right = (k2 < (W - 1))
            k2_not_left = (k2 > 0)

            k1_up = torch.where(k1_not_top, k1 + 1, torch.zeros_like(k1))
            k1_down = torch.where(k1_not_bottom, k1 - 1, torch.ones_like(k1) * (H - 1))
            k2_right = torch.where(k2_not_right, k2 + 1, torch.zeros_like(k2))
            k2_left = torch.where(k2_not_left, k2 - 1, torch.ones_like(k2) * (W - 1))

        k_up = k1_up * W + k2
        k_down = k1_down * W + k2
        k_right = k1 * W + k2_right
        k_left = k1 * W + k2_left

        return k_up, k_down, k_right, k_left

    def z_q_neighbors(self, z):
        """
        Aggregate neighbor centroids in the SOM grid for each z_q.
        Returns [B, 5, D] = [center, up, down, right, left]
        """
        H, W = self.som_dim[0], self.som_dim[1]
        k = self.k(z)  # [B]

        k_up, k_down, k_right, k_left = self.som_neighbors(k)

        k1 = k // W
        k2 = k % W
        k1_up = k_up // W
        k2_up = k_up % W
        k1_down = k_down // W
        k2_down = k_down % W
        k1_right = k_right // W
        k2_right = k_right % W
        k1_left = k_left // W
        k2_left = k_left % W

        z_center = self._embeddings[k1, k2, :]  # [B, D]
        z_up = self._embeddings[k1_up, k2_up, :]
        z_down = self._embeddings[k1_down, k2_down, :]
        z_right = self._embeddings[k1_right, k2_right, :]
        z_left = self._embeddings[k1_left, k2_left, :]

        return torch.stack([z_center, z_up, z_down, z_right, z_left], dim=1)  # [B,5,D]

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

        mu = self.mu
        logvar = self.logvar
        logits = self.logits # [B,784]
        x_flat = x.view(x.shape[0], -1).clamp(0.0, 1.0)

        loss_pix = F.binary_cross_entropy_with_logits(logits, x_flat, reduction="none")  # [B, 784]
        log_lik_loss = loss_pix.sum(dim=1).mean()  # scala

        kl_loss = self._kl_divergence_diag(mu, logvar)
        loss_rec = log_lik_loss + self.prior * kl_loss
        return loss_rec

    def q(self, z):
        """
        Soft assignments between embeddings and centroids (Student's t-kernel).
        Returns [B, H*W].
        """
        dist_flat = self.z_dist_flat(z)
        eps = torch.finfo(dist_flat.dtype).eps
        q = eps + 1.0 / torch.pow(1.0 + dist_flat / self.alpha, (self.alpha + 1.0) / 2.0)
        q = q / q.sum(dim=1, keepdim=True)
        return q

    def q_p(self, x):
        """
        Soft assignments between embeddings and centroids (Student's t-kernel).
        Returns [B, H*W].
        """
        mu, logvar = self._encode(x)
        eps = torch.randn_like(mu)
        z = mu + eps * torch.exp(0.5 * logvar)
        H, W = self.som_dim[0], self.som_dim[1]
        E = self._embeddings  # [H, W, D]
        z_exp = z[:, None, None, :]  # [B,1,1,D]
        diff = z_exp - E[None, :, :, :]  # [B,H,W,D]
        z_dist = (diff * diff).sum(dim=-1)  # [B,H,W]
        z_dist_flat = z_dist.view(z_dist.shape[0], H * W)  # [B, H*W]
        eps = torch.finfo(z_dist_flat.dtype).eps
        q = eps + 1.0 / torch.pow(1.0 + z_dist_flat / self.alpha, (self.alpha + 1.0) / 2.0)
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

    def loss_commit(self, z, eps: float = 1e-10):
        """
        Commitment loss: KL(P || Q) with per-sample sum, then batch mean.
        """
        q = self.q(z)                          # [B, K], soft assignments
        p = self.p                              # [B, K], set via set_p() beforehand
        p = p.to(q.dtype).to(q.device)
        # Clamp BEFORE logs to avoid -inf and match TF clip_by_value placement
        p_safe = p.clamp_min(eps)
        q_safe = q.clamp_min(eps)
        # Per-sample KL(P||Q): sum over K, then mean over batch
        per_sample_kl = (p_safe * (p_safe.log() - q_safe.log())).sum(dim=1)
        return per_sample_kl.mean()

    @torch.no_grad()
    def target_distribution(self, q_np):
        """
        Compute target distribution from soft assignments (on numpy, to match original usage).
        """
        p = q_np ** 2 / (q_np.sum(axis=0, keepdims=False))
        p = p / p.sum(axis=1, keepdims=True)
        return p

    def loss_som(self, z):
        """
        SOM neighbor regularization using detached q for neighbors.
        """
        H, W = self.som_dim[0], self.som_dim[1]

        k = torch.arange(H * W, device=self._embeddings.device, dtype=torch.long)  # [H*W]
        k_up, k_down, k_right, k_left = self.som_neighbors(k)
        z_ng = z.detach()
        q_t = self.q(z_ng).transpose(0, 1)  # [H*W, B]
        q_up = q_t[k_up].transpose(0, 1)
        q_down = q_t[k_down].transpose(0, 1)
        q_right = q_t[k_right].transpose(0, 1)
        q_left = q_t[k_left].transpose(0, 1)
        q_neighbours = torch.stack([q_up, q_down, q_right, q_left], dim=2)  # [B, H*W, 4]

        eps = torch.finfo(q_neighbours.dtype).eps
        q_neighbours_log_sum = torch.sum(torch.log(q_neighbours + eps), dim=-1)  # [B, H*W]

        new_q = self.q(z)  # [B, H*W]
        q_n = q_neighbours_log_sum * new_q.detach()
        q_n = torch.sum(q_n, dim=-1)  # [B]
        qq = -torch.mean(q_n)
        return qq

    def loss(self, x, z):
        """
        Aggregate total loss.
        """
        a = self.theta * self.loss_reconstruction_ze(x)
        b = self.gamma * self.loss_commit(z)
        c = self.beta * self.loss_som(z)
        return a + b + c, a, b, c

    def loss_commit_s(self, z_ng):
        """
        Commitment loss for standard SOM initialization: ||z - z_q||^2
        """
        zq = self.z_q(z_ng)
        return torch.mean((z_ng - zq) ** 2)

    def loss_som_s(self, z_ng):
        """
        SOM neighbor loss for standard SOM initialization: ||z - neighbors||^2
        """
        z = z_ng.unsqueeze(1)  # [B,1,D]
        z_ngb = self.z_q_neighbors(z_ng)  # [B,5,D]
        return torch.mean((z - z_ngb) ** 2)

    def loss_a(self, z):
        """
        Clustering loss of standard SOM used for initialization.
        """
        z_ng = z.detach()
        a = self.loss_som_s(z_ng)
        b = self.loss_commit_s(z_ng)
        return a + b, a, b
