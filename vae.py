import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(32, out_ch)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        h = self.conv1(F.silu(x))
        h = self.norm1(h)
        h = self.conv2(F.silu(h))
        h = self.norm2(h)
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, ch: int):
        # capture residual blocks and attention
        super().__init__()
        self.norm = nn.GroupNorm(32, ch)
        self.q = nn.Conv2d(ch, ch, 1)
        self.k = nn.Conv2d(ch, ch, 1)
        self.v = nn.Conv2d(ch, ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        h = self.norm(x)
        b, c, h_dim, w = h.shape
        q = self.q(h).reshape(b, c, h_dim * w).transpose(1, 2)
        k = self.k(h).reshape(b, c, h_dim * w)
        v = self.v(h).reshape(b, c, h_dim * w).transpose(1, 2)
        attn = torch.bmm(q, k) * (c ** -0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, v).transpose(1, 2).reshape(b, c, h_dim, w)
        return x + self.proj(out)


class Encoder(nn.Module):
    def __init__(self, in_ch: int = 3, latent_dim: int = 512,
                 ch_mult: Tuple[int, ...] = (1, 2, 4, 8)):
        super().__init__()
        base_ch = 128
        self.conv_in = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        self.down_blocks = nn.ModuleList()
        ch = base_ch
        for mult in ch_mult:
            out_ch = base_ch * mult
            self.down_blocks.append(nn.ModuleList([
                ResidualBlock(ch, out_ch),
                ResidualBlock(out_ch, out_ch),
                AttentionBlock(out_ch) if mult >= 4 else nn.Identity(),
                nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)
            ]))
            ch = out_ch
        self.mid_block1 = ResidualBlock(ch, ch)
        self.mid_attn = AttentionBlock(ch)
        self.mid_block2 = ResidualBlock(ch, ch)
        self.norm_out = nn.GroupNorm(32, ch)
        self.conv_mu = nn.Conv2d(ch, latent_dim, 1)
        self.conv_logvar = nn.Conv2d(ch, latent_dim, 1)

    def forward(self, x):
        h = self.conv_in(x)

        for res1, res2, attn, down in self.down_blocks:
            h = res1(h)
            h = res2(h)
            h = attn(h)
            h = down(h)

        h = self.mid_block1(h)
        h = self.mid_attn(h)
        h = self.mid_block2(h)

        h = self.norm_out(h)
        h = F.silu(h)

        mu = self.conv_mu(h)
        logvar = self.conv_logvar(h)

        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, out_ch: int = 3, latent_dim: int = 512,
                 ch_mult: Tuple[int, ...] = (8, 4, 2, 1)):
        # reconstruct from sample of latent space
        super().__init__()
        base_ch = 128
        ch = base_ch * ch_mult[0]
        self.conv_in = nn.Conv2d(latent_dim, ch, 3, padding=1)
        self.mid_block1 = ResidualBlock(ch, ch)
        self.mid_attn = AttentionBlock(ch)
        self.mid_block2 = ResidualBlock(ch, ch)
        self.up_blocks = nn.ModuleList()
        for mult in ch_mult:
            out_ch_block = base_ch * mult
            self.up_blocks.append(nn.ModuleList([
                ResidualBlock(ch, out_ch_block),
                ResidualBlock(out_ch_block, out_ch_block),
                AttentionBlock(out_ch_block) if mult >= 4 else nn.Identity(),
                nn.ConvTranspose2d(out_ch_block, out_ch_block, 4, stride=2, padding=1)
            ]))
            ch = out_ch_block
        self.norm_out = nn.GroupNorm(32, ch)
        self.conv_out = nn.Conv2d(ch, out_ch, 3, padding=1)

    def forward(self, z):
        h = self.conv_in(z)
        h = self.mid_block1(h)
        h = self.mid_attn(h)
        h = self.mid_block2(h)
        for res1, res2, attn, up in self.up_blocks:
            h = res1(h)
            h = res2(h)
            h = attn(h)
            h = up(h)
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return torch.tanh(h)


class VAE(nn.Module):
    def __init__(self, in_ch: int = 3, latent_dim: int = 512,
                 ch_mult: Tuple[int, ...] = (1, 2, 4, 8)):
        super().__init__()
        self.encoder = Encoder(in_ch, latent_dim, ch_mult)
        self.decoder = Decoder(in_ch, latent_dim, tuple(reversed(ch_mult)))
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        recon = self.decode(z)
        return recon, mu, logvar

    def sample(self, n: int, device: str = 'cuda', spatial_size: Tuple[int, int] = (8, 8)):
        z = torch.randn(n, self.latent_dim, *spatial_size).to(device)
        return self.decode(z)


def vae_loss(recon_x, x, mu, logvar, beta: float = 1.0):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = VAE(in_ch=3, latent_dim=512, ch_mult=(1, 2, 4, 8)).to(device)
    x = torch.randn(4, 3, 256, 256).to(device)
    recon, mu, logvar = model(x)
    loss = vae_loss(recon, x, mu, logvar, beta=1.0)
    print()
    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {recon.shape}")
    print(f"Latent mu shape: {mu.shape}")
    print(f"Loss: {loss.item():.4f}")
    print()
    samples = model.sample(n=4, device=device, spatial_size=(8, 8))
    print(f"Generated samples shape: {samples.shape}")

