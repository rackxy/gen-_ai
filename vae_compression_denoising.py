"""
Assignment 9: VAE for Compression and Denoising (MNIST)
--------------------------------------------------------
Trains a VAE for latent dimensions 2, 16 and 32 and, for each one:
  Part A: compression ratio (784 / latent_dim)
  Part B: reconstruction of 10 test images + per-image and mean MSE
  Part C: denoising of 10 noisy test images
  Part D: comparison across latent dimensions (results table)

Outputs:
  reconstruction_latent_<d>.png   original vs reconstructed (10 images)
  denoising_latent_<d>.png        clean vs noisy vs denoised (10 images)
  results_table.csv / printed table

If your Assignment 8 VAE is different, replace the VAE class below with it
(keep the forward() returning (recon, mu, logvar)).

Run:  pip install torch torchvision matplotlib pandas
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

EPOCHS = 15
BATCH = 128
NOISE_STD = 0.3
LATENT_DIMS = [2, 16, 32]

# ---------------- Data ----------------
tfm = T.ToTensor()  # pixels in [0, 1]
train_set = torchvision.datasets.MNIST("./data", train=True, download=True, transform=tfm)
test_set = torchvision.datasets.MNIST("./data", train=False, download=True, transform=tfm)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH, shuffle=True)

test_images = torch.stack([test_set[i][0] for i in range(10)]).to(device)  # 10 fixed test images


# ---------------- Model ----------------
class VAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU())
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 784), nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.enc(x.view(-1, 784))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(recon, x, mu, logvar):
    bce = F.binary_cross_entropy(recon, x.view(-1, 784), reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (bce + kld) / x.size(0)


def train_vae(latent_dim):
    model = VAE(latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(1, EPOCHS + 1):
        total = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            opt.zero_grad()
            recon, mu, logvar = model(x)
            loss = vae_loss(recon, x, mu, logvar)
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0)
        print(f"[latent={latent_dim}] epoch {epoch:02d}/{EPOCHS}  loss={total / len(train_set):.2f}")
    model.eval()
    return model


@torch.no_grad()
def reconstruct(model, x):
    """Deterministic reconstruction: encode to mu, decode (no sampling noise)."""
    mu, _ = model.encode(x)
    return model.decode(mu).view(-1, 1, 28, 28)


def mse_per_image(a, b):
    return ((a - b) ** 2).flatten(1).mean(dim=1)


def show_rows(rows, row_titles, path, title):
    n = rows[0].size(0)
    fig, axes = plt.subplots(len(rows), n, figsize=(n * 1.2, len(rows) * 1.4))
    for r, (imgs, name) in enumerate(zip(rows, row_titles)):
        for c in range(n):
            ax = axes[r, c]
            ax.imshow(imgs[c].cpu().squeeze(), cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if c == 0:
                ax.set_title(name, fontsize=8, loc="left")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.show()


# ---------------- Experiments ----------------
results = []
noisy_images = (test_images + NOISE_STD * torch.randn_like(test_images)).clamp(0, 1)

for d in LATENT_DIMS:
    model = train_vae(d)

    # Part A: compression ratio
    ratio = 784 / d

    # Part B: reconstruction + MSE
    recon = reconstruct(model, test_images)
    mse_recon = mse_per_image(recon, test_images)
    print(f"\n[latent={d}] per-image reconstruction MSE:", [round(v, 4) for v in mse_recon.tolist()])
    show_rows([test_images, recon], ["Original", "Reconstructed"],
              f"reconstruction_latent_{d}.png", f"Reconstruction (latent dim = {d})")

    # Part C: denoising (compare denoised output with the CLEAN image)
    denoised = reconstruct(model, noisy_images)
    mse_noisy = mse_per_image(noisy_images, test_images)
    mse_denoised = mse_per_image(denoised, test_images)
    show_rows([test_images, noisy_images, denoised], ["Clean", "Noisy", "Denoised"],
              f"denoising_latent_{d}.png", f"Denoising (latent dim = {d}, noise std = {NOISE_STD})")

    results.append({
        "Latent dim": d,
        "Compression ratio": f"{ratio:.1f}:1",
        "Recon MSE (mean of 10)": round(mse_recon.mean().item(), 4),
        "Noisy-vs-clean MSE": round(mse_noisy.mean().item(), 4),
        "Denoised-vs-clean MSE": round(mse_denoised.mean().item(), 4),
    })

# Part D: results table
df = pd.DataFrame(results)
df["Denoising improved?"] = df["Denoised-vs-clean MSE"] < df["Noisy-vs-clean MSE"]
print("\n=== RESULTS TABLE ===")
print(df.to_string(index=False))
df.to_csv("results_table.csv", index=False)
print("\nSaved results_table.csv and the comparison PNGs.")
print("Fill in the 'Reconstruction quality' and 'Denoising observations' columns "
      "by looking at the saved images (e.g. latent 2: blurry / digits often wrong; "
      "latent 16/32: sharp and close to the original).")
