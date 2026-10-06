"""
Assignment 7: Creating and Fixing Mode Collapse in a DCGAN
-----------------------------------------------------------
Runs two experiments back to back on MNIST (swap the dataset if Assignment 4
used a different one):

  1. BROKEN : tiny generator, NO batch norm, very high learning rate
  2. FIXED  : normal generator + batch norm, one-sided label smoothing,
              different learning rates for G and D (TTUR)

Outputs:
  broken_output.png   -> screenshot for "collapsed output"
  fixed_output.png    -> screenshot for "improved output"
  diversity score printed for both (higher = more varied images)

Run in Google Colab (GPU) or locally:  pip install torch torchvision matplotlib
"""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

Z_DIM = 100
IMG_SIZE = 32
BATCH = 128
EPOCHS = 20

# ---------------- Data ----------------
tfm = T.Compose([T.Resize(IMG_SIZE), T.ToTensor(), T.Normalize((0.5,), (0.5,))])
data = torchvision.datasets.MNIST("./data", train=True, download=True, transform=tfm)
loader = torch.utils.data.DataLoader(data, batch_size=BATCH, shuffle=True, drop_last=True)


# ---------------- Models ----------------
def make_generator(ngf, use_bn):
    def bn(c):
        return nn.BatchNorm2d(c) if use_bn else nn.Identity()
    return nn.Sequential(
        nn.ConvTranspose2d(Z_DIM, ngf * 4, 4, 1, 0, bias=False), bn(ngf * 4), nn.ReLU(True),   # 4x4
        nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False), bn(ngf * 2), nn.ReLU(True),  # 8x8
        nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False), bn(ngf), nn.ReLU(True),          # 16x16
        nn.ConvTranspose2d(ngf, 1, 4, 2, 1, bias=False), nn.Tanh(),                              # 32x32
    )


def make_discriminator(ndf=64):
    return nn.Sequential(
        nn.Conv2d(1, ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, True),                         # 16
        nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, True),  # 8
        nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, True),  # 4
        nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False), nn.Sigmoid(),
    )


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


# ---------------- Training ----------------
def train(name, ngf, use_bn, lr_g, lr_d, real_label, epochs=EPOCHS):
    G = make_generator(ngf, use_bn).to(device)
    D = make_discriminator().to(device)
    G.apply(init_weights)
    D.apply(init_weights)

    opt_g = torch.optim.Adam(G.parameters(), lr=lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=lr_d, betas=(0.5, 0.999))
    bce = nn.BCELoss()
    fixed_noise = torch.randn(64, Z_DIM, 1, 1, device=device)

    for epoch in range(1, epochs + 1):
        for real, _ in loader:
            real = real.to(device)
            b = real.size(0)

            # --- Discriminator step ---
            D.zero_grad()
            real_targets = torch.full((b,), real_label, device=device)  # smoothing if < 1.0
            fake_targets = torch.zeros(b, device=device)
            loss_real = bce(D(real).view(-1), real_targets)

            noise = torch.randn(b, Z_DIM, 1, 1, device=device)
            fake = G(noise)
            loss_fake = bce(D(fake.detach()).view(-1), fake_targets)
            (loss_real + loss_fake).backward()
            opt_d.step()

            # --- Generator step ---
            G.zero_grad()
            loss_g = bce(D(fake).view(-1), torch.ones(b, device=device))
            loss_g.backward()
            opt_g.step()

        print(f"[{name}] epoch {epoch:02d}/{epochs}  "
              f"D_loss={(loss_real + loss_fake).item():.3f}  G_loss={loss_g.item():.3f}")

    G.eval()
    with torch.no_grad():
        samples = G(fixed_noise).cpu()
    return samples


def diversity_score(samples):
    """Mean pairwise L2 distance between generated images.
    Near 0 => all images look the same (mode collapse)."""
    flat = samples.view(samples.size(0), -1)
    return torch.cdist(flat, flat).mean().item()


def save_grid(samples, title, path):
    grid = make_grid(samples, nrow=8, normalize=True, padding=2)
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title(title)
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.show()


if __name__ == "__main__":
    # real-data diversity for reference
    real_batch = next(iter(loader))[0][:64]
    print(f"Diversity of REAL images: {diversity_score(real_batch):.2f}\n")

    # ===== Experiment 1: BROKEN model =====
    # - small generator (ngf=8)      -> reduced capacity
    # - no batch norm in generator   -> unstable training
    # - lr = 0.01 for both           -> far too high
    broken = train("BROKEN", ngf=8, use_bn=False, lr_g=0.01, lr_d=0.01, real_label=1.0)
    save_grid(broken, "BROKEN: no BatchNorm, small G, lr=0.01", "broken_output.png")
    print(f"Diversity (broken): {diversity_score(broken):.2f}\n")

    # ===== Experiment 2: FIXED model =====
    # - BatchNorm restored, normal capacity (ngf=64)
    # - one-sided label smoothing: real label = 0.9
    # - different learning rates: G=0.0001, D=0.0004 (TTUR)
    fixed = train("FIXED", ngf=64, use_bn=True, lr_g=1e-4, lr_d=4e-4, real_label=0.9)
    save_grid(fixed, "FIXED: BatchNorm + label smoothing + different LRs", "fixed_output.png")
    print(f"Diversity (fixed):  {diversity_score(fixed):.2f}")
