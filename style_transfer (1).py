"""
Assignment 6 - CycleGAN vs Neural Style Transfer (VGG19)

Usage:
    python style_transfer.py photo.jpg style.jpg

photo.jpg : your content image (landscape / building / object)
style.jpg : a painting (e.g. Starry Night)

Needs: torch, torchvision, pillow, matplotlib, git (GPU recommended;
works fine in Google Colab: run with  !python style_transfer.py photo.jpg style.jpg )

Output: comparison.png  (original | CycleGAN | Neural Style Transfer)
"""
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

PHOTO = sys.argv[1] if len(sys.argv) > 1 else "photo.jpg"
STYLE = sys.argv[2] if len(sys.argv) > 2 else "style.jpg"
SIZE = 512
ROOT = os.path.abspath(".")


def run(cmd, cwd=None):
    print(">>", cmd)
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)


# ---------------------------------------------------------------
# Step 1: prepare the content image (512x512, divisible by 4)
# ---------------------------------------------------------------
os.makedirs("input_dir", exist_ok=True)
img = Image.open(PHOTO).convert("RGB").resize((SIZE, SIZE))
img.save("input_dir/photo.jpg")
img.save("content_512.png")

# ---------------------------------------------------------------
# Step 2: CycleGAN - pretrained photo -> Monet
# (cycle consistency: G(photo)->monet, F(monet)->photo, F(G(x)) ~ x)
# ---------------------------------------------------------------
repo = "pytorch-CycleGAN-and-pix2pix"
if not os.path.isdir(repo):
    run(f"git clone -q https://github.com/junyanz/{repo}")
run("pip install -q dominate")
run("bash ./scripts/download_cyclegan_model.sh style_monet", cwd=repo)
run(
    "python test.py --dataroot ../input_dir --name style_monet_pretrained "
    "--model test --no_dropout --preprocess none --num_test 1",
    cwd=repo,
)
cyclegan_out = os.path.join(
    ROOT, repo, "results", "style_monet_pretrained",
    "test_latest", "images", "photo_fake.png",
)

# ---------------------------------------------------------------
# Step 3: Neural Style Transfer (Gatys et al., VGG19)
# ---------------------------------------------------------------
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load = T.Compose([T.Resize((SIZE, SIZE)), T.ToTensor()])
content = load(Image.open("content_512.png").convert("RGB")).unsqueeze(0).to(dev)
style = load(Image.open(STYLE).convert("RGB")).unsqueeze(0).to(dev)

mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(-1, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(-1, 1, 1)
vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(dev).eval()
for p in vgg.parameters():
    p.requires_grad_(False)


def gram(x):
    _, c, h, w = x.shape
    f = x.view(c, h * w)
    return f @ f.t() / (c * h * w)


style_layers = {"conv_1", "conv_2", "conv_3", "conv_4", "conv_5"}
content_layers = {"conv_4"}


def features(x):
    x = (x - mean) / std
    feats, i, name = {}, 0, ""
    for layer in vgg:
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f"conv_{i}"
        elif isinstance(layer, nn.ReLU):
            layer = nn.ReLU(inplace=False)
        x = layer(x)
        if isinstance(layer, nn.Conv2d) and name in style_layers | content_layers:
            feats[name] = x
    return feats


c_feats = features(content)
s_grams = {k: gram(v) for k, v in features(style).items() if k in style_layers}

out = content.clone().requires_grad_(True)
opt = optim.LBFGS([out])
STYLE_W, CONTENT_W = 1e6, 1   # raise STYLE_W for stronger style, lower for more content
step = [0]


def closure():
    with torch.no_grad():
        out.clamp_(0, 1)
    opt.zero_grad()
    f = features(out)
    c_loss = sum(((f[k] - c_feats[k]) ** 2).mean() for k in content_layers)
    s_loss = sum(((gram(f[k]) - s_grams[k]) ** 2).mean() for k in style_layers)
    loss = CONTENT_W * c_loss + STYLE_W * s_loss
    loss.backward()
    step[0] += 1
    if step[0] % 50 == 0:
        print(f"step {step[0]}  loss {loss.item():.4f}")
    return loss


while step[0] < 300:
    opt.step(closure)

with torch.no_grad():
    out.clamp_(0, 1)
T.ToPILImage()(out.squeeze(0).cpu()).save("nst_output.png")

# ---------------------------------------------------------------
# Step 4: side-by-side comparison image
# ---------------------------------------------------------------
panels = [
    ("Original", "content_512.png"),
    ("CycleGAN (Monet)", cyclegan_out),
    ("Neural Style Transfer", "nst_output.png"),
]
fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
for ax, (title, path) in zip(axes, panels):
    ax.imshow(Image.open(path))
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
plt.savefig("comparison.png", dpi=200)
print("Saved comparison.png")
