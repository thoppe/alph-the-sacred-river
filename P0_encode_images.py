from dspipe import Pipe
from src import clip
from PIL import Image
import torch
import numpy as np
import pandas as pd

"""
Preprocessing step:
Compute latents for all images in "data/source_images".
Save data to datasets "data/img_latents.npy", "data/img_keys.csv"
"""

device = "cuda" if torch.cuda.is_available() else "cpu"
model, transform = clip.load("ViT-B/32", device=device)

f_img_latents = "data/img_latents.npy"
f_img_keys = "data/img_keys.csv"


def compute(f0, f1):

    try:
        img = Image.open(f0)
        image = transform(img).unsqueeze(0).to(device)
    except Exception as EX:
        print(f"Failed opening {f0} {EX}")
        return False

    with torch.no_grad():
        latents = model.encode_image(image)

    # Pull the latents off the GPU
    latents = latents.detach().cpu().numpy().ravel().astype(float)

    np.save(f1, latents)


"""
Pipe(
    "data/source_images/",
    "data/image_latents",
    input_suffix=".jpg",
    output_suffix=".npy",
    shuffle=True,
)(compute, 1)
"""


def read(f0):
    v = np.load(f0).astype(np.float64)
    return f0.name.split(".npy")[0], v


data = Pipe("data/image_latents/")(read, -1)
keys, V = zip(*data)

df = pd.DataFrame(data=keys, columns=["unsplashID"])
df.set_index("unsplashID").to_csv(f_img_keys)

V = np.array(V).astype(np.float16)

np.save(f_img_latents, V)
