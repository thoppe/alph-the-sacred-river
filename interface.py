import streamlit as st
import requests
from pathlib import Path
import numpy as np
from PIL import Image
import io


# Typically we run this on the same machine
api_url = "http://127.0.0.1:8000/infer"


cache_dest = Path("data/streamlit_image_cache")
cache_dest.mkdir(exist_ok=True, parents=True)


@st.cache(ttl=3600)
def encoding_sentences(lines):
    r = requests.get(api_url, json={"lines": lines})
    return r.json()


def cache_download(image_idx, expected_height=600):
    f_img = cache_dest / f"{image_idx}.jpg"

    if not f_img.exists():
        print(f"Downloading {image_idx}")

        url = f"https://unsplash.com/photos/{image_idx}/download"
        params = {"force": True, "h": expected_height}

        r = requests.get(url, params=params)

        if not r.ok:
            print(f"Failed {image_idx}")
            return None

        with open(f_img, "wb") as FOUT:
            FOUT.write(r.content)
    else:
        pass

    with open(f_img, "rb") as FIN:
        return FIN.read()


# @st.cache(ttl=10*3600)
def get_unsplash_image(image_idx):
    return cache_download(image_idx)


# @st.cache(ttl=3600)
def combine_images(imageIDs, expected_height=600):
    imgs = [get_unsplash_image(idx) for idx in imageIDs]

    pil_imgs = []
    for img in imgs:
        try:
            pil_imgs.append(Image.open(io.BytesIO(img)))
        except Exception:
            pass

    block = []
    for img in pil_imgs:
        w, h = img.size
        frac = h / expected_height

        img = img.resize((int(w / frac), expected_height))
        img = np.array(img)
        block.append(img)

    mh = sum([x.shape[0] for x in block])
    mw = sum([x.shape[1] for x in block])

    grid = np.zeros(shape=(expected_height, mw, 3), dtype=np.uint8)

    running_w = 0
    for img in block:
        h, w = img.shape[:2]
        grid[:, running_w : running_w + w, :] = img
        running_w += w
    return grid


def preprocess_text(textlines):
    lines = textlines.strip().split("\n")
    lines = [" ".join(line.strip().split()) for line in lines if line.strip()]
    title, lines = lines[0], lines[1:]
    return title, lines
