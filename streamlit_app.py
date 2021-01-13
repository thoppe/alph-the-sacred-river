import streamlit as st
import api
import requests
from PIL import Image
from pathlib import Path

import io
import numpy as np

st.set_page_config(layout="wide",page_title=api.app_formal_name,)

clf = api.CLIP()
clf.load()

sess = requests.session()
top_n = 4
expected_height = 600

cache_dest = Path('data/streamlit_image_cache')
cache_dest.mkdir(exist_ok=True, parents=True)

def cache_download(image_idx):
    f_img = cache_dest / f"{image_idx}.jpg"
        
    if not f_img.exists():
        print(f"Downloading {image_idx}")

        url = f"https://unsplash.com/photos/{image_idx}/download"
        params = {"force":True, "h" : expected_height}

        r = sess.get(url, params=params)

        if not r.ok:
            print(f"Failed {image_idx}")
            return None

        with open(f_img, 'wb') as FOUT:
            FOUT.write(r.content)
    else:
        pass
        #print(f"Cached load {image_idx}")

    with open(f_img, 'rb') as FIN:
        return FIN.read()

    

@st.cache(ttl=10*3600)
def get_unsplash_image(image_idx):
    return cache_download(image_idx)


@st.cache(ttl=3600)
def combine_images(imageIDs):
    imgs = [get_unsplash_image(idx) for idx in imageIDs]

    pil_imgs = []
    for img in imgs:
        try:
            pil_imgs.append(Image.open(io.BytesIO(img)))
        except:
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
    lines = textlines.strip().split('\n')
    lines = [' '.join(line.strip().split()) for line in lines if line.strip()]
    title, lines = lines[0], lines[1:]
    return title, lines


# Load presaved poems
known_poems_dest = Path('docs') / 'collected_poems'
known_poems = {}
for f_poem in known_poems_dest.glob('*.txt'):
    with open(f_poem) as FIN:
        title, lines = preprocess_text(FIN.read())

    known_poems[title] = lines

title = 'Ozymandias'
lines = known_poems[title]
results = clf(lines)

st.title(title)

for k, row in enumerate(results):
    st.markdown(f"## *{row['text']}*")
    grid = combine_images(row['unsplashIDs'])
    

    caption = ', '.join([f"{x:0.0f}" for x in row['scores']])
    st.image(grid, caption, use_column_width=True)
