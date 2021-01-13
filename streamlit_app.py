import streamlit as st
import api
import requests
from PIL import Image
from pathlib import Path

import io
import numpy as np

st.set_page_config(
    layout="wide", page_title=api.app_formal_name, initial_sidebar_state="expanded"
)

clf = api.CLIP()
clf.load()

sess = requests.session()
top_n = 4
expected_height = 600

cache_dest = Path("data/streamlit_image_cache")
cache_dest.mkdir(exist_ok=True, parents=True)


def cache_download(image_idx):
    f_img = cache_dest / f"{image_idx}.jpg"

    if not f_img.exists():
        print(f"Downloading {image_idx}")

        url = f"https://unsplash.com/photos/{image_idx}/download"
        params = {"force": True, "h": expected_height}

        r = sess.get(url, params=params)

        if not r.ok:
            print(f"Failed {image_idx}")
            return None

        with open(f_img, "wb") as FOUT:
            FOUT.write(r.content)
    else:
        pass
        # print(f"Cached load {image_idx}")

    with open(f_img, "rb") as FIN:
        return FIN.read()


#@st.cache(ttl=10 * 3600)
def get_unsplash_image(image_idx):
    return cache_download(image_idx)

#@st.cache(ttl=3600)
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
    lines = textlines.strip().split("\n")
    lines = [" ".join(line.strip().split()) for line in lines if line.strip()]
    title, lines = lines[0], lines[1:]
    return title, lines


# Load presaved poems
known_poems_dest = Path("docs") / "collected_poems"
known_poems = {}
for f_poem in known_poems_dest.glob("*.txt"):
    with open(f_poem) as FIN:
        title, lines = preprocess_text(FIN.read())

    known_poems[title] = lines

default_poem = "Ozymandias"
poem_list = list(known_poems.keys())
poem_choice = st.sidebar.selectbox(
    "Select a starting poem", poem_list, index=poem_list.index(default_poem)
)

lines = known_poems[poem_choice]

with st.beta_expander("Customize Poem Text"):
    text_input = st.text_area(
        "Input poem here, one line per image set. The first line will be the title. [Control+Enter] to compute.",
        value="\n".join([poem_choice] + lines),
    )
    poem_choice, lines = preprocess_text(text_input)

results = clf(lines)


st.title(poem_choice)

st.sidebar.markdown("------------------------------------------------------------")
st.sidebar.markdown(
    f"[{api.app_formal_name}](https://github.com/thoppe/alph-the-sacred-river) combines poems and text using [CLIP](https://openai.com/blog/clip) from OpenAI. Images are sourced from the Unsplash [landscape dataset](https://github.com/unsplash/datasets) and featured photos. Photo credits at the bottom."
)

st.sidebar.markdown("Made with 💙 by [@metasemantic](https://twitter.com/metasemantic)")


credits = []
for k, row in enumerate(results):
    line = row["text"]
    st.markdown(f"## *{line}*")
    grid = combine_images(row["unsplashIDs"])

    credits.append(f"*{line}*")

    for image_idx in row["unsplashIDs"]:
        source_url = f"https://unsplash.com/photos/{image_idx}"
        credit = f"[{image_idx}]({source_url})"
        credits.append(credit)
    credits.append("\n")

    # caption = ', '.join([f"{x:0.0f}" for x in row['scores']])
    st.image(grid, use_column_width=True)

with st.beta_expander("Image Credits"):
    st.markdown("\n".join(credits))
