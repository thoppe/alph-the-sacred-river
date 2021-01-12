import fastapi
import torch
import pandas as pd
import numpy as np
from src import clip

from fastapi import FastAPI
from typing import List
from pydantic import BaseModel

app_name = "alph-the-sacred-river"
app_formal_name = "Alph The Sacred River"
__version__ = "0.1.0"


class CLIP:
    def __init__(self):
        pass

    def load(self, f_latents="data/img_latents.npy", f_keys="data/img_keys.csv"):

        # Load the model for inference on the CPU
        self.model, self.transform = clip.load("ViT-B/32", device="cpu")
        self.model.eval()

        self.scale = self._to_numpy(self.model.logit_scale.exp())

        # Load the pre-computed unsplash latent codes
        self.V = np.load(f_latents)
        self.V /= np.linalg.norm(self.V, ord=2, axis=-1, keepdims=True)

        # Load the mapping of the latent codes to the IDs
        self.keys = pd.read_csv(f_keys)["unsplashID"].values

    def _to_numpy(self, x):
        return x.detach().cpu().numpy()

    def encode_text(self, lines):

        with torch.no_grad():
            tokens = clip.tokenize(lines)
            latents = self.model.encode_text(tokens)

        latents = self._to_numpy(latents)
        latents /= np.linalg.norm(latents, ord=2, axis=-1, keepdims=True)
        return latents

    def compute(self, lines):
        TL = self.encode_text(lines)
        logits = (self.scale * self.V).dot(TL.T)
        return logits

    def __call__(self, lines, top_k=4):
        X = self.compute(lines)
        df = pd.DataFrame(data=X, columns=lines, index=self.keys)

        data = []

        for k, sent in enumerate(df.columns):
            dx = df.sort_values(sent, ascending=False)[sent]
            dx = dx[:top_k]

            data.append(
                {"text": sent,
                 "unsplashIDs":dx.index.values.tolist(), "scores": dx.values.tolist()}
            )

        return data


def load_sample_data():
    with open("docs/kubla_khan.txt") as FIN:
        sents = FIN.read().split("\n")

    sents = [" ".join(line.split()) for line in sents if line.strip()]
    return sents


app = FastAPI()

class TextListInput(BaseModel):
    lines: List[str]


@app.get("/")
def root():
    return {
        "app_name": app_name,
        "version": __version__,
    }

@app.get("/infer")
def infer_multi(q: TextListInput):
    return clf(q.lines)


if __name__ == "__main__":

    clf = CLIP()
    clf.load()

    sents = load_sample_data()

    from fastapi.testclient import TestClient

    client = TestClient(app)

    print(client.get("/").json())

    r = client.get("/infer", json={"lines": sents})
    print(r.json())
