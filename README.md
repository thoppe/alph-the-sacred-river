# Alph, The Sacred River
_AI poetic imagery using [CLIP](https://github.com/openai/CLIP) and [Unsplash](https://unsplash.com/)_

## Preprocessing steps for new images

+ Download a bunch of images from Unsplash ([start here!](https://github.com/unsplash/datasets))
+ Save each image as a jpg in `data/source_images/`
+ Run `python P0_encode_images.py` to compute the image latents

## AWS Notes

CPU only install of torch

    pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html