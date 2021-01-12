import streamlit as st
import api
import requests
from PIL import Image

import io
import numpy as np

st.set_page_config(layout="wide",page_title=api.app_formal_name,)
st.title(api.app_formal_name)


clf = api.CLIP()
clf.load()

sess = requests.session()
top_n = 4
expected_height = 600

@st.cache(ttl=10*3600)
def get_unsplash_image(image_idx, h=expected_height):
    print(f"Downloading {image_idx}")
    
    url = f"https://unsplash.com/photos/{image_idx}/download?force=true&h={h}"
    r = sess.get(url)

    if not r.ok:
        print(f"Failed {image_idx}")
        return None
    
    return r.content

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


sample_text = api.load_sample_data()

sample_text = '''The fog has risen from the sea and crowned
The dark, untrodden summits of the coast,
Where roams a voice, in canyons uttermost,
From midnight waters vibrant and profound.
High on each granite altar dies the sound,
Deep as the trampling of an armored host,
Lone as the lamentation of a ghost,
Sad as the diapason of the drowned.
The mountain seems no more a soulless thing,
But rather as a shape of ancient fear,
In darkness and the winds of Chaos born
Amid the lordless heavens' thundering-
A Presence crouched, enormous and austere,
Before whose feet the mighty waters mourn.'''.split('\n')

sample_text = '''My heart leaps up when I behold A rainbow in the sky:
So was it when my life began; So is it now I am a man; 
So be it when I shall grow old, Or let me die!
The Child is father of the Man; And I could wish my days to be
Bound each to each by natural piety.'''.split('\n')

sample_text0 ='''I met a traveller from an antique land,
Who said "Two vast and trunkless legs of stone Stand in the desert
Near them, on the sand, Half sunk a shattered visage lies, whose frown, And wrinkled lip, and sneer of cold command,
Tell that its sculptor well those passions read Which yet survive, stamped on these lifeless things, 
The hand that mocked them, and the heart that fed;
And on the pedestal, these words appear: My name is Ozymandias, King of Kings;
Look on my Works, ye Mighty, and despair! Nothing beside remains. 
Round the decay Of that colossal Wreck, boundless and bare The lone and level sands stretch far away.'''.split('\n')

sample_text = '''Once upon a midnight dreary, while I pondered, weak and weary,
Over many a quaint and curious volume of forgotten lore
While I nodded, nearly napping, suddenly there came a tapping,
As of some one gently rapping, rapping at my chamber door.
Tis some visitor, I muttered, tapping at my chamber door Only this and nothing more.
Ah, distinctly I remember it was in the bleak December;
And each separate dying ember wrought its ghost upon the floor.
Eagerly I wished the morrow; vainly I had sought to borrow
From my books surcease of sorrow - sorrow for the lost Lenore
For the rare and radiant maiden whom the angels name Lenore\nNameless here for evermore.'''.split('\n')

results = clf(sample_text0)

for k, row in enumerate(results):
    st.markdown(f"## *{row['text']}*")
    grid = combine_images(row['unsplashIDs'])

    caption = ', '.join([f"{x:0.0f}" for x in row['scores']])
    st.image(grid, caption, use_column_width=True)
