import streamlit as st
from pathlib import Path

import start_api
from interface import combine_images, encoding_sentences, preprocess_text

app_formal_name = "Alph The Sacred River"

# Start the app in wide-mode
st.set_page_config(
    layout="wide", page_title=app_formal_name, initial_sidebar_state="expanded"
)

# Load presaved poems
known_poems_dest = Path("docs") / "collected_poems"
known_poems = {}
for f_poem in known_poems_dest.glob("*.txt"):
    with open(f_poem) as FIN:
        title, lines = preprocess_text(FIN.read())

    known_poems[title] = lines

# Select a starting poem and display the choices in the sidebar
default_poem = "Ozymandias"
poem_list = list(known_poems.keys())
poem_choice = st.sidebar.selectbox(
    "Select a starting poem", poem_list, index=poem_list.index(default_poem)
)
lines = known_poems[poem_choice]


# If the user has a custom poem, use it here
with st.expander("Customize Poem Text"):
    text_input = st.text_area(
        "Input poem here, one line per image set. The first line will be the title. [Control+Enter] to compute.",
        value="\n".join([poem_choice] + lines),
    )
    poem_choice, lines = preprocess_text(text_input)

# Run the selected poem through the model
results = encoding_sentences(lines)


st.title(poem_choice)
st.sidebar.markdown("-----------------------------------")
st.sidebar.markdown(
    f"[{app_formal_name}](https://github.com/thoppe/alph-the-sacred-river) "
    f"combines poems and text using [CLIP](https://openai.com/blog/clip) from OpenAI. "
    f"Images are sourced from the Unsplash [landscape dataset](https://github.com/unsplash/datasets) "
    "and featured photos. Photo credits at the bottom."
)
st.sidebar.markdown(
    "Made with 💙 by [@metasemantic](https://twitter.com/metasemantic/status/1349446585952989186)"
)

# Show the credits for each photo in an expandable sidebar
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

with st.expander("Image Credits"):
    st.markdown("\n".join(credits))
