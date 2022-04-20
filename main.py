print(help(modules))

import streamlit as st
import pandas as pd
import numpy as np
import sentence_transformers

from utils import dot_score


@st.cache
def load_resources():
    import pickle

    print(222222222222222)
    print(sentence_transformers)
    print(222222222222222)
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    enc_df = pd.read_csv('encoded_titles.csv')
    titles = enc_df['title']
    embeddings = enc_df[[col for col in enc_df.columns if col != 'title']].values
    return model, titles, embeddings.astype(np.float32)


st.title("Healthy Gamer Search Engine")

st.subheader("Improve your mental health with the improved search engine!")

# # #

model, titles, embeddings = load_resources()

query = st.text_input("What's on your mind?")

if query:
    st.write('Results for: ', query)

    query_embedding = model.encode(query)
    scores = dot_score(query_embedding, embeddings)
    scores = scores.numpy()[0]
    ordered = pd.factorize(scores, sort=True)[0]
    df = pd.DataFrame((scores, ordered), index=['score', 'order']).T.sort_values(by='order', ascending=False)
    idx = df.index.values
    relevant_titles = titles[idx]
    scores_sorted = scores[idx]

    for title, score in zip(relevant_titles[:10], scores_sorted[:10]):
        st.write(title, score)

# img = 'https://s3.viva.pl/newsy/zmarl-boo-najpopularniejszy-pies-swiata-561503-GALLERY_BIG.jpg'
# link = '[GitHub](http://github.com)'

# st.write("a logo and text next to eachother")
# col1, mid, col2 = st.columns([1, 10, 20])
# with col1:
#     st.image(img, width=200)
# with col2:
#     st.markdown(link, unsafe_allow_html=True)
#
# st.markdown("""---""")
