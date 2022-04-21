import json

import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers.util import dot_score

from gdrive import download_file_from_google_drive


@st.cache(allow_output_mutation=True)
def load_model():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
    return model


def download_resources():
    import os
    for key in ['embeddings', 'index_map']:
        destination = key + '.csv'
        if destination not in os.listdir('.'):
            file_id = st.secrets[key]
            download_file_from_google_drive(file_id, destination)


@st.cache
def load_resources():
    embs = np.genfromtxt('embeddings.csv', delimiter=',')
    idx_df = pd.read_csv('index_map.csv')
    return embs.astype(np.float32), idx_df


st.title("Healthy Gamer Search Engine")

st.subheader("Improve your mental health with the improved search engine!")

# # #
model = load_model()
download_resources()
embeddings, index_mapping = load_resources()

query = st.text_input("What's on your mind?")

if query:
    st.write('Results for: ', query)

    query_embedding = model.encode(query)
    scores = dot_score(query_embedding, embeddings)
    scores = scores.numpy()[0]
    ordered = pd.factorize(scores, sort=True)[0]
    df = pd.DataFrame((scores, ordered), index=['score', 'order']).T.sort_values(by='order', ascending=False)
    idx = df.index.values
    relevant_titles = index_mapping.loc[idx, ['title', 'name']]
    scores_sorted = scores[idx]

    for title, score in zip(relevant_titles[:10].values, scores_sorted[:10]):
        st.write(title[0])
        st.write(title[1])
        st.write(score)

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
