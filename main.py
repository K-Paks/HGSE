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
    embs = pd.read_csv('embeddings.csv', index_col=[0, 1])
    idx_df = pd.read_csv('index_map.csv')
    return embs.astype(np.float32), idx_df


st.title("Healthy Gamer Search Engine")

st.subheader("Improve your mental health with the improved search engine!")

# # #
model = load_model()
download_resources()
embeddings, index_mapping = load_resources()

query = st.text_input("What's on your mind?")


def video_idcs_to_names(index_mapping, vid, idx):
    item = index_mapping[(index_mapping['video_id'] == vid) & (index_mapping['index'] == idx)]
    title = item['title'].item()
    name = item['name'].item()
    video_id = item['video_id'].item()
    time = item['time'].item()
    time_h = int(time // 3600)
    time_m = int((time - time_h * 3600) // 60)
    time_s = int((time - time_h * 3600 - time_m * 60))
    time_start = f'{time_h:02d}:{time_m:02d}:{time_s:02d}'
    return title, name, video_id, int(time), time_start


if query:
    st.write('Results for: ', query)

    query_embedding = model.encode(query)
    scores = dot_score(query_embedding, embeddings.values)

    score_df = pd.DataFrame(scores.numpy().squeeze(), index=embeddings.index)
    score_df = score_df.reset_index()
    score_df.columns = ['video_id', 'index', 'score']
    score_df = score_df.sort_values(by='score', ascending=False)

    suggested_videos = []
    for video in score_df.values:
        video_id, index, score = video

        if video_id in suggested_videos:
            continue

        suggested_videos.append(video_id)
        title, name, video_id, time_raw, time_start = video_idcs_to_names(index_mapping, vid=video_id, idx=index)

        col1, mid, col2 = st.columns([1, 3, 20])
        with col1:
            st.image(f'https://i.ytimg.com/vi/{video_id}/default.jpg', width=100)
        with col2:
            st.markdown(f'[{title}](https://www.youtube.com/watch?v={video_id})', unsafe_allow_html=True)
            if name != 'None':
                name = name.replace('- ', '')
                st.markdown(f'Relatable part: [{name}](https://youtu.be/{video_id}?t={time_raw}). Starts at {time_start}')
        #
        # st.markdown("""---""")

        if len(suggested_videos) == 10:
            break


