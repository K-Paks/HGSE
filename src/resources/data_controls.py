from pathlib import Path
import os
from os.path import join

import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from gdrive import download_file_from_google_drive


@st.cache(allow_output_mutation=True)
def load_transformer():
    model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
    return model


def download_resources():
    Path('resources').mkdir(exist_ok=True)

    for key in ['embeddings', 'index_map']:
        destination = join('resources', key + '.csv')
        if destination not in os.listdir('resources'):
            file_id = st.secrets[key]
            download_file_from_google_drive(file_id, destination)


@st.cache
def load_resources():
    embs = pd.read_csv(join('resources', 'embeddings.csv'), index_col=[0, 1])
    idx_df = pd.read_csv(join('resources', 'index_map.csv'))
    return embs.astype(np.float32), idx_df
