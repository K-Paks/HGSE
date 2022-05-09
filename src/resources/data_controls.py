from pathlib import Path
import os
from os.path import join

import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from src.utils.gdrive import download_file_from_google_drive


@st.cache(allow_output_mutation=True)
def load_transformer():
    model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
    return model


def download_resources():
    print('NOT CACHING! FIX FOR PROD IN STREAMLIT')
    Path('resources').mkdir(exist_ok=True)

    for key in ['embeddings', 'index_map']:
        destination = join('resources', key + '.csv')
        if destination not in os.listdir('resources'):
            file_id = st.secrets[key]
            download_file_from_google_drive(file_id, destination)


@st.cache
def load_resources(fname):
    if fname == 'embeddings':
        data = pd.read_csv(join('resources', 'embeddings.csv'), index_col=[0, 1])
        data = data.astype(np.float32)
    elif fname == 'mapping':
        data = pd.read_csv(join('resources', 'index_map.csv'))
        # TODO fix in source \/
        data['time'] = data['time'].astype('int')
    else:
        raise Exception('fname not recognized')

    return data
