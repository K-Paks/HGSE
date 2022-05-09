import io
import sqlite3

import pandas as pd
import numpy as np

from src.resources.data_controls import load_resources


def adapt_array(array):
    """
    Using the numpy.save function to save a binary version of the array,
    and BytesIO to catch the stream of data and convert it into a sqlite3.Binary.
    """
    out = io.BytesIO()
    np.save(out, array)
    out.seek(0)

    return sqlite3.Binary(out.read())


def convert_array(blob):
    """
    Using BytesIO to convert the binary version of the array back into a np array.
    """
    out = io.BytesIO(blob)
    out.seek(0)

    return np.load(out)


class DatabaseController:
    def __init__(self):
        self.connection = sqlite3.connect('./sqlite.embedding.db', detect_types=sqlite3.PARSE_DECLTYPES)
        self.cursor = self.connection.cursor()

    def prepare_db(self):
        # Register the new adapters
        sqlite3.register_adapter(np.ndarray, adapt_array)
        sqlite3.register_converter('array', convert_array)

        # Connect to a local database and create tables for the embeddings and the index mapping
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS embeddings (
                       video_id TEXT, 
                       video_part_id INTEGER, 
                       embedding array,
                       PRIMARY KEY (video_id, video_part_id)
                       )''')

        self.cursor.execute('''CREATE TABLE IF NOT EXISTS mapping (
                       video_id TEXT, 
                       video_part_id INTEGER, 
                       title TEXT,
                       fixed_title TEXT,
                       name TEXT,
                       time INT,
                       PRIMARY KEY (video_id, video_part_id)
                       )''')
        self.connection.commit()

    def populate_db(self):
        # if `embeddings` empty, populate
        if not self.check_if_empty('embeddings'):
            print('Populating `embeddings`...')
            embeddings = load_resources('embeddings')

            for emb in embeddings.iterrows():
                video_id, video_part_id = emb[0]
                embedding = emb[1].values
                self.cursor.execute('INSERT INTO embeddings (video_id, video_part_id, embedding) VALUES (?, ?, ?)',
                                    (video_id, video_part_id, embedding))
            self.connection.commit()

        # if `mapping` empty, populate
        if not self.check_if_empty('mapping'):
            print('Populating `mapping`...')
            index_mapping = load_resources('mapping')

            for idx in index_mapping.iterrows():
                items = idx[1]
                fixed_title, name, index, time, title, video_id = items.values
                self.cursor.execute('INSERT INTO mapping (video_id, video_part_id, title, fixed_title, name, time) '
                                    'VALUES (?, ?, ?, ?, ?, ?)',
                                    (video_id, index, title, fixed_title, name, time))
            self.connection.commit()

    def check_if_empty(self, dbname):
        self.cursor.execute(f'SELECT COUNT(*) FROM {dbname}')
        return self.cursor.fetchone()[0]

    def get_embeddings(self):
        self.cursor.execute('SELECT * FROM embeddings')
        data = self.cursor.fetchall()
        # TODO make faster? \/
        df = pd.DataFrame(np.column_stack(list(zip(*data))))
        df[1] = df[1].astype(int)
        df = df.set_index([0, 1])
        df = df.astype(np.float32)
        df.index.names = ['video_id', 'video_part_id']
        return df

    def get_mapping(self):
        self.cursor.execute('SELECT * FROM mapping')
        data = self.cursor.fetchall()
        df = pd.DataFrame(data, columns=['video_id', 'video_part_id', 'title', 'fixed_title', 'name', 'time'])
        return df

