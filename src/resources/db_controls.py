import io
import sqlite3

import numpy as np


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

    def populate_db(self, embeddings, index_mapping):
        print('Populating `embeddings`...')
        for emb in embeddings.iterrows():
            video_id, video_part_id = emb[0]
            embedding = emb[1].values
            self.cursor.execute('INSERT INTO embeddings (video_id, video_part_id, embedding) VALUES (?, ?, ?)',
                                (video_id, video_part_id, embedding))
        self.connection.commit()

        print('Populating `mapping`...')
        for idx in index_mapping.iterrows():
            items = idx[1]
            fixed_title, name, index, time, title, video_id = items.values
            self.cursor.execute('INSERT INTO mapping (video_id, video_part_id, title, fixed_title, name, time) '
                                'VALUES (?, ?, ?, ?, ?, ?)',
                                (video_id, index, title, fixed_title, name, time))
        self.connection.commit()

    def get_embeddings(self):
        self.cursor.execute('SELECT * FROM embeddings')
        return self.cursor.fetchall()

    def get_mapping(self):
        self.cursor.execute('SELECT * FROM mapping')
        return self.cursor.fetchall()
