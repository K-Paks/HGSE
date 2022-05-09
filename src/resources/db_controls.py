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

        # Connect to a local database and create a table for the embeddings

        self.cursor.execute('''CREATE TABLE IF NOT EXISTS embeddings (
                       video_id TEXT, 
                       video_part_id INTEGER, 
                       embedding array,
                       PRIMARY KEY (video_id, video_part_id)
                       )''')
        self.connection.commit()

    def populate_db(self, embeddings):
        for emb in embeddings.iterrows():
            video_id, video_part_id = emb[0]
            embedding = emb[1].values
            self.cursor.execute('INSERT INTO embeddings (video_id, video_part_id, embedding) VALUES (?, ?, ?)',
                                (video_id, video_part_id, embedding))
        self.connection.commit()

    def get_embeddings(self):
        self.cursor.execute('SELECT * FROM embeddings')
        return self.cursor.fetchall()

