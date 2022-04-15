import pickle

from sentence_transformers import util

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

util