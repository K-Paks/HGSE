from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(sentences)
print(embeddings)


from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Sentences we want sentence embeddings for
sentences = ['This is an example sentence', 'That is an example sentence', 'The example sentence is that.', 'I hate cats',
             'This in enormously long sentence however it also is a good example.']

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

# Perform pooling
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

# Normalize embeddings
sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

print("Sentence embeddings:")
print(sentence_embeddings)

from numpy import dot
from numpy.linalg import norm
import numpy as np
question = sentence_embeddings[0]
context = sentence_embeddings[1]
result = dot(question, context)/(norm(question)*norm(context))
print(result)


Z = np.random.rand(10, 4)
B = Z[0:1].T

Z = sentence_embeddings[:1]
B = sentence_embeddings[1:].T

Z_norm = norm(Z, axis=1, keepdims=True)  # Size (n, 1).
B_norm = norm(B, axis=0, keepdims=True)  # Size (1, b).

# Distance matrix of size (b, n).
cosine_similarity = ((Z @ B) / (Z_norm @ B_norm)).T
cosine_similarity


from scipy import spatial
List1 = sentence_embeddings[0]
List2 = sentence_embeddings[1]
result = 1 - spatial.distance.cosine(List1, List2)
print(result)



import os
from os.path import join
import re

DATA_PATH = join('data', 'subtitles_processed')

name = '_Therapy Seems Useless_ _ Dr K Talks.txt'
# name = 'Addressing Claims_ Are We A Cult_.txt'

with open(join(DATA_PATH, name), 'r') as f:
    text = f.readlines()

queue = []
trigrams = []

for l in range(len(text)):
    line = text[l].replace('\n', '')
    if line:
        if len(line.split(' ')) == 1:
            queue[-1] = queue[-1] + f' {line}'
        else:
            queue.append(line)

        if len(queue) == 3:
            trigram = ' '.join(queue)
            trigram = re.sub('\s+', ' ', trigram)

            trigrams.append(trigram)
            queue.pop(0)


# get tokens
query = 'my life has no meaning'
sentences = [query] + trigrams

encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

with torch.no_grad():
    model_output = model(**encoded_input)
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)


Z = sentence_embeddings[:1]
B = sentence_embeddings[1:].T

Z_norm = norm(Z, axis=1, keepdims=True)  # Size (n, 1).
B_norm = norm(B, axis=0, keepdims=True)  # Size (1, b).

# Distance matrix of size (b, n).
cosine_similarity = ((Z @ B) / (Z_norm @ B_norm)).T
cosine_similarity

np.argmax(cosine_similarity)
max(cosine_similarity)



text = ' '.join(text)
text = re.sub('\s+', ' ', text)

QA_input = {
    'question': "I don't think therapy is any helpful.",
    'context': text
}


nlp(QA_input)

