import os
import json
import re

import pandas as pd
import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
from sklearn.pipeline import Pipeline

nltk.download('stopwords')

documents_path = os.path.join('data_capt', 'subtitles')
documents = os.listdir(documents_path)

texts = []
titles = []
for document in documents:
    try:
        doc_path = os.path.join('data_capt', 'subtitles', document)

        with open(doc_path, 'r') as f:
            doc_json = json.load(f)

        # get sentences
        sents = pd.DataFrame(doc_json)['text'].to_numpy()

        # get the document and truncate empty spaces
        doc = ' '.join(sents)
        doc = re.sub('\s+', ' ', doc)
    except Exception:
        print(document)
    # preprocess
    # words = doc.split(' ')
    # stop_words = stopwords.words('english')
    # symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    # for i in symbols:
    #     data_capt = np.char.replace(words, i, ' ')
    # np.char.replace(data_capt, "'", "")
    # porter = PorterStemmer()
    # stemmed = [porter.stem(word) for word in words if word not in stop_words]

    # texts.append(stemmed)
    texts.append(doc)
    titles.append(document[:-5])

# TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer

# tfidf = TfidfVectorizer()
# X = tfidf.fit_transform(texts)
# df = pd.DataFrame(X.T.todense(), index=tfidf.get_feature_names(), columns=titles)

porter_stemmer = PorterStemmer()
stop_words = stopwords.words('english') + ['like', 'right', '__', '_connector_']

def preproc(text):
    text = text.lower()
    text = re.sub("\\s+(in|the|all|for|and|on)\\s+", " _connector_ ", text)  # normalize certain words

    # stem words
    words = re.split("\\s+", text)
    stemmed_words = [porter_stemmer.stem(word=word) for word in words if word not in stop_words]
    return ' '.join(stemmed_words)

pipe = Pipeline([('count', CountVectorizer(stop_words=stop_words, preprocessor=preproc)),
                 ('tfidf', TfidfTransformer())])
pipe.fit(texts)

X = pipe.transform(texts)
feature_names = pipe['count'].get_feature_names()


t = """Last week I managed to study 5 out of 7 days, where I procrastinated for 2 days, but my mind is like: "You could've studied all 7". I know for a fact even if I get to seven days I won't be satisfied, because I will be like: "You could've studied for one more hour" and so on."""
t = "I have trouble speaking to other people and feel very awkward in social situations."
testrow = pipe.transform([t])

testdf = pd.DataFrame(testrow.T.todense(), index=feature_names, columns=['test'])
compdf = pd.DataFrame(X[81].T.todense(), index=feature_names, columns=['comp'])




compdf.sort_values('comp', ascending=False).iloc[:20]
testdf.sort_values('test', ascending=False).iloc[:20]


from sklearn.metrics.pairwise import cosine_similarity

similarities = cosine_similarity(X, testrow)
np.argmax(similarities)
print('pairwise dense output:\n {}\n'.format(similarities))



from numpy import dot
from numpy.linalg import norm
List1 = np.squeeze(np.asarray(X[0].todense()))
List2 = np.squeeze(np.asarray(tt.todense()))
result = dot(List1, List2)/(norm(List1)*norm(List2))
print(result)

List1 = X[0].todense()
List2 = tt.todense().T
result = dot(List1, List2)/(norm(List1)*norm(List2))
print(result)

df.loc[words].sum().sort_values()

#
# DF = {}
# for i in range(len(texts)):
#     tokens = texts[i]
#     for w in tokens:
#         try:
#             DF[w].add(i)
#         except:
#             DF[w] = {i}
#
# for i in DF:
#     DF[i] = len(DF[i])
#
# vocab = [x for x in DF]
#
