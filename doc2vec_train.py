import gensim
import pandas as pd
from gensim.models.doc2vec import Doc2Vec

regenerate = False
corpus = 'corpus/radiology_corpus.cor'

if regenerate:
    data = pd.read_csv("data/mimic-note/radiology.csv").dropna()
    data['charttime'] = pd.to_datetime(data['charttime'])
    data['storetime'] = pd.to_datetime(data['storetime'])
    data['subject_id'] = data['subject_id'].astype('int64')
    data['hadm_id'] = data['hadm_id'].astype('int64')

    data['text'] = data.apply(lambda x: x['text'].replace('\n',' '), axis=1)
    data['text'] = data.apply(lambda x: ' '.join(gensim.utils.simple_preprocess(x['text'])), axis=1)
    with open(corpus, 'w') as f:
        for doc in data['text']:
            f.write(doc + '\n')

model = Doc2Vec(corpus_file=corpus, vector_size=512, epochs=20, seed=0, workers=8, min_count=2)

model.build_vocab(corpus_file=corpus)
model.train(corpus_file=corpus, total_examples=model.corpus_count, total_words=model.corpus_total_words, epochs=20)


model.save('saved_model/doc2vec_model')
print('done')