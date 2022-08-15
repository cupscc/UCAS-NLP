from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

with open('new_corpus.txt','r',encoding='gbk')as f:
    sentences = LineSentence(f)
    model = Word2Vec(sentences,vector_size = 10,window=5,min_count=1,max_vocab_size = 2000,workers=4,epochs= 50)
    model.save('renmin.mdl')
model.load('renmin.mdl')
items = model.wv.most_similar('人民')
# wv下提供了很多工具方法，这里词向量按与传入的单词相似度从高到低排序
for i, item in enumerate(items):
    print(i, item[0], item[1])
tsne = TSNE(n_components=2, init='pca', n_iter=5000)
embed_two = tsne.fit_transform(model.wv[model.wv.index_to_key])
labels = model.wv.index_to_key

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
plt.figure(figsize=(15, 12))
for i, label in enumerate(labels[:200]):
    x, y = embed_two[i, :]
    plt.scatter(x, y)
    plt.annotate(label, (x, y), ha='center', va='top')
plt.savefig('word.png')
plt.show()