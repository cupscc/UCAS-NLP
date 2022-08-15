from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import gensim
model = Word2Vec.load('renmin.mdl')
items = model.wv.most_similar('人民')
# wv下提供了很多工具方法，这里词向量按与传入的单词相似度从高到低排序
for i, item in enumerate(items):
    print(i, item[0], item[1])