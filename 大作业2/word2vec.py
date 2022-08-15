import numpy as np

def one_hot_encode(id,vocab_size):
    res = [0] * vocab_size
    res[id] = 1
    return res

np.random.seed(42)
window_size = 2
word_set = set()
lines = []
with open("new_corpus.txt", "r", encoding="gbk") as f:
    lines = f.read().strip().split("\n")
    null_string = ''
    while(1):
        if(null_string in lines):
            lines.remove(null_string)
        else:
            break
    for line in lines :
        for word in line.split():
            word_set.add(word)
word_size = len(word_set)
word_to_id = {word:i for i,word in enumerate(word_set)}
id_to_word = {word_to_id[word]:word for word in word_to_id}
train_x = []
train_y = []
for line in lines:
    li_words = line.split()
    for i,word in enumerate(li_words):
        for j in range(-window_size,window_size+1):
            if i+j <0 or i+j >len(li_words) -1 or li_words[i+j] ==word:
                continue
            train_x.append(one_hot_encode(word_to_id[word],word_size))
            train_y.append(one_hot_encode(word_to_id[li_words[i+j]],word_size))
def init_network(word_size,n_embedding):
    model = {"w1":np.random.rand(word_size,n_embedding),
             "w2":np.random.rand(n_embedding,word_size)}

model = init_network(len(word_to_id),10)
def softmax(X):
    res = []
    for x in X:
        exp = np.exp(x)
        res.append(exp / exp.sum())
    return res
def forward (model,X,return_cache = True):
    cache = {}
    cache['a1'] = X @ model['w1']
    cache['a2'] = cache['a1'] @ model['w2']
    cache['z'] = softmax(cache['a2'])
    if not return_cache:
        return cache['z']
    return cache
def cross_entropy(z, y):
    return - np.sum(np.log(z) * y)

def backward(model, X, y, alpha):
    cache  = forward(model, X)
    da2 = cache["z"] - y
    dw2 = cache["a1"].T @ da2
    da1 = da2 @ model["w2"].T
    dw1 = X.T @ da1
    assert(dw2.shape == model["w2"].shape)
    assert(dw1.shape == model["w1"].shape)
    model["w1"] -= alpha * dw1
    model["w2"] -= alpha * dw2
    return cross_entropy(cache["z"], y)

import matplotlib.pyplot as plt

plt.style.use("seaborn")

n_iter = 50
learning_rate = 0.05

history = [backward(model, train_x,train_y, learning_rate) for _ in range(n_iter)]

plt.plot(range(len(history)), history, color="skyblue")
plt.show()

learning = one_hot_encode(word_to_id["learning"], len(word_to_id))
result = forward(model, [learning], return_cache=False)[0]

for word in (id_to_word[id] for id in np.argsort(result)[::-1]):
    print(word)
def get_embedding(model, word):
    try:
        idx = word_to_id[word]
    except KeyError:
        print("`word` not in corpus")
    one_hot = one_hot_encode(idx, len(word_to_id))
    return forward(model, one_hot)["a1"]