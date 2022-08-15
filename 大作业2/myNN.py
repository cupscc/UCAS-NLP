import torch
from torch import nn

device  = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
torch.cuda.is_available()
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
            train_x.append(word_to_id[word])
            train_y.append(word_to_id[li_words[i+j]])
class FNN(nn.Module):
    def __init__(self,word_size,window_size):
        super(FNN,self).__init__()
        self.embedding = nn.Embedding(word_size,10)
        self.linear = nn.Linear(10,word_size)
        self.log_softmax = nn.LogSoftmax()
    def forward(self,x):
        x = self.embedding(x)
        x = self.linear(x)
        x = self.log_softmax(x)
        return x

model = FNN(word_size,window_size).to(device)

loss_func = nn.NLLLoss()
lr = 1e-3
optimizer = torch.optim.Adam(model.parameters(),lr=lr)

model.train()
li_loss = []
batch_size = 10000
for epoch in range(0,1):
    for batch in range(0,len(train_x) - batch_size,batch_size):
        word = torch.tensor(train_x[batch:batch+batch_size]).long().to(device)
        label = torch.tensor(train_y[batch:batch + batch_size]).long().to(device)
        out = model(word)
        loss = loss_func(out,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("hello",batch)
    li_loss.append(loss)

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
result = model(torch.tensor([i for i in range(word_size)]).long().to(device))
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
embed_two = tsne.fit_transform(model.embedding.weight.cpu().detach().numpy())
labels = [id_to_word[i] for i in range(200)]
# 这里就查看前200个单词的分布
plt.figure(figsize=(15, 12))
for i, label in enumerate(labels):
    x, y = embed_two[i, :]
    plt.scatter(x, y)
    plt.annotate(label, (x, y), ha='center', va='top')
plt.savefig('fnn.png')
plt.show()