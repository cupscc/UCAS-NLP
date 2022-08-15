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

def one_hot(id,size) :
    x = torch.zeros([size,1],dtype=torch.float)
    x[id][0] = 1
    return x

class RNN(nn.Module):
    def __init__(self,word_size):
        super(RNN,self).__init__()
        self.embed = nn.Embedding(10,word_size)
        self.rnn = nn.RNN(word_size,256,num_layers=1)
        self.linear = nn.Linear(256,word_size)
        self.log_softmax = nn.LogSoftmax()
    def forward(self,x,state):
        X = self.embed(x)
        Y,state = self.rnn(x,state)
        output = self.linear(Y.reshape((-1,Y.shape[-1])))
        return output,state
    def begin_state(self, batch_size=1):
        return  torch.zeros((1 * 1,batch_size,256))

model = RNN(word_size)
model = model.to(device)

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
        if batch == 0:
            out,next_state = model(word,state = RNN.begin_state(self=model))
        else:
            out,next_state = model(word,state = next_state)
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