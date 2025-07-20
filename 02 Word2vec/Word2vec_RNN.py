import torch
import torch.nn as nn
import torch.optim as optim

# 1. 准备数据
corpus = ["I like natural language processing", "I like deep learning"]
# 分词、构建词表
tokens = [s.lower().split() for s in corpus]
vocab = sorted({w for sent in tokens for w in sent})
word2idx = {w:i for i,w in enumerate(vocab)}
idx2word = {i:w for w,i in word2idx.items()}

# 将语料转换为索引列表
data = []
for sent in tokens:
    idxs = [word2idx[w] for w in sent]
    data.extend(idxs)

# 2. 定义模型
class RNNWord2Vec(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(input_size=embed_dim,
                          hidden_size=hidden_dim,
                          num_layers=1,
                          batch_first=True)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, h0):
        # x: [batch, seq_len] of word indices
        e = self.embed(x)             # [batch, seq, E]
        o, hn = self.rnn(e, h0)       # o: [batch, seq, H]
        logits = self.out(o)          # [batch, seq, V]
        return logits, hn

# 3. 超参数
VOCAB_SIZE = len(vocab)
EMBED_DIM  = 10
HIDDEN_DIM = 20
SEQ_LEN    = 3    # 用前三个词预测下一个词
LR         = 0.01
EPOCHS     = 200

model = RNNWord2Vec(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR)

# 4. 构造训练样本：滑动窗口
#    输入是长度为 SEQ_LEN 的上下文，标签是紧接着的下一个词
inputs, targets = [], []
for i in range(len(data) - SEQ_LEN):
    inputs.append(data[i:i+SEQ_LEN])
    targets.append(data[i+SEQ_LEN])
inputs = torch.tensor(inputs, dtype=torch.long)    # [N, SEQ_LEN]
targets = torch.tensor(targets, dtype=torch.long)  # [N]

# 5. 训练循环
for epoch in range(1, EPOCHS+1):
    h0 = torch.zeros(1, inputs.size(0), HIDDEN_DIM)  # 初始隐状态
    logits, _ = model(inputs, h0)         # [N, SEQ_LEN, V]
    # 只取最后一步的输出去预测下一个词
    last_logits = logits[:, -1, :]        # [N, V]
    loss = criterion(last_logits, targets)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch:3d}, Loss: {loss.item():.4f}")

# 6. 训练完成后，你可以这样取词向量：
embeddings = model.embed.weight.data  # [V, E]
for i, w in idx2word.items():
    print(f"{w:>12s}  →  {embeddings[i].tolist()}")
'''
1.RNN循环神经网络是怎么工作的
2.与普通的Word2Vec模型相比，RNN的优势是什么
3.与普通的神经网络相比，RNN的优势是什么

'''