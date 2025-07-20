import torch
import torch.nn as nn
import torch.optim as optim

# 1. 准备数据
corpus = [
    "I like natural language processing",
    "I like deep learning"
]
tokens = [s.lower().split() for s in corpus]
vocab = sorted({w for sent in tokens for w in sent})
word2idx = {w:i for i,w in enumerate(vocab)}
data = []
window = 2  # 上下文窗口大小

for sent in tokens:
    idxs = [word2idx[w] for w in sent]
    for i, tgt in enumerate(idxs):
        ctx = []
        for j in range(i-window, i+window+1):
            if j!=i and 0<=j<len(idxs):
                ctx.append(idxs[j])
        if ctx:
            data.append((ctx, tgt))

# 2. 模型定义：Embedding + Attention + 输出层
class AttnCBOW(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        # 用于计算注意力分数的一个小型前馈网络
        self.attn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, 1, bias=False)
        )
        self.out = nn.Linear(embed_dim, vocab_size)

    def forward(self, ctx_idxs):
        # ctx_idxs: [batch, C]
        emb = self.embed(ctx_idxs)              # [batch, C, E]
        # 计算注意力分数
        # 将 emb 展平到 [batch*C, E]
        scores = self.attn(emb)                 # [batch, C, 1]
        alpha = torch.softmax(scores, dim=1)    # [batch, C, 1]
        # 加权求和得到上下文表示
        ctx_vec = (alpha * emb).sum(dim=1)      # [batch, E]
        logits = self.out(ctx_vec)              # [batch, V]
        return logits

# 3. 超参数与准备训练集
VOCAB_SIZE = len(vocab)
EMBED_DIM  = 50
LR         = 0.01
EPOCHS     = 100

model = AttnCBOW(VOCAB_SIZE, EMBED_DIM)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# 构造张量
inputs = [ctx for ctx, tgt in data]
targets = [tgt for ctx, tgt in data]
inputs = torch.tensor(inputs, dtype=torch.long)   # [N, C]
targets = torch.tensor(targets, dtype=torch.long) # [N]

# 4. 训练循环
for epoch in range(1, EPOCHS+1):
    logits = model(inputs)             # [N, V]
    loss = criterion(logits, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d}, Loss: {loss.item():.4f}")

# 5. 查看训练后的词向量
embeddings = model.embed.weight.data    # [V, E]
for w, idx in word2idx.items():
    vec = embeddings[idx].tolist()
    print(f"{w:>12s} → {vec[:5]}...")   # 只显示前5维
'''
1.为什么视频中说用随机值初始化词向量？
2.掩码自注意力机制是什么
3.双向编码器表征模型
'''