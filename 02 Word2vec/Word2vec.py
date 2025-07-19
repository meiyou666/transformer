import numpy as np

# 语料和词表
corpus = ["I like natural language processing", "I like deep learning"]
vocab = list(set(" ".join(corpus).lower().split()))
word2idx = {w:i for i,w in enumerate(vocab)}
vocab_size = len(vocab)

# 生成样本 (Skip‑gram, 窗口1)
data = []
for s in corpus:
    t = s.lower().split()
    for i, w in enumerate(t):
        if i>0: data.append((word2idx[w], word2idx[t[i-1]]))
        if i<len(t)-1: data.append((word2idx[w], word2idx[t[i+1]]))

# 参数初始化
dim=2
W1 = np.random.randn(vocab_size, dim)
W2 = np.random.randn(dim, vocab_size)
lr=0.1

# 简易训练
for _ in range(100):
    for tgt, ctx in data:
        h = W1[tgt]
        u = W2.T.dot(h)
        exp_u = np.exp(u)
        y = exp_u/exp_u.sum()
        y[ctx] -= 1
        # 权重更新
        W2 -= lr * np.outer(h, y)
        W1[tgt] -= lr * W2.dot(y)

# 输出词向量
print({w: W1[word2idx[w]] for w in vocab})

'''
1.python的语法，能读懂但不熟悉
2.Skip‑gram机制和CBOW具体实现机制和两者优劣对比
3.每个词的embedding表示是不是维度越高越好
4.为什么进行多次训练得到的embedding表示不完全相同
'''