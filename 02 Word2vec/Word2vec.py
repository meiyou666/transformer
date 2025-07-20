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
5.增加训练次数一定能提高模型的精度吗
6.decoder是怎么通过embedding表示生成文字的，为什么由一个embedding表示能生成很多个词
7.Word2vec模型在训练完成后的语料库是static的，在遇到一次多义的时候又是怎么解决的
8.在训练的语料库中，如果出现了同一个单词的一次多义，比如bank是银行的堤坝，是否会导致得到的embedding表示不准确
9.Word2vec的训练实际上也只是将上下文的一些单词建立联系，实际上并没有真正理解单词的含义？

git的使用还停留在初级阶段，没有开发出其全部功能
'''