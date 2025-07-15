from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# 文本数据
docs = [
    "I absolutely loved this movie, it was fantastic!",
    "What a waste of time. Completely boring.",
    "The film had great acting and a wonderful plot.",
    "I didn't enjoy the movie at all.",
    "An unforgettable experience, truly inspiring.",
    "The worst film I've seen in years.",
    "A masterpiece. I would watch it again.",
    "Not my taste, I fell asleep halfway.",
    "Beautiful cinematography and emotional story.",
    "Terrible script and poor performances.",
    "This was an excellent film, very well made.",
    "I regret watching this. It was horrible.",
    "Heartwarming and thought-provoking.",
    "Disappointing. I expected much more.",
    "Absolutely amazing! The cast did a great job.",
    "Awful. The plot made no sense.",
    "Very touching, I cried at the end.",
    "Bad direction, bad acting, bad everything.",
    "A joy to watch, I smiled throughout.",
    "Waste of money and time.",
    "Incredible film. Deserves an award.",
    "This movie is garbage. Don't watch it.",
    "Stunning visuals and great pacing.",
    "Mediocre at best.",
    "Fantastic storytelling and a strong message.",
    "The actors were wooden and uninspired.",
    "Deeply moving and artistically done.",
    "The movie dragged on and was predictable.",
    "One of the best movies I've ever seen.",
    "I walked out halfway through. Horrible."
]

# 标签（对应文本的情感类别）
labels = [
    "positive", "negative", "positive", "negative", "positive",
    "negative", "positive", "negative", "positive", "negative",
    "positive", "negative", "positive", "negative", "positive",
    "negative", "positive", "negative", "positive", "negative",
    "positive", "negative", "positive", "negative", "positive",
    "negative", "positive", "negative", "positive", "negative"
]

# 创建一个文本分类模型：BoW + Naive Bayes
model = make_pipeline(CountVectorizer(), MultinomialNB())

# 训练模型
print("🚀 开始训练模型...")
model.fit(docs, labels)
print("✅ 模型训练完成!")

# 测试文档
test_docs = [
    "I love this film",
    "I hate the movie",
    "The movie was enjoyable",
    "The film is bad",
    "This is a fantastic movie",
]

# 输出预测结果
for doc in test_docs:
    prediction = model.predict([doc])[0]
    print(f"【{doc}】→ Predicted: {prediction}")

'''
1.词汇表是如何构建的？
   词汇表是通过 `CountVectorizer` 自动从训练数据中提取的。它会将每个文档转换为一个向量，向量的每个元素对应于词汇表中的一个词，表示该词在文档中出现的次数
2.模型是如何进行训练的？
    模型使用 `MultinomialNB` 进行训练。它基于词频来计算每个类别的概率，并使用贝叶斯定理进行分类。训练过程中，模型会学习每个词在不同类别中的分布情况
3.如何使用模型进行预测？
    使用 `model.predict()` 方法可以对新文档进行预测。该方法会将文档转换为向量形式，然后使用训练好的模型进行分类，返回预测的类别标签
4.这种方式进行的训练，是不是只要数据集一样，模型就是相同的？
    是的，只要训练数据集相同，模型的结构和参数也会相同。模型的训练过程是确定性的，即相同的数据集会产生相同的模型
5.BoW模型中是怎么解决能否解决一词多义的问题
    BoW模型本身并不处理一词多义的问题。它将每个词视为独立的特征，不考虑上下文。因此，对于一词多义的情况，BoW模型可能无法正确区分不同含义的词。更复杂的模型（如Word2Vec或BERT）可以通过上下文来处理一词多义的问题
6.怎样获取大量的训练数据集？
    获取大量训练数据集可以通过以下几种方式：
    - 使用公开的文本数据集，如IMDB电影评论数据集、20 Newsgroups等
    - 爬取网络上的文本数据，如新闻文章、社交媒体评论等
    - 使用众包平台（如Amazon Mechanical Turk）收集用户生成的内容
    - 利用现有的文本数据进行数据增强，如同义词替换、数据扩充等方法
7.MultinomialNB()的实现原理
'''