import pandas as pd
import numpy as np
import jieba as jb

# 预处理，
news = pd.read_csv('sqlResult.csv',encoding = 'gb18030');
print(news.shape)
print(news.head())

news[news.content.isna()].head()
news = news.dropna(subset=['content'])
print(news.shape)
# 去掉常用词汇，
with open('chinese_stopwords.txt', 'r', encoding= 'utf-8') as file:
    stopwords = [i[:-1] for i in file.readlines()]

def split_text(text):
    text = text.replace(' ', '').replace('\n', '')
    text2 = jb.cut(text)
    result = ' '.join([w for w in text2 if w not in stopwords])
    return result
    
print(news.iloc[0].content)
print(split_text(news.iloc[0].content))

# 提取特征值TF-IDF Term Frequency - In
corpus = list(map(split_text, [str(i) for i in news.content]))
corpus

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(tfidf.toarray(), lable, test_size=0.3)



# 使用二分类进行训练，得到模型，比较模型的precesion, accurancy, recall，选择一个比例高的算法。

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, precision_score, recall_score
y_predict = model.predict(X_test)

print("准确率：",accuracy_score(y_test, y_predict))
print("精确率: ", precision_score(y_test, y_predict))



# 基于得到的模型对数据进行预测，得出有可能是copy的文章

prediction = model.predict(tfidf.toarray())
labels = np.array(lable)
compare_news_index = pd.DataFrame({'prediction' : prediction, 'labels' : labels})
copy_news_index = compare_news_index[(compare_news_index['prediction']==1)&(compare_news_index['labels']==0)].index
xhs_news_index = compare_news_index[(compare_news_index['labels']==1)].index


# 基于特征值，对数据进行聚类
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
nomalizer = Normalizer()
scaled_array = nomalizer.fit_transform(tfidf.toarray())
kmeans = KMeans(n_clusters=25)
k_labels = kmeans.fit_predict(scaled_array)
k_labels

# 在某个聚类的范围内，进行相似度文章的查找，并列出top 10
id_class = {index:class_ for index, class_ in enumerate(k_labels)}

from collections import defaultdict
class_id = defaultdict(set)
for index, class_ in id_class.items():
    if index in xhs_news_index.tolist():
        class_id[class_].add(index)

from sklearn.metrics.pairwise import cosine_similarity
def find_simiar_paper(cpindex, top=10):
    dist_dict = {i:cosine_similarity(tfidf[cpindex], tfidf[i]) for i in class_id[id_class[cpindex]]}
    return sorted(dist_dict.items(), key=lambda x:x[1][0], reverse=True)[:top]

cpindex =3352
print(cpindex in xhs_news_index)
print(cpindex in copy_news_index)

similar_list = find_simiar_paper(cpindex)
print(similar_list)



print(news.iloc[cpindex].content)
print(news.iloc[similar_list[0][0]].content)