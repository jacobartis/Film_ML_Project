import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from matplotlib import pyplot as plt

raw_data = pd.read_csv("movie_statistic_dataset.csv")
print(raw_data.keys())

data_set = []
labels = []

for x in range(len(raw_data.values)):
    entry = [int(raw_data['Production budget $'][x]),int(raw_data['Worldwide gross $'][x]),float(raw_data['runtime_minutes'][x])]
    data_set.append(entry)
    labels.append(raw_data['movie_averageRating'][x])
x_training, x_test, y_training, y_test = train_test_split(data_set,labels,random_state=0)

scores = []
best_k = 0

for k in range(20,250):
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(x_training,y_training)
    score = knn.score(x_test,y_test)
    if score>=0:
        scores.append(knn.score(x_test,y_test))
    if score>=max(scores):
        best_k = k
    print(k)


plt.plot(range(20,250),scores,"*")
plt.show()
print("best k: ", best_k)