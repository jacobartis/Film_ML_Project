import numpy as np
import pandas as pd
import pytorch as pt
from matplotlib import pyplot as plt

data_set = pd.read_csv("movie_statistic_dataset.csv")
print(data_set.keys())

graph = plt.plot(data_set["movie_averageRating"],data_set["Domestic gross $"], '.', alpha=.5, label='cars')

plt.show()