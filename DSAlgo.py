from sklearn import preprocessing as pr
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from timeit import default_timer as timer

from sklearn.cluster import DBSCAN
d1 = pd.read_csv('UDP.csv', nrows=650000, low_memory=False)
d2 = pd.read_csv('Syn.csv', nrows=650000, low_memory=False)

start = timer()
print(d1.columns)
print(d2.columns)

features = d1.columns+d2.columns
D = pd.DataFrame(d1.append(d2))
D = D.select_dtypes('int64')
print(D)
normalized_features = pd.DataFrame(pr.normalize(D, norm='l2', copy=True, axis = 1))
print(normalized_features)
fig = plt.figure(figsize=(80, 80))

print(timer() - start)

print( pd.DataFrame([1]*normalized_features.shape[0]).shape[0])
print(normalized_features.shape[0])

cm = plt.get_cmap('gist_rainbow')
print(timer() - start)

# for i in range(37):
#     for j in range(37):
#         if i != j:
#             fig = plt.figure(figsize=(80, 10))
#             plt.scatter(abs(normalized_features[i]), abs(normalized_features[i]), color = 'g', s = 20, marker = ".")
    
for i in range(37):
    fig = plt.figure(figsize=(80, 10))
    plt.scatter(100* abs(normalized_features[i]), pd.DataFrame([i]*normalized_features.shape[0]), color = 'g', s = 20, marker = ".")
    # plt.hist(abs(normalized_features[i]), color = cm(1.*i/37), edgecolor = 'black', bins = 1)

    # # seaborn histogram
    # sns.distplot(abs(normalized_features[i]), hist=True, kde=True, bins=1, color = cm(1.*i/37))


                 
# plt.close('all')                
# print("Step 1 finished")
# print(timer() - start)
# kde = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(normalized_features)

# print("Step 2 finished")
# print(timer() - start)
# log_dens = kde.score_samples(normalized_features[1].reshape(1,-1))

# print("Step 3 finished")
# print(timer() - start)
# fig.plot(normalized_features[1], np.exp(log_dens)+1, fc='#AAAAFF')

# print("Step 4 finished")
# print(timer() - start)
# print(normalized_features.columns)
# normalized_features.plot(x=normalized_features, y=pd.DataFrame([True]*normalized_features.shape[1]), s= '+')
# len(features)
# features.drop(['Unnamed: 0'],axis = 1)

# print(d1[' Label'].unique())
# print(d1[' Label'].value_counts())
# print(d2[' Label'].value_counts())

# NUM_COLORS = features.shape[0]
# Index = [i for i in range(d1.shape[1])]
# cm = plt.get_cmap('YlOrRd')
# fig = plt.figure(figsize=((2**15 - 1) / 1000, (2**15 - 1) / 1000), dpi=1000)
# ax = d1.plot(color=[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
# # ax.set_prop_cycle(color= [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
# fig.savefig('moreColorsA.png')
# fig.show()
db
