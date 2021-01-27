# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 10:47:12 2021

@author: Siddhartha
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import math

from sklearn import datasets
from sklearn import preprocessing
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

P1 = pd.read_csv('01-12/DrDoS_DNS.csv',nrows=200000)
P1 = P1.sample(n=100,random_state=36)
P2 = pd.read_csv('01-12/DrDoS_LDAP.csv',nrows=200000)
P2 = P2.sample(n=100,random_state=36)
P3 = pd.read_csv('01-12/DrDoS_MSSQL.csv',nrows=200000)
P3 = P3.sample(n=100,random_state=36)
P4 = pd.read_csv('01-12/DrDoS_NetBIOS.csv',nrows=200000)
P4 = P4.sample(n=100,random_state=36)
P5 = pd.read_csv('01-12/DrDoS_NTP.csv',nrows=200000)
P5 = P5.sample(n=100,random_state=36)
P6 = pd.read_csv('01-12/DrDoS_SNMP.csv',nrows=200000)
P6 = P6.sample(n=100,random_state=36)
P7 = pd.read_csv('01-12/DrDoS_SSDP.csv',nrows=200000)
P7 = P7.sample(n=100,random_state=36)
P8 = pd.read_csv('01-12/DrDoS_UDP.csv',nrows=200000)
P8 = P8.sample(n=100,random_state=36)
P9 = pd.read_csv('01-12/Syn.csv',nrows=200000)
P9 = P9.sample(n=100,random_state=36)
P10 = pd.read_csv('01-12/TFTP.csv',nrows=200000)
P10 = P10.sample(n=100,random_state=36)
P11 = pd.read_csv('01-12/UDPLag.csv',nrows=200000)
P11 = P11.sample(n=100,random_state=36)

PDF = pd.concat([P1,P2,P3,P4,P5,P6,P7,P8,P9,P10,P11])

def Pre_process_data(df,col):
    '''
    Input: Data-frame and Column name.
    Operation: Fills the nan values with the minimum value in their respective column.
    Output: Returns the pre-processed data-frame.lets stop h
    '''
    #df['primary_use'] = df['primary_use'].astype("category").cat.codes
    print("Name of column with NaN: "+str(col))
    print(df[col].value_counts(dropna=False, normalize=True).head())
    df[col].replace(np.inf, -1, inplace=True)
    
    return df

def reduce_mem_usage(df):
    '''
    Input - data-frame.
    Operation - Reduce memory usage of the data-frame.
    '''
    start_mem_usg = df.memory_usage().sum() / 1024**2 
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    #NAlist = [] # Keeps track of columns that have missing values filled in. 
    for col in df.columns:
        if df[col].dtype != object:  # Exclude strings            
            # Print current column type
            print("******************************")
            print("Column: ",col)
            print("dtype before: ",df[col].dtype)            
            # make variables for Int, max and min
            IsInt = False
            mx = df[col].max()
            mn = df[col].min()
            #print("min for this col: ",mn)
            #print("max for this col: ",mx)
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(df[col]).all(): 
                #NAlist.append(col)
                df = Pre_process_data(df,col)
                   
            # test if column can be converted to an integer
            asint = df[col].fillna(0).astype(np.int64)
            result = (df[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True            
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        df[col] = df[col].astype(np.uint8)
                    elif mx < 65535:
                        df[col] = df[col].astype(np.uint16)
                    elif mx < 4294967295:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)    
            # Make float datatypes 32 bit
            else:
                df[col] = df[col].astype(np.float32)
            
            # Print new column type
            print("dtype after: ",df[col].dtype)
            print("******************************")
    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = df.memory_usage().sum() / 1024**2 
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return df

PDF = reduce_mem_usage(PDF)

class_names = pd.unique(PDF[' Label'])

PDF = PDF.drop(['Unnamed: 0', 'Flow ID', ' Source IP',' Source Port', 'SimillarHTTP',' Destination IP',' Destination Port',' Timestamp','Fwd Packets/s','Flow Bytes/s'], axis = 1) 

colList = [i for i in PDF.columns]

# We cannot normalize a DataFrame with a column with data type Object
# So, we give ordinal number to each

enc = OrdinalEncoder(dtype = 'int8').fit_transform(PDF)

N = pd.DataFrame(enc, columns = colList)
B = N[' Label']
A = N.drop([' Label'], axis = 1)

B = pd.DataFrame(B, columns = [' Label'])

X = StandardScaler().fit_transform(A)
X_Norm = np.array(preprocessing.normalize(X))
X_norm = pd.DataFrame(X_Norm, columns = colList[0:77])
X_norm[' Label'] = B
y = N[' Label']

X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.20)

scaler = StandardScaler().fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)



# import some data to play with
iris = datasets.load_iris()

# we only take the first two features. We could avoid this ugly
# slicing by using a two-dim dataset
# X = iris.data[:, :2]
# y = iris.target

h = .02  # step size in the mesh

# Create color maps
cmap_light = ['#BA68C8', '#7986CB', '#4FC3F7', '#4DB6AC', '#81C784', 
                             '#AED581', '#FFF176', '#FFD54F', '#FFB74D', '#E0E0E0', '#90A4AE','#EE98FB']
cmap_bold  = ['#8E24AA', '#3949AB', '#039BE5', '#00897B', '#43A047',
                             '#7CB342', '#FDD835', '#FFB300', '#FB8C00', '#757575', '#546E7A','#883997']

for weights in ['uniform', 'distance']:
    # we create an instance of Neighbours Classifier and fit the data.
    clf = KNeighborsClassifier(3, weights=weights)
    
    X = X_norm[[' Protocol', ' Flow Duration']]
    X=np.array(X)
    clf.fit(X, y)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    xx = np.array([[i/300, j/300] for i in range(math.floor(X[:, 0].min()*300), math.ceil(X[:, 0].max()*300), 2) for j in range(math.floor(X[:, 1].min()*300), math.ceil(X[:, 1].max()*300), 2)])
    print("Started")
    Z = clf.predict(xx)
    
    print("Ended")
    
    # Put the result into a color plot
#     Z = Z.reshape((int(1332/2), 2))
    Z=pd.DataFrame(Z)
    plt.figure(figsize=(8, 6))
    #plt.contourf(xx, yy, cmap=cmap_light)
    
    Q = pd.DataFrame(X)
    P = pd.DataFrame(xx)
    
    fig = plt.figure(figsize=(8, 6))
    for i in range(11663):
        print(i)
        plt.scatter(x=P.iloc[i][0], y=P.iloc[i][1], c=cmap_light[int(Z.iloc[i][0])])
        
        
    fig = plt.figure(figsize=(8, 6))
    for i in range(y.size):
        plt.scatter(x=Q.iloc[i][0], y=Q.iloc[i][1], c=cmap_bold[int(y.iloc[i][0])])
    # Plot also the training points
#     sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=class_names, palette=cmap_bold, alpha=1.0, edgecolor="black")
    plt.xlim(-0.5, 0.27)
    plt.ylim(-0.4, 0.37)
    plt.title("11-Class classification (k = %i, weights = '%s')" % (3, weights))
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])

plt.show()