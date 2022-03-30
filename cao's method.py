import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt 

os.chdir("file path")

data = pd.read_csv("data.csv")
df = data.iloc[:,1:]; df.index = data.date
df = df.dropna()

def create_sequences(data, seq_length):
    xs = []
    for i in range(len(data)-seq_length):
        x = data.iloc[i:(i+seq_length)]
        xs.append(x)
    return np.array(xs).reshape(len(xs), seq_length)

def totalCao(data, seq_length):
    X1 = create_sequences(data, seq_length)
    X2 = create_sequences(data, seq_length+1)
    
    def R(i, X):
        x = X[i]
        lst = [np.sum((X[j]-x)**2) for j in range(len(X)) if np.sum((X[j]-x)**2) != 0]
        return min(lst)**(0.5)
            
    def a(i):
        if R(i, X1) == 0:
            print(i, seq_length)
        return R(i, X2)/R(i, X1)
    
    def E(X):     
        N = len(X)
        ans = 0
        #1. a(i, m)
        for i in range(0, N-seq_length):
            ans += a(i)
        ans = ans/(N-seq_length)
        return ans
    return E(X2)/E(X1)
    
lst_seq = {}
n = len(df)//250
q = len(df)%250
for j in range(1):
    col = df.columns[j]
    lst_seq[col] = {}
    for k in range(n):
        lst_seq[col][k] = []
        if k < n-1:
            for seq_length in range(1,30):
                lst_seq[col][k].append(totalCao(df.iloc[:,j][250*k:250*(k+1)], seq_length))
                print(seq_length)
        else:
            for seq_length in range(1,30):
                lst_seq[col][k].append(totalCao(df.iloc[:,j][205*k:], seq_length))
                print(seq_length) 

for k in range(n):
    plt.plot(lst_seq['Nation'][k], label = str(k+1)+"th")
    plt.legend(loc = "best", ncol = k)

lst_seq_pd = pd.DataFrame(lst_seq)
lst_seq_pd.to_excel("E1_data.xlsx")
