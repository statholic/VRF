#!/usr/bin/env python
# coding: utf-8

# In[1]:


from math import log
import pandas as pd
import matplotlib.pyplot as plt 

def d(series,i,j):
    return abs(series[i]-series[j])

cwd = "C:/Users/문승현/Desktop/2022현상논문/VAE/VAE_rate/"
f=pd.read_csv(cwd + "Rate.csv")
f=(f['Korea']-f['Korea'].mean())/f['Korea'].std()
series=[float(i) for i in f]
N=len(series)
eps=float(input('Initial diameter bound: '))
dlist=[[] for i in range(N)]
n=0 #number of nearby pairs found
for i in range(N):
    for j in range(i+1,N):
        if d(series,i,j) < eps:
            n+=1
            print(n)
            for k in range(min(N-i,N-j)):
                if d(series,i+k,j+k) == 0:
                    pass
                else:
                    dlist[k].append(log(d(series,i+k,j+k)))
lyapunov_rate = []
for i in range(len(dlist)):
    if len(dlist[i]):
        lyapunov_rate.append(sum(dlist[i])/len(dlist[i]))
lyapunov_rate = pd.Series(lyapunov_rate)
lyapunov_rate.to_excel(cwd + "lyapunov_rate.xlsx")


# In[2]:


plt.plot(lyapunov_rate)


# In[3]:


lyapunov_rate.max()


# In[ ]:



