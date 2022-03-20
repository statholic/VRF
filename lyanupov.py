from math import log
import pandas as pd
import matplotlib.pyplot as plt 

def d(series,i,j):
    return abs(series[i]-series[j])

# assign your file path at cwd
cwd = ""
f=pd.read_csv(cwd + "data.csv")
f=(f['A']-f['A'].mean())/f['A'].std()
series=[float(i) for i in f]
N=len(series)
eps=float(input('Initial diameter bound: '))
dlist=[[] for i in range(N)]
n=0
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
lyapunov_data = []
for i in range(len(dlist)):
    if len(dlist[i]):
        lyapunov_data.append(sum(dlist[i])/len(dlist[i]))
lyapunov_data = pd.Series(lyapunov_data)
lyapunov_data.to_excel(cwd + "lyapunov_data.xlsx")

plt.plot(lyapunov_data)

lyapunov_data.max()