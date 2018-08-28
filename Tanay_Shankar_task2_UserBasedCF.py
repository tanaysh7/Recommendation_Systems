
# ## User based collaborative Filtering
# 


import pyspark
import csv
import sys
from pyspark import SparkContext
import collections
from itertools import combinations
import numpy as np
import random
import math
import time





sc = SparkContext("local[*]",appName="inf553")
sc.setLogLevel("ERROR")

t0 = time.time()


filename=sys.argv[1]
rdd = sc.textFile(filename) 
rdd = rdd.mapPartitions(lambda x: csv.reader(x))
header = rdd.first() #extract header
data = rdd.filter(lambda row: row != header)   #filter out header


# In[6]:


data = data.map(lambda x:((int(x[0]),int(x[1])),float(x[2])))


# In[7]:


testfilename=sys.argv[2]
rdd1 = sc.textFile(testfilename) #sys.argv[2]
rdd1 = rdd1.mapPartitions(lambda x: csv.reader(x))
heading = rdd1.first() #extract header
test = rdd1.filter(lambda row: row != heading)
test= test.map(lambda x:((int(x[0]),int(x[1])),0))#filter out header


# In[8]:


train=data.subtractByKey(test).cache()


# In[9]:


trainmovies=train.map(lambda x:(x[0][1],1)).groupByKey().map(lambda x:x[0]).sortBy(lambda x:x).collect()
mdict=dict()
for i in range(len(trainmovies)):
    mdict[trainmovies[i]]=i


# In[10]:


testextra=test.filter(lambda x: x[0][1] not in trainmovies)
test=test.filter(lambda x: x[0][1] in trainmovies)


# In[11]:


mcount=len(trainmovies)


# In[12]:


trainudata=train.map(lambda x:(x[0][0],(x[0][1],x[1]))).groupByKey().sortByKey().collect()
matu=dict()
for j in trainudata:
    empty= np.array([np.NaN for n in range(len(trainmovies))])
    for i in list(j[1]):
        empty[mdict[i[0]]]=i[1]
    matu[j[0]]=empty

# In[13]:

traindata=train.map(lambda x:(x[0][1],(x[0][0],x[1]))).groupByKey().sortByKey().collect()
matm=dict()
for j in traindata:
    empty= np.array([np.NaN for n in range(len(matu))])
    for i in list(j[1]):
        empty[i[0]-1]=i[1]
    matm[j[0]]=empty

# In[14]:


testlist=test.map(lambda x:(x[0][1],x[0][0])).groupByKey()


# In[15]:


def similarity(array1,array2,correlatedindices):
    av1=np.nanmean(array1[correlatedindices])
    av2=np.nanmean(array2[correlatedindices])
    num=0
    den1=0
    den2=0                                          
    for i in correlatedindices:
        num+=(array1[i]-av1)*(array2[i]-av2)
        den1+=(array1[i]-av1)**2
        den2+=(array2[i]-av2)**2
    den=(math.sqrt(den1)*(math.sqrt(den2)))
    if den==0:
        return 0
    return num/den                                       


# In[16]:


def similarUsers(record):
    movie = record[0]
    users=record[1]
    userswhorated=np.where(~np.isnan(matm[movie]))[0]
    ans=[]
    for i in users:
        numer=0
        denom=0
        for j in userswhorated:
            corr = np.intersect1d(np.where(~np.isnan(matu[i])),np.where(~np.isnan(matu[j+1])))
            if(len(corr)>0):
                w=similarity(matu[i],matu[j+1],corr)
                numer+=w*(matm[movie][j]-np.nanmean(matu[j+1][corr]))
                denom+=abs(w)
       
    
    #implement nearest neighbour
        if denom == 0:
            numer=0
            denom=1
        op=np.nanmean(matu[i])+(numer/denom)
        
        if op > 5:
            op=5
        if op < 1:
            op=1
            
        ans.append((i,movie,op))
    return ans


# In[17]:


predictions=testlist.flatMap(similarUsers).map( lambda x: ((x[0],x[1]),x[2]) ).cache()


# In[18]:


extraRes=testextra.map(lambda x:(x[0],np.nanmean(matu[x[0][0]])))
kall=extraRes.union(predictions)


# In[19]:


ratesAndPreds = data.map(lambda r: ((r[0][0], r[0][1]), r[1])).join(kall)
MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
print("Root Mean Squared Error = " + str(math.sqrt(MSE)))


# In[20]:


allans=ratesAndPreds.collect()


# In[21]:


final=dict({i:0 for i in range(5)})
for i in allans:
        k=math.floor(abs(i[1][0]-i[1][1]))
        if k>=4:
            k=4
        final[k]+=1


# In[22]:


print final




printans=[(i[0],i[1][1])for i in allans]


# ## Root Mean Squared Error = 0.947287744943
# ### Ratings
# ### {0: 14967, 1: 4429, 2: 722, 3: 127, 4: 11}



with open('Tanay_Shankar_task2_UserBasedCF.txt', 'wb') as f: 
    for i in sorted(printans):
        f.write(str(i).replace('(',"").replace(')',""))
        f.write('\n')
f.close()

t1 = time.time()

total = t1-t0
print 'Time taken :'+str(total)