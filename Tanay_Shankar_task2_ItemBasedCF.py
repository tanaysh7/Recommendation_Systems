
# coding: utf-8

# ## Item based collaborative Filtering
# 

# In[21]:


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





# In[ ]:


sc = SparkContext("local[*]",appName="inf553")
sc.setLogLevel("ERROR")


# In[23]:


filename="ratings"
rdd = sc.textFile("Assignment_03/data/"+filename+".csv",8) #sys.argv[2]
rdd = rdd.mapPartitions(lambda x: csv.reader(x))
header = rdd.first() #extract header
data = rdd.filter(lambda row: row != header)   #filter out header


# In[24]:


data = data.map(lambda x:((int(x[0]),int(x[1])),float(x[2])))


# In[25]:


testfilename="testing_small"
rdd1 = sc.textFile("Assignment_03/data/"+testfilename+".csv",8) #sys.argv[2]
rdd1 = rdd1.mapPartitions(lambda x: csv.reader(x))
heading = rdd1.first() #extract header
test = rdd1.filter(lambda row: row != heading)
test= test.map(lambda x:((int(x[0]),int(x[1])),0))#filter out header


# In[26]:


train=data.subtractByKey(test).cache()


# In[116]:


tdata=data.join(test).map(lambda x:(x[0],x[1][0])).cache()


# In[48]:


userdata=train.map(lambda x: x[0]).groupByKey().map(lambda x: (x[0],list(x[1]))).sortBy(lambda x: x[0]).collect()


# In[27]:


trainmovies=train.map(lambda x:(x[0][1],1)).groupByKey().map(lambda x:x[0]).sortBy(lambda x:x).collect()
mdict=dict()
for i in range(len(trainmovies)):
    mdict[trainmovies[i]]=i


# In[28]:


testextra=test.filter(lambda x: x[0][1] not in trainmovies)
test=test.filter(lambda x: x[0][1] in trainmovies)


# In[29]:


userlist=(range(671))
mcount=len(trainmovies)
ucount=len(userlist)


# In[30]:

traindata=train.map(lambda x:(x[0][1],(x[0][0],x[1]))).groupByKey().sortByKey().collect()
matm=dict()
for j in traindata:
    empty= np.array([np.NaN for n in range(671)])
    for i in list(j[1]):
        empty[i[0]-1]=i[1]
    matm[j[0]]=empty

# In[31]:

trainudata=train.map(lambda x:(x[0][0],(x[0][1],x[1]))).groupByKey().sortByKey().collect()
matu=dict()
for j in trainudata:
    empty= np.array([np.NaN for n in range(len(trainmovies))])
    for i in list(j[1]):
        empty[mdict[i[0]]]=i[1]
    matu[j[0]]=empty

# In[32]:


testlist=test.map(lambda x:(x[0][0],x[0][1])).groupByKey()


# In[33]:


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


# In[164]:


def similarMovies(record):
    movies = record[1]
    user   = record[0]
    moviesratedbyuser=np.where(~np.isnan(matu[user]))[0]
    ans=[]
    for i in movies:
        wlist=[]
        rlist=[]
        denom=0
        numer=0
        for j in moviesratedbyuser:
            corr = np.intersect1d(np.where(~np.isnan(matm[i])),np.where(~np.isnan(matm[trainmovies[j]])))
            if(len(corr)>0):
                w=similarity(matm[i],matm[trainmovies[j]],corr)
                if(w!=0):
                    wlist.append(w)
                    rlist.append(matu[user][j])
        
        numer=np.sum(np.multiply(np.array(wlist),np.array(rlist)))
        denom=np.sum(np.absolute(wlist))            
        
        
        if len(wlist)>20:
            order=list(np.array(wlist).argsort()[::-1][:20])
            numer=np.sum(np.multiply(np.array(wlist)[order],np.array(rlist)[order]))
            denom=np.sum(np.absolute(wlist)[order])
        
        
        if (denom ==0 or numer==0):
            numer=np.nanmean(np.concatenate((matm[i],matu[user]), axis=0))
            denom=1
        op=(numer/denom)
        if op > 5:
            op=5
        if op < 1:
            op=1
        ans.append((user,i,op))
    return ans


# In[64]:



matm_01=dict()
for i in trainmovies:
    empty=[]
    for j in userdata:
        if i in list(j[1]):
            empty.append(1)
        else:
            empty.append(0)
    matm_01[i]=empty


def newhash(nhash,userlist):
    hdict=dict(((i, []) for i in range(nhash)))
    for i in range(nhash):
        k=[(37*i+23*m)%(len(userlist)) for m in userlist]
        hdict[i]=k
    return hdict

hashnum=60
numofband=20
hashlist=newhash(hashnum,range(ucount))

hashmat = dict()
for i in trainmovies:
    hashmat[i]=[ucount for j in range(hashnum) ]
    
for i in matm_01:
    for j in range(ucount):
        if(matm_01[i][j]==1):
            for k in range(hashnum):
                hashmat[i][k]=min(hashmat[i][k],hashlist[k][j])
                

def LSH(chunk):
    op=dict()
    chunk=list(chunk)
    opjaccard=[]
    for j in range(len(chunk[0])):
        key=tuple(chunk[i][j] for i in range(len(chunk)))
        if(key) in op:
            op[key].append(trainmovies[j])
        else:
            op[key]=[trainmovies[j]]
    op=[list(combinations(i,2)) for i in op.values() if len(i)>1]
    return iter(item for sublist in op for item in sublist)

def jaccard(a,b):
    unn=0
    intr=0
    for i in range(len(a)):
        sum=a[i]+b[i]
        if(sum>=1):
            unn+=1
            if(sum==2):
                intr+=1
    if unn==0:
        return 0
    return float(intr)/unn


# In[67]:



bands=sc.parallelize(np.array([hashmat[i] for i in trainmovies]).T,numofband)
finalans=bands.mapPartitions(LSH).map(lambda x:(x,1)).groupByKey().map(lambda x:x[0]).map(lambda i: (i,jaccard(matm_01[i[0]],matm_01[i[1]])) ).filter(lambda x: x[1]>=0.5).collect()

# In[118]:


jaccardmovies={ i[0] for i in finalans}


# In[138]:


def similarLSHMovies(record):
    movies = record[1]
    user   = record[0]
    moviesratedbyuser=np.where(~np.isnan(matu[user]))[0]
    ans=[]
    for i in movies:
        wlist=[]
        rlist=[]
        denom=0
        numer=0
        for j in moviesratedbyuser:
            if tuple(sorted((i,j))) in jaccardmovies:
                corr = np.intersect1d(np.where(~np.isnan(matm[i])),np.where(~np.isnan(matm[trainmovies[j]])))
                if(len(corr)>0):
                    w=similarity(matm[i],matm[trainmovies[j]],corr)
                    if(w!=0):
                        wlist.append(w)
                        rlist.append(matu[user][j])
        
        
        numer=np.sum(np.multiply(np.array(wlist),np.array(rlist)))
        denom=np.sum(np.absolute(wlist)) 
      
        if (denom ==0 or numer==0):
            numer=np.nanmean(np.concatenate((matm[i],matu[user]), axis=0))
            denom=1
        op=(numer/denom)
        if op > 5:
            op=5
        if op < 1:
            op=1
        ans.append((user,i,op))
    return ans


# In[165]:
extraRes=testextra.map(lambda x:(x[0],np.nanmean(matu[x[0][0]])))

t0 = time.time()

predictions=testlist.flatMap(similarMovies).map( lambda x: ((x[0],x[1]),x[2]) ).cache()

kall=extraRes.union(predictions)


ratesAndPreds = tdata.map(lambda r: ((r[0][0], r[0][1]), r[1])).join(kall)
MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
print("Root Mean Squared Error without LSH = " + str(math.sqrt(MSE)))


allans=ratesAndPreds.collect()

final=dict({i:0 for i in range(5)})
for i in allans:
        k=math.floor(abs(i[1][0]-i[1][1]))
        if k>=4:
            k=4
        final[k]+=1

		


print "Predictions without LSH"
print final
		
		
t1 = time.time()

total = t1-t0
print 'Time taken :'+str(total)


t0 = time.time()

predictionsLSH=testlist.flatMap(similarLSHMovies).map( lambda x: ((x[0],x[1]),x[2]) ).cache()


kallLSH=extraRes.union(predictionsLSH)




ratesAndPredsLSH = tdata.map(lambda r: ((r[0][0], r[0][1]), r[1])).join(kallLSH)
MSE = ratesAndPredsLSH.map(lambda r: (r[1][0] - r[1][1])**2).mean()

print("Root Mean Squared Error with LSH = " + str(math.sqrt(MSE)))

allansLSH=ratesAndPredsLSH.collect()

finalLSH=dict({i:0 for i in range(5)})
for i in allansLSH:
        k=math.floor(abs(i[1][0]-i[1][1]))
        if k>=4:
            k=4
        finalLSH[k]+=1
		
print "Predictions with LSH"
print finalLSH


t1 = time.time()

total = t1-t0
print 'Time taken :'+str(total)






# In[ ]:




# In[132]:


printans=[(i[0],i[1][1])for i in allansLSH]


# In[133]:


with open('Tanay_Shankar_task2_ItemBasedCF.txt', 'wb') as f: 
    for i in sorted(printans):
        f.write(str(i).replace('(',"").replace(')',""))
        f.write('\n')
f.close()


