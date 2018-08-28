
import pyspark
import csv
import sys
from pyspark import SparkContext
import collections
from itertools import combinations
import numpy as np
import random
import time




sc = SparkContext("local[*]",appName="inf553")
sc.setLogLevel("ERROR")


# In[23]:

t0 = time.time()

filename=sys.argv[1]
rdd = sc.textFile(filename)
rdd = rdd.mapPartitions(lambda x: csv.reader(x))
header = rdd.first() #extract header
data = rdd.filter(lambda row: row != header)   #filter out header


data = data.map(lambda x:(int(x[0]),int(x[1])))
distmovies=data.map(lambda x: x[1]).distinct().sortBy(lambda x: x).collect()



data=data.groupByKey().map(lambda x: (x[0],list(x[1]))).sortBy(lambda x: x[0])
datas=data.collect()
distusers= [i[0] for i in datas]


mcount=len(distmovies)
ucount=len(distusers)



matm=dict()
for i in distmovies:
    empty=[]
    for j in datas:
        if i in j[1]:
            empty.append(1)
        else:
            empty.append(0)
    matm[i]=empty



def newhash(nhash,userlist):
    hdict=dict(((i, []) for i in range(nhash)))
    for i in range(nhash):
        k=[(7*i+23*m)%(len(userlist)) for m in userlist]
        hdict[i]=k
    return hdict



hashnum=60
numofband=20
hashlist=newhash(hashnum,range(ucount))


hashmat = dict()
for i in distmovies:
    hashmat[i]=[ucount for j in range(hashnum) ]




for i in matm:
    for j in range(ucount):
        if(matm[i][j]==1):
            for k in range(hashnum):
                hashmat[i][k]=min(hashmat[i][k],hashlist[k][j])


bands=sc.parallelize(np.array([hashmat[i] for i in distmovies]).T,numofband)




def LSH(chunk):
    op=dict()
    chunk=list(chunk)
    opjaccard=[]
    for j in range(len(chunk[0])):
        key=tuple(chunk[i][j] for i in range(len(chunk)))
        if(key) in op:
            op[key].append(distmovies[j])
        else:
            op[key]=[distmovies[j]]
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
    return float(intr)/unn


finalans=bands.mapPartitions(LSH).distinct().map(lambda i: (i,jaccard(matm[i[0]],matm[i[1]])) ).filter(lambda x: x[1]>=0.5).collect()



print "Total number of pairs :"
print len(finalans)



with open('Tanay_Shankar_SimilarMovie_Jaccard.txt', 'wb') as f: 
    for i in finalans:
        f.write(str(i).replace('(',"").replace(')',""))
        f.write('\n')
f.close()

t1 = time.time()

total = t1-t0
print 'Time taken :'+str(total)

