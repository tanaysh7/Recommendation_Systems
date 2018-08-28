
# coding: utf-8

# In[19]:


import pyspark
import csv
import sys
from pyspark import SparkContext
import collections
from itertools import combinations
import numpy as np
import random
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
import math
import time




sc = SparkContext("local[*]",appName="inf553")
sc.setLogLevel("ERROR")


# In[23]:

t0 = time.time()


filename=sys.argv[1]
rdd = sc.textFile(filename) 
rdd = rdd.mapPartitions(lambda x: csv.reader(x))
header = rdd.first() #extract header
data = rdd.filter(lambda row: row != header)
data= data.map(lambda x:((int(x[0]),int(x[1])),float(x[2])))   #filter out header



# In[59]:


testfilename=sys.argv[2]
rdd1 = sc.textFile(testfilename) #sys.argv[2]
rdd1 = rdd1.mapPartitions(lambda x: csv.reader(x))
heading = rdd1.first() #extract header
test = rdd1.filter(lambda row: row != heading)
test= test.map(lambda x:((int(x[0]),int(x[1])),0))#filter out header



# In[60]:


train=data.subtractByKey(test).map(lambda x: Rating(x[0][0],x[0][1],x[1])).repartition(32).cache()
test=test.map(lambda x:x[0])


# In[61]:


data=data.map(lambda x: Rating(x[0][0],x[0][1],x[1]))


# In[62]:


traincount=train.count()
testcount=test.count()


# In[63]:


# Build the recommendation model using Alternating Least Squares
rank = 5
numIterations = 10
model = ALS.train(train, rank, numIterations,0.1)


# In[64]:


predictions = model.predictAll(test).map(lambda r: ((r[0], r[1]), r[2]))
ratesAndPreds = data.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
print("Root Mean Squared Error = " + str(math.sqrt(MSE)))

# Save and load model
#model.save(sc, "target/tmp/myCollaborativeFilter")
#sameModel = MatrixFactorizationModel.load(sc, "target/tmp/myCollaborativeFilter")


# ## Small
# ### Root Mean Squared Error = 0.948174507981
# ### {0: 13892, 1: 4020, 2: 710, 3: 102, 4: 9}
# ## Big
# ### Root Mean Squared Error = 0.828714545399
# ### {0: 3206075, 1: 745786, 2: 86074, 3: 8176, 4: 220}

# In[65]:


final=dict({i:0 for i in range(5)})


# In[66]:


for i in ratesAndPreds.collect():
    k=math.floor(abs(i[1][0]-i[1][1]))
    if k>=4:
        k=4
    final[k]+=1


# In[67]:


print final


# In[68]:


with open('Tanay_Shankar_ModelBasedCF.txt', 'wb') as f: 
    for i in ratesAndPreds.map(lambda x:(x[0],x[1][1])).sortBy(lambda x:(x[0],x[1]),ascending=True).collect():
        f.write(str(i).replace('(',"").replace(')',""))
        f.write('\n')
f.close()

t1 = time.time()

total = t1-t0
print 'Time taken :'+str(total)

