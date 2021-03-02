import math
import csv
import time
import numpy as np
import collections
from mpi4py import MPI
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score



#===================================================Devide data set to further scattering=====================
def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    return out
#===================================================Count lables for recalculating means of centroids=====================
def addCounter(counter1, counter2, datatype):
    for item in counter2:
        counter1[item] += counter2[item]
    return counter1

# comm = MPI.COMM_WORLD
# size = comm.Get_size()
# rank = comm.Get_rank()
global dimensions, num_clusters, num_points,dimensions,data,flag
num_clusters=0


#===============================================reading and preparing data set======================

# print("Enter the number of clusters you want to make: ")
num_clusters = 1 # input()
num_clusters = int(num_clusters)
start_time = time.time() 										#turn on a timer which allows us to estimate performane of this algorithm

with open('3D_spatial_network.csv','r') as f:
  # with open('3D_spatial_network.csv','rb') as f:
	reader = csv.reader(f)
	data = list(reader) #make array

data.pop(0)
print(len(data))
'''
data.pop(0)
for i in range (len(data)):
	data[i].pop(0)
data=data[0:10000]
data=np.array(data).astype(np.float)

kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(data).labels_
# 	print('data',[ data[i] for i in [indices] ])
# 	data=np.array ([[2,10],[2,5],[8,4],[5,8],[7,5],[6,4],[1,2],[4,9]])
#====================================================================================================

#================================================Initialize centroids matrix=========================
initial=[]
for i in range(num_clusters):
	initial.append(data[i])
initial=np.vstack(initial)
#====================================================================================================

num_points = len(data)                                    #number of rows
dimensions = len(data[0])                                 #number of columns
#chunks = [ [] for _ in range(size) ]

#for i, chunk in enumerate(data):
#	chunks[i % size].append(chunk)
chunks=chunkIt(data,size)								  #deviding data set on parts for further scattering
'''
