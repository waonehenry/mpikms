# Parallel-implementation-of-k-means-
The main idea of this research work was to get acquainted with the basic concepts of parallel programming, apply parallelism in k-means clustering algorithm, and thereby estimate the boost of performance.
The idea was realized on Python(v 2.7.14) program language. It was decided increase performance of one by using open MPI Message Passing Interface library project. Implementation open MPI reached through the ”mpi4py”Python package.
At the beginning of this research work it was calculated Davies–Bouldin index for differ- ent number of clusters to estimate the best one (number of clusters). It was found the best number of cluster for each data sets that were presented here. Further there were written and shown the parallel and sequential k-means algorithms in Python language using ”mpi4py” package. To validate these programs (to show that written algorithms work correctly) it was carried out comparing of clustering vectors (clustering vectors it’s the vector that consists of the labels were obtained after clustering) between handwritten k-means functions and k-means from ”sklearn.cluster” Python library. The comparison was done using Adjust rand index from ”sklearn.metrics” Python library. After it was carried out series of experiments for parallel and sequential algorithms on data sets of different sizes. The result of testing showed us that par- allel algorithm has more faster run-time of code than sequential, also It was obtained the best number of processes that gives us the best run-time of code and performance boost, respectively.
# mpikms
