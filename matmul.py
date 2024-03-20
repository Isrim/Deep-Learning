import matplotlib.pyplot as plt
import numpy as np
from numba import jit
import time

@jit(nopython=True, cache=True, fastmath=False)
def matmul_naive(A, B, M, N, K):
	C = np.zeros((M,N))
	for i in range(0,M):
		for j in range(0,N):
			for k in range(0,K):
				C[i,j] += A[i,k] * B[k,j]
	return C



@jit(nopython=True, cache=True, fastmath=False)
def matmul_numpy_sum(A, B, M, N, K):
	C = np.zeros((M,N))
	for i in range(0,M):
		for j in range(0,N):
			C[i,j] = np.sum(A[i,:]*B[:,j])
				
	return C


def plotFLOPS():
    X = np.arange(256,2049,256)
    Y = []
    for i in X:
        A = (np.random.random((i,i))-0.5)*0.1
        B = (np.random.random((i,i))-0.5)-0.1
        t1 = time.time()
        matmul_naive(A,B,i,i,i)
        t2 = time.time()
        Y.append(i**3/(t2-t1))
    plt.plot(X,Y)
    plt.xlabel("taille matrice")
    plt.ylabel("FLOPS")
    plt.show()
    
M = 2048
N = 2048
K = 2048

    
np.random.seed(0)

A = (np.random.random((M,K))-0.5)*0.1
B = (np.random.random((K,N))-0.5)-0.1

#### Select one ####
#C = matmul_naive(A,B,M,N,K) # time = [3.630,3.625,3.607,3.613,3.624] # timeNumba = [0.641,0.626,0.620,0.602,0.621] matrice de taille 256
#C = matmul_numpy_sum(A, B, M, N, K) #time = [0.236,0.237,0.237,0.238,0.238] #timeNumba = [0.364,0.361,0.355,0.368,0.357] matrice de taille 256 
#C = A@B #time = [0.612,0.611,0.610,0.614,0.606] matrice de taille 2048
#C = np.matmul(A,B) #time = [0.360,0.383,0.384,0.387,0.385] matrice de taille 2048
#C = np.dot(A,B) #time = [0.613,0.603,0.616,0.612,0.616] matrice de taille 2048

#plotFLOPS()

#print (C[M//2,N//2])









