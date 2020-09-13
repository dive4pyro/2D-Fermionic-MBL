from numpy import argsort, array
import numpy as np

'''
this file contains code that converts a particle number conserving matrix from
block-diagonal form to the "usual" occupation number basis

the code below is probably quite unclear but I can write notes on it if needed.
'''


#produces an array of particle numbers corresponding to basis states
def ParticleNum(N):
    a = array([0,1])
    for i in range(N-1):
        a = np.append(a,a+1)
    return a

def permute_order(N):
    particleNumbers = ParticleNum(N)
    return argsort(argsort(particleNumbers))
'''
the double argsort is there because we want the INVERSE of the permutation,
that is we want to go away from block diagonal rather than to block diagonal
(in this case the 'desired' permutation is P^-1,where P is the "to block diagonal permutation")
'''


def permuteBasis(H,order): #where order is the inverse of the desired permutation
    H = H[order] #permute rows
    H = H.T[order].T #permute columns
    return H
