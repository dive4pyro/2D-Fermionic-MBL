from numpy import argsort, array, append
from expm_taylor import torch_expm
import torch
'''
this file contains code that converts a particle number conserving matrix from
block-diagonal form to the "usual" occupation number basis


How this works: (consider N = 3, with Hilbert space dim = 8, but the idea should generalize easily)

The "usual" basis consists of basis vectors: { |000⟩, |001⟩, |010⟩, |011⟩, |100⟩, |101⟩, |110⟩, |111⟩ }

These basis vectors have particle numbers {0,1,1,2,1,2,2,3} respectively.  A particle-number conserving operator will not be block diagonal in this basis.
However, if we define a new basis by shuffling the basis vectors so that the respective particle numbers are {0,1,1,1,2,2,2,3}, then particle-number conserving
operators will be diagonal.

Our situation here is that we want particle-number conserving unitaries, but we want to work in the "usual" basis.  So first we generate random block diagonal
matrices, and then apply the appropriate inverse basis permutation to get the "usual" basis non-block-diagonal matrices.
'''

#produces an array of particle numbers corresponding to basis states
def ParticleNum(N):
    a = array([0,1])
    for i in range(N-1):
        a = append(a,a+1)
    return a
#example: ParticleNum(3) returns array([0,1,1,2,1,2,2,3])

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
 
 ####################################################################################   
'''function to generate the untaries in the quantum circuit ansatz.
creates a random unitary with particle number conservation'''
def generate_unitary(A):
    #A is a list of [4x4 matrix, 6x6 matrix, 4x4 matrix]

    #construct the block diagonal random unitary matrix
    u = torch.zeros([16,16])
    u[0,0] = 1.       # 0 particle sector
    u[15,15] = 1      # 4 particle sector
    u[1:5,1:5] = torch_expm(A[0]-A[0].T)    # 1 particle sector
    u[5:11,5:11] = torch_expm(A[1]-A[1].T)  # 2 particle sector
    u[11:15,11:15] = torch_expm(A[2]-A[2].T)# 3 particle sector

    #transform to the usual basis (u will no longer be block diagonal)
    u = permuteBasis(u,permute_order(4))
    return u.reshape(2,2,2,2,2,2,2,2)   
    
    
