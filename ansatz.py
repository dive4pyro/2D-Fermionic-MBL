from numpy import argsort, array, append
from expm_taylor import torch_expm
import torch
from hamiltonian import *

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
    if d==2:
        a = array([0,1])
    if d==3:
        a = array([0,1,1])
    if d==4:
        a = array([0,1,1,2])
    for i in range(N-1):
        x = a.copy()
        a = append(a,x+1)
        if d>2:
            a = append(a,x+1)
        if d>3:
            a = append(a,x+2)
    return a
#example: for d = 2, ParticleNum(3) returns array([0,1,1,2,1,2,2,3])


#here N is the number of sites of the hilbert space that we are permuting
#we always have N=4, regardless of d, since the unitaries have 4 legs.
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
def generate_unitary2(A):
    #A is a list of [4x4 matrix, 6x6 matrix, 4x4 matrix]
    #note 1+4+6+4+1=16
    #construct the 16x16 block diagonal random unitary matrix
    u = torch.zeros([16,16],device=dev)
    u[0,0] = 1.                            #0 particle sector: 1x1 matrix
    u[1:5,1:5] = torch_expm(A[0]-A[0].T)   #1 particle sector: 4x4 matrix
    u[5:11,5:11] = torch_expm(A[1]-A[1].T) #2 particle sector: 6x6 matrix
    u[11:15,11:15]=torch_expm(A[2]-A[2].T) #3 particle sector: 4x4 matrix
    u[15,15] = 1                           #4 particle sector: 1x1 matrix

    #transform to the usual basis (u will no longer be block diagonal)
    u = permuteBasis(u,permute_order(4))
    return u.reshape(2,2,2,2,2,2,2,2)

def generate_unitary3(A):
    #A is a list of [8x8 , 24x24, 32x32, 16x16 matrices]
    #note 1+8+24+32+16=81
    #construct the 81x81 block diagonal random unitary matrix
    u = torch.zeros([81,81],device=dev)
    u[0,0] = 1.                            #0 particle sector: 1x1 matrix
    u[1:9,1:9] = torch_expm(A[0]-A[0].T)   #1 particle sector: 8x8 matrix
    u[9:33,9:33] = torch_expm(A[1]-A[1].T) #2 particle sector: 24x24 matrix
    u[33:65,33:65]=torch_expm(A[2]-A[2].T) #3 particle sector: 32x32 matrix
    u[65:81,65:81]=torch_expm(A[3]-A[3].T) #4 particle sector: 16x16 matrix

    #transform to the usual basis (u will no longer be block diagonal)
    u = permuteBasis(u,permute_order(4))
    return u.reshape(3,3,3,3,3,3,3,3)

def generate_unitary4(A):
    #A is a list of [8x8 , 28x28, 56x56, 70x70 ,56x56, 28x28, 8x8 matrices]
    #note 1+8+28+56+70+56+28+8+1=256
    #construct the 256x256 block diagonal random unitary matrix
    u = torch.zeros([256,256],device=dev)
    u[0,0] = 1.                                  #0 particle sector: 1x1 matrix
    u[1:9,1:9] = torch_expm(A[0]-A[0].T)         #1 particle sector: 8x8 matrix
    u[9:37,9:37] = torch_expm(A[1]-A[1].T)       #2 particle sector: 28x28 matrix
    u[37:93,37:93] = torch_expm(A[2]-A[2].T)     #3 particle sector: 56x56 matrix
    u[93:163,93:163] = torch_expm(A[3]-A[3].T)   #4 particle sector: 70x70 matrix
    u[163:219,163:219] = torch_expm(A[4]-A[4].T) #5 particle sector: 56x56 matrix
    u[219:247,219:247] = torch_expm(A[5]-A[5].T) #6 particle sector: 28x28 matrix
    u[247:255,247:255] = torch_expm(A[6]-A[6].T) #7 particle sector: 8x8 matrix
    u[255,255]= 1.                               #8 particle sector: 1x1 matrix

    #transform to the usual basis (u will no longer be block diagonal)
    u = permuteBasis(u,permute_order(4))
    return u.reshape(4,4,4,4,4,4,4,4)

if d==2:
    generate_unitary = generate_unitary2
if d==3:
    generate_unitary = generate_unitary3
if d==4:
    generate_unitary = generate_unitary4
