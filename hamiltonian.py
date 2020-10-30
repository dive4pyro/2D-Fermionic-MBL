import numpy as np
import torch
from numpy import pi,cos,floor,kron,zeros,eye,array
###########################################################################
'''change/edit parameters here'''
N=6
dev = 'cuda:0' #if using GPU
#dev = 'cpu'   #uncomment this if you don't have cuda installed

##############################################################################
'''for the Hamiltonian'''
bx=.721; by=.693
U=J=delta=1.

cdag = array([[0.,0],[1,0]])
c = array([[0.,1],[0,0]])
nhat = array([[0.,0],[0,1]])

sz = torch.tensor([[1.,0],
            [0,-1]],device=dev)

c_ = torch.tensor(c, dtype=torch.float,device=dev)
c_dag = torch.tensor(cdag, dtype=torch.float,device=dev)
nHat = torch.tensor(nhat, dtype=torch.float,device=dev)

def W(m,n):
    return delta*(cos(2*pi*bx*m)+cos(2*pi*by*n))

'''
this returns the hamiltonian term h_k  at position (x,y),
where (x,y) is the coordinate of the LEFT or LOWER site
'''

def hamiltonian(coordinates):#coordinates = (x,y) (a tuple)
    m = coordinates[0]%N
    n = coordinates[1]%N
    h = kron(cdag,c) + kron(c,cdag) + kron(nhat, .5*W(m,n)*eye(2) + U*nhat)
    return torch.tensor(h.reshape(2,2,2,2),dtype=torch.float,device=dev)
