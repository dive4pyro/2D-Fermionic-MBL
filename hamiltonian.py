import numpy as np
import torch
from numpy import pi,cos,floor,kron,zeros,eye,array
###########################################################################
'''change/edit parameters here'''
N=6
dev = 'cuda:0' #if using GPU
dev = 'cpu'   #uncomment this if you don't have cuda installed
d=2
##############################################################################
'''for the Hamiltonian'''
bx=.721; by=.693
U=J=delta=1.


if d==2:
    cDag = torch.tensor([[0,0.],
                         [1,0]],device=dev)

    c = torch.tensor([[0,1.],
                      [0,0]],device=dev)

    nHat = torch.tensor([[0,0.],
                         [0,1]],device=dev)

    sCross = torch.tensor([[1.,0],
                           [0,-1]],device=dev)

if d==3:
    cUpDag = torch.tensor([[0,0,0],
                           [1,0,0],
                           [0,0,0.]],device=dev)

    cDownDag = torch.tensor([[0,0,0],
                             [0,0,0],
                             [1,0,0.]],device=dev)
    cUp = cUpDag.T
    cDown = cDownDag.T
    nHat = torch.diag(torch.tensor([0,1,1.],device=dev))

if d==4:
    cUpDag = torch.tensor([[0,0,0,0],
                           [1,0,0,0],
                           [0,0,0,0],
                           [0,0,1,0.]],device=dev)

    cDownDag = torch.tensor([[0,0,0,0],
                             [0,0,0,0],
                             [1,0,0,0],
                             [0,1,0,0.]],device=dev)
    cUp = cUpDag.T
    cDown = cDownDag.T
    nUp = torch.diag(torch.tensor([0,1,0,1.],device=dev))
    nDown = torch.diag(torch.tensor([0,0,1,1.],device=dev))
    nHat = torch.diag(torch.tensor([0,1,1,2.],device=dev))

def W(m,n):
    return delta*(cos(2*pi*bx*m)+cos(2*pi*by*n))

#pytorch doesn't have a built in Kron
def torchKron(A, B):
    return torch.einsum("ab,cd->acbd", A, B).reshape(A.size(0)*B.size(0), A.size(1)*B.size(1))


I = torch.eye(d,device=dev)

def h_operators(m,n):
    if d==2:
        return [[nHat,0.5*W(m,n)*I + U*nHat],[cDag,c],[c,cDag]]
    if d==3:
        return [[0.5*W(m,n)*nHat,I],[cUpDag,cUp],[cUp,cUpDag],[cDownDag,cDown],[cDown,cDownDag]]
    if d==4:
        return [[0.5*(U*nUp@nDown + W(m,n)*nHat),I],[cUpDag,cUp],[cUp,cUpDag],[cDownDag,cDown],[cDown,cDownDag]]

'''
this returns the hamiltonian term h_k  at position (x,y),
where (x,y) is the coordinate of the LEFT or LOWER site
'''
def hamiltonian(coordinates):#coordinates = (x,y) (a tuple)
    m = coordinates[0]%N
    n = coordinates[1]%N

    h=0
    for term in h_operators(m,n):
     h+= torchKron(term[0],term[1])
    return h.reshape(d,d,d,d)
