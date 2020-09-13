'''
Code to find the ground state of the Ising model,

 H  = sum -J sz sz - g sx

with an MPS ansatz, using gradient-based optimization (bfgs)
with automatic differentiaiton in PyTorch.

Then, we compare to exact diagonalization.

Intended to be a demonstration of how the optimization aspect of the 2D MBL
code will work.  Of course in 'real life' for a case like this one would rather
use DMRG.

Note that this is only for N=4 sites!
I wrote the main contraction (for "psiHpsi" and "psipsi" below) using one big einsum
function, which is very easy, but at the cost of not being able to change N.
'''
import torch

#first, construct the Hamiltonian MPO

#pauli matrices
sx = torch.tensor([[0,1],[1,0]])
sz = torch.tensor([[1,0],[0,-1]])

#order of indices is (left, right, top, bottom) for MPO tensors
#and (vertical index, left, right) for MPS tensors.

#Ising model H  = sum -J sz sz - g sx
J = 1
g = 1

#MPS bond dimension
D = 4

vL = torch.tensor([0.,0,1])
vR = torch.tensor([1.,0,0])

h = torch.zeros([3,3,2,2])
h[0,0] = h[2,2] = torch.eye(2)
h[1,0] = sz
h[2,0] = -g*sx
h[2,1] = -J*sz

hL = torch.einsum('i,iabc',vL,h)
hR = torch.einsum('i,aibc',vR,h)
###########################################################################
# A is a list of the four MPS tensors, i.e. our variables to optimize
#initialize with random tensors
A = []
def Atensor(dim):
    return torch.rand(dim,requires_grad = True)
A = [Atensor((2,D)),Atensor((2,D,D)),Atensor((2,D,D)),Atensor((2,D))]

#define the loss function, i.e. energy, in terms of our variables
def energy():
    psiHpsi = torch.einsum('ja,kab,lbc,mc,djn,deko,eflp,fmq,ng,ogh,phi,qi',
                    A[0],A[1],A[2],A[3],hL,h,h,hR,A[0],A[1],A[2],A[3])

    psipsi = torch.einsum('ia,jab,kbc,lc,id,jde,kef,lf',
                        A[0],A[1],A[2],A[3],A[0],A[1],A[2],A[3])

    return psiHpsi/psipsi


#define the "optimizer" object.  We use the LBFGS (quasi Newton) algorithm
optimizer = torch.optim.LBFGS(A,max_iter=40)#maximum # of iterations

#the "closure" is basically something we have to include to the optimizer to recompute
#the loss function multiple times
def closure():
        optimizer.zero_grad()
        loss = energy()
        loss.backward()
        return loss
################################################################################
print('initial (random state) energy = ', energy())

#now run the optimization
optimizer.step(closure)
print('energy after optimization = ', energy())

#################################################################################
#now for exact diagonalization
#construct the Hamiltonian matrix
from numpy import zeros,kron, eye, linalg as la
def Hmatrix(N,J=1,g=1):
    szsz = kron(sz,sz)
    H = zeros([2**N,2**N])
    for n in range(N):
        H-= g*kron(kron(eye(2**n),sx),eye(2**(N-n-1)))
    for n in range(N-1):
        H-= J*kron(kron(eye(2**n),szsz),eye(2**(N-n-2)))
    return H

def GSenergy(H):
    energies,states = la.eig(H)
    return min(energies)

print('actual ground state energy = ',GSenergy(Hmatrix(4)))
