import numpy as np
from numpy import pi,cos,floor,kron,zeros
from numpy.random import rand
from scipy.linalg import expm
import torch
from contraction import *
from expm_taylor import *
from block_diag import *

'''for the Hamiltonian'''
bx=.721; by=.693
U=J=delta=1.

cdag = torch.tensor([[0.,0],[1,0]])
c = torch.tensor([[0.,1],[0,0]])
nhat = torch.tensor([[0.,0],[0,1]])

def f(m,n):
    return delta*(cos(2*pi*bx*m)+cos(2*pi*by*n))

def h(m,n):
    return (kron(cdag,c) + kron(c,cdag) + kron(nhat,.5*f(m,n)*eye(2)*U*nhat)).reshape(2,2,2,2)


'''function to generate the untaries in the quantum circuit ansatz.
creates a random unitary with particle number conservation'''
def generate_unitary():
    a1 = torch.rand(4,4)# 1 particle sector
    a2 = torch.rand(6,6)# 2 particle sector
    a3 = torch.rand(4,4)# 3 particle sector

    #construct the block diagonal random unitary matrix
    u = torch.zeros([16,16])
    u[0,0] = 1.       # 0 particle sector
    u[15,15] = 1      # 4 particle sector
    u[1:5,1:5] = torch_expm(a1-a1.T)    # 1 particle sector
    u[5:11,5:11] = torch_expm(a2-a2.T)  # 2 particle sector
    u[11:15,11:15] = torch_expm(a3-a3.T)# 3 particle sector

    #transform to the usual basis (u will no longer be block diagonal)
    u = permuteBasis(u,permute_order(4))
    return u.reshape(2,2,2,2,2,2,2,2)

#to take transpose, i.e. "flip upside down"
def dagger(u):
    return u.reshape(16,16).T.reshape(2,2,2,2,2,2,2,2)

####################################################################################
'''main part of the code'''
#initialize all the to-be-optimized variables i.e. the unitaries
#store them in 2D lists
# u = top layer, v = bottom layer
u = []; v = []
for i in range(int(N/2)):
    u.append([])
    v.append([])
    for j in range(int(N/2)):
        u[i].append(generate_unitary())
        v[i].append(generate_unitary())



'''
need to write the following section
'''

def figure_of_merit():
    fig_of_merit = 0
    '''

    when completed,this will probably be the most nontrivial part of the code

    a very rough outline is as follows:


    for i in sites:
        get the Z tensor

        for hk in lightcone(site i):
            for hl in lightcone(site i):

                find the relevant A,B,C,D tensors
                fig_of_merit += trace calculation(...)
    '''
    return fig_of_merit




optimizer = torch.optim.LBFGS([u,v],max_iter=40)

def closure():
        optimizer.zero_grad()
        loss = figure_of_merit()
        loss.backward()
        return loss
################################################################################

#now run the optimization
optimizer.step(closure)

'''
At this point, export the optimized unitaries to a file, or perform further
calculations/checks, etc.

'''
