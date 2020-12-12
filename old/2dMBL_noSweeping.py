'''
Main Code
(optimize all unitaries at once)
local dim d=2, with overall particle number conservation
'''
from block_diag import *
from FOM_terms import *

####################################################################################
#initialize all the to-be-optimized variables
#store them in 2D lists
'''actual variables Au and Av
these are the underlying variables which must be passed to the optimizer later
Au and Av each is an (N/2) x (N/2) list, and each element of is a list [4x4 matrix, 6x6 matrix, 4x4 matrix]
this list is turned into a 16x16 unitary matrix (and then reshaped to tensor)
via the generate_unitary function in block_diag.py
'''
#initialize mxm matrix
def initialize(m):
    return torch.zeros(m,m,requires_grad=True,device=dev)

Au = []; Av = []
for i in range(int(N/2)):
    Au.append([])
    Av.append([])
    for j in range(int(N/2)):
        Au[i].append([initialize(4),initialize(6),initialize(4)])
        Av[i].append([initialize(4),initialize(6),initialize(4)])

def f_plaq(upper_unitaries,lower_unitary,x,y):
    Z1 = torch.einsum('abcdijkl,im,mjklefgh',lower_unitary,sz,dagger(lower_unitary))
    Z2 = torch.einsum('abcdijkl,jm,imklefgh',lower_unitary,sz,dagger(lower_unitary))
    Z3 = torch.einsum('abcdijkl,km,ijmlefgh',lower_unitary,sz,dagger(lower_unitary))
    Z4 = torch.einsum('abcdijkl,lm,ijkmefgh',lower_unitary,sz,dagger(lower_unitary))

    f_plaquette = 0
    for Z in [Z1,Z2,Z3,Z4]:
        for f in [f1,f2,f3,f4,f5,f6,f7]:
            f_plaquette += f(upper_unitaries,Z,x,y)

    return f_plaquette

def figure_of_merit():
    # unitaries: u = top layer, v = bottom layer
    u = []; v = []
    for i in range(int(N/2)):
        u.append([])
        v.append([])
        for j in range(int(N/2)):
            u[i].append(generate_unitary(Au[i][j]))
            v[i].append(generate_unitary(Av[i][j]))

    fig_of_merit = 0
    for i in range(int(N/2)):
        for j in range(int(N/2)):
            fig_of_merit+=f_plaq([ u[i][j], u[(i+1)%int(N/2)][j], u[(i+1)%int(N/2)][(j+1)%int(N/2)], u[i][(j+1)%int(N/2)] ], v[i][j] ,i,j)

    fig_of_merit *= -1
    return fig_of_merit

optimizer = torch.optim.LBFGS(sum(sum(Au+Av,[]),[]),max_iter=10)

def closure():
        optimizer.zero_grad()
        loss = figure_of_merit()
        loss.backward()
        return loss
################################################################################
from time import time
start = time()


#now run the optimization
print(optimizer.step(closure))
print(figure_of_merit())
print(time()-start)

'''
At this point, export the optimized unitaries to a file, or perform further
calculations/checks, etc.
'''
