'''
Main Code
(optimize FOM terms from one plaquette at a time)
d=2, with overall particle number conservation
'''
import torch
from contraction import *
from expm_taylor import *
from block_diag import *
from FOM_terms import *

N = 6

####################################################################################
#initialize all the to-be-optimized variables
#store them in 2D lists
'''actual variables Au and Av
these are the underlying variables which must be passed to the optimizer later
Au and Av each is an (N/2) x (N/2) list, and each element of is a list [4x4 matrix, 6x6 matrix, 4x4 matrix]
this list is turned into a 16x16 unitary matrix (and then reshaped to tensor) 
via the generate_unitary function in block_diag.py
'''
Au = []; Av = []
for i in range(int(N/2)):
    Au.append([])
    Av.append([])
    for j in range(int(N/2)):
        Au[i].append([torch.zeros(4,4,requires_grad=True),torch.zeros(6,6,requires_grad=True),torch.zeros(4,4,requires_grad=True)])
        Av[i].append([torch.zeros(4,4,requires_grad=True),torch.zeros(6,6,requires_grad=True),torch.zeros(4,4,requires_grad=True)])


for step in range(3):
    fom = 0
    for i in range(int(N/2)):
        for j in range(int(N/2)):
            def figure_of_merit():
                # unitaries: u = top layer, v = bottom layer

                unitaries = [generate_unitary(Au[i][j]), generate_unitary(Au[(i+1)%int(N/2)][j]),
                           generate_unitary(Au[i][(j+1)%int(N/2)]), generate_unitary(Au[(i+1)%int(N/2)][(j+1)%int(N/2)])]
                v = generate_unitary(Av[i][j])


                fig_of_merit = f_plaq(unitaries, v ,i,j)

                fig_of_merit *= -1
                return fig_of_merit
            params = [Au[i][j], Au[(i+1)%int(N/2)][j], Au[i][(j+1)%int(N/2)], Au[(i+1)%int(N/2)][(j+1)%int(N/2)], Av[i][j]]
            optimizer = torch.optim.LBFGS(sum(params,[]),max_iter=1)

            def closure():
                optimizer.zero_grad()
                loss = figure_of_merit()
                loss.backward()
                return loss
            fom+=optimizer.step(closure)

    print('step ',step+1,'   ',fom)

################################################################################
'''
At this point, export the optimized unitaries to a file, or perform further
calculations/checks, etc.'''
