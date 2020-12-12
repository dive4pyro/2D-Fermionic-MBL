'''
Main Code
with overall particle number conservation imposed
'''
from ansatz import *
from FOM_terms import *
from time import time
import gc
#optmization algorithm and no. steps
optim='bfgs'
Nsteps=16

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
        if d==2:
            Au[i].append([initialize(4),initialize(6),initialize(4)])
            Av[i].append([initialize(4),initialize(6),initialize(4)])
        if d==3:
            Au[i].append([initialize(8),initialize(24),initialize(32),initialize(16)])
            Av[i].append([initialize(8),initialize(24),initialize(32),initialize(16)])
        if d==4:
            Au[i].append([initialize(8),initialize(28),initialize(56),initialize(70),initialize(56),initialize(28),initialize(8)])
            Av[i].append([initialize(8),initialize(28),initialize(56),initialize(70),initialize(56),initialize(28),initialize(8)])

#######################################################################################################
#begin main loop
for step in range(Nsteps+1):
    fom = 0
    s=time()
    for i in range(int(N/2)):
        for j in range(int(N/2)):
            def figure_of_merit():
                # unitaries: u = top layer, v = bottom layer
                unitaries = [generate_unitary(Au[i][j]), generate_unitary(Au[(i+1)%int(N/2)][j]),
                           generate_unitary(Au[i][(j+1)%int(N/2)]), generate_unitary(Au[(i+1)%int(N/2)][(j+1)%int(N/2)])]
                v = generate_unitary(Av[i][j])
                x = 2*i+1; y = 2*j+1
                fig_of_merit = f_supersite(unitaries, v ,x,y)

                fig_of_merit *= -1
                return fig_of_merit

            params = Au[i][j], Au[(i+1)%int(N/2)][j], Au[i][(j+1)%int(N/2)], Au[(i+1)%int(N/2)][(j+1)%int(N/2)], Av[i][j]
            if optim== 'bfgs':
                optimizer = torch.optim.LBFGS(sum(params,[]),max_iter=1)
                def closure():
                    optimizer.zero_grad()
                    loss = figure_of_merit()
                    loss.backward()
                    return loss
                if step<Nsteps:
                    fom+=optimizer.step(closure)
                    print('supersite (',i,',',j,') done')
                else:
                    fom += figure_of_merit()
            if optim =='adam':
                if step<Nsteps:
                    optimizer = torch.optim.Adam(sum(params,[]),lr = .04-.01*step if step<4 else .005)
                    optimizer.zero_grad()
                    loss = figure_of_merit()
                    fom+=loss
                    loss.backward()
                    optimizer.step()
                    print('supersite (',i,',',j,') done')
                else:
                    fom += figure_of_merit()
            gc.collect()
    print('after step ',step,'   ',fom/d**16)
    print((time()-s)/60, ' mins')
################################################################################
'''
At this point, export the optimized unitaries to a file, or perform further
calculations/checks, etc.'''
from numpy import zeros, save
U = zeros([int(N/2),int(N/2),d**4,d**4])
V = zeros([int(N/2),int(N/2),d**4,d**4])
for i in range(int(N/2)):
    for j in range(int(N/2)):
        U[i,j] = generate_unitary(Au[i][j]).detach().numpy().reshape(d**4,d**4)
        V[i,j] = generate_unitary(Av[i][j]).detach().numpy().reshape(d**4,d**4)
save('U_d'+str(d),U)
save('V_d'+str(d),V)
