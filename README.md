./PytorchTest/mps.py is a quick demonstration of how PyTorch works, using finding the ground state of the Ising model as an example.


Here is the 2D fermionic MBL code. (now completed!)
(for spinless fermions on an arbitrarily sized PBC lattice)

The main file is '2dMBL.py'   The other files are supporting files.  'contraction.py' is basically the same as before.

Basically, I think the code can be broken down into three main parts:

1. Setting up the parameters and optimization (2dMBL.py)
2. the function for the calculation of the trace term in the figure of merit (contraction.py)
3. looping over the sites (and then Hamiltonian terms, on each site) to set up all of the trace terms in the figure of merit. (FOM_terms.py)




