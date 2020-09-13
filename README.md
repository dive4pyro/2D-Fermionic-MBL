Here is an unfinished version of the 2D fermionic MBL code.
(for spinless fermions on an arbitrarily sized PBC lattice)

The main file is '2dMBL.py'   The other files are supporting files.  'contraction.py' is basically the same as that in my earlier email.

Basically, I think the code can be broken down into three main parts:

1. Setting up the parameters and optimization
2. the function for the calculation of the trace term in the figure of merit (contraction.py)
3. looping over the sites (and then Hamiltonian terms, on each site) to set up all of the trace terms in the figure of merit.

current status is:
1. and 2. are now done.  3. (probably the most diffcult part) hasn't been started yet.


In ./PytorchTest/mps.py is a demonstration of how PyTorch works, using finding the ground state of the Ising model as an example.
