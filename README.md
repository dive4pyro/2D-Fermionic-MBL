UPDATE Dec 2020: this is the new rewritten version.   Now with d=2,3, and 4. 
(see bottom of this file for a brief python/pytorch 'introduction')

# Here is the 2D fermionic MBL code. 

The main file is 'mbl2D.py'   The other files are supporting files.

TO RUN THE CODE: store all the files in the same directory.  Modify parameters in the top part of hamiltonian.py   Run mbl2D.py

Basically, the code can be broken down into a few main parts:

1. Setting up the overal parameters and optimization process (2dMBL.py, hamiltonian.py)
2. Setting the ansatz/unitiaries, enforcing particle number conservation via block diagonal matrices (ansatz.py)
3. the trace_calculation function (contraction.py)
4. looping over the sites (and then Hamiltonian terms, on each site) to set up all of the trace terms in the figure of merit. (FOM_terms.py)
5. expm_taylor.py contains code that implements a matrix exponential for Pytorch (a workaround since there's no torch.expm.  I didn't write this myself; found it one someone's github page)


## Some comments about Python:
Python code is usually super easy and obvious but here are perhaps a few things worth mentioning:

 - Python is usually object oriented but we make NO use of classes here, just liberal use of function definitions.

 - installations: to run this code you need Anaconda and Pytorch.

 - How to run python (for the actual runs we'll probably do it in a more organized way like via shell script, but this how I usually work):  
Put the files in one directory.  cd to that directory and run "python" to open the Python interpreter. To run a file, the command is ```exec(open('filename.py').read())```   (yes, it's quite annoying.  It used to be simply ```run('filename.py)``` but alas, it was changed in Python3)

 - one line if statements: the statement ```x = a if c==d else b``` is equivalent to:
```
if c==d:
    x = a
else:
    x = b
```
 - you can work with functions like variables.  E.g. the following code:
```
def square(x):
    return x**2
  
def cube(x):
    return x**3
  
powers = [square,cube]

for power in powers:
    print(power(3))
```
will print out 9, 27.

 - defualt function arguments: we can have, for example
```
def f(a=2,b=2,c=3):
    #insert code here
```	

then calling f() will do f(1,2,3). We may specify arguments, e.g. f(b=5,c=10) will give f(1,5,10).   		

 - dictionaries as function inputs.
Dictionaries are basically lists of entries in curly braces where each entry is a string paired with some quantity/object.
e.g. ```{ 'sigmax': array([[0,1],[1,0]]) , 'sigmaz': array([[1,0],[0,-1]]) }``` is a dictonary. 

Our use for dictionaries will be as a tool to feed into the trace_calculation function.

Here's how it works in general.

Let's say we have a function 
```
def f(a,b):
    #insert code here
```
and let's say we want to find f(x,y), where x and y are two defined variables

we can write ```f(**{'a':x,'b':y})```.  This will tell the function to use x for a, and y for b.

How the trace_calculation function works:  we define it as
```
def trace_calculation(Z, A1=id,B1=id,C1=id,D1=id,A2=id,B2=id,C2=id,D2=id,A1odd=False,
A2odd=False,B1odd=False,B2odd=False,C1odd=False,C2odd=False,D1odd=False,D2odd=False):
    #code here    
```
So the first argument will be  the Z tensor, the other tensors are by default set to identity, and assumed to be even.

let's say in our process of calculating the figure of merit, we need to calculate the trace for a case where only A1, B2, and D2 are not identity, and B2 and D2 are odd.  Then we need to call ```trace_calculation(Z,**dict)```, where ```dict = {'A1':A1, 'B2':B2, 'D2':D2, 'B2odd':True, 'D2odd':True}```
(and where Z, A1, B2, D2 are the calculated tensors)


Basically, the overall algorithm involves looping over the hamiltonian terms to find the contributions to the figure of merit, and at each iteration, produce the dictionary to feed to trace_calculation.

## Some comments about Pytorch:

./PytorchTest/mps.py is a quick demonstration of how PyTorch works, using finding the ground state of the Ising model as an example.

That contains all we need to know.  The one difference in the 2D MBL problem is that we have sweeping.
We work with torch tensors, which are NOT numpy arrays but basically behave the same way.  For every numpy.something() there usually is a torch.something()
(A major exception being, as mentioned above, numpy.linalg.expm().)

About GPU/cuda:  to use GPUs, make sure to install Pytorch with cuda  (nothing involved, just check the box/type in the command)
then to run on GPU: initialize each tensor with ```device='cuda:0'```.   To use CPU, omit that or type ```device='cpu'```
	











