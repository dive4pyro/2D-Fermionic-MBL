from hamiltonian import *
from contraction import *


if d==2:
    sigmas = [torch.diag(torch.tensor([1,-1.],device=dev))]
if d==3:
    sigmas = [torch.diag(torch.tensor([1,1,-1.],device=dev))]#,
              #torch.diag(torch.tensor([1,-1,1.],device=dev))]
if d==4:
    sigmas = [torch.diag(torch.tensor([1,-1,1,1.],device=dev))]#,
              #torch.diag(torch.tensor([1,1,-1,1.],device=dev)),
              #torch.diag(torch.tensor([1,1,1,-1.],device=dev))]

def f_supersite(upper_unitaries,lower_unitary,x,y):
    fom_supersite = 0
    for s in sigmas:
        Z1 = torch.einsum('abcdijkl,im,mjklefgh',lower_unitary,s,dagger(lower_unitary))
        Z2 = torch.einsum('abcdijkl,jm,imklefgh',lower_unitary,s,dagger(lower_unitary))
        Z3 = torch.einsum('abcdijkl,km,ijmlefgh',lower_unitary,s,dagger(lower_unitary))
        Z4 = torch.einsum('abcdijkl,lm,ijkmefgh',lower_unitary,s,dagger(lower_unitary))

        for Z in [Z1,Z2,Z3,Z4]:
            for f in [fa,fb,fc,fd]:
                fom_supersite += f(upper_unitaries,Z,x,y)

    return fom_supersite


#to take transpose, i.e. "flip upside down"
def dagger(u):
    return u.reshape(d**4,d**4).T.reshape(d,d,d,d,d,d,d,d)

ABCD = ['A','B','C','D']


num_h_terms = 3 if d==2 else 5

def fa(unitaries,Z,x,y):
    trace = 0
    for quadrant1 in [0,1,2,3]:
        dict1 = tensor_dictionary_square(unitaries,quadrant1,x,y,layer=1)
        for quadrant2 in [0,1,2,3]:
            dict2 = tensor_dictionary_square(unitaries,quadrant2,x,y,layer=2)
            trace+=trace_calculation(Z,**{**dict1,**dict2})
    return trace

def fb(unitaries,Z,x,y):
    trace  = 0
    for quadrant1 in [0,1,2,3]:
        dict1 = tensor_dictionary_square(unitaries,quadrant1,x,y,layer=1)
        for position in [1,2,3,4,5,6,7,8]:
            for term in range(num_h_terms):
                dict2 = tensor_dictionary_across(unitaries,position,term,x,y,layer=2)
                trace += trace_calculation(Z,**{**dict1,**dict2})
    return 2*trace

def fc(unitaries,Z,x,y):
    Trace = 0
    for position1 in [1,2,3,4,5,6,7,8]:
        for term1 in range(num_h_terms):
            dict1 = tensor_dictionary_across(unitaries,position1,term1,x,y)
            for position2 in range(position1,9):
                for term2 in range(num_h_terms):
                    dict2 = tensor_dictionary_across(unitaries,position2,term2,x,y,layer=2)
                    dict = {**dict1,**dict2}
                    if position1==position2:
                        Trace += trace_calculation(Z,**dict)
                    elif position1<position2:
                        Trace += 2*trace_calculation(Z,**dict)
    return Trace
################################################################################
#the 4 useful "sum functions"
def sum1(u,op):
    return torch.einsum('abcdijkl,im,mjklefgh',dagger(u),op,u)

def sum2(u,op):
    return torch.einsum('abcdijkl,jm,imklefgh',dagger(u),op,u)

def sum3(u,op):
    return torch.einsum('abcdijkl,km,ijmlefgh',dagger(u),op,u)

def sum4(u,op):
    return torch.einsum('abcdijkl,lm,ijkmefgh',dagger(u),op,u)

sum_funcs = [sum1,sum2,sum3,sum4]
################################################################################

def fd(unitaries,Z,x,y):
    '''
       ___________   side 2
       |         |
       |         |
       |         |
       |_________|

    side 1

    we need to consider the 2 'sides' since in each case the way the 'hanging' little h
    operators are decomposed (for fd_add) or partial traced (for f_subtract) differently
    '''
    #this is the 'add' part
    def fd_add(quadrant,position,coord,side):
        m = coord[0]%N
        n = coord[1]%N
        if d==2:
            h = [[c,cDag,0.5*W(m,n)*I + U*nHat],[cDag, c, nHat]]
        if d==3:
            h = [[cUp,cUpDag,cDown,cDownDag,I],[cUpDag, cUp,cDownDag,cDown, 0.5*W(m,n)*nHat]]
        if d==4:
            h = [[cUp,cUpDag,cDown,cDownDag,I],[cUpDag, cUp,cDownDag,cDown, 0.5*(U*nUp@nDown + W(m,n)*nHat)]]
        Trace = 0
        for term1 in range(num_h_terms):
            for term2 in range(num_h_terms):
                X1odd = X2odd = True
                X1 = sum_funcs[position](unitaries[quadrant],h[side%2][term1])
                if term1==2: X1odd=False
                X2 = sum_funcs[position](unitaries[quadrant],h[side%2][term2])
                if term2==2: X2odd=False
                dict = {ABCD[quadrant]+'1':X1,ABCD[quadrant]+'2':X2, ABCD[quadrant]+'1odd':X1odd, ABCD[quadrant]+'2odd':X2odd}
                Trace += torch.trace(h[side-1][term1]@h[side-1][term2])*trace_calculation(Z,**dict)
        return Trace

    #this is the 'subtract' part
    def fd_subtract(quadrant,position,coord,side):
        #take partial trace of the h operator
        if side==1: hprime = torch.einsum('iaib',hamiltonian(coord))
        if side==2: hprime = torch.einsum('aibi',hamiltonian(coord))
        X = sum_funcs[position](unitaries[quadrant],hprime)
        dict ={ABCD[quadrant]+'1':X,ABCD[quadrant]+'2':X}
        return trace_calculation(Z,**dict)


    '''
    explicitly go through all 16 positions, applying the above functions and 'manually inputting'
    the quadrant, position, (x,y) coordinate, and side (1 or 2) in each.
    '''
    return (1/2)*( fd_add(0,1,(x,y-2),1) + fd_add(0,0,(x-1,y-2),1) + fd_add(0,0,(x-2,y-1),1) + fd_add(0,3,(x-2,y),1) +\
    fd_add(2,0,(x-2,y+1),1) + fd_add(2,3,(x-2,y+2),1) + \
    fd_add(2,3,(x-1,y+2),2) + fd_add(2,2,(x,y+2),2) + fd_add(3,3,(x+1,y+2),2) + 2*fd_add(3,2,(x+2,y+2),2) +\
    fd_add(3,1,(x+2,y+1),2) + fd_add(1,2,(x+2,y),2) + fd_add(1,1,(x+2,y-1),2) +\
    fd_add(1,1,(x+2,y-2),1) + fd_add(1,0,(x+1,y-2),1) ) \
    \
    -(1/4)*( fd_subtract(0,1,(x,y-2),1) + fd_subtract(0,0,(x-1,y-2),1) + fd_subtract(0,0,(x-2,y-1),1) + fd_subtract(0,3,(x-2,y),1) +\
    fd_subtract(2,0,(x-2,y+1),1) + fd_subtract(2,3,(x-2,y+2),1) + \
    fd_subtract(2,3,(x-1,y+2),2) + fd_subtract(2,2,(x,y+2),2) + fd_subtract(3,3,(x+1,y+2),2) + 2*fd_subtract(3,2,(x+2,y+2),2) +\
    fd_subtract(3,1,(x+2,y+1),2) + fd_subtract(1,2,(x+2,y),2) + fd_subtract(1,1,(x+2,y-1),2) +\
    fd_subtract(1,1,(x+2,y-2),1) + fd_subtract(1,0,(x+1,y-2),1) )

#tensor_dictionary_square for fa and fb
def tensor_dictionary_square(unitaries,quadrant,x,y,layer):
    def h_plaquette(x,y):
        return torch.einsum('abef,cg,dh',hamiltonian((x-1,y-1)),I,I) +\
        torch.einsum('bcfg,ae,dh',hamiltonian((x,y-1)),I,I)+\
        torch.einsum('dchg,ae,bf',hamiltonian((x-1,y)),I,I) + \
        torch.einsum('aiek,bjim,cdjn,kmfl,lngh',hamiltonian((x-1,y-1)),swap,swap,swap,swap)
    if quadrant==0:
        hquad = h_plaquette(x,y) + .5*torch.einsum('iaie,bf,cg,dh',hamiltonian((x-1,y-2)),I,I,I) + \
        .5*torch.einsum('iaie,bf,cg,dh',hamiltonian((x-2,y-1)),I,I,I) + \
        .5*torch.einsum('ibif,ae,cg,dh',hamiltonian((x,y-2)),I,I,I) + \
        .5*torch.einsum('idih,ae,bf,cg',hamiltonian((x-2,y)),I,I,I)
    if quadrant==1:
        hquad = h_plaquette(x+2,y) + .5*torch.einsum('iaie,bf,cg,dh',hamiltonian((x+1,y-2)),I,I,I) + \
        .5*torch.einsum('ibif,ae,cg,dh',hamiltonian((x+2,y-2)),I,I,I) + \
        .5*torch.einsum('bifi,ae,cg,dh',hamiltonian((x+2,y-1)),I,I,I) + \
        .5*torch.einsum('cigi,ae,bf,dh',hamiltonian((x+2,y)),I,I,I)
    if quadrant==2:
        hquad = h_plaquette(x,y+2) + .5*torch.einsum('iaie,bf,cg,dh',hamiltonian((x-2,y+1)),I,I,I) + \
        .5*torch.einsum('idih,ae,bf,cg',hamiltonian((x-2,y+2)),I,I,I) + \
        .5*torch.einsum('dihi,ae,bf,cg',hamiltonian((x-1,y+2)),I,I,I) + \
        .5*torch.einsum('cigi,ae,bf,dh',hamiltonian((x,y+2)),I,I,I)
    if quadrant==3:
        hquad = h_plaquette(x+2,y+2) + .5*torch.einsum('bifi,ae,cg,dh',hamiltonian((x+2,y+1)),I,I,I) + \
        2*.5*torch.einsum('cigi,ae,bf,dh',hamiltonian((x+2,y+2)),I,I,I) + \
        .5*torch.einsum('dihi,ae,bf,cg',hamiltonian((x+1,y+2)),I,I,I)

    X = torch.einsum('abcdijkl,ijklmnop,mnopefgh',dagger(unitaries[quadrant]),hquad,unitaries[quadrant])
    return {ABCD[quadrant]+str(layer):X}

#tensor_dictionary_across for fb and fc
def tensor_dictionary_across(unitaries,position,term,x,y,layer=1):
    #for a given term in a given "across" h operator, return the dictionary
    #with the 2 tensors and their labels, and odd/evenness
    def operators_dict(term,m,n,quad1,quad2,sum_func1,sum_func2):
        dic = {} if term==0 else {ABCD[quad1]+str(layer)+'odd':True,ABCD[quad2]+str(layer)+'odd':True}
        operators = h_operators(m,n)[term]
        X1 = sum_func1(unitaries[quad1],operators[0])
        X2 = sum_func2(unitaries[quad2],operators[1])
        return {ABCD[quad1]+str(layer):X1,ABCD[quad2]+str(layer):X2,**dic}

    '''
    similar to the code for fd, here we explicitly go through the 8 positions,
    'manually inputting' the coordinate, position, quadrants, term data in each case.
    '''
    if position==1:
        return operators_dict(term,x, (y-1)%N,0,1,sum2,sum1)

    if position ==2:
        return operators_dict(term,x,y,0,1,sum3,sum4)

    if position ==3:
        return operators_dict(term,(x-1)%N, y,0,2,sum4,sum1)

    if position ==4:
        return operators_dict(term, x,y ,0,2,sum3,sum2)

    if position ==5:
        return operators_dict(term, (x+1)%N, y ,1,3,sum4,sum1)

    if position==6:
        return operators_dict(term, (x+2)%N, y ,1,3,sum3,sum2)

    if position==7:
        return operators_dict(term, x, (y+1)%N ,2,3,sum2,sum1)

    if position==8:
        return operators_dict(term, x, (y+2)%N ,2,3,sum3,sum4)
