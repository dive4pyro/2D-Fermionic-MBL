from hamiltonian import *
from contraction import *

'''
below is the code for the FOM contributions f1,...,f7
basically, the idea is loop over all the positions that the little h hamiltonian operator
can be in, with two "main loops", one for the upper h_k and one for the lower h_l

for each h_k, h_l, use the "tensor dictionary" function to produce the input to the trace_calculation function
'''
def f1(unitaries,Z,x,y):
    Trace = 0
    for quadrant1 in [0,1,2,3]:
        for position1 in [0,1,2,3]:
            dict1 = tensor_dictionary_plaq(unitaries,quadrant1,position1,x,y)
            for quadrant2 in [0,1,2,3]:
                for position2 in [0,1,2,3]:
                    dict2 = tensor_dictionary_plaq(unitaries,quadrant2,position2,x,y,layer=2)
                    dict = {**dict1,**dict2}
                    #so we get something like e.g. dict = {'A1':X1,'B2':X2}
                    # then we would have
                    # trace_calculation(**dict) = trace_calculation(A1=X1,B2=X2, all others identity)
                    Trace += trace_calculation(Z,Z,**dict)
    return Trace


def f2(unitaries,Z,x,y):
    Trace = 0
    for position1 in [1,2,3,4,5,6,7,8]:
        for term1 in [1,2,3]:
            dict1 = tensor_dictionary_pp(unitaries,position1,term1,x,y)
            for position2 in range(position1,9):
                for term2 in [1,2,3]:
                    dict2 = tensor_dictionary_pp(unitaries,position2,term2,x,y,layer=2)
                    dict = {**dict1,**dict2}
                    if position1==position2:
                        Trace += trace_calculation(Z,Z,**dict)
                    elif position1<position2:
                        Trace += 2*trace_calculation(Z,Z,**dict)
    return Trace


def f3(unitaries,Z,x,y):
    Trace = 0
    for quadrant1 in [0,1,2,3]:
        for position1 in [0,1,2,3]:
            dict1 = tensor_dictionary_plaq(unitaries,quadrant1,position1,x,y)
            for position2 in [1,2,3,4,5,6,7,8]:
                for term2 in [1,2,3]:
                    dict2 = tensor_dictionary_pp(unitaries,position2,term2,x,y,layer=2)
                    dict = {**dict1,**dict2}
                    Trace += trace_calculation(Z,Z,**dict)
    return 2*Trace

def f4(unitaries,Z,x,y):
    Trace = 0
    for quadrant1 in [0,1,2,3]:
        for position1 in [0,1,2,3]:
            dict1 = tensor_dictionary_plaq(unitaries,quadrant1,position1,x,y)
            for position2 in range(1,17):#[1,2,3,...,16]
                dict2 = tensor_dictionary_edge(unitaries,position2,x,y,layer=2)
                dict = {**dict1,**dict2}
                Trace += trace_calculation(Z,Z,**dict)
    return Trace

def f5(unitaries,Z,x,y):
    Trace = 0
    for position1 in [1,2,3,4,5,6,7,8]:
        for term1 in [1,2,3]:
            dict1 = tensor_dictionary_pp(unitaries,position1,term1,x,y)
            for position2 in range(1,17):#[1,2,3,...,16]
                dict2 = tensor_dictionary_edge(unitaries,position2,x,y,layer=2)
                dict = {**dict1,**dict2}
                Trace += trace_calculation(Z,Z,**dict)
    return Trace

def f6(unitaries,Z,x,y):
    Trace = 0
    for position1 in range(1,17):#[2,3,...,16]
        dict1 = tensor_dictionary_edge(unitaries,position1,x,y)
        for position2 in range(1,position1):#[1,2,3,...,position1-1]
            dict2 = tensor_dictionary_edge(unitaries,position2,x,y,layer=2)
            dict = {**dict1,**dict2}
            Trace += trace_calculation(Z,Z,**dict)
    return 2*(1/4)*Trace


''' warning: the code from here on below is probably highly unreadable '''
def f7(unitaries,Z,x,y):
    '''
       ___________   side 2
       |         |
       |         |
       |         |
       |_________|

    side 1
    '''
    def f7part(quadrant,position,coord,side):
        m = coord[0]%N
        n = coord[1]%N
        ABCD = ['A','B','C','D']
        h = [[c_,c_dag,0.5*W(m,n)*torch.eye(2,device=dev) + U*nHat],[c_dag, c_, nHat]]
        Trace = 0
        for term1 in [0,1,2]:
            for term2 in [0,1,2]:
                X1odd = X2odd = True
                X1 = sum_funcs[position](unitaries[quadrant],h[side%2][term1])
                if term1==2: X1odd=False
                X2 = sum_funcs[position](unitaries[quadrant],h[side%2][term2])
                if term2==2: X2odd=False
                dict = {ABCD[quadrant]+'1':X1,ABCD[quadrant]+'2':X2, ABCD[quadrant]+'1odd':X1odd, ABCD[quadrant]+'2odd':X2odd}
                Trace += torch.trace(h[side-1][term1]@h[side-1][term2])*trace_calculation(Z,Z,**dict)
        return Trace

    Trace = f7part(0,1,(x,y-2),1) + f7part(0,0,(x-1,y-2),1) + f7part(0,0,(x-2,y-1),1) + f7part(0,2,(x-2,y),1) +\
    f7part(2,0,(x-2,y+1),1) + f7part(2,3,(x-2,y+2),1) + \
    f7part(2,3,(x-1,y+2),2) + f7part(2,2,(x,y+2),2) + f7part(3,3,(x+1,y+2),2) + 2*f7part(3,2,(x+2,y+2),2) +\
    f7part(3,1,(x+2,y+1),2) + f7part(1,2,(x+2,y),2) + f7part(1,1,(x+2,y-1),2) +\
    f7part(1,1,(x+2,y-2),1) + f7part(1,0,(x+1,y-2),1)

    return (1/2)*Trace


############################################################################
'''
below is the code for the tensor dictionaries (probably also quite unreadable)
'''

#to take transpose, i.e. "flip upside down"
def dagger(u):
    return u.reshape(16,16).T.reshape(2,2,2,2,2,2,2,2)

'''
f1: contribution from where both hk and hl lie inside a plaquette
'''
quad = [(0,0),(2,0),(0,2),(2,2)]
pos = [(0,0),(1,0),(0,1),(0,0)]
def mn(quadrant,position):
    return tuple(array(quad[quadrant]) + array(pos[position]))

def tensor_dictionary_plaq(unitaries,quadrant,position,x,y,layer=1):

    def Xh12(quadrant):
        u = unitaries[quadrant]
        h = hamiltonian(array((x-1,y-1))+mn(quadrant,position=0))
        return torch.einsum('abcdijkl,ijmn,mnklefgh',dagger(u),h,u)

    def Xh23(quadrant):
        u = unitaries[quadrant]
        h = hamiltonian((x-1,y-1)+mn(quadrant,position=1))
        return torch.einsum('abcdijkl,jkmn,imnlefgh',dagger(u),h,u)

    def Xh34(quadrant):
        u = unitaries[quadrant]
        h = hamiltonian((x-1,y-1)+mn(quadrant,position=2))
        return torch.einsum('abcdijkl,klmn,ijmnefgh',dagger(u),h,u)

    def Xh14(quadrant):
        u = unitaries[quadrant]
        h = hamiltonian((x-1,y-1)+mn(quadrant,position=3))
        u = torch.einsum('abijefgh,cdij',u,swap)
        u = torch.einsum('aijdefgh,bcij',u,swap)
        return torch.einsum('abcdijkl,ijmn,mnklefgh',dagger(u),h,u)

    getX = [Xh12,Xh23,Xh34,Xh14]

    if layer==1:
        labels=['A1','B1','C1','D1']
    if layer==2:
        labels=['A2','B2','C2','D2']

    X = getX[position](quadrant)
    return {labels[quadrant]:X}


def sum1(u,op):
    return torch.einsum('abcdijkl,im,mjklefgh',dagger(u),op,u)

def sum2(u,op):
    return torch.einsum('abcdijkl,jm,imklefgh',dagger(u),op,u)

def sum3(u,op):
    return torch.einsum('abcdijkl,km,ijmlefgh',dagger(u),op,u)

def sum4(u,op):
    return torch.einsum('abcdijkl,lm,ijkmefgh',dagger(u),op,u)

sum_funcs = [sum1,sum2,sum3,sum4]


def tensor_dictionary_pp(unitaries,position,term,x,y,layer=1):
    if term==1:
        operators = [c_dag,c_]
    if term==2:
        operators = [c_,c_dag]


    if position==1:
        if term==3:
            m,n = x, (y-1)%N
            operators = [nHat,0.5*W(m,n)*torch.eye(2,device=dev) + U*nHat]
            d = {}

        if term==1 or term ==2:
            d = {'A'+str(layer)+'odd':True,'B'+str(layer)+'odd':True}

        A = torch.einsum('abcdijkl,jm,imklefgh',dagger(unitaries[0]),operators[0],unitaries[0])
        B = torch.einsum('abcdijkl,im,mjklefgh',dagger(unitaries[1]),operators[1],unitaries[1])
        return {'A'+str(layer):A,'B'+str(layer):B,**d}

    if position ==2:
        if term==3:
            m,n = x, y
            operators = [nHat,0.5*W(m,n)*torch.eye(2,device=dev) + U*nHat]
            d = {}

        if term==1 or term ==2:
            d = {'A'+str(layer)+'odd':True,'B'+str(layer)+'odd':True}

        A = torch.einsum('abcdijkl,km,ijmlefgh',dagger(unitaries[0]),operators[0],unitaries[0])
        B = torch.einsum('abcdijkl,lm,ijkmefgh',dagger(unitaries[1]),operators[1],unitaries[1])
        return {'A'+str(layer):A,'B'+str(layer):B,**d}

    if position ==3:
        if term==3:
            m,n = (x-1)%N, y
            operators = [nHat,0.5*W(m,n)*torch.eye(2,device=dev) + U*nHat]
            d = {}

        if term==1 or term ==2:
            d = {'A'+str(layer)+'odd':True,'C'+str(layer)+'odd':True}

        A = torch.einsum('abcdijkl,lm,ijkmefgh',dagger(unitaries[0]),operators[0],unitaries[0])
        C = torch.einsum('abcdijkl,im,mjklefgh',dagger(unitaries[2]),operators[1],unitaries[2])
        return {'A'+str(layer):A,'C'+str(layer):C,**d}

    if position ==4:
        if term==3:
            m,n = x, y
            operators = [nHat,0.5*W(m,n)*torch.eye(2,device=dev) + U*nHat]
            d = {}

        if term==1 or term ==2:
            d = {'A'+str(layer)+'odd':True,'C'+str(layer)+'odd':True}

        A = torch.einsum('abcdijkl,km,ijmlefgh',dagger(unitaries[0]),operators[0],unitaries[0])
        C = torch.einsum('abcdijkl,jm,imklefgh',dagger(unitaries[2]),operators[1],unitaries[2])
        return {'A'+str(layer):A,'C'+str(layer):C,**d}

    if position ==5:
        if term==3:
            m,n = (x+1)%N, y
            operators = [nHat,0.5*W(m,n)*torch.eye(2,device=dev) + U*nHat]
            d = {}

        if term==1 or term ==2:
            d = {'B'+str(layer)+'odd':True,'D'+str(layer)+'odd':True}

        B = torch.einsum('abcdijkl,lm,ijkmefgh',dagger(unitaries[1]),operators[0],unitaries[1])
        D = torch.einsum('abcdijkl,im,mjklefgh',dagger(unitaries[3]),operators[1],unitaries[3])
        return {'B'+str(layer):B,'D'+str(layer):D,**d}

    if position==6:
        if term==3:
            m,n = (x+2)%N, y
            operators = [nHat,0.5*W(m,n)*torch.eye(2,device=dev) + U*nHat]
            d = {}

        if term==1 or term ==2:
            d = {'B'+str(layer)+'odd':True,'D'+str(layer)+'odd':True}

        B = torch.einsum('abcdijkl,km,ijmlefgh',dagger(unitaries[1]),operators[0],unitaries[1])
        D = torch.einsum('abcdijkl,jm,imklefgh',dagger(unitaries[3]),operators[1],unitaries[3])
        return {'B'+str(layer):B,'D'+str(layer):D,**d}

    if position==7:
        if term==3:
            m,n = x, (y+1)%N
            operators = [nHat,0.5*W(m,n)*torch.eye(2,device=dev) + U*nHat]
            d = {}

        if term==1 or term ==2:
            d = {'C'+str(layer)+'odd':True,'D'+str(layer)+'odd':True}

        C = torch.einsum('abcdijkl,jm,imklefgh',dagger(unitaries[2]),operators[0],unitaries[2])
        D = torch.einsum('abcdijkl,im,mjklefgh',dagger(unitaries[3]),operators[1],unitaries[3])
        return {'C'+str(layer):C,'D'+str(layer):D,**d}

    if position==8:
        if term==3:
            m,n = x, (y+2)%N
            operators = [nHat,0.5*W(m,n)*torch.eye(2,device=dev) + U*nHat]
            d = {}

        if term==1 or term ==2:
            d = {'A'+str(layer)+'odd':True,'B'+str(layer)+'odd':True}

        C = torch.einsum('abcdijkl,km,ijmlefgh',dagger(unitaries[2]),operators[0],unitaries[2])
        D = torch.einsum('abcdijkl,lm,ijkmefgh',dagger(unitaries[3]),operators[1],unitaries[3])
        return {'C'+str(layer):C,'D'+str(layer):D,**d}


def partialTraceL(h):
    return torch.einsum('iaib',h)


def partialTraceR(h):
    return torch.einsum('aibi',h)



def tensor_dictionary_edge(unitaries,position,x,y,layer=1):
    if position == 1:
        h = partialTraceL(hamiltonian((x,y-2)))
        A = torch.einsum('abcdijkl,jm,imklefgh',dagger(unitaries[0]),h,unitaries[0])
        return {'A'+str(layer):A}
    if position == 2:
        h = partialTraceL(hamiltonian((x-1,y-2)))
        A = torch.einsum('abcdijkl,im,mjklefgh',dagger(unitaries[0]),h,unitaries[0])
        return {'A'+str(layer):A}
    if position == 3:
        h = partialTraceL(hamiltonian((x-2,y-1)))
        A = torch.einsum('abcdijkl,im,mjklefgh',dagger(unitaries[0]),h,unitaries[0])
        return {'A'+str(layer):A}
    if position == 4:
        h = partialTraceL(hamiltonian((x-2,y)))
        A = torch.einsum('abcdijkl,lm,ijkmefgh',dagger(unitaries[0]),h,unitaries[0])
        return {'A'+str(layer):A}
    if position == 5:
        h = partialTraceL(hamiltonian((x-2,y+1)))
        C = torch.einsum('abcdijkl,im,mjklefgh',dagger(unitaries[2]),h,unitaries[2])
        return {'C'+str(layer):C}
    if position == 6:
        h = partialTraceL(hamiltonian((x-2,y+2)))
        C = torch.einsum('abcdijkl,lm,ijkmefgh',dagger(unitaries[2]),h,unitaries[2])
        return {'C'+str(layer):C}
    if position == 7:
        h = partialTraceR(hamiltonian((x-1,y+2)))
        C = torch.einsum('abcdijkl,lm,ijkmefgh',dagger(unitaries[2]),h,unitaries[2])
        return {'C'+str(layer):C}
    if position == 8:
        h = partialTraceR(hamiltonian((x,y+2)))
        C = torch.einsum('abcdijkl,km,ijmlefgh',dagger(unitaries[2]),h,unitaries[2])
        return {'C'+str(layer):C}
    if position == 9:
        h = partialTraceR(hamiltonian((x+1,y+2)))
        D = torch.einsum('abcdijkl,lm,ijkmefgh',dagger(unitaries[3]),h,unitaries[3])
        return {'D'+str(layer):D}
    if position == 10 or position==11:
        h = partialTraceR(hamiltonian((x+2,y+2)))
        D = torch.einsum('abcdijkl,km,ijmlefgh',dagger(unitaries[3]),h,unitaries[3])
        return {'D'+str(layer):D}
    if position == 12:
        h = partialTraceR(hamiltonian((x+2,y+1)))
        D = torch.einsum('abcdijkl,jm,imklefgh',dagger(unitaries[3]),h,unitaries[3])
        return {'D'+str(layer):D}
    if position == 13:
        h = partialTraceR(hamiltonian((x+2,y)))
        B = torch.einsum('abcdijkl,km,ijmlefgh',dagger(unitaries[1]),h,unitaries[1])
        return {'B'+str(layer):B}
    if position == 14:
        h = partialTraceR(hamiltonian((x+2,y-1)))
        B = torch.einsum('abcdijkl,jm,imklefgh',dagger(unitaries[1]),h,unitaries[1])
        return {'B'+str(layer):B}
    if position == 15:
        h = partialTraceL(hamiltonian((x+2,y-2)))
        B = torch.einsum('abcdijkl,jm,imklefgh',dagger(unitaries[1]),h,unitaries[1])
        return {'B'+str(layer):B}
    if position == 16:
        h = partialTraceL(hamiltonian((x+1,y-2)))
        B = torch.einsum('abcdijkl,im,mjklefgh',dagger(unitaries[1]),h,unitaries[1])
        return {'B'+str(layer):B}
