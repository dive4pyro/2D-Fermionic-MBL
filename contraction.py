'''This is the code for the trace_calculation function,
i.e. the fermionic version of figure. 8 in the Wahl-Pal-Simon paper.

(note fig. 8 in the paper applies directly the code here BUT the labels
C and D need to be swapped)

See "main trace operation" notes for figures and detailed explanations
Everything in the code here refers directly to the notes and vis versa'''

import torch
from hamiltonian import *

#make the swap tensor
swap = torch.zeros([d,d,d,d],device=dev)

for i in range(d):
    for j in range(d):
        if i==j==1 or i==j==2:
            swap[i,j,j,i] = -1
        else:
            swap[i,j,j,i] = 1

id = torch.eye(d**4,device=dev).reshape(d,d,d,d,d,d,d,d)

if d==2:
    sCross = torch.diag(torch.tensor([1,-1.],device=dev))
if d==3:
    sCross = torch.diag(torch.tensor([1,-1,-1.],device=dev))
if d==4:
    sCross = torch.diag(torch.tensor([1,-1,-1,1.],device=dev))

'''
(assuming PBC) all terms in the figure of merit will involve the calculation below

First, combine all the little u's with sigmas and Hamiltonian terms as appropriate to
form the 10 tensors A1,B1,C1,D1,A2,B2,C2,D2, and Z, in the form of Fig. 1
'''

#check if two tensors A and B are equal
#returns true if A==B and false if A!=B
def equal(A,B):
    return (A==B).min()
################################################################################
'''this is the optimized version.  that is, at various points it checks whether
tensors are identities or not, and if so, does not perform certain contraction operations

unfortunately it turns out that the speedup is negligible in the d=2 or 3 case,
and marginal in the d=4 case, but oh well...'''
def trace_calculation(Z, A1=id,B1=id,C1=id,D1=id,A2=id,B2=id,C2=id,D2=id,A1odd=False,
A2odd=False,B1odd=False,B2odd=False,C1odd=False,C2odd=False,D1odd=False,D2odd=False):
    ###############################################################
    A_id = B_id = C_id = D_id = False
    #STEP 1: get A and D
    if equal(A1,A2) and equal(A1,id):
        A_id = True
    else:
        if not equal(A1,id):
            A1 = torch.einsum('cdij,abijefkl,klgh',swap,A1,swap)
        if not equal(A2,id):
            A2 = torch.einsum('cdij,abijefkl,klgh',swap,A2,swap)
        A = torch.einsum('ijkdlmna,lmnbijkc',A1,A2)
    if equal(D1,D2) and equal(D2,id):
        D_id=True
    else:
        D = torch.einsum('dijkalmn,blmncijk',D1,D2)
    # now we have Fig. 2
    ###############################################################
    #STEP 2: drag the lines

    #if A is odd
    if A1odd!=A2odd:    #note != is simply the XOR operator
        B1 = torch.einsum('ai,bj,ck,dl,ijklefgh',sCross,sCross,sCross,sCross,B1)

    #if D is odd
    if D1odd!=D2odd:    #note != is simply the XOR operator
        C1 = torch.einsum('ai,bj,ck,dl,ijklefgh',sCross,sCross,sCross,sCross,C1)

    #Now we have Fig. 3

    if B1odd:
        if A_id:
            A = d**3*torch.einsum('ad,bc',sCross,sCross)
            A_id = False
        else:
            A = torch.einsum('ia,jb,ijcd',sCross,sCross,A)
    if B2odd:
        if A_id:
            A = d**3*torch.einsum('ad,bc',sCross,sCross)
            A_id = False
        else:
            A = torch.einsum('ic,jd,abij',sCross,sCross,A)

    if C1odd:
        if D_id:
            D = d**3*torch.einsum('ad,bc',sCross,sCross)
            D_id = False
        else:
            D = torch.einsum('ia,jb,ijcd',sCross,sCross,D)
    if C2odd:
        if D_id:
            D = d**3*torch.einsum('ad,bc',sCross,sCross)
            D_id = False
        else:
            D = torch.einsum('ic,jd,abij',sCross,sCross,D)
    #now we have Fig. 4
    ###############################################################
    #STEP 3
    #get B and C
    if equal(B1,id) and equal(B2,id):
        B_id = True
    else:
        B = torch.einsum('ijkdlmna,lmnbijkc',B1,B2)

    if equal(C1,id) and equal(C2,id):
        C_id = True
    else:
        if not equal(C1,id):
            C1 = torch.einsum('abij,ijcdklgh,klef',swap,C1,swap)
        if not equal(C2,id):
            C2 = torch.einsum('abij,ijcdklgh,klef',swap,C2,swap)
        C = torch.einsum('dijkalmn,blmncijk',C1,C2)
    #now we have fig. 5
    ###############################################################
    Z1 = Z.clone(); Z2 =Z.clone()
    if not B_id or not A_id:
        Z1 = torch.einsum('abij,ijcdefgh',swap,Z1)
        Z1 = torch.einsum('ejai,ibcdjfgh',swap,Z1)

        Z2 = torch.einsum('ijef,abcdijgh',swap,Z2)
        Z2 = torch.einsum('ejai,ibcdjfgh',swap,Z2)

    if not C_id or not D_id:
        Z1 = torch.einsum('cdij,abijefgh',swap,Z1)
        Z1 = torch.einsum('jhid,abciefgj',swap,Z1)

        Z2 = torch.einsum('ijgh,abcdefij',swap,Z2)
        Z2 = torch.einsum('jhid,abciefgj',swap,Z2)

    Z1 = d**3*Z1 if B_id else torch.einsum('abcdijgh,ijfe',Z1,B)
    Z1 = d**3*Z1 if C_id else torch.einsum('abcdefij,jigh',Z1,C)
    Z1 = d**3*Z1 if A_id else torch.einsum('jicdefgh,ijab',Z1,A)
    Z1 = d**3*Z1 if D_id else torch.einsum('abijefgh,ijdc',Z1,D)

    return torch.einsum('abcdijkl,ijklabcd',Z1,Z2)


###############################################################################################################3
'''this is the 'non-optimized' version'''
def trace_calculation_not_optimized(Z, A1=id,B1=id,C1=id,D1=id,A2=id,B2=id,C2=id,D2=id,A1odd=False,
A2odd=False,B1odd=False,B2odd=False,C1odd=False,C2odd=False,D1odd=False,D2odd=False):
    ###############################################################
    #STEP 1: get A and D
    A1 = torch.einsum('cdij,abijefkl,klgh',swap,A1,swap)
    A2 = torch.einsum('cdij,abijefkl,klgh',swap,A2,swap)
    A = torch.einsum('ijkdlmna,lmnbijkc',A1,A2)
    D = torch.einsum('dijkalmn,blmncijk',D1,D2)
    # now we have Fig. 2
    ###############################################################
    #STEP 2: drag the lines

    #if A is odd
    if A1odd!=A2odd:    #note != is simply the XOR operator
        B1 = torch.einsum('ai,bj,ck,dl,ijklefgh',sCross,sCross,sCross,sCross,B1)

    #if D is odd
    if D1odd!=D2odd:    #note != is simply the XOR operator
        C1 = torch.einsum('ai,bj,ck,dl,ijklefgh',sCross,sCross,sCross,sCross,C1)

    #Now we have Fig. 3

    if B1odd:
        A = torch.einsum('ia,jb,ijcd',sCross,sCross,A)
    if B2odd:
        A = torch.einsum('ic,jd,abij',sCross,sCross,A)

    if C1odd:
        D = torch.einsum('ia,jb,ijcd',sCross,sCross,D)
    if C2odd:
        D = torch.einsum('ic,jd,abij',sCross,sCross,D)
    #now we have Fig. 4
    ###############################################################
    #STEP 3
    #get B and C
    B = torch.einsum('ijkdlmna,lmnbijkc',B1,B2)
    C1 = torch.einsum('abij,ijcdklgh,klef',swap,C1,swap)
    C2 = torch.einsum('abij,ijcdklgh,klef',swap,C2,swap)
    C = torch.einsum('dijkalmn,blmncijk',C1,C2)
    #now we have fig. 5
    ###############################################################
    Z1 = torch.einsum('abij,ijcdefgh',swap,Z)
    Z1 = torch.einsum('ejai,ibcdjfgh',swap,Z1)
    Z1 = torch.einsum('cdij,abijefgh',swap,Z1)
    Z1 = torch.einsum('jhid,abciefgj',swap,Z1)

    Z2 = torch.einsum('ijef,abcdijgh',swap,Z)
    Z2 = torch.einsum('ejai,ibcdjfgh',swap,Z2)
    Z2 = torch.einsum('ijgh,abcdefij',swap,Z2)
    Z2 = torch.einsum('jhid,abciefgj',swap,Z2)

    Z1 = torch.einsum('abcdijgh,ijfe',Z1,B)
    Z1 = torch.einsum('abcdefij,jigh',Z1,C)
    Z1 = torch.einsum('jicdefgh,ijab',Z1,A)
    Z1 = torch.einsum('abijefgh,ijdc',Z1,D)

    return torch.einsum('abcdijkl,ijklabcd',Z1,Z2)
