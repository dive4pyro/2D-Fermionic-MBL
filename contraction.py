'''
This is the code for the main trace operation, i.e. the fermionic version of
figure. 8 in the Wahl-Pal-Simon paper.

See "main trace operation" notes for figures and detailed explanations
Everything in the code here refers directly to the notes and vis versa

'''
import torch
from hamiltonian import *

#make the swap tensor
swap = torch.zeros([2,2,2,2],device=dev)
swap[0,0,0,0] = swap[0,1,1,0] = swap[1,0,0,1] = 1.
swap[1,1,1,1]= -1

id = torch.eye(2**4,device=dev).reshape(2,2,2,2,2,2,2,2)

'''
(assuming PBC) all terms in the figure of merit will involve the calculation below

First, combine all the little u's with sigmas and Hamiltonian terms as appropriate to
form the 10 tensors A1,B1,C1,D1,Z1,A2,B2,C2,D2, and Z2, in the form of Fig. 1
'''
def trace_calculation(Z1, Z2, A1=id,B1=id,C1=id,D1=id,A2=id,B2=id,C2=id,D2=id,A1odd=False,
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
        B1 = torch.einsum('ai,bj,ck,dl,ijklefgh',sz,sz,sz,sz,B1)

    #if D is odd
    if D1odd!=D2odd:    #note != is simply the XOR operator
        C1 = torch.einsum('ai,bj,ck,dl,ijklefgh',sz,sz,sz,sz,C1)

    #Now we have Fig. 3

    if B1odd:
        A = torch.einsum('ia,jb,ijcd',sz,sz,A)
    if B2odd:
        A = torch.einsum('ic,jd,abij',sz,sz,A)

    if C1odd:
        D = torch.einsum('ia,jb,ijcd',sz,sz,D)
    if C2odd:
        D = torch.einsum('ic,jd,abij',sz,sz,D)
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
    Z1 = torch.einsum('abij,cdkl,ijklefgh',swap,swap,Z1)
    Z1 = torch.einsum('cjbi,lgkf,iaekjdhl',swap,swap,Z1)

    Z2 = torch.einsum('abcdijkl,ijef,klgh',Z2,swap,swap)
    Z2 = torch.einsum('cjbi,lgkf,iaekjdhl',swap,swap,Z2)
    Z1 = torch.einsum('abcdefmn,mngh',Z1,C)
    Z = torch.einsum('abijefop,klcdopgh,ijkl',Z1,Z2,B)
    #Z = torch.einsum('abijefmn,klcdopgh,ijkl,mnop',Z1,Z2,B,C)
    return torch.einsum('abcd,abcdijkl,ijkl',A,Z,D)
