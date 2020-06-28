###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################
import os
import sys
path = os.getcwd().split('MiningSubjectiveSubgraphPatterns')[0]+'MiningSubjectiveSubgraphPatterns/'
if path not in sys.path:
    sys.path.append(path)
import numpy as np
import math
import networkx as nx

from src.Utils.Measures import getCodeLength, getCodeLengthParallel, getDirectedSubgraph
from src.Utils.Measures import computeDescriptionLength, computeInterestingness
from src.Patterns.Pattern import Pattern

class EvaluateMerge:
    # Now this data structure should contain all the possible removals
    # along with pattern number as key and Information Gain and list of nodes as value
    def __init__(self, gtype='U', isSimple=True, l=6):
        self.Data = dict()
        self.gtype = gtype
        self.isSimple = isSimple
        self.l = l # possible types (give number) of action, default is 6
        print('initialized EvaluateMerge')

    def evaluateAllConstraintPairs(self, G, PD):
        keys = list(PD.lprevUpdate.keys())
        for i1 in range(len(keys)-1):
            for i2 in range(i1+1, len(keys)):
                self.evaluateConstraintPair( G, PD, keys[i1], keys[i2] )
        return

    def evaluateConstraintPair(self, G, PD, k1, k2):
        if self.gtype == 'U':
            NL1 = PD.lprevUpdate[k1][1]
            NL2 = PD.lprevUpdate[k2][1]
            NL_F = list(set(NL1).union(set(NL2)))
            H = G.subgraph(NL_F)
            if nx.is_connected(H):
                self.computeParametersU( H, NL_F, PD, k1, k2 )
        elif self.gtype == 'D':
            inNL1 = PD.lprevUpdate[k1][1]
            inNL2 = PD.lprevUpdate[k2][1]
            outNL1 = PD.lprevUpdate[k1][2]
            outNL2 = PD.lprevUpdate[k2][2]
            inNL_F = list(set(inNL1).union(set(inNL2)))
            outNL_F = list(set(outNL1).union(set(outNL2)))
            HD = getDirectedSubgraph( G, inNL_F, outNL_F, self.isSimple )
            H = nx.Graph(HD)
            if nx.is_connected(H):
                self.computeParametersD( HD, inNL_F, outNL_F, PD, k1, k2 )
        return

    def computeParametersU(self, H, NL, PD, k1, k2):
        nlambda = PD.updateDistribution( H, idx=None, val_retrun='return', case=3, dropLidx=[k1, k2] ) #// TODO: handle this issue, code it !!!!!!!!
        codeLengthC = getCodeLengthParallel( H, PD, gtype=self.gtype, case=2, NL=NL, isSimple=self.isSimple )
        codeLengthCprime = getCodeLengthParallel( H, PD, gtype=self.gtype, case=5, NL=NL, isSimple=self.isSimple, dropLidx=[k1, k2], nlambda=nlambda )
        IC = codeLengthC - codeLengthCprime
        DL = computeDescriptionLength( dlmode=8, excActionType=False, l=6, gtype=self.gtype,W=H.number_of_nodes(), kw=H.number_of_edges(), C=len(PD.lprevUpdate) )
        IG = computeInterestingness( IC, DL, mode=2 )
        if IG > 0:
            P = Pattern(H)
            P.setIC_dssg(IC)
            P.setDL(DL)
            P.setI(IG)
            P.setPrevOrder([k1,k2])
            P.setPatType('Merge')
            P.setLambda(nlambda)
            self.Data[i] = P
        return

    def computeParametersD(self, H, inNL, outNL, PD, k1, k2):
        nlambda = PD.updateDistribution( H, idx=None, val_retrun='return', case=3, dropLidx=[k1, k2] ) #// TODO: handle this issue, code it !!!!!!!!
        codeLengthC = getCodeLengthParallel( H, PD, gtype=self.gtype, case=2, inNL=inNL, outNL=outNL, isSimple=self.isSimple )
        codeLengthCprime = getCodeLengthParallel( H, PD, gtype=self.gtype, case=5, inNL=inNL, outNL=outNL, isSimple=self.isSimple, dropLidx=[k1, k2], nlambda=nlambda )
        IC = codeLengthC - codeLengthCprime
        DL = computeDescriptionLength( dlmode=8, excActionType=False, l=6, gtype=self.gtype, WI=inNL, WO=outNL, kw=H.number_of_edges(), C=len(PD.lprevUpdate) )
        IG = computeInterestingness( IC, DL, mode=2 )
        if IG > 0:
            P = Pattern(H)
            P.setIC_dssg(IC)
            P.setDL(DL)
            P.setI(IG)
            P.setPrevOrder([k1,k2])
            P.setPatType('Merge')
            P.setLambda(nlambda)
            self.Data[i] = P
        return