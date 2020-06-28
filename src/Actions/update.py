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

class EvaluateUpdate:
    # Now this data structure should contain all the possible removals
    # along with pattern number as key and Information Gain and list of nodes as value
    def __init__(self, gtype='U', isSimple=True, l=6):
        self.Data = dict()
        self.gtype = gtype
        self.isSimple = isSimple
        self.l = l # possible types (give number) of action, default is 6
        print('initialized EvaluateUpdate')

    def evaluateAllConstraints(self, G, PD):
        self.Data = dict()
        for i in PD.lprevUpdate.keys():
            self.evaluateConstraint(G, PD, i)
        return

    def evaluateConstraint(self, G, PD, i):
        if self.gtype == 'U':
            NL = PD.lprevUpdate[i][1]
            H = G.subgraph(NL)
            nlambda = PD.updateDistribution( H, idx=None, val_retrun='return', case=3, dropLidx=[i] ) #// ToDo: Add code to coumpute new lambda

            codeLengthC = getCodeLengthParallel(G, PD, NL=NL, case=2, isSimple=self.isSimple, gtype=self.gtype) #now case is 1 as none of teh lambdas shall be removed
            codeLengthCprime = getCodeLengthParallel(G, PD, NL=NL, case=5, dropLidx=[i], nlambda=nlambda, isSimple=self.isSimple, gtype=self.gtype)  #now case is 4 as one lambda is to be dropped to compute new codelength

            IC = codeLengthC - codeLengthCprime
            DL = computeDescriptionLength(dlmode=5, gtype=self.gtype, C=len(PD.lprevUpdate), W=len(NL), l=self.l, excActionType=False)
            IG = computeInterestingness(IC, DL, mode=2)

            if IG > 0:
                H = G.subgraph(NL)
                P = Pattern(H)
                P.setIC_dssg(IC)
                P.setDL(DL)
                P.setI(IG)
                P.setPrevOrder(i)
                P.setPatType('Update')
                P.setLambda(nlambda)
                self.Data[i] = P
        else:
            inNL = PD.lprevUpdate[i][1]
            outNL = PD.lprevUpdate[i][2]

            HD = getDirectedSubgraph( G, inNL, outNL, self.isSimple )

            nlambda = PD.updateDistribution( HD, idx=None, val_retrun='return', case=3, dropLidx=[i] ) #// ToDo: Add code to coumpute new lambda

            codeLengthC = getCodeLengthParallel(G, PD, inNL=inNL, outNL=outNL, case=1, isSimple=self.isSimple, gtype=self.gtype) #now case is 1 as none of teh lambdas shall be removed
            codeLengthCprime = getCodeLengthParallel(G, PD, inNL=inNL, outNL=outNL, case=5, dropLidx=[i], nlambda=nlambda, isSimple=self.isSimple, gtype=self.gtype)  #now case is 4 as one lambda is to be dropped to compute new codelength

            IC = codeLengthC - codeLengthCprime
            DL = computeDescriptionLength(dlmode=4, gtype=self.gtype, C=len(PD.lprevUpdate), WI=inNL, WO=outNL, l=self.l, excActionType=False)
            IG = computeInterestingness(IC, DL, mode=2)

            if IG > 0:
                H = getDirectedSubgraph(G, inNL, outNL, isSimple=self.isSimple)
                P = Pattern(H)
                P.setIC_dssg(IC)
                P.setDL(DL)
                P.setI(IG)
                P.setPrevOrder(i)
                P.setPatType('Update')
                P.setLambda(nlambda)
                self.Data[i] = P
        return