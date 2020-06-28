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
###################################################################################################################################################################
class EvaluateRemove:
    # Now this data structure should contain all the possible removals
    # along with pattern number as key and Information Gain and list of nodes as value
    def __init__(self, gtype='U', isSimple=True, l=6):
        self.Data = dict()
        self.gtype = gtype
        self.isSimple = isSimple
        self.l = l # possible types (give number) of action, default is 6
        print('Initialized EvaluateRemove')
###################################################################################################################################################################
    def evaluateAllConstraints(self, G, PD): #arguments would be the distribution and output would all possibilities or no output
        self.Data = dict()
        for i in PD.lprevUpdate.keys():
            self.evaluateConstraint(G, PD, i)
        return
###################################################################################################################################################################
    def evaluateConstraint(self, G, PD, i):
        if self.gtype == 'U':
            NL = PD.lprevUpdate[i][1]

            codeLengthC = getCodeLengthParallel(G, PD, NL=NL, case=2, isSimple=self.isSimple, gtype=self.gtype) #now case is 1 as none of teh lambdas shall be removed
            codeLengthCprime = getCodeLengthParallel(G, PD, NL=NL, case=4, dropLidx=[i], isSimple=self.isSimple, gtype=self.gtype)  #now case is 4 as one lambda is to be dropped to compute new codelength

            IC = codeLengthC - codeLengthCprime
            DL = computeDescriptionLength(dlmode=4, gtype=self.gtype, C=len(PD.lprevUpdate), l=self.l)
            IG = computeInterestingness(IC, DL, mode=2)

            if IG > 0:
                H = G.subgraph(NL)
                P = Pattern(H)
                P.setIC_dssg(IC)
                P.setDL(DL)
                P.setI(IG)
                P.setPrevOrder(i)
                P.setPatType('Remove')
                P.setLambda(PD.lprevUpdate[i][0])
                self.Data[i] = P
        else:
            inNL = PD.lprevUpdate[i][1]
            outNL = PD.lprevUpdate[i][2]

            codeLengthC = getCodeLengthParallel(G, PD, inNL=inNL, outNL=outNL, case=1, isSimple=self.isSimple, gtype=self.gtype) #now case is 1 as none of teh lambdas shall be removed
            codeLengthCprime = getCodeLengthParallel(G, PD, inNL=inNL, outNL=outNL, case=4, dropLidx=[i], isSimple=self.isSimple, gtype=self.gtype)  #now case is 4 as one lambda is to be dropped to compute new codelength

            IC = codeLengthC - codeLengthCprime
            DL = computeDescriptionLength(dlmode=4, gtype=self.gtype, C=len(PD.lprevUpdate), l=self.l, excActionType=False)
            IG = computeInterestingness(IC, DL, mode=2)

            if IG > 0:
                H = getDirectedSubgraph(G, inNL, outNL, isSimple=self.isSimple)
                P = Pattern(H)
                P.setIC_dssg(IC)
                P.setDL(DL)
                P.setI(IG)
                P.setPrevOrder(i)
                P.setPatType('Remove')
                P.setLambda(PD.lprevUpdate[i][0])
                self.Data[i] = P
###################################################################################################################################################################
    def updateConstraintEvaluation(self, G, PD, i, condition = 1):
        #Things to note P shall have the prev_order(identifier of the constraint) correct
        if condition == 1: #update only description length
            DL = computeDescriptionLength(dlmode=4, gtype=self.gtype, C=len(PD.lprevUpdate), l=self.l, excActionType=False)
            IG = computeInterestingness(self.Data[i].IC_dssg, DL, mode=2)
            self.Data[i].setDL(DL)
            self.Data[i].setI(IG)
        elif condition == 2: #update codelength and description length
            self.evaluateConstraint(G, PD, i)
        return

    def checkAndUpdateAllPossibilities(self, G, PD, prevPat):
        if self.gtype == 'U':
            for k,v in self.Data.items():
                if len(set(v.NL).intersection(set(prevPat.NL))) > 1:
                    self.updateConstraintEvaluation(G, PD, prevPat.prev_order, 2)
                else:
                    self.updateConstraintEvaluation(G, PD, prevPat.prev_order, 1)
        else:
            for k,v in self.Data.items():
                inInt = len(set(v.inNL).intersection(set(prevPat.inNL)))
                outInt = len(set(v.outNL).intersection(set(prevPat.outNL)))
                if inInt > 0 and outInt > 0:
                    self.updateConstraintEvaluation(G, PD, prevPat.prev_order, 2)
                else:
                    self.updateConstraintEvaluation(G, PD, prevPat.prev_order, 1)
        return
###################################################################################################################################################################
    def getBestOption(self):
        if len(self.Data) < 1:
            return None
        else:
            bestR = max(self.Data.items(), key=lambda x: x[1].I)
            return bestR[1]
###################################################################################################################################################################
	def updateDistribution(self, PD, bestR):
		# Now here we remove the knowledge of any such pattern, hence we remove the lambda associated 
		# with the pattern
		output = PD.lprevUpdate.pop(bestR.prev_order, None)
		if output is None:
			print("Something is fishy")
		else: # This to check if the poped item is correct or not
			if output[0] != bestR.la:
				print("Something is more fishy")
		return
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################