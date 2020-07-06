import os
import sys
path = os.getcwd().split('MiningSubjectiveSubgraphPatterns')[0]+'MiningSubjectiveSubgraphPatterns/'
if path not in sys.path:
	sys.path.append(path)
import numpy as np
import networkx as nx
import math
from src.BackgroundDistributions.PDClass import PDClass
##################################################################################################################################################################
class UniDistSimple(PDClass):
    def __init__(self, G):
        super().__init__(G)
        self.findDistribution()
##################################################################################################################################################################
    def findDistribution(self):
        self.ps = self.density
        if 	self.density == 1.0:
            self.la = float('inf')
        else:
            self.la = 0.5*math.log(self.density/(1-self.density))
        return
##################################################################################################################################################################
    def getAB(self):
        mSmallestLambda = np.min(self.la)
        mLargestLambda = np.max(self.la)

        if math.fabs(mSmallestLambda) > math.fabs(mLargestLambda):
            a = -3*math.fabs(mSmallestLambda)
            b = 3*math.fabs(mSmallestLambda)
        else:
            a = -3*math.fabs(mLargestLambda)
            b = 3*math.fabs(mLargestLambda)
        return a,b
###################################################################################################################################################################
    def getExpectationFromPOS(self, a):
        return a
###################################################################################################################################################################
    def getExpectationFromExpLambda(self, a):
        return a/(1+a)
###################################################################################################################################################################
    def getExpectation(self, i, j, **kwargs):
        kwargs['isSimple'] = True
        p = self.getPOS(i, j, **kwargs)
        E = self.getExpectationFromPOS(p)
        return E
##################################################################################################################################################################
    def explambda(self, i, j):
        expL = math.exp(self.la)*math.exp(self.la)
        return expL
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################