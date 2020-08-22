import os
import sys
path = os.getcwd().split('MiningSubjectiveSubgraphPatterns')[0]+'MiningSubjectiveSubgraphPatterns/'
if path not in sys.path:
	sys.path.append(path)
import numpy as np
import networkx as nx
import math
from src.BackgroundDistributions.PDClass import PDClass
###################################################################################################################################################################
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
            self.la = 0.5*math.log(self.density/(1+self.density))
        return
###################################################################################################################################################################
    def explambda(self, i, j):
        expL = math.exp(self.la)*math.exp(self.la)
        return expL
###################################################################################################################################################################
    def getAB(self):
        mSmallestLambda = np.min(self.la)
        mLargestLambda = np.max(self.la)

        epsilon = 1e-7

        if math.fabs(mSmallestLambda) > math.fabs(mLargestLambda):
            a = epsilon
            b = 3*math.fabs(mSmallestLambda)
        else:
            a = epsilon
            b = 3*math.fabs(mLargestLambda)
        return a,b
###################################################################################################################################################################
    def getExpectationFromExpLambda(self, a):
        if a > 0.9999999999:
            a = 0.9999999999
        return a/(1-a)
###################################################################################################################################################################
    def getExpectationFromPOS(self, a):
        return (1-a)/a
###################################################################################################################################################################
    def getExpectation(self, i, j, **kwargs):
        if i==j:
            return 0
        kwargs['isSimple'] = False
        p = self.getPOS(i, j, **kwargs)
        if p==0:
            raise Exception('i:{}, j:{}\n lprevUpdate: {}'.format(i,j,self.lprevUpdate))
        E = self.getExpectationFromPOS(p)
        return E
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################