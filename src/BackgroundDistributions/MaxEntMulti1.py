import numpy as np
import networkx as nx
import math
import os
import sys
path = os.getcwd().split('MiningSubjectiveSubgraphPatterns')[0]+'MiningSubjectiveSubgraphPatterns/'
if path not in sys.path:
	sys.path.append(path)
from src.BackgroundDistributions.PDClass import PDClass
###################################################################################################################################################################
class MaxEntMulti1U(PDClass):
    def __init__(self, G = None):
        super().__init__(G)
        self.la = None
        self.jrows = None
        self.errors = None
        self.ps = None
        self.gla = None
        self.lprevUpdate = {}
        self.degrees = None
        if G is not None:
            self.findMaxEntDistribution()
###################################################################################################################################################################
    def findMaxEntDistribution(self):
        self.degrees = np.array(list(dict(sorted(dict(self.G.degree()).items())).values()))
        n = len(self.degrees)
        ######################
        prows = self.degrees/n
        prowsunique,irows,self.jrows,vrows = np.unique(prows, return_index=True, return_inverse=True, return_counts=True)
        nunique = len(prowsunique)
        bins = np.zeros(nunique)
        if irows[0] == 1:
            bins[0] == 1
        for i in range(1, nunique):
            if irows[i-1] == irows[i]-1:
                bins[i] = 1
        self.la = -np.ones(nunique)
        h = np.zeros(nunique)
        nit = 1000
        tol = 1e-14
        self.errors = np.empty(0)
        ######################
        lb = -5
        for k in range(nit):
            E = np.multiply(np.outer(np.ones(nunique).T, np.exp(self.la/2)),np.outer(np.exp(self.la/2), np.ones(nunique).T))
            ps = np.divide(E, 1-E)
            self.gla = np.multiply(-n*prowsunique+np.dot(ps, vrows)-np.diag(ps), vrows)
            self.errors = np.append(self.errors, np.linalg.norm(self.gla))

            H = 1/2*np.dot(np.dot(np.diag(vrows), np.divide(E, np.square(1-E))), np.diag(vrows))
            H = H + np.diag(np.sum(H, 0)) - 2*np.diag(np.divide(np.diag(H), vrows))
            H = H + np.dot(np.trace(H) / nunique,1e-10)

            deltala = np.linalg.solve(- H,self.gla)

            fbest = 0;
            errorbest = self.errors[k];

            for f in np.logspace(lb,1,20):
                latry=self.la+f*deltala
                Etry = np.multiply(np.outer(np.ones(nunique).T, np.exp(latry/2)),np.outer(np.exp(latry/2), np.ones(nunique).T))
                if np.max(np.max(Etry - np.diag(np.multiply(bins, np.diag(Etry))))) >= 1:
                    break
                pstry = np.divide(Etry, 1-Etry)
                glatry = np.multiply(-n*prowsunique+np.dot(pstry, vrows)-np.diag(pstry), vrows)
                errortry = np.linalg.norm(glatry)
                if errortry < errorbest:
                    fbest = f
                    errorbest = errortry
            if fbest == 0:
                if lb>-1000:
                    lb = lb*2
                else:
                    break

            self.la = self.la+fbest*deltala;

            if self.errors[k]/n < tol:
                break

        E = np.multiply(np.outer(np.ones(nunique).T, np.exp(self.la/2)),np.outer(np.exp(self.la/2), np.ones(nunique).T))
        self.ps = np.divide(E, 1-E)
        self.gla = np.multiply(-n*prowsunique+np.dot(self.ps, vrows)-np.diag(self.ps), vrows)
        self.errors = np.append(self.errors, np.linalg.norm(self.gla))
###################################################################################################################################################################
    def getAB(self):
        mSmallestLambda = np.min(self.la)
        mLargestLambda = np.max(self.la)

        epsilon = 1e-8

        if math.fabs(mSmallestLambda) > math.fabs(mLargestLambda):
            a = epsilon
            b = 3*math.fabs(mSmallestLambda)
        else:
            a = epsilon
            b = 3*math.fabs(mLargestLambda)
        return a,b
###################################################################################################################################################################
    def getExpectationFromExpLambda(self, a):
        if 1-a < 1e-10:
            return 1e10
        return a/(1-a)
###################################################################################################################################################################
    def getExpectationFromPOS(self, a):
        return (1-a)/a
###################################################################################################################################################################
    def getExpectation(self, i, j, **kwargs):
        if i==j:
            return 0.0
        p = self.getPOS(i, j, **kwargs)
        # if p < 1e-2:
        #     print(i, j, p)
        E = self.getExpectationFromPOS(p)
        return E
###################################################################################################################################################################
    def explambda(self, i, j):
        expL = math.exp(self.la[self.jrows[i]]/2)*math.exp(self.la[self.jrows[j]]/2)
        return expL
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################


class MaxEntMulti1D(PDClass):
    def __init__(self, G = None):
        super().__init__(G)
        self.tp = 'D'
        self.la = None
        self.mu = None
        self.jrows = None
        self.jcols = None
        self.errors = None
        self.ps = None
        self.gla = None
        self.lprevUpdate = {}
        self.indegrees = None
        self.outdegrees = None
        if G is not None:
            self.findMaxEntDistribution()
###################################################################################################################################################################
    def findMaxEntDistribution(self):
        self.indegrees = np.array(list(dict(sorted(dict(self.G.in_degree()).items())).values()))
        self.outdegrees = np.array(list(dict(sorted(dict(self.G.out_degree()).items())).values()))
        n = len(self.indegrees)
        m = len(self.outdegrees)

        fac = math.log(self.density/(1+self.density))

        prows = self.outdegrees
        prowsunique, irows, self.jrows, vrows = np.unique(prows, return_index=True, return_inverse=True, return_counts=True)
        rownunique = len(prowsunique)
        self.la = -math.fabs(fac)*np.ones(rownunique)
        rowh = np.zeros(rownunique)

        pcols = self.indegrees
        pcolsunique, icols, self.jcols, vcols = np.unique(pcols, return_index=True, return_inverse=True, return_counts=True)
        colnunique = len(pcolsunique)
        self.mu = -math.fabs(fac)*np.ones(colnunique)
        colh = np.zeros(colnunique)

        loops = np.outer(np.zeros(rownunique), np.zeros(colnunique).T)

        for i in range(rownunique):
            for j in range(colnunique):
                loops[i][j] = len(set(np.where(self.jrows==i)[0]).intersection(set(np.where(self.jcols==j)[0])))
        finalmat=np.outer(vrows, vcols)-np.array(loops)

        print('Sizes----Vrows:',len(vrows),' Vcols', len(vcols))

        nit = 1000
        tol = 1e-14
        self.errors = np.empty(0)

        lb = -5
        for k in range(nit):
            E = np.multiply(np.outer(np.exp(self.la/2), np.ones(colnunique).T),np.outer(np.ones(rownunique).T, np.exp(self.mu/2)))
            self.ps = np.divide(E, 1-E)

            gla_t = np.multiply(self.ps, finalmat)

            gla_r = np.sum(gla_t, 1) - np.multiply(prowsunique, vrows)
            gla_c = np.sum(gla_t, 0) - np.multiply(pcolsunique, vcols)

            self.gla = np.append(gla_r, gla_c)

            self.errors = np.append(self.errors, np.linalg.norm(self.gla))

            H_t = np.divide(E, np.square(1-E))
            H_t = np.multiply(H_t, finalmat)

            H_r = np.diag(np.sum(H_t,1))
            H_c = np.diag(np.sum(H_t,0))

            H = np.append(np.append(H_r, H_t, 1), np.append(H_t.T, H_c, 1), 0)

            delta = np.linalg.lstsq(- H, self.gla, rcond=max(H.shape)*np.finfo(H.dtype).eps)[0]
            deltala = delta[0:rownunique]
            deltamu = delta[rownunique:rownunique+colnunique+1]
            fbest = 0;
            errorbest = self.errors[k];

            for f in np.logspace(lb,1,20):
                latry=self.la+f*deltala
                mutry=self.mu+f*deltamu
                flag = True
                for ind1 in range(len(latry)):
                    for ind2 in range(len(mutry)):
                        if latry[ind1]+mutry[ind2]>-1e-15 and finalmat[ind1][ind2]>0.0001:
                            flag = False
                if flag:
                    Etry = np.multiply(np.outer(np.exp(latry/2), np.ones(colnunique).T),np.outer(np.ones(rownunique).T, np.exp(mutry/2)))
                    pstry = np.divide(Etry, 1-Etry)

                    gla_ttry = np.multiply(pstry, finalmat)

                    gla_rtry = np.sum(gla_ttry, 1) - np.multiply(prowsunique, vrows)
                    gla_ctry = np.sum(gla_ttry, 0) - np.multiply(pcolsunique, vcols)

                    gla_try = np.append(gla_rtry, gla_ctry)

                    errortry = np.linalg.norm(gla_try)

                    if errortry < errorbest:
                        fbest = f
                        errorbest = errortry
            if fbest == 0:
                if lb>-1000:
                    lb = lb*2
                else:
                    break

            self.la = self.la+fbest*deltala;
            self.mu = self.mu+fbest*deltamu;

            if self.errors[k] < tol:
                break

        E = np.multiply(np.outer(np.exp(self.la/2), np.ones(colnunique).T),np.outer(np.ones(rownunique).T, np.exp(self.mu/2)))
        self.ps = np.divide(E, 1-E)
        gla_t = np.multiply(self.ps, finalmat)

        gla_r = np.sum(gla_t, 1) - np.multiply(prowsunique, vrows)
        gla_c = np.sum(gla_t, 0) - np.multiply(pcolsunique, vcols)

        self.gla = np.append(gla_r, gla_c)

        self.errors = np.append(self.errors, np.linalg.norm(self.gla))
###################################################################################################################################################################
    def getAB(self):
        mSmallestLambda = np.min(np.array(list(set(self.la).union(set(self.mu)))))
        mLargestLambda = np.max(np.array(list(set(self.la).union(set(self.mu)))))

        epsilon = 1e-8

        if math.fabs(mSmallestLambda) > math.fabs(mLargestLambda):
            a = epsilon
            b = 3*math.fabs(mSmallestLambda)
        else:
            a = epsilon
            b = 3*math.fabs(mLargestLambda)
        return a,b
###################################################################################################################################################################
    def getExpectationFromExpLambda(self, a):
        return a/(1-a)
###################################################################################################################################################################
    def getExpectationFromPOS(self, a):
        if a < 1e-10:
            a = 1e-10
        if a > 1.0 - 1e-10:
            a = 1.0 - 1e-10
        return (1-a)/a
###################################################################################################################################################################
    def getExpectation(self, i, j, **kwargs):
        kwargs['isSimple'] = False
        p = self.getPOS(i, j, **kwargs)
        E = self.getExpectationFromPOS(p)
        return E
###################################################################################################################################################################
    def explambda(self, i, j):
        expL = math.exp(self.la[self.jrows[i]]/2)*math.exp(self.mu[self.jcols[j]]/2)
        return expL
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################