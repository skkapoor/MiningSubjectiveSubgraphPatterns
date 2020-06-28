import numpy as np
import math
import networkx as nx
from PDClass import PDClass
###################################################################################################################################################################
class MaxEntMulti2U(PDClass):
    def __init__(self, G = None):
        super().__init__()
        self.la = None
        self.mu = None
        self.degreeNeighbor = None
        self.degrees = None
        self.Neighbors = None
        self.jrows = None
        self.errors = None
        self.ps_la = None
        self.ps_mu = None
        self.gla = None
        self.lprevUpdate = {}
        if G is not None:
            self.findMaxEntDistribution()
###################################################################################################################################################################
    def findMaxEntDistribution(self):
        self.degrees = np.array(list(dict(sorted(dict(self.G.degree()).items())).values()))
        self.Neighbors = []
        for i in range(self.G.number_of_nodes()):
            self.Neighbors.append(len(list(self.G.neighbors(i))))
        self.Neighbors = np.array(self.Neighbors)
        self.degreeNeighbor = []
        for i in range(len(self.degrees)):
            self.degreeNeighbor.append(tuple([self.degrees[i], self.Neighbors[i]]))
        self.degreeNeighbor = np.array(self.degreeNeighbor)
        ##############################
        prows = self.degreeNeighbor
        prowsunique,irows,self.jrows,vrows = np.unique(prows, axis=0, return_index=True, return_inverse=True, return_counts=True)
        nunique = len(prowsunique)
        self.la = -np.ones(nunique)
        self.mu = -np.ones(nunique)
        h = np.zeros(nunique)
        nit = 1000
        tol = 1e-14
        self.errors = np.empty(0)
        ##############################
        lb = -5 
        for k in range(nit):
            R = np.multiply(np.outer(np.ones(nunique).T, np.exp(self.la/2)),np.outer(np.exp(self.la/2), np.ones(nunique).T))
            S = np.multiply(np.outer(np.ones(nunique).T, np.exp(self.mu/2)),np.outer(np.exp(self.mu/2), np.ones(nunique).T))

            ps_la = np.divide(np.multiply(R,S), np.multiply(1-R, 1-np.multiply(R,1-S)))
            ps_mu = np.divide(np.multiply(R,S), 1-np.multiply(R,1-S))

            gla_la = np.multiply(-prowsunique[:,0]+np.dot(ps_la, vrows)-np.diag(ps_la), vrows)
            gla_mu = np.multiply(-prowsunique[:,1]+np.dot(ps_mu, vrows)-np.diag(ps_mu), vrows)

            self.gla = np.append(gla_la, gla_mu)
            self.errors = np.append(self.errors, np.linalg.norm(self.gla))

            H1_u1 = np.dot(np.dot(np.diag(vrows), np.divide(np.multiply(np.multiply(R,S), 1 - np.multiply(np.square(R), 1-S)), np.square(np.multiply(1-R, 1-np.multiply(R,1-S))))), np.diag(vrows)) 
            H1_u2 = np.diag(np.sum(H1_u1, 0)) - np.diag(np.divide(np.diag(H1_u1), vrows))
            H1 = H1_u1 + H1_u2

            H2_u1 = np.dot(np.dot(np.diag(vrows), np.divide(np.multiply(np.multiply(R,S), 1 - R), np.square(1-np.multiply(R,1-S)))), np.diag(vrows)) 
            H2_u2 = np.diag(np.sum(H2_u1, 0)) - np.diag(np.divide(np.diag(H2_u1), vrows))
            H2 = H2_u1 + H2_u2

            H3_u1 = np.dot(np.dot(np.diag(vrows), np.divide(np.multiply(R,S), np.square(1-np.multiply(R,1-S)))), np.diag(vrows)) 
            H3_u2 = np.diag(np.sum(H3_u1, 0)) - np.diag(np.divide(np.diag(H3_u1), vrows))
            H3 = H3_u1 + H3_u2

            H = 0.5 * np.append(np.append(H1, H3, 1), np.append(H3, H2, 1), 0)

            delta = np.linalg.lstsq(- H, self.gla, rcond=max(H.shape)*np.finfo(H.dtype).eps)[0]
            delta_la = delta[0:nunique]
            delta_mu = delta[nunique:nunique+nunique+1]

            fbest = 0;
            errorbest = self.errors[k];

            for f in np.logspace(lb,1,20):
                latry=self.la+f*delta_la
                mutry=self.mu+f*delta_mu

                Rtry = np.multiply(np.outer(np.ones(nunique).T, np.exp(latry/2)),np.outer(np.exp(latry/2), np.ones(nunique).T))
                Stry = np.multiply(np.outer(np.ones(nunique).T, np.exp(mutry/2)),np.outer(np.exp(mutry/2), np.ones(nunique).T))

                ps_latry = np.divide(np.multiply(Rtry,Stry), np.multiply(1-Rtry, 1-np.multiply(Rtry,1-Stry)))
                ps_mutry = np.divide(np.multiply(Rtry,Stry), 1-np.multiply(Rtry,1-Stry))

                gla_latry = np.multiply(-prowsunique[:,0]+np.dot(ps_latry, vrows)-np.diag(ps_latry), vrows)
                gla_mutry = np.multiply(-prowsunique[:,1]+np.dot(ps_mutry, vrows)-np.diag(ps_mutry), vrows)

                glatry = np.append(gla_latry, gla_mutry)
                errortry = np.linalg.norm(glatry)
                if errortry < errorbest:
                    fbest = f
                    errorbest = errortry
            if fbest == 0:
                if lb>-1000:
                    lb = lb*2
                else:
                    break

            self.la = self.la+fbest*delta_la
            self.mu = self.mu+fbest*delta_mu

            if self.errors[k] < tol:
                break

        R = np.multiply(np.outer(np.ones(nunique).T, np.exp(self.la/2)),np.outer(np.exp(self.la/2), np.ones(nunique).T))
        S = np.multiply(np.outer(np.ones(nunique).T, np.exp(self.mu/2)),np.outer(np.exp(self.mu/2), np.ones(nunique).T))

        self.ps_la = np.divide(np.multiply(R,S), np.multiply(1-R, 1-np.multiply(R,1-S)))
        self.ps_mu = np.divide(np.multiply(R,S), 1-np.multiply(R,1-S))

        gla_la = np.multiply(-prowsunique[:,0]+np.dot(ps_la, vrows)-np.diag(ps_la), vrows)
        gla_mu = np.multiply(-prowsunique[:,1]+np.dot(ps_mu, vrows)-np.diag(ps_mu), vrows)

        self.gla = np.append(gla_la, gla_mu)
        self.errors = np.append(self.errors, np.linalg.norm(self.gla))
###################################################################################################################################################################
    def explambda(self, i, j): #This is indeed explambdaR
        R = math.exp(self.la[self.jrows[i]]/2)*math.exp(self.la[self.jrows[j]]/2) 
        return R
###################################################################################################################################################################
    def explambdaS(self, i, j):
        S = math.exp(self.mu[self.jrows[i]]/2)*math.exp(self.mu[self.jrows[j]]/2)
        return S
###################################################################################################################################################################
    def returnExpectation(self, R, S):
        E = R*S/ ((1-R)*(1-R*(1-S)))
        return E
###################################################################################################################################################################
	def getExpectation(self, i, j, **kwargs):
		kwargs['isSimple'] = False
		R = self.getPOS(i, j, **kwargs)
		S = self.explambdaS(i, j)
		E = self.returnExpectation(R, S)
		return E
###################################################################################################################################################################
    def updateDistribution(self, pat, idx): #lprevUpdate = list() Each item is a tuple (a, b); a = lambda; b = listofnodes()
        numNodes = pat.number_of_nodes()
        numEdges = pat.G.number_of_edges()
        nodes = sorted(list(pat.G.nodes()))

        mSmallestLambda = np.min(self.la)
        mLargestLambda = np.max(self.la)

        epsilon = 1e-7

        if math.fabs(mSmallestLambda) > math.fabs(mLargestLambda):
            a = epsilon
            b = 4*math.fabs(mSmallestLambda)
        else:
            a = epsilon
            b = 4*math.fabs(mLargestLambda)

        expLambdaR = [None]*numNodes
        expLambdaS = [None]*numNodes
        for i in range(numNodes):
            expLambdaR[i] = [0.0]*numNodes
            expLambdaS[i] = [0.0]*numNodes

        for i in range(numNodes):
            for j in range(i+1, numNodes):
                expLambdaR[i][j] = self.explambdaIncLprev(nodes[i], nodes[j])
                expLambdaS[i][j] = self.explambdaS(nodes[i], nodes[j])
                if math.fabs(b) > math.fabs(math.log(expLambdaR[i][j])):
                    b = math.fabs(math.log(expLambdaR[i][j]))

        b = b - epsilon

        while b-a > 1e-11:
            f_a = 0.0
            f_b = 0.0
            f_c = 0.0
            c = (a+b)/2

            for i in range(numNodes):
                for j in range(i+1, numNodes):
                    try:
                        v_aR=expLambdaR[i][j]*math.exp(a)
                        v_bR=expLambdaR[i][j]*math.exp(b)
                        v_cR=expLambdaR[i][j]*math.exp(c)
                        f_a+=self.returnExpectation(v_aR, expLambdaS[i][j])
                        f_b+=self.returnExpectation(v_bR, expLambdaS[i][j])
                        f_c+=self.returnExpectation(v_cR, expLambdaS[i][j])
                    except OverflowError as error:
                        print(error,a,b)


            f_a=f_a-numEdges
            f_b=f_b-numEdges
            f_c=f_c-numEdges

            print('f_a:', f_a, '\t at a:', a)
            print('f_c:', f_c, '\t at c:', c)
            print('f_b:', f_b, '\t at b:', b,'\n')

            if f_c < 0:
                a = c
            else:
                b = c

        lambdac = round((a + b) / 2, 10)
        self.lprevUpdate[count] = tuple([lambdac, nodes])

        f_c = 0.0
        for i in range(numNodes):
            for j in range(i+1, numNodes):
                v_cR=expLambdaR[i][j]*math.exp(lambdac)
                f_c+=self.returnExpectation(v_cR, expLambdaS[i][j])

        f_c = f_c-numEdges
        # print('Final lamdba: ',lambdac, f_c, numEdges)
        return
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################


class  MaxEntMulti2D(PDClass):
    def __init__(self, G = None):
        super().__init__(G)
        self.tp = 'D'
        self.la_r = None
        self.la_c = None
        self.mu_r = None
        self.mu_c = None
        self.jrows = None
        self.jcols = None
        self.errors = None
        self.ps_la = None
        self.ps_mu = None
        self.gla = None
        self.lprevUpdate = {}
        self.indegrees = None
        self.outdegrees = None
        self.predcount = None
        self.succount = None
        self.inpred = None
        self.outsucc = None
        if G is not None:
            self.findMaxEntDistribution()
###################################################################################################################################################################
	def findMaxEntDistribution(self):
		self.indegrees = np.array(list(dict(sorted(dict(self.G.in_degree()).items())).values()))
		self.outdegrees = np.array(list(dict(sorted(dict(self.G.out_degree()).items())).values()))

		fac = math.log(nx.density(self.G)/(1+nx.density(self.G)))

		self.predcount = []
		for i in range(self.G.number_of_nodes()):
			self.predcount.append(len(list(self.G.predecessors(i))))
		self.predcount = np.array(self.predcount)

		self.succount = []
		for i in range(self.G.number_of_nodes()):
			self.succount.append(len(list(self.G.successors(i))))
		self.succount = np.array(self.succount)

		self.inpred = []
		for i in range(len(self.indegrees)):
			self.inpred.append(tuple([self.indegrees[i], self.predcount[i]]))
		self.inpred = np.array(self.inpred)

		self.outsucc = []
		for i in range(len(self.outdegrees)):
			self.outsucc.append(tuple([self.outdegrees[i], self.succount[i]]))
		self.outsucc = np.array(self.outsucc)

		n = len(self.indegrees)
		m = len(self.outdegrees)

		prows = self.outsucc
		prowsunique, irows, self.jrows, vrows = np.unique(prows, axis=0, return_index=True, return_inverse=True, return_counts=True)
		rownunique = len(prowsunique)
		self.la_r = -math.fabs(fac)*np.ones(rownunique)
		self.mu_r = np.zeros(rownunique)
		rowh = np.zeros(rownunique)

		pcols = self.inpred
		pcolsunique, icols, self.jcols, vcols = np.unique(pcols, axis=0, return_index=True, return_inverse=True, return_counts=True)
		colnunique = len(pcolsunique)
		self.la_c = -math.fabs(fac)*np.ones(colnunique)
		self.mu_c = np.zeros(colnunique)
		colh = np.zeros(colnunique)

		loops = np.outer(np.zeros(rownunique), np.zeros(colnunique).T)

		for i in range(rownunique):
			for j in range(colnunique):
				loops[i][j] = len(set(np.where(self.jrows==i)[0]).intersection(set(np.where(self.jcols==j)[0])))

		finalmat = np.outer(vrows, vcols) - loops

		nit = 1000
		tol = 1e-14
		self.errors = np.empty(0)

		lb = -5
		for k in range(nit):
			R = np.multiply(np.outer(np.exp(self.la_r/2), np.ones(colnunique).T),np.outer(np.ones(rownunique).T, np.exp(self.la_c/2)))
			S = np.multiply(np.outer(np.exp(self.mu_r/2), np.ones(colnunique).T),np.outer(np.ones(rownunique).T, np.exp(self.mu_c/2)))

			self.ps_la = np.divide(np.multiply(R,S), np.multiply(1-R, 1-np.multiply(R,1-S)))
			self.ps_mu = np.divide(np.multiply(R,S), 1-np.multiply(R,1-S))

			gla_t_la = np.multiply(self.ps_la, finalmat)

			gla_r_la = np.sum(gla_t_la, 1) - np.multiply(prowsunique[:,0], vrows)
			gla_c_la = np.sum(gla_t_la, 0) - np.multiply(pcolsunique[:,0], vcols)

			gla_t_mu = np.multiply(self.ps_mu, finalmat)

			gla_r_mu = np.sum(gla_t_mu, 1) - np.multiply(prowsunique[:,1], vrows)
			gla_c_mu = np.sum(gla_t_mu, 0) - np.multiply(pcolsunique[:,1], vcols)

			self.gla = np.append(np.append(gla_r_la, gla_c_la), np.append(gla_r_mu, gla_c_mu))

			self.errors = np.append(self.errors, np.linalg.norm(self.gla))

			H1_u = np.divide(np.multiply(np.multiply(R,S), 1 - np.multiply(np.square(R), 1-S)), np.square(np.multiply(1-R, 1-np.multiply(R,1-S))))
			H2_u = np.divide(np.multiply(np.multiply(R,S), 1 - R), np.square(1-np.multiply(R,1-S)))
			H3_u = np.divide(np.multiply(R,S), np.square(1-np.multiply(R,1-S)))
			H1_t = np.multiply(H1_u, finalmat)
			H2_t = np.multiply(H2_u, finalmat)
			H3_t = np.multiply(H3_u, finalmat)

			H1 = np.diag(np.sum(H1_t, 1))
			H2 = np.diag(np.sum(H1_t, 0))
			H3 = np.diag(np.sum(H2_t, 1))
			H4 = np.diag(np.sum(H2_t, 0))

			H5 = H1_t
			H6 = np.diag(np.sum(H3_u, 1))
			H7 = H3_u
			H8 = H7.T
			H9 = np.diag(np.sum(H3_u, 0))
			H10 = H2_u

			R1 = np.append(np.append(H1, H5, 1), np.append(H6, H7, 1), 1) 
			R2 = np.append(np.append(H5.T, H2, 1), np.append(H8, H9, 1), 1)
			R3 = np.append(np.append(H6.T, H8.T, 1), np.append(H3, H10, 1), 1)
			R4 = np.append(np.append(H7.T, H9.T, 1), np.append(H10.T, H4, 1), 1)

			H = np.append(np.append(R1, R2, 0), np.append(R3, R4, 0), 0)

			delta = np.linalg.lstsq(- H, self.gla, rcond=max(H.shape)*np.finfo(H.dtype).eps)[0]
			deltala_r = delta[0:rownunique]
			deltala_c = delta[rownunique:rownunique+colnunique]
			deltamu_r = delta[rownunique+colnunique:2*rownunique+colnunique]
			deltamu_c = delta[2*rownunique+colnunique:2*rownunique+2*colnunique]


			fbest = 0;
			errorbest = self.errors[k];

			for f in np.logspace(lb,1,20):
				la_rtry=self.la_r+f*deltala_r
				la_ctry=self.la_c+f*deltala_c
				mu_rtry=self.mu_r+f*deltamu_r
				mu_ctry=self.mu_c+f*deltamu_c

				flag = True
				for ind1 in range(len(la_rtry)):
					for ind2 in range(len(la_ctry)):
						if la_rtry[ind1]+la_ctry[ind2]>-1e-15 and finalmat[ind1][ind2]>0.0001:
							flag = False

				if flag:
					Rtry = np.multiply(np.outer(np.exp(la_rtry/2), np.ones(colnunique).T),np.outer(np.ones(rownunique).T, np.exp(la_ctry/2)))
					Stry = np.multiply(np.outer(np.exp(mu_rtry/2), np.ones(colnunique).T),np.outer(np.ones(rownunique).T, np.exp(mu_ctry/2)))

					ps_latry = np.divide(np.multiply(Rtry,Stry), np.multiply(1-Rtry, 1-np.multiply(Rtry,1-Stry)))
					ps_mutry = np.divide(np.multiply(Rtry,Stry), 1-np.multiply(Rtry,1-Stry))

					gla_t_latry = np.multiply(ps_latry, finalmat)

					gla_r_latry = np.sum(gla_t_latry, 1) - np.multiply(prowsunique[:,0], vrows)
					gla_c_latry = np.sum(gla_t_latry, 0) - np.multiply(pcolsunique[:,0], vcols)

					gla_t_mutry = np.multiply(ps_mutry, finalmat)

					gla_r_mutry = np.sum(gla_t_mutry, 1) - np.multiply(prowsunique[:,1], vrows)
					gla_c_mutry = np.sum(gla_t_mutry, 0) - np.multiply(pcolsunique[:,1], vcols)

					glatry = np.append(np.append(gla_r_latry, gla_c_latry), np.append(gla_r_mutry, gla_c_mutry))

					errortry = np.linalg.norm(glatry)

					if errortry < errorbest:
						fbest = f
						errorbest = errortry
			if fbest == 0:
				if lb>-1000:
					lb = lb*2
				else:
					break

			self.la_r = self.la_r+fbest*deltala_r;
			self.la_c = self.la_c+fbest*deltala_c;
			self.mu_r = self.mu_r+fbest*deltamu_r;
			self.mu_c = self.mu_c+fbest*deltamu_c;

			if self.errors[k]/n < tol:
				break

		R = np.multiply(np.outer(np.exp(self.la_r/2), np.ones(colnunique).T),np.outer(np.ones(rownunique).T, np.exp(self.la_c/2)))
		S = np.multiply(np.outer(np.exp(self.mu_r/2), np.ones(colnunique).T),np.outer(np.ones(rownunique).T, np.exp(self.mu_c/2)))

		self.ps_la = np.divide(np.multiply(R,S), np.multiply(1-R, 1-np.multiply(R,1-S)))
		self.ps_mu = np.divide(np.multiply(R,S), 1-np.multiply(R,1-S))

		gla_t_la = np.multiply(self.ps_la, finalmat)

		gla_r_la = np.sum(gla_t_la, 1) - np.multiply(prowsunique[:,0], vrows)
		gla_c_la = np.sum(gla_t_la, 0) - np.multiply(pcolsunique[:,0], vcols)

		gla_t_mu = np.multiply(self.ps_mu, finalmat)

		gla_r_mu = np.sum(gla_t_mu, 1) - np.multiply(prowsunique[:,1], vrows)
		gla_c_mu = np.sum(gla_t_mu, 0) - np.multiply(pcolsunique[:,1], vcols)

		self.gla = np.append(np.append(gla_r_la, gla_c_la), np.append(gla_r_mu, gla_c_mu))

		self.errors = np.append(self.errors, np.linalg.norm(self.gla))
###################################################################################################################################################################
	def explambda(self, i, j): #This is indeed explambdaR
		if i==j:
			return 0
		R = math.exp(self.la_r[self.jrows[i]]/2)*math.exp(self.la_c[self.jcols[j]]/2)
		return R
###################################################################################################################################################################
	def explambdaIncLprevS(self, i, j):
		if i==j:
			return 0
		S = math.exp(self.mu_r[self.jrows[i]]/2)*math.exp(self.mu_c[self.jcols[j]]/2)
		return S
###################################################################################################################################################################
    def returnExpectation(self, R, S):
        E = R*S/ ((1-R)*(1-R*(1-S)))
        return E
###################################################################################################################################################################
	def getExpectation(self, i, j, **kwargs):
		kwargs['isSimple'] = False
		R = self.getPOS(i, j, **kwargs)
		S = self.explambdaS(i, j)
		E = self.returnExpectation(R, S)
		return E
###################################################################################################################################################################
    def updateBackground(self, pat, idx): #lprevUpdate = list() Each item is a tuple (a, b); a = lambda; b = listofnodes()
		mSmallestLambda = np.min(np.array(list(set(self.la_r).union(set(self.la_c)))))
		mLargestLambda = np.max(np.array(list(set(self.la_r).union(set(self.la_c)))))

		epsilon = 1e-7

		if math.fabs(mSmallestLambda) > math.fabs(mLargestLambda):
			a = epsilon
			b = 4*math.fabs(mSmallestLambda)
		else:
			a = epsilon
			b = 4*math.fabs(mLargestLambda)


		inL = dict(pat.in_degree())
		outL = dict(pat.out_degree())

		inNL = []
		outNL = []
		for k,v in inL.items():
			if v!=0:
				inNL.append(k)
		for k,v in outL.items():
			if v!=0:
				outNL.append(k)

		numInNodes = len(inNL)
		numOutNodes = len(outNL)
		numEdges = pat.number_of_edges()

		expLambdaR = [None]*numOutNodes
		expLambdaS = [None]*numOutNodes
		for i in range(numOutNodes):
			expLambdaR[i] = [0.0]*numInNodes
			expLambdaS[i] = [0.0]*numInNodes

		for i in range(numOutNodes):
			for j in range(numInNodes):
				expLambdaS[i][j] = self.explambdaS(outNL[i], inNL[j])
				expLambdaR[i][j] = self.explambdaIncLprev(outNL[i], inNL[j])
				if outNL[i]!=inNL[j]:
					if expLambdaR[i][j]>0.0 and math.fabs(b) > math.fabs(math.log(expLambdaR[i][j])):
						b = math.fabs(math.log(expLambdaR[i][j]))
				else:
					expLambdaR[i][j] = 0

		b = b - epsilon

		while b-a > 1e-15:
			f_a = 0.0
			f_b = 0.0
			f_c = 0.0
			c = (a+b)/2

			for i in range(numOutNodes):
				for j in range(numInNodes):
					try:
						v_aR=expLambdaR[i][j]*math.exp(a)
						v_bR=expLambdaR[i][j]*math.exp(b)
						v_cR=expLambdaR[i][j]*math.exp(c)
						f_a+=self.returnExpectation(v_aR, expLambdaS[i][j])
						f_b+=self.returnExpectation(v_bR, expLambdaS[i][j])
						f_c+=self.returnExpectation(v_cR, expLambdaS[i][j])
					except OverflowError as error:
						print(error,a,b)


			f_a=f_a-numEdges
			f_b=f_b-numEdges
			f_c=f_c-numEdges

			print('f_a:', f_a, '\t at a:', a)
			print('f_c:', f_c, '\t at c:', c)
			print('f_b:', f_b, '\t at b:', b,'\n')

			if f_c < 0:
				a = c
			else:
				b = c

		lambdac = round((a + b) / 2, 10)
		self.lprevUpdate[count] = tuple([lambdac, inNL, outNL])

		f_c = 0.0
		for i in range(numOutNodes):
			for j in range(numInNodes):
				v_cR=expLambdaR[i][j]*math.exp(lambdac)
				f_c+=self.returnExpectation(v_cR, expLambdaS[i][j])

		f_c = f_c-numEdges
		# print('Final lamdba: ',lambdac, f_c, numEdges)
		return
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################