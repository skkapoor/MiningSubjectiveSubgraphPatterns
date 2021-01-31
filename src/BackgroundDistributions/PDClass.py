###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################
import numpy as np
import networkx as nx
import math
##################################################################################################################################################################
class PDClass:
    """
    This is the abstract class for background distribution
    """
    def __init__(self, G):
        """
        initialization function

        Parameters
        ----------
        G : networkx graph
            input graph
        """
        self.G = G.copy()
        self.density = nx.density(self.G)
        self.la = None
        self.mu = None
        self.ps = None
        self.lprevUpdate = dict()
        if G.is_directed():
            self.tp = 'D'
        else:
            self.tp = 'U'
##################################################################################################################################################################
    def findDistribution(self):
        """
        abstract function to compute the background distribution
        """
        return
##################################################################################################################################################################
    def getExpectationFromPOS(self, a):
        """
        abstract function to compute expectation from probability of success. This function shall be override depending on the requirement.

        Parameters
        ----------
        a : float
            probability of success

        Returns
        -------
        float
            computed expectation
        """
        return a/(1+a)
##################################################################################################################################################################
    def getAB(self):
        """
        abstract function to compute A and B parameters required to update the distribution.
        """
        return
##################################################################################################################################################################
    def updateDistribution(self, pat, idx=None, val_return='save', case=2, dropLidx=None):
        """
        function to update the background distribution

        Parameters
        ----------
        pat : nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph
            an input networkx graph pattern
        idx : int, optional
            identifier to be used for the new constraint/pattern
        case : int, optional
            case==1 if original lambda are used and new lambdas for new pattern are not used
            case==2, if all lambdas are used
            case==3, if all lambdas are used but some specific lambdas are dropped, by default 2
        val_return : str, optional
            use 'save' to update the BD and save the new lambda, else use 'return' to return the new lambda with saving, by default 'save'
        dropLidx : list, optional
            list of lambdas' identifier to be dropped

        Returns
        -------
        double
            corresponding Lagrangian multiplier
        """
        a,b = self.getAB()

        numNodes = None
        numEdges = None
        nodes = None

        inNL = None
        outNL = None
        numInNodes = None
        numOutNodes = None

        if pat.is_directed():
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
        else:
            numNodes = pat.number_of_nodes()
            numEdges = pat.number_of_edges()
            nodes = sorted(list(pat.nodes()))

        expLambda = None
        if case == 1:
            if pat.is_directed():
                expLambda = [None]*numOutNodes
                for i in range(numOutNodes):
                    expLambda[i] = [0.0]*numInNodes
                for i in range(numOutNodes):
                    for j in range(numInNodes):
                        expLambda[i][j] = self.explambda(outNL[i], inNL[j])
            else:
                expLambda = [None]*numNodes
                for i in range(numNodes):
                    expLambda[i] = [0.0]*numNodes
                for i in range(numNodes):
                    for j in range(i+1, numNodes):
                        expLambda[i][j] = self.explambda(nodes[i], nodes[j])

        elif case == 2:
            if pat.is_directed():
                expLambda = [None]*numOutNodes
                for i in range(numOutNodes):
                    expLambda[i] = [0.0]*numInNodes
                for i in range(numOutNodes):
                    for j in range(numInNodes):
                        expLambda[i][j] = self.explambdaIncLprev(outNL[i], inNL[j])
            else:
                expLambda = [None]*numNodes
                for i in range(numNodes):
                    expLambda[i] = [0.0]*numNodes
                for i in range(numNodes):
                    for j in range(i+1, numNodes):
                        expLambda[i][j] = self.explambdaIncLprev(nodes[i], nodes[j])

        elif case == 3:
            if pat.is_directed():
                expLambda = [None]*numOutNodes
                for i in range(numOutNodes):
                    expLambda[i] = [0.0]*numInNodes
                for i in range(numOutNodes):
                    for j in range(numInNodes):
                        expLambda[i][j] = self.explambdaIncLprevButDropSomeLas(outNL[i], inNL[j], dropLidx)
            else:
                expLambda = [None]*numNodes
                for i in range(numNodes):
                    expLambda[i] = [0.0]*numNodes
                for i in range(numNodes):
                    for j in range(i+1, numNodes):
                        expLambda[i][j] = self.explambdaIncLprevButDropSomeLas(nodes[i], nodes[j], dropLidx)

        # if pat.is_multigraph() and pat.is_directed():
        #     for i in range(numOutNodes):
        #         for j in range(numInNodes):
        #             if expLambda[i][j]!=0 and math.fabs(b) > math.fabs(math.log(expLambda[i][j])):
        #                 b = math.fabs(math.log(expLambda[i][j])) - 1e-13
        # elif pat.is_multigraph():
        #     for i in range(numNodes):
        #         for j in range(i+1, numNodes):
        #             if math.fabs(b) > math.fabs(math.log(expLambda[i][j])):
        #                 b = math.fabs(math.log(expLambda[i][j])) - 1e-13

        while b-a > 1e-11:
            f_a = 0.0
            f_b = 0.0
            f_c = 0.0
            c = round((a + b) / 2, 12)

            if pat.is_directed():
                for i in range(numOutNodes):
                    for j in range(numInNodes):
                        try:
                            if i!=j:
                                v_a=expLambda[i][j]*math.exp(a)
                                v_b=expLambda[i][j]*math.exp(b)
                                v_c=expLambda[i][j]*math.exp(c)
                                f_a+=self.getExpectationFromExpLambda(v_a)
                                f_b+=self.getExpectationFromExpLambda(v_b)
                                f_c+=self.getExpectationFromExpLambda(v_c)
                        except OverflowError as error:
                            print(error,a,b)
            else:
                for i in range(numNodes):
                    for j in range(i+1, numNodes):
                        try:
                            v_a=expLambda[i][j]*math.exp(a)
                            v_b=expLambda[i][j]*math.exp(b)
                            v_c=expLambda[i][j]*math.exp(c)
                            f_a+=self.getExpectationFromExpLambda(v_a)
                            f_b+=self.getExpectationFromExpLambda(v_b)
                            f_c+=self.getExpectationFromExpLambda(v_c)
                        except OverflowError as error:
                            print(error,a,b)

            # print('idx:',idx,'f_c: ',f_c, 'a: ', a, 'b: ',b,'f_a: ',f_a,'f_b: ',f_b, 'numEdges: ', numEdges)

            f_a=f_a-numEdges
            f_b=f_b-numEdges
            f_c=f_c-numEdges

            if f_c < 0:
                a = c
            else:
                b = c
        lambdac = round((a + b) / 2, 10)
        if 'save' in val_return:
            if pat.is_directed():
                self.lprevUpdate[idx] = tuple([lambdac, inNL, outNL, numEdges])
            else:
                self.lprevUpdate[idx] = tuple([lambdac, nodes, numEdges])

        return lambdac
##################################################################################################################################################################
    def explambda(self, i, j):
        """
        abstract function to compute exponent to the power to some quatity.

        Parameters
        ----------
        i : float
            input parameter 1
        j : float
            input parameter 2
        """
        return
##################################################################################################################################################################
    def explambdaMultiplier(self, i, j):
        """
        function to compute extra multipliers if the background has been updated a number of times.

        Parameters
        ----------
        i : float
            input parameter 1
        j : float
            input parameter 2

        Returns
        -------
        float
            computed multiplier
        """
        if self.tp == 'U':
            r = 1.0
            for k,v in self.lprevUpdate.items():
                if i in v[1] and j in v[1]:
                    r *= math.exp(v[0])
        elif self.tp == 'D':
            r = 1.0
            for k,v in self.lprevUpdate.items():
                if i in v[2] and j in v[1]:
                    r *= math.exp(v[0])
        else:
            r = 1.0
            print('Invalid Graph Type in explambdaMultiplier() function')
        return r
##################################################################################################################################################################
    def explambdaIncLprev(self, i, j):
        """
        function to compute exponent to the power to some quatity included all the multipliers which are result of updation of BD.

        Parameters
        ----------
        i : float
            input parameter 1
        j : float
            input parameter 2

        Returns
        -------
        float
            value
        """
        expL = self.explambda(i, j)
        expL *= self.explambdaMultiplier(i, j)
        return expL
##################################################################################################################################################################
    def explambdaIncLprevButDropSomeLas(self, i, j, dropLidx):
        """
        function to compute exponent to the power to some quatity included all the multipliers which are result of updation of BD but dropping some of the Lagrangian multipliers.

        Parameters
        ----------
        i : float
            input parameter 1
        j : float
            input parameter 2
        dropLidx : list
            identifiers of Lagrangian multipliers to be dropped

        Returns
        -------
        float
            value
        """
        expL = self.explambda(i, j)
        r = self.explambdaMultiplier(i, j)
        if self.tp == 'U':
            for k in dropLidx:
                v = self.lprevUpdate[k]
                if i in v[1] and j in v[1]:
                    r /= math.exp(v[0])
        elif self.tp == 'D':
            for k in dropLidx:
                v = self.lprevUpdate[k]
                if i in v[2] and j in v[1]:
                    r /= math.exp(v[0])
        else:
            r = 1.0
            print('Invalid Graph Type in explambdaIncLprevButDropList() function')
        expL *= r
        return expL
###################################################################################################################################################################
    def getPOS(self, i, j, **kwargs):
        """
        function to compute probability of success.

        Parameters
        ----------
        i : float
            input parameter 1
        j : float
            input parameter 2
        dropLidx : list
            identifiers of Lagrangian multipliers to be dropped

        Returns
        -------
        float
            value
        """
        if i==j:
            return 0.0
        case=2
        dropLidx=None
        nlambda=0.0
        isSimple=True
        if 'case' in kwargs:
            case = kwargs['case']
        if 'dropLidx' in kwargs:
            dropLidx = kwargs['dropLidx']
        if 'nlambda' in kwargs:
            nlambda = kwargs['nlambda']
        if 'isSimple' in kwargs:
            isSimple = kwargs['isSimple']
        expL = 0.0
        if case==1:
            #Original Exp Lambda
            expL = self.explambda(i, j)
        elif case==2 or case==3:
            #Including new added lambdas
            expL = self.explambdaIncLprev(i, j)
        elif case==4 or case==5:
            # Excluding some (a list of) Lambdas
            expL = self.explambdaIncLprevButDropSomeLas(i, j, dropLidx)

        if case==3 or case==5:
            #Adding one new Lambda
            expL *= math.exp(nlambda)

        pos = 1.0 - expL
        if isSimple:
            pos = expL/(1.0+expL)
        # if pos < 1e-7:
        #     raise Exception('pos less than epsilon', i, j)
        # if pos > 1.0:
        #     raise Exception('pos more than one', i, j)
        return pos
###################################################################################################################################################################
    def getExpectationFromExpLambda(self, a):
        """
        abstract function to compute expectation from exponent to the power of some quantity.

        Parameters
        ----------
        a : float
            input value

        Returns
        -------
        float
            value
        """
        return a/(1+a)
###################################################################################################################################################################
    def getExpectationFromPOS(self, a):
        """
        abstract function to compute expectation from probability of success.

        Parameters
        ----------
        a : float
            input value

        Returns
        -------
        float
            value
        """
        return a
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################