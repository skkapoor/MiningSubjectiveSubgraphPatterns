###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################
import os
import sys
path = os.getcwd().split('MiningSubjectiveSubgraphPatterns')[0]+'MiningSubjectiveSubgraphPatterns/'
if path not in sys.path:
    sys.path.append(path)
import networkx as nx

from src.Utils.Measures import getCodeLength, getCodeLengthParallel, getDirectedSubgraph
from src.Utils.Measures import computeDescriptionLength, computeInterestingness
from src.Patterns.Pattern import Pattern
###################################################################################################################################################################
class EvaluateUpdate:
    """
    This data structure shall contain all the possible updates
    along with pattern number as key and other parameters as value
    """
    def __init__(self, gtype='U', isSimple=True, l=6, imode=2):
        """
        initialization function

        Parameters
        ----------
        gtype : str, optional
            Input Graph type, 'U': Undirected, 'D': Directed, by default 'U'
        isSimple : bool, optional
            if input graph is a simple graph then True else False if it is a multigraph, by default True
        l : int, optional
            Total number of unique action types that can be performed, by default 6
        imode : int, optional
            Interestingness mode--- 1: fraction, 2: Difference, by default 2
        """
        self.Data = dict()
        self.gtype = gtype
        self.isSimple = isSimple
        self.l = l # possible types (give number) of action, default is 6
        self.imode = imode
        print('initialized EvaluateUpdate')
###################################################################################################################################################################
    def evaluateAllConstraints(self, G, PD):
        """
        function to evaluate all constraints and make a list of candidate constraints which are feasible to update

        Parameters
        ----------
        G : Networkx Graph
            Input Graph
        PD : PDClass
            Background distribution
        """
        self.Data = dict()
        for i in PD.lprevUpdate.keys():
            self.evaluateConstraint(G, PD, i)
        return
###################################################################################################################################################################
    def evaluateConstraint(self, G, PD, id):
        """
        function to evaluate if a constraint is a feasible candidate for update

        Parameters
        ----------
        G : Networkx Graph
            Input Graph
        PD : PDClass
            Background Distribution
        id : int
            identifier of a constraint to be evaluated
        """
        if self.gtype == 'U':
            NL = PD.lprevUpdate[id][1]
            H = G.subgraph(NL)
            if nx.number_connected_components(H) == 1:
                Params = dict()
                Params['Pat'] = Pattern(H)
                nlambda = PD.updateDistribution( Params['Pat'].G, idx=None, val_return='return', case=3, dropLidx=[id] ) #// ToDo: Add code to coumpute new lambda
                Params['codeLengthC'] = getCodeLengthParallel( Params['Pat'].G, PD, NL=Params['Pat'].NL, case=2, isSimple=self.isSimple, gtype=self.gtype ) #now case is 1 as none of teh lambdas shall be removed
                Params['codeLengthCprime'] = getCodeLengthParallel( Params['Pat'].G, PD, NL=Params['Pat'].NL, case=5, dropLidx=[id], nlambda=nlambda, isSimple=self.isSimple, gtype=self.gtype )  #now case is 4 as one lambda is to be dropped to compute new codelength
                Params['Pat'].setIC_dssg( Params['codeLengthC'] - Params['codeLengthCprime'] )
                Params['Pat'].setDL( computeDescriptionLength( dlmode=5, gtype=self.gtype, C=len(PD.lprevUpdate), W=Params['Pat'].NCount, l=self.l, excActionType=False, kw=Params['Pat'].ECount, kws=Params['Pat'].kws, isSimple=self.isSimple ) )
                Params['Pat'].setI( computeInterestingness( Params['Pat'].IC_dssg, Params['Pat'].DL, mode=self.imode) )

                if Params['Pat'].I > 0:# in original paper we updated only those patterns for which density increases, but here we do not
                # if Params['Pat'].I > 0 and Params['Pat'].ECount > PD.lprevUpdate[id][2]: # uncomment this line and comment able line for original paper version
                    Params['Pat'].setPrevOrder(id)
                    Params['Pat'].setPatType('update')
                    Params['Pat'].setLambda(nlambda)
                    self.Data[id] = Params
        else:
            inNL = PD.lprevUpdate[id][1]
            outNL = PD.lprevUpdate[id][2]
            HD = getDirectedSubgraph( G, inNL, outNL, self.isSimple )
            H = nx.Graph(HD)
            if nx.number_connected_components(H) == 1:
                Params = dict()
                Params['Pat'] = Pattern(HD)
                nlambda = PD.updateDistribution( Params['Pat'].G, idx=None, val_return='return', case=3, dropLidx=[id] ) #// ToDo: Add code to coumpute new lambda
                Params['codeLengthC'] = getCodeLengthParallel(Params['Pat'].G, PD, inNL=Params['Pat'].inNL, outNL=Params['Pat'].outNL, case=1, isSimple=self.isSimple, gtype=self.gtype) #now case is 1 as none of teh lambdas shall be removed
                Params['codeLengthCprime'] = getCodeLengthParallel(Params['Pat'].G, PD, inNL=Params['Pat'].inNL, outNL=Params['Pat'].outNL, case=5, dropLidx=[id], nlambda=nlambda, isSimple=self.isSimple, gtype=self.gtype)  #now case is 4 as one lambda is to be dropped to compute new codelength
                Params['Pat'].setIC_dssg( Params['codeLengthC'] - Params['codeLengthCprime'] )
                Params['Pat'].setDL( computeDescriptionLength( dlmode=5, gtype=self.gtype, C=len(PD.lprevUpdate), WI=Params['Pat'].inNL, WO=Params['Pat'].outNL, l=self.l, excActionType=False, kw=Params['Pat'].ECount, kws=Params['Pat'].kws, isSimple=self.isSimple ) )
                Params['Pat'].setI( computeInterestingness( Params['Pat'].IC_dssg, Params['Pat'].DL, mode=self.imode) )

                if Params['Pat'].I > 0: # in original paper we updated only those patterns for which density increases, but here we do not
                # if Params['Pat'].I > 0 and Params['Pat'].ECount > PD.lprevUpdate[id][3]: # uncomment this line and comment able line for original paper version
                    Params['Pat'].setPrevOrder(id)
                    Params['Pat'].setPatType('update')
                    Params['Pat'].setLambda(nlambda)
                    self.Data[id] = Params
        return
###################################################################################################################################################################
    def updateConstraintEvaluation(self, G, PD, id, condition=1):
        """
        function to now evaluate and update a possible candidate

        Parameters
        ----------
        G : Networkx Graph
            Input Graph
        PD : PDClass
            Background Distribution
        id : int
            identifier of candidate to the updated
        condition : int, optional
            1 if codelength does not changes, else 2, by default 1
        """
        if condition == 1:
            if self.gtype == 'U':
                DL = computeDescriptionLength( dlmode=5, gtype=self.gtype, C=len(PD.lprevUpdate), W=self.Data[id]['Pat'].NCount, l=self.l, excActionType=False, kw=self.Data[id]['Pat'].ECount, kws=self.Data[id]['Pat'].kws, isSimple=self.isSimple )
                IG = computeInterestingness( self.Data[id]['Pat'].IC_dssg, DL, mode=2 )
                self.Data[id]['Pat'].setDL(DL)
                self.Data[id]['Pat'].setI(IG)
            else:
                DL = computeDescriptionLength( dlmode=5, gtype=self.gtype, C=len(PD.lprevUpdate), WI=self.Data[id]['Pat'].inNL, WO=self.Data[id]['Pat'].outNL, l=self.l, excActionType=False, kw=self.Data[id]['Pat'].ECount, kws=self.Data[id]['Pat'].kws, isSimple=self.isSimple )
                IG = computeInterestingness( self.Data[id]['Pat'].IC_dssg, DL, mode=2 )
                self.Data[id]['Pat'].setDL(DL)
                self.Data[id]['Pat'].setI(IG)
        elif condition == 2:
            self.evaluateConstraint(G, PD, id)
        return
###################################################################################################################################################################
    def checkAndUpdateAllPossibilities(self, G, PD, prevPat):
        """
        function to update the parameters associated to each possible candidates

        Parameters
        ----------
        G : Networkx Graph
            Input Graph
        PD : PDClass
            Background distribution
        prevPat : Pattern
            Pattern  corresponding to the previously performed action. Note that this pattern shall contains the set of nodes that are involved in previous action, 
            both as prior and posterior
        """
        ### removing candidate if any other action was performed on it ###
        if 'update' not in prevPat.pat_type:
            if prevPat.pat_type in ['merge']:
                for p in prevPat.prev_order:
                    if p in self.Data:
                        del self.Data[p]
            elif prevPat.pat_type in ['shrink', 'split', 'remove']:
                if prevPat.prev_order in self.Data:
                    del self.Data[prevPat.prev_order]
        if self.gtype == 'U':
            for k,v in self.Data.items():
                if len(set(v['Pat'].NL).intersection(set(prevPat.NL))) > 1:
                    self.updateConstraintEvaluation(G, PD, k, 2)
                else:
                    self.updateConstraintEvaluation(G, PD, k, 1)
        else:
            for k,v in self.Data.items():
                inInt = len(set(v['Pat'].inNL).intersection(set(prevPat.inNL)))
                outInt = len(set(v['Pat'].outNL).intersection(set(prevPat.outNL)))
                if inInt > 1 and outInt > 1:
                    self.updateConstraintEvaluation(G, PD, k, 2)
                else:
                    self.updateConstraintEvaluation(G, PD, k, 1)
        return
###################################################################################################################################################################
    def getBestOption(self):
        """
        function to return the best candidate to update

        Returns
        -------
        dict
            dictionary containing a Pattern, and the two corresponding codelength associated to the pattern, i.e., prior and posterior to performing update action.
        """
        if len(self.Data) < 1:
            return None
        else:
            bestR = max(self.Data.items(), key=lambda x: x[1]['Pat'].I)
            return bestR[1]
###################################################################################################################################################################
    def updateDistribution(self, PD, bestU):
        """
        function to update background distribution.
        * Now here we remove the knowledge of pervious pattern which is now updated and add the knowledge of pattern which is the result of update
        * hence we remove the previous lambda associated with the pattern and add a new lambda for updated pattern

        Parameters
        ----------
        PD : PDClass
            Background distribution
        bestM : dcit
            last update action details
        """
        del self.Data[bestU['Pat'].prev_order]
        out = PD.lprevUpdate.pop(bestU['Pat'].prev_order, None)
        if not out:
            print('Something is fishy')
        else:
            PD.updateDistribution( bestU['Pat'].G, idx=bestU['Pat'].cur_order, val_return='save', case=2 )
        return
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################