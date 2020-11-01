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
class EvaluateMerge:
    """
    This data structure shall contain all the possible mergers
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
        self.curAdds = list()
        print('initialized EvaluateMerge')
###################################################################################################################################################################
    def evaluateAllConstraintPairs(self, G, PD):
        """
        function to evaluate all constraint pairs and make a list of candidate constraint pairs which are feasible to merge

        Parameters
        ----------
        G : Networkx Graph
            Input Graph
        PD : PDClass
            Background distribution
        """
        keys = list(PD.lprevUpdate.keys())
        for i1 in range(len(keys)-1):
            for i2 in range(i1+1, len(keys)):
                self.evaluateConstraintPair( G, PD, keys[i1], keys[i2] )
        return
###################################################################################################################################################################
    def evaluateConstraintPair(self, G, PD, k1, k2):
        """
        function to evaluate if a constraint pair is a feasible candidate for merge

        Parameters
        ----------
        G : Networkx Graph
            Input Graph
        PD : PDClass
            Background Distribution
        k1 : int
            identifier of first constraint
        k2 : int
            identifier of second constraint
        """
        if self.gtype == 'U':
            NL1 = PD.lprevUpdate[k1][1]
            NL2 = PD.lprevUpdate[k2][1]
            NL_F = list(set(NL1).union(set(NL2)))
            H = G.subgraph(NL_F)
            if nx.is_connected(H):
                P = Pattern(H)
                self.computeParametersU( P, PD, min(k1, k2), max(k1, k2) )
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
                P = Pattern(H)
                self.computeParametersD( P, PD, min(k1, k2), max(k1, k2) )
        return
###################################################################################################################################################################
    def computeParametersU(self, P, PD, k1, k2):
        """
        Utility function to compute paramters for a potential candidate constraint pair when Input graph is undirected

        Parameters
        ----------
        P : Pattern
            Input patter by merging two constraints
        PD : PDClass
            Background Distribution
        k1 : int
            identifier of first constraint
        k2 : int
            identifier of second constraint
        """
        Params = dict()
        Params['Pat'] = P
        nlambda = PD.updateDistribution( Params['Pat'].G, idx=None, val_return='return', case=3, dropLidx=[k1, k2] ) #// TODO: handle this issue, code it !!!!!!!!
        Params['codeLengthC'] = getCodeLengthParallel( Params['Pat'].G, PD, gtype=self.gtype, case=2, NL=Params['Pat'].NL, isSimple=self.isSimple )
        Params['codeLengthCprime'] = getCodeLengthParallel( Params['Pat'].G, PD, gtype=self.gtype, case=5, NL=Params['Pat'].NL, isSimple=self.isSimple, dropLidx=[k1, k2], nlambda=nlambda )
        Params['Pat'].setIC_dssg( Params['codeLengthC'] - Params['codeLengthCprime'] )
        Params['Pat'].setDL( computeDescriptionLength( dlmode=8, excActionType=False, l=6, gtype=self.gtype, W=Params['Pat'].NCount, kw=Params['Pat'].ECount, C=len(PD.lprevUpdate), kws=Params['Pat'].kws, isSimple=self.isSimple ) )
        Params['Pat'].setI( computeInterestingness( Params['Pat'].IC_dssg, Params['Pat'].DL, mode=self.imode) )
        if Params['Pat'].I > 0:
            Params['Pat'].setPrevOrder((int(k1),int(k2)))
            Params['Pat'].setPatType('merge')
            Params['Pat'].setLambda(nlambda)
            if int(k1) in self.curAdds and int(k2) in self.curAdds:
                raise Exception('ADD ADD MERGE EVALUATE HUA')
            self.Data[(k1,k2)] = Params
        return
###################################################################################################################################################################
    def computeParametersD(self, P, PD, k1, k2):
        """
        Utility function to compute paramters for a potential candidate constraint pair when Input graph is directed

        Parameters
        ----------
        P : Pattern
            Input patter by merging two constraints
        PD : PDClass
            Background Distribution
        k1 : int
            identifier of first constraint
        k2 : int
            identifier of second constraint
        """
        Params = dict()
        Params['Pat'] = P
        nlambda = PD.updateDistribution( Params['Pat'].G, idx=None, val_return='return', case=3, dropLidx=[k1, k2] ) #// TODO: handle this issue, code it !!!!!!!!
        Params['codeLengthC'] = getCodeLengthParallel( Params['Pat'].G, PD, gtype=self.gtype, case=2, inNL=Params['Pat'].inNL, outNL=Params['Pat'].outNL, isSimple=self.isSimple )
        Params['codeLengthCprime'] = getCodeLengthParallel( Params['Pat'].G, PD, gtype=self.gtype, case=5, inNL=Params['Pat'].inNL, outNL=Params['Pat'].outNL, isSimple=self.isSimple, dropLidx=[k1, k2], nlambda=nlambda )
        Params['Pat'].setIC_dssg( Params['codeLengthC'] - Params['codeLengthCprime'] )
        Params['Pat'].setDL( computeDescriptionLength( dlmode=8, excActionType=False, l=6, gtype=self.gtype, WI=Params['Pat'].inNL, WO=Params['Pat'].outNL, kw=Params['Pat'].ECount, C=len(PD.lprevUpdate), kws=Params['Pat'].kws, isSimple=self.isSimple ) )
        Params['Pat'].setI( computeInterestingness( Params['Pat'].IC_dssg, Params['Pat'].DL, mode=self.imode) )
        if Params['Pat'].I > 0:
            Params['Pat'].setPrevOrder((int(k1),int(k2)))
            Params['Pat'].setPatType('merge')
            Params['Pat'].setLambda(nlambda)
            self.Data[(k1,k2)] = Params
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
                DL = computeDescriptionLength( dlmode=8, excActionType=False, l=self.l, gtype=self.gtype, W=self.Data[id]['Pat'].NCount, kw=self.Data[id]['Pat'].ECount, C=len(PD.lprevUpdate), kws=self.Data[id]['Pat'].kws, isSimple=self.isSimple )
                IG = computeInterestingness( self.Data[id]['Pat'].IC_dssg, DL, mode=self.imode )
                self.Data[id]['Pat'].setDL(DL)
                self.Data[id]['Pat'].setI(IG)
            else:
                DL = computeDescriptionLength( dlmode=8, excActionType=False, l=6, gtype=self.gtype, WI=self.Data[id]['Pat'].inNL, WO=self.Data[id]['Pat'].outNL, kw=self.Data[id]['Pat'].ECount, C=len(PD.lprevUpdate), kws=self.Data[id]['Pat'].kws, isSimple=self.isSimple )
                IG = computeInterestingness( self.Data[id]['Pat'].IC_dssg, DL, mode=self.imode )
                self.Data[id]['Pat'].setDL(DL)
                self.Data[id]['Pat'].setI(IG)
        elif condition == 2:
            self.evaluateConstraintPair( G, PD, id[0], id[1] )
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
            Pattern corresponding to the previously performed action. Note that this pattern shall contains the set of nodes that are involved in previous action,
            both as prior and posterior
        """
        keyUpdate = []
        if self.gtype == 'U':
            for k,v in self.Data.items():
                if len(set(v['Pat'].NL).intersection(set(prevPat.NL))) > 1:
                    keyUpdate.append(k)
                else:
                    self.updateConstraintEvaluation(G, PD, k, 1)
        else:
            for k,v in self.Data.items():
                inInt = len(set(v['Pat'].inNL).intersection(set(prevPat.inNL)))
                outInt = len(set(v['Pat'].outNL).intersection(set(prevPat.outNL)))
                if inInt > 1 and outInt > 1:
                    keyUpdate.append(k)
                else:
                    self.updateConstraintEvaluation(G, PD, k, 1)
        for k in keyUpdate:
            del self.Data[k]
            self.updateConstraintEvaluation(G, PD, k, 2)
        return
###################################################################################################################################################################
    def doProcessWithNewConstraint(self, G, PD, prevPat):
        """
        This function append into the list of candidate merges if a new candidate merge is feasible after performing an action

        Parameters
        ----------
        G : Networkx Graph
            Input Graph
        PD : PDClass
            Background distribution
        prevPat : Pattern, list of Patterns
            Pattern corresponding to the previously performed action. Note that this pattern shall contains the final set of nodes that are result of previous action

        * Adding new candidates mainly depends on the type of the last performed action, if it is a add, merge, shrink and update then ofcourse there can be new candidates
        * After remove obviously there are no possiblities
        * However if there is a split action then the situation changes and for each new pattern we check if a new candidate merge is possible or not
        """
        if isinstance(prevPat, list): #if last action is split then prevPat shall be a list of patterns
            pkeys = list()
            for p in prevPat:
                pkeys.append(p.cur_order)
            keys = list(set(PD.lprevUpdate.keys()) - set(pkeys))
            for pk in pkeys:
                for k in keys:
                    self.evaluateConstraintPair( G, PD, min(k, pk), max(k, pk) )
        else:
            if 'remove' in prevPat.pat_type:
                return
            else:
                if 'add' in prevPat.pat_type:
                    self.curAdds.append(prevPat.cur_order)
                keys = list(set(PD.lprevUpdate.keys()) - set([prevPat.cur_order]))
                for k in keys:
                    if not (k in self.curAdds and prevPat.cur_order in self.curAdds):
                        self.evaluateConstraintPair( G, PD, min(k, prevPat.cur_order), max(k, prevPat.cur_order) )
        return
###################################################################################################################################################################
    def removeCandidates(self, prevPat):
        """
        This function remove the iineligible candidates after an action is performed.
        * Now ineligible candidates are those for which one part has been now updated/removed/merged/shrinked/splited

        Parameters
        ----------
        prevPat : Pattern
            Pattern which is the outcome of last perfromed action
        """
        keys = self.Data.keys()
        pkeys = None
        if isinstance(prevPat.prev_order, tuple) or isinstance(prevPat.prev_order, list) or isinstance(prevPat.prev_order, set):
            pkeys = set(prevPat.prev_order)
        else:
            pkeys = set([prevPat.prev_order])
        delkeys = []
        for k in keys:
            if len(set(k).intersection(pkeys)) == 1:
                delkeys.append(k)
        for k in delkeys:
            del self.Data[k]
        return
###################################################################################################################################################################
    def updateDistribution(self, PD, bestM):
        """
        function to update background distribution.
        * Now here we remove the knowledge of pervious two patterns which are merged and add the knowledge of pattern which is the result of merger
        * hence we remove the lambdas associated with the two pattern and add a new lambda for merged pattern

        Parameters
        ----------
        PD : PDClass
            Background distribution
        bestM : dict
            last merge action details
        """
        del self.Data[tuple([bestM['Pat'].prev_order[0], bestM['Pat'].prev_order[1]])]
        out1 = PD.lprevUpdate.pop(int(bestM['Pat'].prev_order[0]), None)
        out2 = PD.lprevUpdate.pop(int(bestM['Pat'].prev_order[1]), None)
        if not out1 or not out2:
            print('Something is fishy')
        else:
            la = PD.updateDistribution( bestM['Pat'].G, idx=bestM['Pat'].cur_order, val_return='save', case=2 )
            bestM['Pat'].setLambda(la)
        return
###################################################################################################################################################################
    def getBestOption(self):
        """
        function to return the best candidate to merge

        Returns
        -------
        dict
            dictionary containing a Pattern, and the two corresponding codelength associated to the pattern, i.e., prior and posterior to performing merge action.
        """
        if len(self.Data) < 1:
            return None
        else:
            bestM = max(self.Data.items(), key=lambda x: x[1]['Pat'].I)
            if bestM[1]['Pat'].prev_order[int(0)] in self.curAdds and bestM[1]['Pat'].prev_order[int(1)] in self.curAdds:
                raise Exception('ADD ADD MERGE RETURN HUA')
            return bestM[1]
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################