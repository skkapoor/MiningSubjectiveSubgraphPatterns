###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################
import os
import sys
path = os.getcwd().split('MiningSubjectiveSubgraphPatterns')[0]+'MiningSubjectiveSubgraphPatterns/'
if path not in sys.path:
	sys.path.append(path)

from src.Utils.Measures import getCodeLength, getCodeLengthParallel, getDirectedSubgraph
from src.Utils.Measures import computeDescriptionLength, computeInterestingness
from src.Patterns.Pattern import Pattern
###################################################################################################################################################################
class EvaluateRemove:
    """
    This data structure shall contain all the possible removal candidates
    along with pattern number as key and other information as value
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
        print('Initialized EvaluateRemove')
###################################################################################################################################################################
    def evaluateAllConstraints(self, G, PD): #arguments would be the distribution and output would all possibilities or no output
        """
        function to evaluate all constraint and make a list of candidate constraints which are feasible to remove

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
        function to evaluate if a constraint is a feasible candidate for remove

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
            Params = dict()
            NL = PD.lprevUpdate[id][1]
            H = G.subgraph(NL)
            Params['Pat'] = Pattern(H)
            Params['codeLengthC'] = getCodeLengthParallel(Params['Pat'].G, PD, NL=Params['Pat'].NL, case=2, isSimple=self.isSimple, gtype=self.gtype) #now case is 1 as none of teh lambdas shall be removed
            Params['codeLengthCprime'] = getCodeLengthParallel(G, PD, NL=NL, case=4, dropLidx=[id], isSimple=self.isSimple, gtype=self.gtype)  #now case is 4 as one lambda is to be dropped to compute new codelength
            Params['Pat'].setIC_dssg( Params['codeLengthC'] - Params['codeLengthCprime'] )
            Params['Pat'].setDL( computeDescriptionLength(dlmode=4, gtype=self.gtype, C=len(PD.lprevUpdate), l=self.l) )
            Params['Pat'].setI( computeInterestingness( Params['Pat'].IC_dssg, Params['Pat'].DL, mode=self.imode) )

            if Params['Pat'].I > 0:
                Params['Pat'].setPrevOrder(id)
                Params['Pat'].setPatType('remove')
                Params['Pat'].setLambda(PD.lprevUpdate[id][0])
                self.Data[id] = Params
        else:
            Params = dict()
            inNL = PD.lprevUpdate[id][1]
            outNL = PD.lprevUpdate[id][2]
            HD = getDirectedSubgraph(G, inNL, outNL, self.isSimple)
            Params['Pat'] = Pattern(HD)
            Params['codeLengthC'] = getCodeLengthParallel(G, PD, inNL=inNL, outNL=outNL, case=1, isSimple=self.isSimple, gtype=self.gtype) #now case is 1 as none of teh lambdas shall be removed
            Params['codeLengthCprime'] = getCodeLengthParallel(G, PD, inNL=inNL, outNL=outNL, case=4, dropLidx=[id], isSimple=self.isSimple, gtype=self.gtype)  #now case is 4 as one lambda is to be dropped to compute new codelength
            Params['Pat'].setIC_dssg( Params['codeLengthC'] - Params['codeLengthCprime'] )
            Params['Pat'].setDL( computeDescriptionLength(dlmode=4, gtype=self.gtype, C=len(PD.lprevUpdate), l=self.l, excActionType=False) )
            Params['Pat'].setI( computeInterestingness( Params['Pat'].IC_dssg, Params['Pat'].DL, mode=self.imode) )

            if Params['Pat'].I > 0:
                Params['Pat'].setPrevOrder(id)
                Params['Pat'].setPatType('remove')
                Params['Pat'].setLambda(PD.lprevUpdate[id][0])
                self.Data[id] = Params
###################################################################################################################################################################
    def updateConstraintEvaluation(self, G, PD, id, condition = 1):
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

        *Things to note P shall have the prev_order(identifier of the constraint) correct
        """
        if condition == 1: #update only description length
            DL = computeDescriptionLength(dlmode=4, gtype=self.gtype, C=len(PD.lprevUpdate), l=self.l, excActionType=False)
            IG = computeInterestingness(self.Data[id]['Pat'].IC_dssg, DL, mode=2)
            self.Data[id]['Pat'].setDL(DL)
            self.Data[id]['Pat'].setI(IG)
        elif condition == 2: #update codelength and description length
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
        if prevPat.pat_type is not 'remove':
            if prevPat.pat_type in ['merge']:
                for p in prevPat.prev_order:
                    if p in self.Data:
                        del self.Data[p]
            elif prevPat.pat_type in ['shrink', 'split', 'update']:
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
        function to return the best candidate to remove

        Returns
        -------
        dict
            dictionary containing a Pattern, and the two corresponding codelength associated to the pattern, i.e., prior and posterior to performing remove action.
        """
        if len(self.Data) < 1:
            return None
        else:
            bestR = max(self.Data.items(), key=lambda x: x[1]['Pat'].I)
            return bestR[1]
###################################################################################################################################################################
    def updateDistribution(self, PD, bestR):
        """
        function to update background distribution. Now here we remove the knowledge of any such pattern, hence we remove the lambda associated with the pattern.

        Parameters
        ----------
        PD : PDClass
            Background distribution
        bestR : dict
            last remove action details
        """
        del self.Data[bestR['Pat'].prev_order]
        out = PD.lprevUpdate.pop(bestR['Pat'].prev_order, None)
        if out is None:
            print("Something is fishy")
        return
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################