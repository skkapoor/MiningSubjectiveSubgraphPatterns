###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################
import os
import sys
path = os.getcwd().split('MiningSubjectiveSubgraphPatterns')[0]+'MiningSubjectiveSubgraphPatterns/'
if path not in sys.path:
	sys.path.append(path)
import ray
import time

from src.Utils.Measures import getCodeLength, getCodeLengthParallel, getDirectedSubgraph
from src.Utils.Measures import computeDescriptionLength, computeInterestingness
from src.Patterns.Pattern import Pattern
from src.HillClimbers.HC_v4 import runGivenSeeds, getSeeds, getAllInterestBasedSeeds, evaluateSetSeedsInterest

@ray.remote
def checkIntersectionU(seedDet, seedL, NL):
    IL = set()
    for s in seedL:
        if len(set(seedDet[s][1]).intersection(set(NL))) > 1:
            IL.add(s)
    return IL

@ray.remote
def checkIntersectionD(seedDet, seedL, inNL, outNL):
    IL = set()
    for s in seedL:
        if len(set(seedDet[s][1]).intersection(set(inNL))) > 0 and len(set(seedDet[s][2]).intersection(set(outNL))):
            IL.add(s)
    return IL
###################################################################################################################################################################
class EvaluateAdd:
    """
    This data structure shall contain all the possible addition candidates
    along with pattern number as key and other information as value
    """
    def __init__(self, gtype='U', isSimple=True, l=6, ic_mode=1, imode=2, minsize=2, seedType='interest', seedRuns=10, q=0.01, incEdges=False):
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
        icmode : int, optional
            Mode for information content--- 1: IC_ssg, 2: AD (aggregate deviation), 3: IC_dsimp, by default 1
        imode : int, optional
            Interestingness mode--- 1: fraction, 2: Difference, by default 2
        minsize : int, optional
            Minimum size of pattern, by default 2
        seedType : str, optional
            Type of seed run for the hill climber, it can be "all", "uniform", "degree" or "interest", by default "interest"
        seedRuns : int, optional
            Minimum size of pattern, by default 2
        q : float, optional
            Expected size of pattern, given as a factor of original size of the input graph, ranges 0.0-1.0, by default 0.01
        incEdges : bool, optional
            True in edges to be encoded for description length else false, by default false
        """
        self.Data = dict()
        self.seeds = list()
        self.gtype = gtype
        self.isSimple = isSimple
        self.l = l # possible types (give number) of action, default is 6
        self.imode = imode
        self.icmode = ic_mode
        self.minsize = minsize
        self.q = q
        self.seedMode = seedType
        self.seedRuns = seedRuns
        self.incEdges = incEdges
        self.allSeedDet = None
        print('Initialized EvaluateAdd')

    def getInteresectionSeeds(self, PrevPat):
        SL = set()
        keysL = list(self.allSeedDet.keys())
        seedDet = ray.put(self.allSeedDet)
        ln = int(len(keysL)/os.cpu_count())+1
        if self.gtype=='U':
            P_NL = ray.put(PrevPat.NL)
            Res = [checkIntersectionU.remote(seedDet, keysL[i:min(i+ln, len(keysL))], P_NL) for i in range(0, len(keysL), ln)]
        else:
            P_inNL = ray.put(PrevPat.inNL)
            P_outNL = ray.put(PrevPat.outNL)
            Res = [checkIntersectionD.remote(seedDet, keysL[i:min(i+ln, len(keysL))], P_inNL, P_outNL) for i in range(0, len(keysL), ln)]
        res = ray.get(Res)
        for r in res:
            SL = SL.union(r)
        return SL
###################################################################################################################################################################
    def evaluateNew(self, G, PD):
        """
        function to evaluate all independent seed runs

        Parameters
        ----------
        G : networkx graph
            input graph
        PD : PDClass
            Input background distribution
        """
        ft1 = st1 = ft2 = st2 = 0.0
        if 'interest' in self.seedMode:
            st1 = time.time()
            self.allSeedDet = getAllInterestBasedSeeds(G, PD, self.q, self.seedMode, self.seedRuns, self.icmode, self.gtype, self.isSimple, self.incEdges)
            topKseeds = list(sorted(self.allSeedDet.items(), key = lambda kv: kv[1][0], reverse=True))[:min(len(self.allSeedDet), self.seedRuns)]
            print('topKseeds', topKseeds)
            self.seeds = []
            for ts in topKseeds:
                self.seeds.append(ts[0])
            st2 = ft1 = time.time()
            self.Data = runGivenSeeds(G, PD, self.q, self.seeds, self.icmode, self.gtype, self.isSimple, self.incEdges)
            ft2 = time.time()
        else:
            self.seeds = getSeeds(G, PD, self.q, self.seedMode, self.seedRuns, self.icmode, self.gtype, self.isSimple, self.incEdges)
            self.Data = runGivenSeeds(G, PD, self.q, self.seeds, self.icmode, self.gtype, self.isSimple, self.incEdges)
        # print('IN EA seeds: ', self.seeds)
        # print(self.seeds)
        # print(self.Data)
        print('################Time in Evaluate new for finding Top-k interest based seeds: {}'.format(ft1-st1)+'\n################Time in Evaluate new for running hill climber for Top-k interest based seeds: {}'.format(ft2-st2))
        return

    def evaluateSecond(self, G, PD, PrevPat):
        """
        function to evaluate all independent seed runs without checking Interest value of independent seed

        Parameters
        ----------
        G : networkx graph
            input graph
        PD : PDClass
            Input background distribution
        """
        ft1 = st1 = ft2 = st2 = 0.0
        if 'interest' in self.seedMode:
            st1 = time.time()
            inSecSeed = self.getInteresectionSeeds(PrevPat)
            inSecSeedDet = evaluateSetSeedsInterest(G, PD, inSecSeed, self.q, self.icmode, self.gtype, self.isSimple, self.incEdges)
            self.allSeedDet.update(inSecSeedDet)
            nseeds = sorted(self.allSeedDet, key = lambda kv: self.allSeedDet[kv][0], reverse=True)[:min(len(self.allSeedDet), self.seedRuns)]
            ft1 = time.time()
            if len(set(self.seeds).intersection(set(nseeds))) < len(self.seeds):
                self.seeds = nseeds[:]
                st2 = time.time()
                self.Data = runGivenSeeds(G, PD, self.q, self.seeds, self.icmode, self.gtype, self.isSimple, self.incEdges)
                ft2 = time.time()
        else:
            print('Comming here')
            self.seeds = getSeeds(G, PD, self.q, self.seedMode, self.seedRuns, self.icmode, self.gtype, self.isSimple, self.incEdges)
            self.Data = runGivenSeeds(G, PD, self.q, self.seeds, self.icmode, self.gtype, self.isSimple, self.incEdges)
        print('################Time in Evaluate Second for finding Top-k interest based seeds: {}'.format(ft1-st1)+'\n################Time in Evaluate new for running hill climber for Top-k interest based seeds: {}'.format(ft2-st2))
        # print('IN EA seeds: ', self.seeds)
        return
###################################################################################################################################################################
    def checkAndUpdateAllPossibilities(self, G, PD, PrevPat):
        """
        function to update the candidate list

        Parameters
        ----------
        G : networkx graph
            input graph
        PD : PDClass
            Input background distribution
        PrevPat : src.Patterns.Pattern
            resultant pattern of last performed action
        """
        #First if the last action is add then we are required to find a new pattern again, that is, running a hill climber for top-k seeds fresh.
        if len(self.Data) < 1:
            self.evaluateSecond(G, PD, PrevPat)
        else:
            #second if there is an overlap of nodes affected (to specific a node-pair) then we find a new pattern and run the hill climber for top-k seeds fresh.
            bestPattern = max(self.Data, key=lambda x: x.I)
            if self.gtype == 'U':
                if len(set(bestPattern.NL).intersection(set(PrevPat.NL))) > 1:
                    self.evaluateSecond(G, PD, PrevPat)
            else:
                inInt = len(set(bestPattern.inNL).intersection(set(PrevPat.inNL)))
                outInt = len(set(bestPattern.outNL).intersection(set(PrevPat.outNL)))
                if inInt > 1 and outInt > 1:
                    self.evaluateSecond(G, PD, PrevPat)
        return
###################################################################################################################################################################
    def getBestOption(self, G, PD):
        """
        function to return the best candidate to add

        Parameters
        ----------
        G : networkx graph
            input graph
        PD : PDClass
            Input background distribution

        Returns
        -------
        dict
            dictionary containing a Pattern to add and correspoding prior and posterior codelengths
        """
        if len(self.Data) > 0:
            bestPattern = max(self.Data, key=lambda x: x.I)
            codeLengthC = None
            codeLengthCprime = None
            DL = None
            dlmode = 3
            if self.gtype == 'U':
                nlambda = PD.updateDistribution(bestPattern.G, None, 'return', 2, None)
                codeLengthC = getCodeLengthParallel(G, PD, NL=bestPattern.NL, case=2, gtype=self.gtype, isSimple=self.isSimple)
                codeLengthCprime = getCodeLengthParallel(G, PD, NL=bestPattern.NL, case=3, gtype=self.gtype, isSimple=self.isSimple, nlambda=nlambda)
                DL = computeDescriptionLength(dlmode=dlmode, V=G.number_of_nodes(), W=bestPattern.NCount, kw=bestPattern.ECount, q=self.q, isSimple=self.isSimple, kws=bestPattern.kws, excActionType=False, l=self.l)
            else:
                nlambda = PD.updateDistribution(bestPattern.G, None, 'return', 2, None)
                codeLengthC = getCodeLengthParallel(G, PD, NL=bestPattern.NL, case=2, gtype=self.gtype, isSimple=self.isSimple)
                codeLengthCprime = getCodeLengthParallel(G, PD, inNL=bestPattern.inNL, outNL=bestPattern.outNL, case=3, gtype=self.gtype, isSimple=self.isSimple, nlambda=nlambda)
                DL = computeDescriptionLength(dlmode=dlmode, V=G.number_of_nodes(), WI=bestPattern.inNL, WO=bestPattern.outNL, kw=bestPattern.ECount, q=self.q, isSimple=self.isSimple, kws=bestPattern.kws, excActionType=False, l=self.l)
            IC_dssg = codeLengthC - codeLengthCprime
            bestPattern.setIC_dssg(IC_dssg)
            bestPattern.setDL(DL)
            bestPattern.setI( computeInterestingness(bestPattern.IC_dssg, bestPattern.DL, mode=self.imode) )
            bestPattern.setPatType('add')
            Params = dict()
            Params['Pat'] = bestPattern
            Params['codeLengthC'] = codeLengthC
            Params['codeLengthCprime'] = codeLengthCprime
            return Params
        else:
            return None
###################################################################################################################################################################
    def updateDistribution(self, PD, bestA):
        """
        function to update background distribution.
        * Now here we add a new lambda for the added pattern.

        Parameters
        ----------
        PD : PDClass
            Background distribution
        bestA : dict
            last added action details
        """
        self.Data = []
        la = PD.updateDistribution( bestA['Pat'].G, idx=bestA['Pat'].cur_order, val_return='save', case=2 )
        bestA['Pat'].setLambda(la)
        return
###################################################################################################################################################################
    def printCands(self):
        """
        function to print all current candidates
        """
        if len(self.Data):
            for k in range(len(self.Data)):
                print('\t\t', k, self.Data[k])
                print('\n')
        return
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################