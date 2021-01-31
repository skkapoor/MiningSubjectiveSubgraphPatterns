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

class HillClimber:
    def __init__(self, G, q, seedMode, nKseed, mode = 1, gtype = 'U', isSimple=True, incEdge = False):
        self.G = G
        self.q = q
        self.seedMode = seedMode
        self.nKseed = nKseed
        self.icmode = mode
        self.gtype = gtype
        self.isSimple = isSimple
        self.incEdge = incEdge
        self.Data = dict()
        self.seeds = list()
        self.allSeedDet = None
        self.first = True

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

    def evaluateNew(self, PD):
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
            self.allSeedDet = getAllInterestBasedSeeds(self.G, PD, self.q, self.seedMode, self.nKseed, self.icmode, self.gtype, self.isSimple, self.incEdge)
            topKseeds = list(sorted(self.allSeedDet.items(), key = lambda kv: kv[1][0], reverse=True))[:min(len(self.allSeedDet), self.nKseed)]
            print('topKseeds', topKseeds)
            self.seeds = []
            for ts in topKseeds:
                self.seeds.append(ts[0])
            st2 = ft1 = time.time()
            self.Data = runGivenSeeds(self.G, PD, self.q, self.seeds, self.icmode, self.gtype, self.isSimple, self.incEdge)
            ft2 = time.time()
        else:
            self.seeds = getSeeds(self.G, PD, self.q, self.seedMode, self.nKseed, self.icmode, self.gtype, self.isSimple, self.incEdge)
            self.Data = runGivenSeeds(self.G, PD, self.q, self.seeds, self.icmode, self.gtype, self.isSimple, self.incEdge)
        # print('IN EA seeds: ', self.seeds)
        # print(self.seeds)
        # print(self.Data)
        print('################Time in Evaluate new for finding Top-k interest based seeds: {}'.format(ft1-st1)+'\n################Time in Evaluate new for running hill climber for Top-k interest based seeds: {}'.format(ft2-st2))
        return 

    def evaluateSecond(self, PD, PrevPat):
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
            inSecSeedDet = evaluateSetSeedsInterest(self.G, PD, inSecSeed, self.q, self.icmode, self.gtype, self.isSimple, self.incEdge)
            self.allSeedDet.update(inSecSeedDet)
            topKseeds = list(sorted(self.allSeedDet.items(), key = lambda kv: kv[1][0], reverse=True))[:min(len(self.allSeedDet), self.nKseed)]
            ft1 = time.time()
            print('topKseeds', topKseeds)
            self.seeds = []
            for ts in topKseeds:
                self.seeds.append(ts[0])
            st2 = ft1 = time.time()
            self.Data = runGivenSeeds(self.G, PD, self.q, self.seeds, self.icmode, self.gtype, self.isSimple, self.incEdge)
            ft2 = time.time()
        else:
            print('Comming here')
            self.seeds = getSeeds(self.G, PD, self.q, self.seedMode, self.nKseed, self.icmode, self.gtype, self.isSimple, self.incEdge)
            self.Data = runGivenSeeds(self.G, PD, self.q, self.seeds, self.icmode, self.gtype, self.isSimple, self.incEdge)
        print('################Time in Evaluate Second for finding Top-k interest based seeds: {}'.format(ft1-st1)+'\n################Time in Evaluate new for running hill climber for Top-k interest based seeds: {}'.format(ft2-st2))
        # print('IN EA seeds: ', self.seeds)
        return

    def getBestOption(self, PD, PrevPat = None):
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
        if self.first:
            self.evaluateNew(PD)
            self.first = False
        else:
            self.evaluateSecond(PD, PrevPat)

        bestPattern = max(self.Data, key=lambda x: x.I)
        return bestPattern