###### This file contains the implementation of hill climber which is parallalised using Ray framework #########
##### Following points are running in parallel:
##### 1. Seed selection
##### 2. Seed Runs
import os
import sys
path = os.getcwd().split('MiningSubjectiveSubgraphPatterns')[0]+'MiningSubjectiveSubgraphPatterns/'
if path not in sys.path:
	sys.path.append(path)
import math
import numpy as np
import copy
import networkx as nx

from src.Patterns.Pattern import Pattern
from src.Utils.Measures import computeSumOfEdgeProbablity, IC_SSG, NW, NW_D
from src.Utils.Measures import computeDescriptionLength, computeInterestingness
from src.Utils.Measures import computeSumOfEdgeProbablityBetweenNodeAndList
from src.Utils.Measures import computeSumOfExpectations, computeSumOfExpectationsBetweenNodeAndList
from src.Utils.Measures import computePWparameters, computePWparametersBetweenNodeAndList, computeMinPOSBetweenNodeAndList
from src.Utils.Measures import AD, IC_DSIMP, getDirectedSubgraph

# import psutil
import ray
import time
##################################################################################################################################################################
@ray.remote(num_return_vals=2)
def computeInterestParallel(LNit, Gid, PDid, mode, gtype, isSimple, incEdge, q):
    # G = Gid
    # PD = ray.get(PDid)
    curlist = list(set(Gid.neighbors(LNit)).union(set([LNit])))
    H = Gid.subgraph(curlist)
    if len(curlist)>1:
        ic = 0.0
        dl = 0.0
        dlmode = 1
        if incEdge:
            dlmode = 2
        if mode == 1:
            pw = computeSumOfEdgeProbablity(PDid, gtype=gtype, NL=curlist, isSimple=isSimple)
            ic = IC_SSG(3, pw=pw, W=H)
            dl = computeDescriptionLength(dlmode=dlmode, V=Gid.number_of_nodes(), W=H.number_of_nodes(), kw=H.number_of_edges(), q=q)
        elif mode == 2:
            mu_w = computeSumOfExpectations(PDid, gtype=gtype, NL=curlist, isSimple=isSimple)
            ic = AD(H.number_of_edges(), mu_w)
            kws = nx.Graph(H).number_of_edges()
            dl = computeDescriptionLength(dlmode=dlmode, V=Gid.number_of_nodes(), W=H.number_of_nodes(), kw=H.number_of_edges(), q=q, kws=kws)
        elif mode == 3:
            mu_w, p0 = computePWparameters(PDid, gtype=gtype, NL=curlist, isSimple=isSimple)
            ic = IC_DSIMP(H.number_of_edges(), NW(len(curlist)), mu_w, p0)
            kws = nx.Graph(H).number_of_edges()
            dl = computeDescriptionLength(dlmode=dlmode, V=Gid.number_of_nodes(), W=H.number_of_nodes(), kw=H.number_of_edges(), q=q, kws=kws)
        interestValue = computeInterestingness(ic, dl)
    return LNit, interestValue
##################################################################################################################################################################
@ray.remote(num_return_vals=2)
def computeInterestParallelD(LNit, Gid, PDid, mode, gtype, isSimple, incEdge, q):
    # G = Gid
    # PD = ray.get(PDid)
    curlistOut = list(set(Gid.predecessors(LNit)).union(set([LNit])))
    curlistIn = list(set(Gid.successors(LNit)).union(set([LNit])))
    H = getDirectedSubgraph(Gid, curlistIn, curlistOut, isSimple)
    if len(curlistIn)>1 and len(curlistOut)>1:
        ic = 0.0
        dl = 0.0
        dlmode = 1
        if incEdge:
            dlmode = 2
        if mode == 1:
            pw = computeSumOfEdgeProbablity(PDid, gtype=gtype, inNL=curlistIn, outNL=curlistOut, isSimple=isSimple)
            ic = IC_SSG(3, pw=pw, W=H)
            dl = computeDescriptionLength(dlmode=dlmode, V=Gid.number_of_nodes(), WI=curlistIn, WO=curlistOut, kw=H.number_of_edges(), q=q)
        elif mode == 2:
            mu_w = computeSumOfExpectations(PDid, gtype=gtype, inNL=curlistIn, outNL=curlistOut, isSimple=isSimple)
            ic = AD(H.number_of_edges(), mu_w)
            kws = nx.DiGraph(H).number_of_edges()
            dl = computeDescriptionLength(dlmode=dlmode, V=Gid.number_of_nodes(), WI=curlistIn, WO=curlistOut, kw=H.number_of_edges(), q=q, kws=kws)
        elif mode == 3:
            mu_w, p0 = computePWparameters(PDid, gtype=gtype, inNL=curlistIn, outNL=curlistOut, isSimple=isSimple)
            ic = IC_DSIMP(H.number_of_edges(), NW_D(curlistIn, curlistOut), mu_w, p0)
            kws = nx.DiGraph(H).number_of_edges()
            dl = computeDescriptionLength(dlmode=dlmode, V=Gid.number_of_nodes(), WI=curlistIn, WO=curlistOut, kw=H.number_of_edges(), q=q, kws=kws)
        interestValue = computeInterestingness(ic, dl)
    return LNit, interestValue
##################################################################################################################################################################
def getSeeds(G, PD, q, seedMode, nKseed, mode, gtype, isSimple, incEdge):
    """Function to get seeds to run the hill climber

    Raises:
        Exception 1: if nKseed != mNumNodes:
            raise Exception("Number of seeds should be equal to number of nodes here.")
        Exception: raise Exception('no valid seed mode given')

    Returns:
        list: seed node's list
    """
    mNumNodes = G.number_of_nodes()
    seedNodes = [None]*nKseed
    if 'all' in seedMode:
        if nKseed != mNumNodes:
            raise Exception("Number of seeds should be equal to number of nodes here.")

        for r in range(nKseed):
            seedNodes[r] = r

    elif 'uniform' in seedMode:
        randoml = list(G.nodes())
        np.random.shuffle(randoml)
        for r in range(nKseed):
            seedNodes[r] = randoml[r]

    elif 'degree' in seedMode:
        degreeList = sorted(dict(G.degree()).items(), key=lambda kv: kv[1], reverse=True)
        for r in range(nKseed):
            seedNodes[r] = degreeList[r][0]

    elif 'interest' in seedMode:
        ListNode = sorted(list(G.nodes()))
        Gid = ray.put(G)
        PDid = ray.put(PD)
        nterestListids = None
        if gtype == 'U':
            interestListids = [computeInterestParallel.remote(LNit, Gid, PDid, mode, gtype, isSimple, incEdge, q) for LNit in ListNode]
        else:
            interestListids = [computeInterestParallelD.remote(LNit, Gid, PDid, mode, gtype, isSimple, incEdge, q) for LNit in ListNode]
        interestList = []
        for ii in interestListids:
            interestList.append(tuple([ray.get(ii[0]), ray.get(ii[1])]))
        interestList = sorted(interestList, key=lambda kv: kv[1], reverse=True)
        mRange = min([nKseed, len(interestList)])
        seedNodes = [0]*mRange
        for r in range(mRange):
            # print(r, interestList[r][0])
            if interestList[r][0] is None:
                print(r, interestList[r][0])
            seedNodes[r] = interestList[r][0]

    else:
        raise Exception('no valid seed mode given')
    return seedNodes
##################################################################################################################################################################
def extendPatternUtil(pattern, candidates, G, PD, q, mode, gtype, isSimple, incEdge):
    """Util function to check for the best candidate node to add

    Args:
        pattern (Pattern): input subgraph pattern
        candidates (dict): dictionary of candidates that can be added with parameters as values

    Returns:
        int, dict: best node id, dictionary of parameters cmputed for the input node
    """
    best_node = None
    params = None
    ic = 0.0
    dl = 0.0

    dlmode = 1
    if incEdge:
        dlmode = 2

    if mode == 1:
        best_node, params = max(candidates.items(), key=lambda x: computeInterestingness( IC_SSG( 1, pw=pattern.sumPOS+x[1]['pw_surplus'], kw=pattern.ECount+x[1]['kw_surplus'], nw=NW(pattern.NCount+1) ),\
            computeDescriptionLength( dlmode=dlmode, V=G.number_of_nodes(), W=pattern.NCount+1, kw=pattern.ECount+x[1]['kw_surplus'], q=q ) ) )
        params['pw_new'] = pattern.sumPOS + params['pw_surplus']
        params['kw_new'] = pattern.ECount + params['kw_surplus']
        params['nw_new'] = NW(pattern.NCount+1)
        ic = IC_SSG(1, pw=params['pw_new'], kw=params['kw_new'], nw=params['nw_new'])
        dl = computeDescriptionLength(dlmode=dlmode, V=G.number_of_nodes(), W=pattern.NCount+1, kw=params['kw_new'], q=q)
    elif mode == 2:
        best_node, params = max(candidates.items(), key=lambda x: computeInterestingness( AD( pattern.ECount+x[1]['kw_surplus'], pattern.expectedEdges + x[1]['mu_w_surplus'] ),\
            computeDescriptionLength( dlmode=dlmode, V=G.number_of_nodes(), W=pattern.NCount+1, kw=pattern.ECount+x[1]['kw_surplus'], q=q, kws=pattern.kws+x[1]['kws_surplus'], isSimple = isSimple ) ) )
        params['mu_w_new'] = pattern.expectedEdges + params['mu_w_surplus']
        params['kw_new'] = pattern.ECount + params['kw_surplus']
        params['kws_new'] = pattern.ECount + params['kws_surplus']
        ic = AD(params['kw_new'], params['mu_w_new'])
        dl = computeDescriptionLength(dlmode=dlmode, V=G.number_of_nodes(), W=pattern.NCount+1, kw=params['kw_new'], q=q, isSimple=isSimple, kws=params['kws_new'])
    elif mode == 3:
        best_node, params = max(candidates.items(), key=lambda x: computeInterestingness( IC_DSIMP( pattern.ECount+x[1]['kw_surplus'], NW(pattern.NCount+1), pattern.expectedEdges + x[1]['mu_w_surplus'], min(pattern.minPOS,x[1]['p0_surplus']) ),\
            computeDescriptionLength( dlmode=dlmode, V=G.number_of_nodes(), W=pattern.NCount+1, kw=pattern.ECount+x[1]['kw_surplus'], q=q, kws=pattern.kws+x[1]['kws_surplus'], isSimple = isSimple ) ) )
        params['mu_w_new'] = pattern.expectedEdges + params['mu_w_surplus']
        params['p0_new'] = min(pattern.minPOS, params['p0_surplus'])
        params['kw_new'] = pattern.ECount + params['kw_surplus']
        params['kws_new'] = pattern.ECount + params['kws_surplus']
        params['nw_new'] = NW(pattern.NCount+1)
        ic = IC_DSIMP(params['kw_new'], params['nw_new'], params['mu_w_new'], params['p0_new'])
        dl = computeDescriptionLength(dlmode=dlmode, V=G.number_of_nodes(), W=pattern.NCount+1, kw=params['kw_new'], q=q, isSimple=isSimple, kws=params['kws_new'])
    I = computeInterestingness(ic, dl)

    params['ic'] = ic
    params['dl'] = dl
    params['I'] = I

    return best_node, params
##################################################################################################################################################################
def extendPatternFinal(pattern, nodeToAdd, params, G, PD, q, mode, gtype, isSimple, incEdge):
    """function to update the subgraph pattern while adding the node resultng in maximum increase of interestingness value

    Args:
        pattern (Pattern): input subgraph pattern
        nodeToAdd (int): node id of the vertex to be added
        params (dict): corresponding parameters of the node to be added

    Returns:
        Pattern: updated subgraph pattern
    """
    H = nx.Graph()
    if not isSimple:
        H = nx.MultiGraph()
    for p in pattern.NL:
        if G.has_edge(p, nodeToAdd):
            H.add_edges_from([tuple([p, nodeToAdd])]*G.number_of_edges(p, nodeToAdd))
    pattern.updateGraph(H)
    if mode == 1:
        pattern.setSumPOS(params['pw_new'])
        pattern.setIC_ssg(params['ic'])
    elif mode == 2:
        pattern.setAD(params['ic'])
        pattern.setExpectedEdges(params['mu_w_new'])
    elif mode == 3:
        pattern.setIC_dsimp(params['ic'])
        pattern.setExpectedEdges(params['mu_w_new'])
        pattern.setMinPOS(params['p0_new'])
    pattern.setDL(params['dl'])
    pattern.setI(params['I'])
    return pattern
##################################################################################################################################################################
def shrinkPatternUtil(pattern, nodeToCheck, G, PD, q, mode, gtype, isSimple, incEdge):
    """Util function to check for the best candidate node to remove

    Args:
        pattern (Pattern): input subgraph pattern
        nodeToCheck (int): node id of the vertex to check for removal

    Returns:
        dict: dictionary of parameters cmputed for the input node
    """
    kw_deficit  = 0
    kws_deficit = 0
    for p in pattern.NL:
        kw_deficit  += G.number_of_edges(p, nodeToCheck)
        kws_deficit  += (G.number_of_edges(p, nodeToCheck)>0)
    curNL = pattern.NL

    params = dict()
    ic = 0.0
    dl = 0.0
    dlmode = 1
    if incEdge:
        dlmode = 2
    if mode == 1:
        params['pw_deficit'] = computeSumOfEdgeProbablityBetweenNodeAndList(PD, nodeToCheck, curNL, gtype=gtype, isSimple=isSimple)
        params['pw_new'] = pattern.sumPOS - params['pw_deficit']
        params['kw_new'] = pattern.ECount - kw_deficit
        params['nw_new'] = NW(len(curNL)-1)
        ic = IC_SSG(1, pw=params['pw_new'], kw=params['kw_new'], nw=params['nw_new'])
        dl = computeDescriptionLength(dlmode=dlmode, V=G.number_of_nodes(), W=len(curNL)-1, kw=params['kw_new'], q=q)
    elif mode == 2:
        params['mu_w_deficit'] = computeSumOfExpectationsBetweenNodeAndList(PD, nodeToCheck, curNL, gtype=gtype, isSimple=isSimple)
        params['mu_w_new'] = pattern.expectedEdges - params['mu_w_deficit']
        params['kw_new'] = pattern.ECount - kw_deficit
        params['kws_new'] = pattern.kws - kws_deficit
        ic = AD(params['kw_new'], params['mu_w_new'])
        dl = computeDescriptionLength(dlmode=dlmode, V=G.number_of_nodes(), W=len(curNL)-1, kw=params['kw_new'], q=q, isSimple=isSimple, kws=params['kws_new'])
    else:
        params['kw_new'] = pattern.ECount - kw_deficit
        params['kws_new'] = pattern.kws - kws_deficit
        params['mu_w_deficit'], params['p0_deficit'] = computePWparametersBetweenNodeAndList(PD, nodeToCheck, curNL, gtype=gtype, isSimple=isSimple)
        if pattern.minPOS == params['p0_deficit']:
            curNL.remove(nodeToCheck)
            params['mu_w_new'], params['p0_new'] = computePWparameters(PD, gtype=gtype, NL=curNL, isSimple=isSimple)
        else:
            params['p0_new'] = pattern.minPOS
            params['mu_w_new'] = pattern.expectedEdges - params['mu_w_deficit']
        ic = IC_DSIMP(params['kw_new'], params['nw_new'], params['mu_w_new'], params['p0_new'])
        dl = computeDescriptionLength(dlmode=dlmode, V=G.number_of_nodes(), W=len(curNL)-1, kw=params['kw_new'], q=q, isSimple=isSimple, kws=params['kws_new'])
    I = computeInterestingness(ic, dl)
    params['ic'] = ic
    params['dl'] = dl
    params['I'] = I
    return params
##################################################################################################################################################################
def shrinkPatternFinal(pattern, nodeToRemove, params, G, PD, q, mode, gtype, isSimple, incEdge):
    """function to update the subgraph pattern while removing the node resulting in maximum increase of interestingness value

    Args:
        pattern (Pattern): input subgraph pattern
        nodeToAdd (int): node id of the vertex to be removed
        params (dict): corresponding parameters of the node to be removed

    Returns:
        Pattern: updated subgraph pattern
    """
    pattern.removeNode(nodeToRemove)
    if mode == 1:
        pattern.setSumPOS(params['pw_new'])
        pattern.setIC_ssg(params['ic'])
    elif mode == 2:
        pattern.setAD(params['ic'])
        pattern.setExpectedEdges(params['mu_w_new'])
    elif mode == 3:
        pattern.setIC_dsimp(params['ic'])
        pattern.setExpectedEdges(params['mu_w_new'])
        pattern.setMinPOS(params['p0_new'])
    pattern.setDL(params['dl'])
    pattern.setI(params['I'])
    return pattern
##################################################################################################################################################################
def updateCandidateAddition(best_node, bestPattern, candidates, G, PD, q, mode, gtype, isSimple, incEdge):
    neighborsA = list(G.neighbors(best_node))
    if best_node in candidates:
        del candidates[best_node]
    else:
        print('\t\tWarning:', best_node, 'is not in the candidate list')

    #update parameters of previously present candidate list
    if mode == 1:
        for u in candidates.keys():
            candidates[u]['kw_surplus'] += G.number_of_edges(best_node, u)
            candidates[u]['pw_surplus'] += PD.getPOS(best_node, u, isSimple=isSimple)
    elif mode == 2:
        for u in candidates.keys():
            candidates[u]['kw_surplus'] += G.number_of_edges(best_node, u)
            candidates[u]['kws_surplus'] += (G.number_of_edges(best_node, u)>0)
            candidates[u]['mu_w_surplus'] += PD.getExpectation(best_node, u, isSimple=isSimple)
    elif mode == 3:
        for u in candidates.keys():
            candidates[u]['kw_surplus'] += G.number_of_edges(best_node, u)
            candidates[u]['kws_surplus'] += (G.number_of_edges(best_node, u)>0)
            candidates[u]['mu_w_surplus'] += PD.getExpectation(best_node, u, isSimple=isSimple)
            candidates[u]['p0_surplus'] = min(candidates[u]['p0_surplus'], PD.getPOS(best_node, u, isSimple=isSimple))

    pot_add = set(neighborsA) - set(candidates.keys()).union(set(bestPattern.NL))
    #update candidate list by adding new candidates
    if mode == 1:
        for u in pot_add:
            candidates[u] = dict()
            candidates[u]['kw_surplus'] = 0
            candidates[u]['pw_surplus'] = 0.0
            for v in bestPattern.NL:
                candidates[u]['kw_surplus'] += G.number_of_edges(v, u)
                candidates[u]['pw_surplus'] += PD.getPOS(v, u, isSimple=isSimple)
    elif mode == 2:
        for u in pot_add:
            candidates[u] = dict()
            candidates[u]['kw_surplus'] = 0
            candidates[u]['kws_surplus'] = 0
            candidates[u]['mu_w_surplus'] = 0.0
            for v in bestPattern.NL:
                candidates[u]['kw_surplus'] += G.number_of_edges(v, u)
                candidates[u]['kws_surplus'] += (G.number_of_edges(v, u)>0)
                candidates[u]['mu_w_surplus'] += PD.getExpectation(v, u, isSimple=isSimple)
    elif mode == 3:
        for u in pot_add:
            candidates[u] = dict()
            candidates[u]['kw_surplus'] = 0
            candidates[u]['kws_surplus'] = 0
            candidates[u]['mu_w_surplus'] = 0.0
            candidates[u]['p0_surplus'] = float("inf")
            for v in bestPattern.NL:
                candidates[u]['kw_surplus'] += G.number_of_edges(v, u)
                candidates[u]['kws_surplus'] += (G.number_of_edges(v, u)>0)
                candidates[u]['mu_w_surplus'] += PD.getExpectation(v, u, isSimple=isSimple)
                candidates[u]['p0_surplus'] = min(candidates[u]['p0_surplus'], PD.getPOS(v, u, isSimple=isSimple))

    return candidates
##################################################################################################################################################################
def updateCandidateDeletion(best_node, bestPattern, candidates, G, PD, q, mode, gtype, isSimple, incEdge):


    # remove irrelevant keys in candidates
    neighborsR = set(G.neighbors(best_node))
    pNL = set(bestPattern.NL)
    for nR in neighborsR:
        if len(set(G.neighbors(nR)).intersection(pNL)) == 0:
            if nR in candidates:
                del candidates[nR]

    # update rest of the keys in candidates
    if mode == 1:
        for u in candidates.keys():
            candidates[u]['kw_surplus'] -= G.number_of_edges(best_node, u)
            candidates[u]['pw_surplus'] -= PD.getPOS(best_node, u, isSimple=isSimple)
    elif mode == 2:
        for u in candidates.keys():
            candidates[u]['kw_surplus'] -= G.number_of_edges(best_node, u)
            candidates[u]['kws_surplus'] -= (G.number_of_edges(best_node, u)>0)
            candidates[u]['mu_w_surplus'] -= PD.getExpectation(best_node, u, isSimple=isSimple)
    elif mode == 3:
        for u in candidates.keys():
            candidates[u]['kw_surplus'] -= G.number_of_edges(best_node, u)
            candidates[u]['kws_surplus'] -= (G.number_of_edges(best_node, u)>0)
            candidates[u]['mu_w_surplus'] -= PD.getExpectation(best_node, u, isSimple=isSimple)
            cur_minPOS = PD.getPOS(best_node, u, isSimple=isSimple)
            if abs(cur_minPOS - candidates[u]['p0_surplus']) < 1e-9:
                candidates[u]['p0_surplus'] = computeMinPOSBetweenNodeAndList(PD, u, list(pNL), gtype=gtype, isSimple=isSimple)

    return candidates
##################################################################################################################################################################
def climbOneStep(pattern, candidates, G, PD, q, mode, gtype, isSimple, incEdge):
        """function to climb one step at a time, i.e., either addition of node or removal from the subgraph pattern

        Args:
            pattern (Pattern): input subgraph pattern
            candidates (set): set of candidate node that can be added such that the subgraph remains connected

        Returns:
            Pattern, list, str: Updated subgraph pattern, updated candidate list and operation performed
        """
        # print('In climb one step')
        # print('\tSize of candidates: ', len(candidates))
        # print('\tBefore operation')
        # print('\teP: ', pattern.sumPOS, 'interest: ', pattern.I)

        operation = 'none'
        nodeAddedFinal = None
        nodeRemovedFinal = None

        # Extend Pattern
        best_params = dict()
        best_params['I'] = pattern.I
        best_node = None

        best_node, best_params = extendPatternUtil(pattern, candidates, G, PD, q, mode, gtype, isSimple, incEdge)

        # for cand in candidates:
        #     cur = extendPatternUtil(pattern, cand)
        #     if best_params['I'] < cur['I']:
        #         best_params = cur.copy()
        #         best_node = cand

        if best_params['I'] > pattern.I:
            # print('Added', best_node)
            bestPattern = extendPatternFinal(pattern, best_node, best_params, G, PD, q, mode, gtype, isSimple, incEdge)
            nodeAddedFinal = best_node
            operation = 'addition'

        if 'none' in operation and pattern.NCount > 2:
            for cand in pattern.NL:
                cur = shrinkPatternUtil(pattern, cand, G, PD, q, mode, gtype, isSimple, incEdge)
                if best_params['I'] < cur['I']:
                    best_params = cur.copy()
                    best_node = cand

            if best_params['I'] > pattern.I:
                bestPattern = shrinkPatternFinal(pattern, best_node, best_params, G, PD, q, mode, gtype, isSimple, incEdge)
                nodeRemovedFinal = best_node
                operation = 'deletion'

        # update candidate list now
        # print(operation, "\t", bestPattern.NL)

        if 'addition' in operation:
            candidates = updateCandidateAddition(nodeAddedFinal, bestPattern, candidates, G, PD, q, mode, gtype, isSimple, incEdge)

        elif 'deletion' in operation:
            candidates = updateCandidateDeletion(nodeRemovedFinal, bestPattern, candidates, G, PD, q, mode, gtype, isSimple, incEdge)

        # return relevant information
        # print('\tAfter Operation ', operation)
        # print('\tpw: ', pattern.sumPOS, 'interest: ', pattern.I)
        return pattern, candidates, operation
##################################################################################################################################################################
def extendPatternUtilD(pattern, candidates, dir_mode, G, PD, q, mode, gtype, isSimple, incEdge):
    # add one node a time and check for best gain
    # count edges from nodeToCheck to pattern
    # dir_mode (int): required id gtype is 'D"; 1 - from node to list (evaluating outnode) and 2 from list to node (evaluating innode)

    ######### Check in-node and out-node addition ############
    best_node = None
    params = None
    ic = 0.0
    dl = 0.0

    dlmode = 1
    if incEdge:
        dlmode = 2

    wi_count = pattern.InNCount
    wo_count = pattern.OutNCount

    if dir_mode == 1:
        wo_count += 1
    else:
        wi_count += 1

    if mode == 1:
        best_node, params = max(candidates.items(), key=lambda x: computeInterestingness( IC_SSG( 1, pw=pattern.sumPOS+x[1]['pw_surplus'], kw=pattern.ECount+x[1]['kw_surplus'], nw=pattern.nw+x[1]['nw_surplus'] ),\
            computeDescriptionLength( dlmode=dlmode, V=G.number_of_nodes(), WI=wi_count, WO=wo_count, nw=pattern.nw+x[1]['nw_surplus'], kw=pattern.ECount+x[1]['kw_surplus'], q=q ) ) )
        params['pw_new'] = pattern.sumPOS + params['pw_surplus']
        params['kw_new'] = pattern.ECount + params['kw_surplus']
        params['nw_new'] = pattern.nw + params['nw_surplus']
        ic = IC_SSG(1, pw=params['pw_new'], kw=params['kw_new'], nw=params['nw_new'])
        dl = computeDescriptionLength(dlmode=dlmode, V=G.number_of_nodes(), WI=wi_count, WO=wo_count, nw=params['nw_new'], kw=params['kw_new'], q=q)
    elif mode == 2:
        best_node, params = max(candidates.items(), key=lambda x: computeInterestingness( AD( pattern.ECount+x[1]['kw_surplus'], pattern.expectedEdges + x[1]['mu_w_surplus'] ),\
            computeDescriptionLength( dlmode=dlmode, V=G.number_of_nodes(), WI=wi_count, WO=wo_count, nw=pattern.nw+x[1]['nw_surplus'], kw=pattern.ECount+x[1]['kw_surplus'], q=q, isSimple=isSimple, kws=pattern.kws+x[1]['kws_surplus'] ) ) )
        params['mu_w_new'] = pattern.expectedEdges + params['mu_w_surplus']
        params['kw_new'] = pattern.ECount + params['kw_surplus']
        params['kws_new'] = pattern.ECount + params['kws_surplus']
        params['nw_new'] = pattern.nw + params['nw_surplus']
        ic = AD(params['kw_new'], params['mu_w_new'])
        dl = computeDescriptionLength(dlmode=dlmode, V=G.number_of_nodes(), WI=wi_count, WO=wo_count, nw=params['nw_new'], kw=params['kw_new'], q=q, isSimple=isSimple, kws=params['kws_new'])
    elif mode == 3:
        best_node, params = max(candidates.items(), key=lambda x: computeInterestingness( IC_DSIMP( pattern.ECount+x[1]['kw_surplus'], pattern.nw+x[1]['nw_surplus'], pattern.expectedEdges + x[1]['mu_w_surplus'], min(pattern.minPOS,x[1]['p0_surplus']) ),\
            computeDescriptionLength( dlmode=dlmode, V=G.number_of_nodes(), WI=wi_count, WO=wo_count, nw=pattern.nw+x[1]['nw_surplus'], kw=pattern.ECount+x[1]['kw_surplus'], q=q, isSimple=isSimple, kws=pattern.kws+x[1]['kws_surplus'] ) ) )
        params['mu_w_new'] = pattern.expectedEdges + params['mu_w_surplus']
        params['p0_new'] = min(pattern.minPOS, params['p0_surplus'])
        params['kw_new'] = pattern.ECount + params['kw_surplus']
        params['kws_new'] = pattern.ECount + params['kws_surplus']
        params['nw_new'] = pattern.nw + params['nw_surplus']
        ic = IC_DSIMP(params['kw_new'], params['nw_new'], params['mu_w_new'], params['p0_new'])
        dl = computeDescriptionLength(dlmode=dlmode, V=G.number_of_nodes(), WI=wi_count, WO=wo_count, nw=params['nw_new'], kw=params['kw_new'], q=q, isSimple=isSimple, kws=params['kws_new'])
    I = computeInterestingness(ic, dl)

    params['ic'] = ic
    params['dl'] = dl
    params['I'] = I

    return best_node, params
##################################################################################################################################################################
def extendPatternFinalD(pattern, nodeToAdd, params, typeOfAddition, G, PD, q, mode, gtype, isSimple, incEdge):
    H = nx.DiGraph()
    if not  isSimple:
        H = nx.MultiDiGraph()
    if 'in' in typeOfAddition:
        for p in pattern.outNL:
            if G.has_edge(p, nodeToAdd):
                H.add_edges_from([tuple([p, nodeToAdd])]*G.number_of_edges(p, nodeToAdd))
    else:
        for p in pattern.inNL:
            if G.has_edge(nodeToAdd, p):
                H.add_edges_from([tuple([nodeToAdd, p])]*G.number_of_edges(nodeToAdd, p))
    pattern.updateGraph(H)
    if mode == 1:
        pattern.setSumPOS(params['pw_new'])
        pattern.setIC_ssg(params['ic'])
    elif mode == 2:
        pattern.setAD(params['ic'])
        pattern.setExpectedEdges(params['mu_w_new'])
    elif mode == 3:
        pattern.setIC_dsimp(params['ic'])
        pattern.setExpectedEdges(params['mu_w_new'])
        pattern.setMinPOS(params['p0_new'])
    pattern.setDL(params['dl'])
    pattern.setI(params['I'])
    return pattern
##################################################################################################################################################################
def shrinkPatternUtilD(pattern, nodeToCheck, dir_mode, G, PD, q, mode, gtype, isSimple, incEdge):
    # remove node at a time, compute final interestingness and return the updated pattern
    # count edges from nodeToCheck to pattern

    ########### Check in-node or out-node removal ############
    kw_deficit  = 0
    kws_deficit  = 0
    params = dict()
    ic = 0.0
    dl = 0.0
    dlmode = 1
    if incEdge:
        dlmode = 2
    curInNL = pattern.inNL[:]
    curOutNL = pattern.outNL[:]
    curFunc = None
    if dir_mode == 1:
        for p in pattern.inNL:
            if nodeToCheck!=p:
                kw_deficit += int(G.number_of_edges(nodeToCheck, p))
                kws_deficit += (G.number_of_edges(nodeToCheck, p)>0)
        curOutNL.remove(nodeToCheck)
        curFunc = curInNL
    else:
        for p in pattern.outNL:
            if nodeToCheck!=p:
                kw_deficit += G.number_of_edges(p, nodeToCheck)
                kws_deficit += (G.number_of_edges(p, nodeToCheck)>0)
        curInNL.remove(nodeToCheck)
        curFunc = curOutNL

    if mode == 1:
        params['pw_deficit'] = computeSumOfEdgeProbablityBetweenNodeAndList(PD, nodeToCheck, curFunc, dir_mode=dir_mode, gtype=gtype, isSimple=isSimple)
        params['pw_new'] = pattern.sumPOS - params['pw_deficit']
        params['kw_new'] = pattern.ECount - kw_deficit
        params['nw_new'] = NW_D(curInNL, curOutNL)
        ic = IC_SSG(1, pw=params['pw_new'], kw=params['kw_new'], nw=params['nw_new'])
        dl = computeDescriptionLength(dlmode=dlmode, V=G.number_of_nodes(), WI=curInNL, WO=curOutNL, kw=params['kw_new'], q=q)
    elif mode == 2:
        params['mu_w_deficit'] = computeSumOfExpectationsBetweenNodeAndList(PD, nodeToCheck, curFunc, dir_mode=dir_mode, gtype=gtype, isSimple=isSimple)
        params['mu_w_new'] = pattern.expectedEdges - params['mu_w_deficit']
        params['kw_new'] = pattern.ECount - kw_deficit
        params['kws_new'] = pattern.kws - kws_deficit
        ic = AD(params['kw_new'], params['mu_w_new'])
        dl = computeDescriptionLength(dlmode=dlmode, V=G.number_of_nodes(), WI=curInNL, WO=curOutNL, kw=params['kw_new'], q=q, isSimple=isSimple, kws=params['kws_new'])
    else:
        params['kw_new'] = pattern.ECount - kw_deficit
        params['kws_new'] = pattern.kws - kws_deficit
        params['nw_new'] = NW_D(curInNL, curOutNL)
        params['mu_w_deficit'], params['p0_deficit'] = computePWparametersBetweenNodeAndList(PD, nodeToCheck, curFunc, dir_mode=dir_mode, gtype=gtype, isSimple=isSimple)
        if pattern.minPOS == params['p0_deficit']:
            params['mu_w_new'], params['p0_new'] = computePWparameters(PD, gtype=gtype, inNL = curInNL, outNL = curOutNL, isSimple=isSimple)
        else:
            params['p0_new'] = pattern.minPOS
            params['mu_w_new'] = pattern.expectedEdges - params['mu_w_deficit']
        ic = IC_DSIMP(params['kw_new'], params['nw_new'], params['mu_w_new'], params['p0_new'])
        dl = computeDescriptionLength(dlmode=dlmode, V=G.number_of_nodes(), WI=curInNL, WO=curOutNL, kw=params['kw_new'], q=q, isSimple=isSimple, kws=params['kws_new'])
    I = computeInterestingness(ic, dl)
    params['ic'] = ic
    params['dl'] = dl
    params['I'] = I
    return params
##################################################################################################################################################################
def shrinkPatternFinalD(pattern, nodeToRemove, params, typeOfDeletion, G, PD, q, mode, gtype, isSimple, incEdge):
    if 'in' in typeOfDeletion:
        pattern.removeInNode(nodeToRemove)
    else:
        pattern.removeOutNode(nodeToRemove)
    if mode == 1:
        pattern.setSumPOS(params['pw_new'])
        pattern.setIC_ssg(params['ic'])
    elif mode == 2:
        pattern.setAD(params['ic'])
        pattern.setExpectedEdges(params['mu_w_new'])
    elif mode == 3:
        pattern.setIC_dsimp(params['ic'])
        pattern.setExpectedEdges(params['mu_w_new'])
        pattern.setMinPOS(params['p0_new'])
    pattern.setDL(params['dl'])
    pattern.setI(params['I'])
    return pattern
##################################################################################################################################################################
def updateCandidateAdditionD(best_node, bestPattern, candidatesIn, candidatesOut, typeOfAddition, G, PD, q, mode, gtype, isSimple, incEdge):
    if 'in' in typeOfAddition:
        del candidatesIn[best_node]
        #update parameters of previously present candidateOut list as new in-node is added
        if mode == 1:
            for u in candidatesOut.keys():
                if u != best_node:
                    candidatesOut[u]['kw_surplus'] += G.number_of_edges(u, best_node)
                    candidatesOut[u]['pw_surplus'] += PD.getPOS(u, best_node, isSimple=isSimple)
                    candidatesOut[u]['nw_surplus'] += 1
        elif mode == 2:
            for u in candidatesOut.keys():
                if u != best_node:
                    candidatesOut[u]['kw_surplus'] += G.number_of_edges(u, best_node)
                    candidatesOut[u]['kws_surplus'] += (G.number_of_edges(u, best_node)>0)
                    candidatesOut[u]['mu_w_surplus'] += PD.getExpectation(u, best_node, isSimple=isSimple)
                    candidatesOut[u]['nw_surplus'] += 1
        elif mode == 3:
            for u in candidatesOut.keys():
                if u != best_node:
                    candidatesOut[u]['kw_surplus'] += G.number_of_edges(u, best_node)
                    candidatesOut[u]['kws_surplus'] += (G.number_of_edges(u, best_node)>0)
                    candidatesOut[u]['mu_w_surplus'] += PD.getExpectation(u, best_node, isSimple=isSimple)
                    candidatesOut[u]['p0_surplus'] = min(candidatesOut[u]['p0_surplus'], PD.getPOS(u, best_node, isSimple=isSimple))
                    candidatesOut[u]['nw_surplus'] += 1
        #### similarly new out candidates are now introduced and added in the list as in-node is added
        pot_add = set(G.predecessors(best_node)) - set(candidatesOut.keys()).union(set(bestPattern.outNL))
        if mode == 1:
            for u in pot_add:
                candidatesOut[u] = dict()
                candidatesOut[u]['kw_surplus'] = 0
                candidatesOut[u]['nw_surplus'] = 0
                candidatesOut[u]['pw_surplus'] = 0.0
                for v in bestPattern.inNL:
                    if u != v:
                        candidatesOut[u]['kw_surplus'] += G.number_of_edges(u, v)
                        candidatesOut[u]['pw_surplus'] += PD.getPOS(u, v, isSimple=isSimple)
                        candidatesOut[u]['nw_surplus'] += 1
        elif mode == 2:
            for u in pot_add:
                candidatesOut[u] = dict()
                candidatesOut[u]['kw_surplus'] = 0
                candidatesOut[u]['kws_surplus'] = 0
                candidatesOut[u]['nw_surplus'] = 0
                candidatesOut[u]['mu_w_surplus'] = 0.0
                for v in bestPattern.inNL:
                    if u != v:
                        candidatesOut[u]['kw_surplus'] += G.number_of_edges(u, v)
                        candidatesOut[u]['kws_surplus'] += (G.number_of_edges(u, v)>0)
                        candidatesOut[u]['mu_w_surplus'] += PD.getExpectation(u, v, isSimple=isSimple)
                        candidatesOut[u]['nw_surplus'] += 1
        elif mode == 3:
            for u in pot_add:
                candidatesOut[u] = dict()
                candidatesOut[u]['kw_surplus'] = 0
                candidatesOut[u]['kws_surplus'] = 0
                candidatesOut[u]['nw_surplus'] = 0
                candidatesOut[u]['mu_w_surplus'] = 0.0
                candidatesOut[u]['p0_surplus'] = float("inf")
                for v in bestPattern.inNL:
                    if u != v:
                        candidatesOut[u]['kw_surplus'] += G.number_of_edges(u, v)
                        candidatesOut[u]['kws_surplus'] += (G.number_of_edges(u, v)>0)
                        candidatesOut[u]['mu_w_surplus'] += PD.getExpectation(u, v, isSimple=isSimple)
                        candidatesOut[u]['p0_surplus'] = min(candidatesOut[u]['p0_surplus'], PD.getPOS(u, v, isSimple=isSimple))
                        candidatesOut[u]['nw_surplus'] += 1
    else:
        del candidatesOut[best_node]
        #update parameters of previously present candidateIn list as new out-node is added
        if mode == 1:
            for u in candidatesIn.keys():
                if u != best_node:
                    candidatesIn[u]['kw_surplus'] += G.number_of_edges(best_node, u)
                    candidatesIn[u]['pw_surplus'] += PD.getPOS(best_node, u, isSimple=isSimple)
                    candidatesIn[u]['nw_surplus'] += 1
        elif mode == 2:
            for u in candidatesIn.keys():
                if u != best_node:
                    candidatesIn[u]['kw_surplus'] += G.number_of_edges(best_node, u)
                    candidatesIn[u]['kws_surplus'] += (G.number_of_edges(best_node, u)>0)
                    candidatesIn[u]['mu_w_surplus'] += PD.getExpectation(best_node, u, isSimple=isSimple)
                    candidatesIn[u]['nw_surplus'] += 1
        elif mode == 3:
            for u in candidatesIn.keys():
                if u != best_node:
                    candidatesIn[u]['kw_surplus'] += G.number_of_edges(best_node, u)
                    candidatesIn[u]['kws_surplus'] += (G.number_of_edges(best_node, u)>0)
                    candidatesIn[u]['mu_w_surplus'] += PD.getExpectation(best_node, u, isSimple=isSimple)
                    candidatesIn[u]['p0_surplus'] = min(candidatesIn[u]['p0_surplus'], PD.getPOS(best_node, u, isSimple=isSimple))
                    candidatesIn[u]['nw_surplus'] += 1
        #### similarly new "in" candidates are now introduced and added in the list as out-node is added
        pot_add = set(G.successors(best_node)) - set(candidatesIn.keys()).union(set(bestPattern.inNL))
        if mode == 1:
            for u in pot_add:
                candidatesIn[u] = dict()
                candidatesIn[u]['kw_surplus'] = 0
                candidatesIn[u]['nw_surplus'] = 0
                candidatesIn[u]['pw_surplus'] = 0.0
                for v in bestPattern.outNL:
                    if u != v:
                        candidatesIn[u]['kw_surplus'] += G.number_of_edges(v, u)
                        candidatesIn[u]['pw_surplus'] += PD.getPOS(v, u, isSimple=isSimple)
                        candidatesIn[u]['nw_surplus'] += 1
        elif mode == 2:
            for u in pot_add:
                candidatesIn[u] = dict()
                candidatesIn[u]['kw_surplus'] = 0
                candidatesIn[u]['kws_surplus'] = 0
                candidatesIn[u]['nw_surplus'] = 0
                candidatesIn[u]['mu_w_surplus'] = 0.0
                for v in bestPattern.outNL:
                    if u != v:
                        candidatesIn[u]['kw_surplus'] += G.number_of_edges(v, u)
                        candidatesIn[u]['kws_surplus'] += (G.number_of_edges(v, u)>0)
                        candidatesIn[u]['mu_w_surplus'] += PD.getExpectation(v, u, isSimple=isSimple)
                        candidatesIn[u]['nw_surplus'] += 1
        elif mode == 3:
            for u in pot_add:
                candidatesIn[u] = dict()
                candidatesIn[u]['kw_surplus'] = 0
                candidatesIn[u]['kws_surplus'] = 0
                candidatesIn[u]['nw_surplus'] = 0
                candidatesIn[u]['mu_w_surplus'] = 0.0
                candidatesIn[u]['p0_surplus'] = float("inf")
                for v in bestPattern.outNL:
                    if u != v:
                        candidatesIn[u]['kw_surplus'] += G.number_of_edges(v, u)
                        candidatesIn[u]['kws_surplus'] += (G.number_of_edges(v, u)>0)
                        candidatesIn[u]['mu_w_surplus'] += PD.getExpectation(v, u, isSimple=isSimple)
                        candidatesIn[u]['p0_surplus'] = min(candidatesIn[u]['p0_surplus'], PD.getPOS(v, u, isSimple=isSimple))
                        candidatesIn[u]['nw_surplus'] += 1

    return candidatesIn, candidatesOut
##################################################################################################################################################################
def updateCandidateDeletionD(best_node, bestPattern, candidatesIn, candidatesOut, typeOfDeletion, G, PD, q, mode, gtype, isSimple, incEdge):
    if 'in' in typeOfDeletion:
        #since an IN node is removed thus OUT candidate list is updated

        #First remove irrelevant key in OUT candidate list
        outNeighbor = list(G.predecessors(best_node))
        for oN in outNeighbor:
            if len(set(G.successors(oN)).intersection(bestPattern.inNL)) == 0:
                if oN in candidatesOut:
                    del candidatesOut[oN]
        #update rest of the keys
        if mode == 1:
            for u in candidatesOut.keys():
                if u != best_node:
                    candidatesOut[u]['kw_surplus'] -= G.number_of_edges(u, best_node)
                    candidatesOut[u]['pw_surplus'] -= PD.getPOS(u, best_node, isSimple=isSimple)
                    candidatesOut[u]['nw_surplus'] -= 1
        elif mode == 2:
            for u in candidatesOut.keys():
                if u != best_node:
                    candidatesOut[u]['kw_surplus'] -= G.number_of_edges(u, best_node)
                    candidatesOut[u]['kws_surplus'] -= (G.number_of_edges(u, best_node)>0)
                    candidatesOut[u]['mu_w_surplus'] -= PD.getExpectation(u, best_node, isSimple=isSimple)
                    candidatesOut[u]['nw_surplus'] -= 1
        elif mode == 3:
            for u in candidatesOut.keys():
                if u != best_node:
                    candidatesOut[u]['kw_surplus'] -= G.number_of_edges(u, best_node)
                    candidatesOut[u]['kws_surplus'] -= (G.number_of_edges(u, best_node)>0)
                    candidatesOut[u]['mu_w_surplus'] -= PD.getExpectation(u, best_node, isSimple=isSimple)
                    candidatesOut[u]['nw_surplus'] -= 1
                    cur_minPOS = PD.getPOS(u, best_node, isSimple=isSimple)
                    if abs(cur_minPOS - candidatesOut[u]['p0_surplus']) < 1e-9:
                        candidatesOut[u]['p0_surplus'] = computeMinPOSBetweenNodeAndList(PD, u, bestPattern.inNL, dir_mode=1, gtype=gtype, isSimple=isSimple)
    else:
        #since an OUT node is removed thus IN candidate list is updated

        #First remove irrelevant keys in IN candidate list
        inNeighbor = list(G.successors(best_node))
        for iN in inNeighbor:
            if len(set(G.predecessors(oN)).intersection(bestPattern.outNL)) == 0:
                if iN in candidatesIn:
                    del candidatesIn[iN]
        #update rest of the keys
        if mode == 1:
            for u in candidatesIn.keys():
                if u != best_node:
                    candidatesIn[u]['kw_surplus'] -= G.number_of_edges(best_node, u)
                    candidatesIn[u]['pw_surplus'] -= PD.getPOS(best_node, u, isSimple=isSimple)
                    candidatesIn[u]['nw_surplus'] -= 1
        elif mode == 2:
            for u in candidatesIn.keys():
                if u != best_node:
                    candidatesIn[u]['kw_surplus'] -= G.number_of_edges(best_node, u)
                    candidatesIn[u]['kws_surplus'] -= (G.number_of_edges(best_node, u)>0)
                    candidatesIn[u]['mu_w_surplus'] -= PD.getExpectation(best_node, u, isSimple=isSimple)
                    candidatesIn[u]['nw_surplus'] -= 1
        elif mode == 3:
            for u in candidatesIn.keys():
                if u != best_node:
                    candidatesIn[u]['kw_surplus'] -= G.number_of_edges(best_node, u)
                    candidatesIn[u]['kws_surplus'] -= (G.number_of_edges(best_node, u)>0)
                    candidatesIn[u]['mu_w_surplus'] -= PD.getExpectation(best_node, u, isSimple=isSimple)
                    candidatesIn[u]['nw_surplus'] -= 1
                    cur_minPOS = PD.getPOS(best_node, u, isSimple=isSimple)
                    if abs(cur_minPOS - candidatesIn[u]['p0_surplus']) < 1e-9:
                        candidatesIn[u]['p0_surplus'] = computeMinPOSBetweenNodeAndList(PD, u, bestPattern.outNL, dir_mode=2, gtype=gtype, isSimple=isSimple)

    return candidatesIn, candidatesOut
##################################################################################################################################################################
def climbOneStepD(pattern, candidatesIn, candidatesOut, G, PD, q, mode, gtype, isSimple, incEdge):
    operation = 'none'
    bestPattern = pattern.copy()
    nodeAddedFinal = None
    nodeRemovedFinal = None

    # Extend Pattern
    bestInParams = dict()
    bestInParams['I'] = pattern.I
    bestInNode = None

    bestOutParams = dict()
    bestOutParams['I'] = pattern.I
    bestOutNode = None

    #Check all possible in-node addition
    bestInNode, bestInParams = extendPatternUtilD(pattern, candidatesIn, 2, G, PD, q, mode, gtype, isSimple, incEdge)

    #Check all possible out-node addition
    bestOutNode, bestOutParams = extendPatternUtilD(pattern, candidatesOut, 1, G, PD, q, mode, gtype, isSimple, incEdge)

    #Perform best addition
    if bestInParams['I'] > pattern.I or bestOutParams['I'] > pattern.I:
        if bestInParams['I'] > bestOutParams['I']:
            bestPattern = extendPatternFinal(pattern, bestInNode, bestInParams, 'in', G, PD, q, mode, gtype, isSimple, incEdge)
            nodeAddedFinal = bestInNode
            operation = 'inaddition'
        else:
            bestPattern = extendPatternFinal(pattern, bestOutNode, bestOutParams, 'out', G, PD, q, mode, gtype, isSimple, incEdge)
            nodeAddedFinal = bestOutNode
            operation = 'outaddition'


    # If no extension, shrink pattern
    if 'none' in operation:
        #Check all possible in-node removal
        if pattern.InNCount > 1:
            for node in pattern.inNL:
                curIn = shrinkPatternUtilD(pattern, node, 2, G, PD, q, mode, gtype, isSimple, incEdge)
                if bestInParams['I'] < curIn['I']:
                    bestInParams = curIn.copy()
                    bestInNode = node

        #Check all possible out-node removal
        if pattern.OutNCount>1:
            for node in pattern.outNL:
                curOut = shrinkPatternUtilD(pattern, node, 1, G, PD, q, mode, gtype, isSimple, incEdge)
                if bestOutParams['I'] < curOut['I']:
                    bestOutParams = curOut.copy()
                    bestOutNode = node

        #Perform best removal
        if bestInParams['I'] > pattern.I or bestOutParams['I'] > pattern.I:
            if bestInParams['I'] > bestOutParams['I']:
                bestPattern = shrinkPatternFinal(pattern, bestInNode, bestInParams, 'in', G, PD, q, mode, gtype, isSimple, incEdge)
                nodeRemovedFinal = bestInNode
                operation = 'indeletion'
            else:
                bestPattern = shrinkPatternFinal(pattern, bestOutNode, bestOutParams, 'out', G, PD, q, mode, gtype, isSimple, incEdge)
                nodeRemovedFinal = bestOutNode
                operation = 'outdeletion'

    # update candidate list now
    # print(operation, "\t", bestPattern.NL)

    if 'addition' in operation:
        if 'in' in operation:
            candidatesIn, candidateOut = updateCandidateAdditionD(nodeAddedFinal, bestPattern, candidatesIn, candidatesOut, 'in', G, PD, q, mode, gtype, isSimple, incEdge)
        else:
            candidatesIn, candidateOut = updateCandidateAdditionD(nodeAddedFinal, bestPattern, candidatesIn, candidatesOut, 'out', G, PD, q, mode, gtype, isSimple, incEdge)
    elif 'deletion' in operation:
        if 'in' in operation:
            candidatesIn, candidateOut = updateCandidateDeletionD(nodeAddedFinal, bestPattern, candidatesIn, candidatesOut, 'in', G, PD, q, mode, gtype, isSimple, incEdge)
        else:
            candidatesIn, candidateOut = updateCandidateDeletionD(nodeAddedFinal, bestPattern, candidatesIn, candidatesOut, 'out', G, PD, q, mode, gtype, isSimple, incEdge)

    # return relevant information
    print('Best Pattern eP: ', bestPattern.sumPOS, 'interest: ', bestPattern.I)
    # print("inNL", bestPattern.inNL)
    # print("outNL", bestPattern.outNL)
    print("Operation", operation, bestInNode, bestOutNode)
    return bestPattern, candidatesIn, candidatesOut, operation
##################################################################################################################################################################
@ray.remote
def searchPattern(seed, G, PD, q, mode, gtype, isSimple, incEdge):
    """function to search for a pattern starting from a given seed subgraph

    Args:
        seed (int): seed subgraph of one node

    Returns:
        Pattern: found pattern for given seed
    """
    pattern = None
    if gtype == 'U':
        candidates = dict()
        if mode == 1: #we require two parameters, which are, number of edges and pw_surplus from node to pattern (subgraph)
            for u in list(G.neighbors(seed)):
                candidates[u] = dict()
                candidates[u]['kw_surplus'] = G.number_of_edges(seed, u)
                candidates[u]['pw_surplus'] = PD.getPOS(seed, u, isSimple=isSimple)
        elif mode == 2: #we require again two parameters: number of edges and expectedEdge_surplus from node to pattern (subgraph)
            for u in list(G.neighbors(seed)):
                candidates[u] = dict()
                candidates[u]['kw_surplus'] = G.number_of_edges(seed, u)
                candidates[u]['kws_surplus'] = (G.number_of_edges(seed, u)>0)
                candidates[u]['mu_w_surplus'] = PD.getExpectation(seed, u, isSimple=isSimple)
        elif mode == 3: #we require three parameters: number of edges, expectedEdge_surplus and minp from node to pattern (subgraph)
            for u in list(G.neighbors(seed)):
                candidates[u] = dict()
                candidates[u]['kw_surplus'] = G.number_of_edges(seed, u)
                candidates[u]['kws_surplus'] = (G.number_of_edges(seed, u)>0)
                candidates[u]['mu_w_surplus'] = PD.getExpectation(seed, u, isSimple=isSimple)
                candidates[u]['p0_surplus'] = PD.getPOS(seed, u, isSimple=isSimple)
        else:
            raise Exception('Invalid mode provided')

        term = False

        pattern = Pattern(G.subgraph(seed))
        pattern.setPatType('Found')
        # while termination==false climb one step at a time
        while not term:
            pattern, candidates, operation = climbOneStep(pattern, candidates, G, PD, q, mode, gtype, isSimple, incEdge)
            # print(operation)
            if 'none' in operation:
                term = True
    else:
        candidatesIn = dict()
        candidateOut = dict()
        if mode == 1: #we require two parameters, which are, number of edges and pw_surplus from node to pattern (subgraph)
            for u in list(G.successors(seed)):
                candidatesIn[u] = dict()
                candidatesIn[u]['kw_surplus'] = G.number_of_edges(seed, u)
                candidatesIn[u]['pw_surplus'] = PD.getPOS(seed, u, isSimple=isSimple)
                candidatesIn[u]['nw_surplus'] = 1
        elif mode == 2: #we require again two parameters: number of edges and expectedEdge_surplus from node to pattern (subgraph)
            for u in list(G.successors(seed)):
                candidatesIn[u] = dict()
                candidatesIn[u]['kw_surplus'] = G.number_of_edges(seed, u)
                candidatesIn[u]['kws_surplus'] = (G.number_of_edges(seed, u)>0)
                candidatesIn[u]['mu_w_surplus'] = PD.getExpectation(seed, u, isSimple=isSimple)
                candidatesIn[u]['nw_surplus'] = 1
        elif mode == 3: #we require three parameters: number of edges, expectedEdge_surplus and minp from node to pattern (subgraph)
            for u in list(G.successors(seed)):
                candidatesIn[u] = dict()
                candidatesIn[u]['kw_surplus'] = G.number_of_edges(seed, u)
                candidatesIn[u]['kws_surplus'] = (G.number_of_edges(seed, u)>0)
                candidatesIn[u]['mu_w_surplus'] = PD.getExpectation(seed, u, isSimple=isSimple)
                candidatesIn[u]['p0_surplus'] = PD.getPOS(seed, u, isSimple=isSimple)
                candidatesIn[u]['nw_surplus'] = 1

        if mode == 1: #we require two parameters, which are, number of edges and pw_surplus from node to pattern (subgraph)
            for u in list(G.predecessors(seed)):
                candidatesOut[u] = dict()
                candidatesOut[u]['kw_surplus'] = G.number_of_edges(u, seed)
                candidatesOut[u]['pw_surplus'] = PD.getPOS(u, seed, isSimple=isSimple)
                candidatesOut[u]['nw_surplus'] = 1
        elif mode == 2: #we require again two parameters: number of edges and expectedEdge_surplus from node to pattern (subgraph)
            for u in list(G.predecessors(seed)):
                candidatesOut[u] = dict()
                candidatesOut[u]['kw_surplus'] = G.number_of_edges(u, seed)
                candidatesOut[u]['kws_surplus'] = (G.number_of_edges(u, seed)>0)
                candidatesOut[u]['mu_w_surplus'] = PD.getExpectation(u, seed, isSimple=isSimple)
                candidatesOut[u]['nw_surplus'] = 1
        elif mode == 3: #we require three parameters: number of edges, expectedEdge_surplus and minp from node to pattern (subgraph)
            for u in list(G.predecessors(seed)):
                candidatesOut[u] = dict()
                candidatesOut[u]['kw_surplus'] = G.number_of_edges(u, seed)
                candidatesOut[u]['kws_surplus'] = (G.number_of_edges(u, seed)>0)
                candidatesOut[u]['mu_w_surplus'] = PD.getExpectation(u, seed, isSimple=isSimple)
                candidatesOut[u]['p0_surplus'] = PD.getPOS(u, seed, isSimple=isSimple)
                candidatesOut[u]['nw_surplus'] = 1

        term = False
        pattern = Pattern(G.subgraph(seed))
        pattern.setPatType('Found')
        # while termination==false climb one step at a time
        while not term:
            pattern, candidatesIn, candidatesOut, operation = climbOneStepD(pattern, candidatesIn, candidatesOut, G, PD, q, mode, gtype, isSimple, incEdge)
            if 'none' in operation:
                term = True

    #return pattern and other information
    return pattern
##################################################################################################################################################################
def findBestPattern(G, PD, q, seedMode, nKseed, mode = 1, gtype = 'U', isSimple=True, incEdge = False):
    seedNodes = getSeeds(G, PD, q, seedMode, nKseed, mode, gtype, isSimple, incEdge)
    bestPattern = Pattern(nx.Graph())
    if gtype == 'U' and not isSimple:
        bestPattern = Pattern(nx.MultiGraph())
    if gtype == 'D':
        bestPattern = Pattern(nx.DiGraph())
        if not isSimple:
            bestPattern = Pattern(nx.MultiDiGraph())

    Gid = ray.put(G)
    PDid = ray.put(PD)

    Results = ray.get([searchPattern.remote(seed, Gid, PDid, q, mode, gtype, isSimple, incEdge) for seed in seedNodes])
    print('res length', len(Results))
    bestPattern = max(Results, key=lambda x: x.I)

    return bestPattern
##################################################################################################################################################################
def runNKseeds(G, PD, q, seedMode, nKseed, mode = 1, gtype = 'U', isSimple=True, incEdge = False):
    seedNodes = getSeeds(G, PD, q, seedMode, nKseed, mode, gtype, isSimple, incEdge)

    Gid = ray.put(G)
    PDid = ray.put(PD)

    Results = ray.get([searchPattern.remote(seed, Gid, PDid, q, mode, gtype, isSimple, incEdge) for seed in seedNodes])
    print('res length', len(Results))

    return Results
##################################################################################################################################################################
def runGivenSeeds(G, PD, q, seedNodes, mode = 1, gtype = 'U', isSimple=True, incEdge = False):
    Gid = ray.put(G)
    PDid = ray.put(PD)

    Results = ray.get([searchPattern.remote(seed, Gid, PDid, q, mode, gtype, isSimple, incEdge) for seed in seedNodes])
    print('res length', len(Results))

    return Results
##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################


# def addNewPatternList(NL, count, Pattype):
#     pattern = Pattern.Pattern(G.subgraph(NL).copy(), NL, 0, 0, PD.computeEdgeProbabilityListNodesIncLprev(NL), PD.computeInterestingnessListNodesIncLprev(NL), count)
#     pattern.updateGraphProperties()
#     PD.updateBackground(pattern, count)
#     pattern.setPatType(Pattype)
#     return pattern, PD.copy()

# def addNewPattern(pattern, count):
#     PD.updateBackground(pattern, count)
#     return PD.copy()