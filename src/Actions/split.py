###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################
import os
import sys
path = os.getcwd().split('MiningSubjectiveSubgraphPatterns')[0]+'MiningSubjectiveSubgraphPatterns/'
if path not in sys.path:
    sys.path.append(path)
import numpy as np
import math
import networkx as nx

from src.Utils.Measures import getCodeLength, getCodeLengthParallel, getDirectedSubgraph
from src.Utils.Measures import computeDescriptionLength, computeInterestingness
from src.Utils.Measures import LN, DL_Edges, ncr
from src.Patterns.Pattern import Pattern

class EvaluateSplit:
    # Now this data structure should contain all the possible removals
    # along with pattern number as key and Information Gain and list of nodes as value
    def __init__(self, gtype='U', isSimple=True, l=6, minsize=2):
        self.Data = dict()
        self.gtype = gtype
        self.isSimple = isSimple
        self.l = l # possible types (give number) of action, default is 6
        self.minsize = minsize
        print('initialized EvaluateSplit')

    def evaluateAllConstraints(self, G, PD):
        self.Data = dict()
        for i in PD.lprevUpdate.keys():
            self.evaluateConstraint(G, PD, i)
        return

    def evaluateConstraint(self, G, PD, i):
        if self.gtype == 'U':
            self.processAsU(G, PD, i)
        elif self.gtype == 'D':
            self.processAsD(G, PD, i)
        return

    def processAsU(self, G, PD, i):
        NL = PD.lprevUpdate[i][1]
        H = G.subgraph(NL)
        components = nx.connected_component_subgraphs(H, copy=True)
        fcomponents = dict()
        it = 0
        for comp in components:
            if comp.number_of_nodes() > self.minsize:
                fcomponents[it] = comp
        if len(fcomponents) > 1: #* If components are more than one then only we can split this pattern
            baseParams = dict()
            baseParams['Pat'] = Pattern(H)
            baseParams['NodesInc'] = 0
            compPats = dict()
            nodes_union = set()
            for k,v in fcomponents.items():
                compPats[k] = Pattern(v)
                baseParams['NodesInc'] += v.number_of_nodes()
                nodes_union = nodes_union.union(set(compPats[k].NL))
            baseParams['compos'] = compPats
            baseParams['excludedNL'] = list( set(baseParams['Pat'].NL) - nodes_union )
            baseParams['codeLengthC'] = getCodeLengthParallel( H, PD, gtype=self.gtype, case=2, isSimple=self.isSimple, NL=baseParams['Pat'].NL )
            baseParams['codeLengthCprime'] = self.computeCodeLengthSplitU(G, PD, 2, baseParams, i) #// Todo : write code for this part
            baseParams['Pat'].setIC_dssg( baseParams['codeLengthC'] - baseParams['codeLengthCprime'] )
            baseParams['Pat'].setDL( computeDescriptionLength( case=7, C=len(PD.lprevUpdate), gtype=self.gtype, WS=baseParams['Pat'].NCount, compos=baseParams['compos'] ) )
            baseParams['Pat'].setI( computeInterestingness( baseParams['Pat'].IC_dssg, baseParams['Pat'].DL, mode=2 ) )

            #now try reducing each component
            FinalParams = baseParams
            for k in baseParams['compos'].keys():
                FinalParams = self.getReducedComponentU(G, PD, FinalParams, i, k)

            #compute new lambdas for each new pattern/component
            #// Todo: write code
            for k,v in FinalParams['compos'].items():
                v.setLambda( PD.updateDistribution( pat=v.G, idx=None, val_retrun='return', case=3, dropLidx=[i]) )
            FinalParams['codeLengthCprime'] = self.computeCodeLengthSplitU(G, PD, 3, FinalParams, i) #// Todo : write code for this part
            FinalParams['Pat'].setIC_dssg( FinalParams['codeLengthC'] - FinalParams['codeLengthCprime'] )
            FinalParams['Pat'].setDL( computeDescriptionLength( case=7, C=len(PD.lprevUpdate), gtype=self.gtype, WS=FinalParams['Pat'].NCount, compos=FinalParams['compos'], excActionType=False, l=self.l ) )
            FinalParams['Pat'].setI( computeInterestingness( FinalParams['Pat'].IC_dssg, FinalParams['Pat'].DL, mode=2 ) )
            # Now set these values to all component patterns
            #// Todo: Write Code
            for k,v in FinalParams['compos'].items():
                v.setIC_dssg( FinalParams['Pat'].IC_dssg )
                v.setDL( FinalParams['Pat'].DL )
                v.setI( FinalParams['Pat'].I )
                v.setPrevOrder(i)
            self.Data[i] = FinalParams
        return

    def getReducedComponentU(self, G, PD, FinalParams, Lid, k):
        doshrink = True
        while doshrink:
            doshrink = False
            bestRNode = None
            bestCLprime = None
            bestDL = None
            bestI = FinalParams['Pat'].I
            for node in FinalParams['compos'][k].NL:
                curCLprime = FinalParams['codeLengthCprime'] - self.computeCLgainRemoveNodeU(G, PD, list(set(FinalParams['compos'][k].NL)-set([node])), node, [Lid])
                curIC = FinalParams['codeLengthC'] - curCLprime
                curDL = FinalParams['Pat'].DL - self.getDescriptionLengthChangeU(FinalParams['compos'][k], node, FinalParams['Pat'].NCount, FinalParams['NodesInc'])
                curI =  computeInterestingness( curIC, curDL, mode=2 )
                if curI > bestI:
                    bestRNode = node
                    bestCLprime = curCLprime
                    bestDL = curDL
                    bestI = curI
            if bestI > FinalParams['Pat'].I:
                FinalParams['codeLengthCprime'] = bestCLprime
                FinalParams['compos'][k].removeNode(bestRNode)
                FinalParams['Pat'].setIC_dssg( FinalParams['codeLengthC'] - FinalParams['codeLengthCprime'] )
                FinalParams['Pat'].setDL( computeDescriptionLength( case=7, C=len(PD.lprevUpdate), gtype=self.gtype, WS=FinalParams['Pat'].NCount, compos=FinalParams['compos'], excActionType=False, l=self.l ) )
                FinalParams['Pat'].setI( computeInterestingness( FinalParams['Pat'].IC_dssg, FinalParams['Pat'].DL, mode=2 ) )
                FinalParams['NodesInc'] -= 1
                FinalParams['excludedNL'].append(bestRNode)
                doshrink = True

        return FinalParams

    def computeCLgainRemoveNodeU(self, G, PD, nodes, node, dropLidx):
        CL_I = 0.0
        CL_F = 0.0
        for i in nodes:
            pos_I = PD.getPOS(i, node, case=2, isSimple=self.isSimple)
            pos_F = PD.getPOS(i, node, case=4, isSimple=self.isSimple, dropLidx=dropLidx)
            numE = G.number_of_edges(i, node)
            if self.isSimple:
                CL_I += math.log2(math.pow(1.0-pos_I, 1.0-numE)*math.pow(pos_I, numE))
                CL_F += math.log2(math.pow(1.0-pos_F, 1.0-numE)*math.pow(pos_F, numE))
            else:
                CL_I += math.log2(math.pow(1.0-pos_I, numE)*pos_I)
                CL_F += math.log2(math.pow(1.0-pos_F, numE)*pos_F)
        return -CL_I - (-CL_F)

    def getDescriptionLengthChangeU(self, Pat, node, WI, W):
        change = 0.0
        change += LN(Pat.NCount) - LN(Pat.NCount - 1)
        change += ncr(WI, W) - ncr(WI, W-1)
        pDLEdge = DL_Edges(Pat.nw, Pat.ECount)
        nECount = Pat.ECount
        for k in Pat.NL:
            nECount -= Pat.G.number_of_edges(k, node)
        nDLEdge = DL_Edges(Pat.nw, nECount)
        change += pDLEdge - nDLEdge
        return change

    def computeCodeLengthSplitU(self, G, PD, condition, Params, Lidx=None):
        codeLength = 0.0
        if condition == 1:
            codelength = getCodeLengthParallel( G, PD, gtype=self.gtype, case=2, isSimple=self.isSimple, NL=Params['Pat'].NL )
            return codelength
        elif condition == 2:
            # intra-component codelength computation
            for k,v in Params['compos'].items():
                codelength += getCodeLengthParallel( G, PD, gtype=self.gtype, case=2, isSimple=self.isSimple, NL=v.NL )
            # inter-component codelength computation, i.e., one list of the component and other list of all the rest of the node
            keys = Params['compos'].keys()
            for k1 in range(len(keys)-1):
                for k2 in range(k1+1, len(keys)):
                    codelength += getCodeLengthParallel( G, PD, gtype='D', case=4, isSimple=self.isSimple, inNL=Params['compos'][k1].NL, outNL=Params['compos'][k2].NL, dropLidx=[Lidx] )
            # compute for excluded nodes
            if len(Params['excludedNL']) > 0:
                codelength += getCodeLengthParallel( G, PD, gtype=self.gtype, case=4, isSimple=self.isSimple, NL=Params['excludedNL'], dropLidx=[Lidx] )
                for k,v in Params['compos'].items():
                    codelength += getCodeLengthParallel( G, PD, gtype='D', case=4, isSimple=self.isSimple, inNL=v.NL, outNL=Params['excludedNL'], dropLidx=[Lidx] )
        elif condition == 3:
            # intra-component codelength computation
            for k,v in Params['compos'].items():
                codelength += getCodeLengthParallel( G, PD, gtype=self.gtype, case=5, isSimple=self.isSimple, NL=v.NL, dropLidx=[Lidx], nlambda=v.la )
            # inter-component codelength computation, i.e., one list of the component and other list of all the rest of the node
            keys = Params['compos'].keys()
            for k1 in range(len(keys)-1):
                for k2 in range(k1+1, len(keys)):
                    codelength += getCodeLengthParallel( G, PD, gtype='D', case=4, isSimple=self.isSimple, inNL=Params['compos'][k1].NL, outNL=Params['compos'][k2].NL, dropLidx=[Lidx] )
            # compute for excluded nodes
            if len(Params['excludedNL']) > 0:
                codelength += getCodeLengthParallel( G, PD, gtype=self.gtype, case=4, isSimple=self.isSimple, NL=Params['excludedNL'], dropLidx=[Lidx] )
                for k,v in Params['compos'].items():
                    codelength += getCodeLengthParallel( G, PD, gtype='D', case=4, isSimple=self.isSimple, inNL=v.NL, outNL=Params['excludedNL'], dropLidx=[Lidx] )
        return codelength

    def processAsD(self, G, PD, i):
        inNL = PD.lprevUpdate[i][1]
        outNL = PD.lprevUpdate[i][2]
        HD = getDirectedSubgraph( G, inNL, outNL, self.isSimple )
        H = nx.Graph(HD)
        components = nx.connected_component_subgraphs(H, copy=True)
        fcomponents = dict()
        it = 0
        for comp in components:
            if comp.number_of_nodes() > self.minsize:
                C_inNL = list(set(comp.nodes()).intersection(set(inNL)))
                C_outNL = list(set(comp.nodes()).intersection(set(outNL)))
                fcomponents[it] = getDirectedSubgraph(G, C_inNL, C_outNL, self.isSimple)
        if len(fcomponents) > 1: #* If components are more than one then only we can split this pattern
            baseParams = dict()
            baseParams['Pat'] = Pattern(HD)
            baseParams['inNodesInc'] = 0
            baseParams['outNodesInc'] = 0
            compPats = dict()
            in_nodes_union = set()
            for k,v in fcomponents.items():
                compPats[k] = Pattern(v)
                baseParams['inNodesInc'] += compPats[k].InNCount
                baseParams['outNodesInc'] += compPats[k].OutNCount
                in_nodes_union = in_nodes_union.union(set(compPats[k].inNL))
                out_nodes_union = out_nodes_union.union(set(compPats[k].outNL))
            baseParams['compos'] = compPats
            baseParams['excludedInNL'] = list( set(baseParams['Pat'].inNL) - in_nodes_union )
            baseParams['excludedOutNL'] = list( set(baseParams['Pat'].outNL) - out_nodes_union )
            baseParams['codeLengthC'] = getCodeLengthParallel( H, PD, gtype=self.gtype, case=2, isSimple=self.isSimple, inNL=baseParams['Pat'].inNL, outNL=baseParams['Pat'].outNL )
            baseParams['codeLengthCprime'] = self.computeCodeLengthSplitD(G, PD, 2, baseParams, i) #// Todo : write code for this part
            baseParams['Pat'].setIC_dssg( baseParams['codeLengthC'] - baseParams['codeLengthCprime'] )
            baseParams['Pat'].setDL( computeDescriptionLength( case=7, C=len(PD.lprevUpdate), gtype=self.gtype, WIS=baseParams['Pat'].InNCount, WOS=baseParams['Pat'].OutNCount, compos=baseParams['compos'] ) )
            baseParams['Pat'].setI( computeInterestingness( baseParams['Pat'].IC_dssg, baseParams['Pat'].DL, mode=2 ) )

            #now try reducing each component
            FinalParams = baseParams
            for k in baseParams['compos'].keys():
                FinalParams = self.getReducedComponentD(G, PD, FinalParams, i, k)

            #compute new lambdas for each new pattern/component
            #// Todo: write code
            for k,v in FinalParams['compos'].items():
                v.setLambda( PD.updateDistribution( pat=v.G, idx=None, val_retrun='return', case=3, dropLidx=[i]) )
            FinalParams['codeLengthCprime'] = self.computeCodeLengthSplitD(G, PD, 3, FinalParams, i) #// Todo : write code for this part
            FinalParams['Pat'].setIC_dssg( FinalParams['codeLengthC'] - FinalParams['codeLengthCprime'] )
            FinalParams['Pat'].setDL( computeDescriptionLength( case=7, C=len(PD.lprevUpdate), gtype=self.gtype, WIS=FinalParams['Pat'].InNCount, WOS=FinalParams['Pat'].OutNCount, compos=FinalParams['compos'], excActionType=False, l=self.l ) )
            FinalParams['Pat'].setI( computeInterestingness( FinalParams['Pat'].IC_dssg, FinalParams['Pat'].DL, mode=2 ) )
            # Now set these values to all component patterns
            #// Todo: Write Code
            for k,v in FinalParams['compos'].items():
                v.setIC_dssg( FinalParams['Pat'].IC_dssg )
                v.setDL( FinalParams['Pat'].DL )
                v.setI( FinalParams['Pat'].I )
                v.setPrevOrder(i)
            self.Data[i] = FinalParams
        return

    def getReducedComponentD(self, G, PD, FinalParams, Lid, k):
        doshrink = True
        count_remove_nodes = 0
        while doshrink:
            doshrink = False
            bestRNode = None
            bestCLprime = None
            bestDL = None
            bestI = FinalParams['Pat'].I
            bestType = None
            for node in FinalParams['compos'][k].inNL:
                curCLprime = FinalParams['codeLengthCprime'] - self.computeCLgainRemoveNodeD( G, PD, FinalParams['compos'][k].outNL, node, [Lid], 1 ) #// Todo: check this code
                curIC = FinalParams['codeLengthC'] - curCLprime
                curDL = FinalParams['Pat'].DL - self.getDescriptionLengthChangeD( FinalParams['compos'][k], node, FinalParams['Pat'].InNCount, FinalParams['inNodesInc'], 1 ) #// Todo: check this code
                curI =  computeInterestingness( curIC, curDL, mode=2 )
                if curI > bestI:
                    bestRNode = node
                    bestCLprime = curCLprime
                    bestDL = curDL
                    bestI = curI
                    bestType = 'in'
            for node in FinalParams['compos'][k].outNL:
                curCLprime = FinalParams['codeLengthCprime'] - self.computeCLgainRemoveNodeD( G, PD, FinalParams['compos'][k].inNL, node, [Lid], 2 ) #// Todo: check this code
                curIC = FinalParams['codeLengthC'] - curCLprime
                curDL = FinalParams['Pat'].DL - self.getDescriptionLengthChangeD( FinalParams['compos'][k], node, FinalParams['Pat'].OutNCount, FinalParams['outNodesInc'], 2 ) #// Todo: check this code
                curI =  computeInterestingness( curIC, curDL, mode=2 )
                if curI > bestI:
                    bestRNode = node
                    bestCLprime = curCLprime
                    bestDL = curDL
                    bestI = curI
                    bestType = 'out'
            if bestI > FinalParams['Pat'].I:
                FinalParams['codeLengthCprime'] = bestCLprime
                if 'in' in bestType:
                    FinalParams['compos'][k].removeInNode(bestRNode)
                    FinalParams['inNodesInc'] -= 1
                    FinalParams['excludedInNL'].append(bestRNode)
                else:
                    FinalParams['compos'][k].removeOutNode(bestRNode)
                    FinalParams['outNodesInc'] -= 1
                    FinalParams['excludedOutNL'].append(bestRNode)
                FinalParams['Pat'].setIC_dssg( FinalParams['codeLengthC'] - FinalParams['codeLengthCprime'] )
                FinalParams['Pat'].setDL( computeDescriptionLength( case=7, C=len(PD.lprevUpdate), gtype=self.gtype, WIS=FinalParams['Pat'].InNCount, WOS=FinalParams['Pat'].OutNCount, compos=FinalParams['compos'], excActionType=False, l=self.l ) )
                FinalParams['Pat'].setI( computeInterestingness( FinalParams['Pat'].IC_dssg, FinalParams['Pat'].DL, mode=2 ) )
                count_remove_nodes += 1
                doshrink = True

        return FinalParams

    def computeCLgainRemoveNodeD(self, G, PD, nodes, node, dropLidx, dir):
        CL_I = 0.0
        CL_F = 0.0
        if dir==1:
            for i in nodes:
                pos_I = PD.getPOS(node, i, case=2, isSimple=self.isSimple)
                pos_F = PD.getPOS(node, i, case=4, isSimple=self.isSimple, dropLidx=dropLidx)
                numE = G.number_of_edges(node, i)
                if self.isSimple:
                    CL_I += math.log2(math.pow(1.0-pos_I, 1.0-numE)*math.pow(pos_I, numE))
                    CL_F += math.log2(math.pow(1.0-pos_F, 1.0-numE)*math.pow(pos_F, numE))
                else:
                    CL_I += math.log2(math.pow(1.0-pos_I, numE)*pos_I)
                    CL_F += math.log2(math.pow(1.0-pos_F, numE)*pos_F)
        elif dir==2:
            for i in nodes:
                pos_I = PD.getPOS(i, node, case=2, isSimple=self.isSimple)
                pos_F = PD.getPOS(i, node, case=4, isSimple=self.isSimple, dropLidx=dropLidx)
                numE = G.number_of_edges(i, node)
                if self.isSimple:
                    CL_I += math.log2(math.pow(1.0-pos_I, 1.0-numE)*math.pow(pos_I, numE))
                    CL_F += math.log2(math.pow(1.0-pos_F, 1.0-numE)*math.pow(pos_F, numE))
                else:
                    CL_I += math.log2( math.pow( 1.0-pos_I, numE )*pos_I )
                    CL_F += math.log2( math.pow( 1.0-pos_F, numE )*pos_F )
        return -CL_I - (-CL_F)

    def getDescriptionLengthChangeD(self, Pat, node, WI, W, tp):
        change = 0.0
        if tp == 1:
            change += LN(Pat.InNCount) - LN(Pat.InNCount - 1)
            change += ncr(WI, W) - ncr(WI, W-1)
            pDLEdge = DL_Edges(Pat.nw, Pat.ECount)
            nECount = Pat.ECount
            for k in Pat.outNL:
                nECount -= Pat.G.number_of_edges(k, node)
            nDLEdge = DL_Edges(Pat.nw-Pat.OutNCount, nECount)
            change += pDLEdge - nDLEdge
        else:
            change += LN(Pat.OutNCount) - LN(Pat.OutNCount - 1)
            change += ncr(WI, W) - ncr(WI, W-1)
            pDLEdge = DL_Edges(Pat.nw, Pat.ECount)
            nECount = Pat.ECount
            for k in Pat.inNL:
                nECount -= Pat.G.number_of_edges(node, k)
            nDLEdge = DL_Edges(Pat.nw-Pat.InNCount, nECount)
            change += pDLEdge - nDLEdge
        return change

    def computeCodeLengthSplitD(self, G, PD, condition, Params, Lidx=None):
        codeLength = 0.0
        if condition == 1:
            codelength = getCodeLengthParallel( G, PD, gtype=self.gtype, case=2, isSimple=self.isSimple, inNL=Params['Pat'].inNL, outNL=Params['Pat'].outNL )
            return codelength
        elif condition == 2:
            # intra-component codelength computation
            for k,v in Params['compos'].items():
                codelength += getCodeLengthParallel( G, PD, gtype=self.gtype, case=2, isSimple=self.isSimple, inNL=v.inNL, outNL=v.outNL )
            # inter-component codelength computation, i.e., one list of the component and other list of all the rest of the node
            keys = Params['compos'].keys()
            for k1 in range(len(keys)-1):
                for k2 in range(k1+1, len(keys)):
                    codelength += getCodeLengthParallel( G, PD, gtype=self.gtype, case=4, isSimple=self.isSimple, inNL=Params['compos'][k1].inNL, outNL=Params['compos'][k2].outNL, dropLidx=[Lidx] )
                    codelength += getCodeLengthParallel( G, PD, gtype=self.gtype, case=4, isSimple=self.isSimple, inNL=Params['compos'][k1].outNL, outNL=Params['compos'][k2].inNL, dropLidx=[Lidx] )
            # compute for excluded nodes
            if len(Params['excludedInNL']) > 0:
                for k,v in Params['compos'].items():
                    codelength += getCodeLengthParallel( G, PD, gtype=self.gtype, case=4, isSimple=self.isSimple, inNL=Params['excludedInNL'], outNL=v.outNL, dropLidx=[Lidx] )
            if len(Params['excludedOutNL']) > 0:
                for k,v in Params['compos'].items():
                    codelength += getCodeLengthParallel( G, PD, gtype=self.gtype, case=4, isSimple=self.isSimple, inNL=v.inNL, outNL=Params['excludedOutNL'], dropLidx=[Lidx] )
            if len(Params['excludedInNL']) > 0 and len(Params['excludedOutNL']) > 0:
                codelength += getCodeLengthParallel( G, PD, gtype=self.gtype, case=4, isSimple=self.isSimple, inNL=Params['excludedInNL'], outNL=Params['excludedOutNL'], dropLidx=[Lidx] )
        elif condition == 3:
            # intra-component codelength computation
            for k,v in Params['compos'].items():
                codelength += getCodeLengthParallel( G, PD, gtype=self.gtype, case=5, isSimple=self.isSimple, inNL=v.inNL, outNL=v.outNL, dropLidx=[Lidx], nlambda=v.la )
            # inter-component codelength computation, i.e., one list of the component and other list of all the rest of the node
            keys = Params['compos'].keys()
            for k1 in range(len(keys)-1):
                for k2 in range(k1+1, len(keys)):
                    codelength += getCodeLengthParallel( G, PD, gtype=self.gtype, case=4, isSimple=self.isSimple, inNL=Params['compos'][k1].inNL, outNL=Params['compos'][k2].outNL, dropLidx=[Lidx] )
                    codelength += getCodeLengthParallel( G, PD, gtype=self.gtype, case=4, isSimple=self.isSimple, inNL=Params['compos'][k1].outNL, outNL=Params['compos'][k2].inNL, dropLidx=[Lidx] )
            # compute for excluded nodes
            if len(Params['excludedInNL']) > 0:
                for k,v in Params['compos'].items():
                    codelength += getCodeLengthParallel( G, PD, gtype=self.gtype, case=4, isSimple=self.isSimple, inNL=Params['excludedInNL'], outNL=v.outNL, dropLidx=[Lidx] )
            if len(Params['excludedOutNL']) > 0:
                for k,v in Params['compos'].items():
                    codelength += getCodeLengthParallel( G, PD, gtype=self.gtype, case=4, isSimple=self.isSimple, inNL=v.inNL, outNL=Params['excludedOutNL'], dropLidx=[Lidx] )
            if len(Params['excludedInNL']) > 0 and len(Params['excludedOutNL']) > 0:
                codelength += getCodeLengthParallel( G, PD, gtype=self.gtype, case=4, isSimple=self.isSimple, inNL=Params['excludedInNL'], outNL=Params['excludedOutNL'], dropLidx=[Lidx] )
        return codelength