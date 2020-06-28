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
from src.Patterns.Pattern import Pattern

class EvaluateShrink:
    # Now this data structure should contain all the possible removals
    # along with pattern number as key and Information Gain and list of nodes as value
    def __init__(self, gtype='U', isSimple=True, l=6, minsize=2):
        self.Data = dict()
        self.gtype = gtype
        self.isSimple = isSimple
        self.l = l # possible types (give number) of action, default is 6
        self.minsize = minsize
        print('initialized EvaluateShrink')

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

        if len(fcomponents) == 1: # * if valid components is more than 1 than split shall be performed
            baseParams = dict()
            baseParams['Pat'] = Pattern(H)
            baseParams['codeLengthC'] = getCodeLengthParallel( H, PD, gtype=self.gtype, case=2, isSimple=self.isSimple, NL=baseParams['Pat'].NL )
            baseParams['codeLengthCprime'] = baseParams['codeLengthC']
            baseParams['Pat'].setIC_dssg( baseParams['codeLengthC'] - baseParams['codeLengthCprime'] )
            baseParams['Pat'].setDL( computeDescriptionLength( case=6, C=len(PD.lprevUpdate), gtype=self.gtype, WS=baseParams['Pat'].NCount, W=baseParams['Pat'].NCount, kw=baseParams['Pat'].ECount ))
            baseParams['Pat'].setI( computeInterestingness( baseParams['Pat'].IC_dssg, baseParams['Pat'].DL, mode=2 ) )

            curPat = fcomponents[0]

            bestParams = None
            if curPat.number_of_nodes < baseParams['Pat'].NCount:
                bestParams = dict()
                bestParams['Pat'] = Pattern(curPat)
                bestParams['codeLengthCprime']  = computeCodeLengthShrinkU(self, G, PD, 2, baseParams, bestParams, i)
                bestParams['Pat'].setIC_dssg( baseParams['codeLengthC'] - bestParams['codeLengthCprime'] )
                bestParams['Pat'].setDL( computeDescriptionLength( case=6, C=len(PD.lprevUpdate), gtype=self.gtype, WS=baseParams['Pat'].NCount, W=bestParams['Pat'].NCount, kw=bestParams['Pat'].ECount ) )
                bestParams['Pat'].setI( computeInterestingness( bestParams['Pat'].IC_dssg, bestParams['Pat'].DL, mode=2 ) )
            else:
                bestParams = baseParams

            # * Now reduce the only component in fcomponents
            FinalParams = self.getReducedSubgraphU(G, PD, baseParams, bestParams, i)
            if bestParams['Pat'].I > FinalParams['Pat'].I:
                self.Data[i] = FinalParams['Pat']

    def getReducedSubgraphU(self, G, PD, baseParams, bestParams, Lid):
        """[summary]

        Args:
            G (networkx graph): [description]
            PD (PDClass): [description]
            NL (list): [description]
            baseParams (dict): [description]
            bestParams (dict): [description]
        """
        doshrink = True
        count_remove_nodes = 0
        FinalParams = bestParams
        while doshrink: #continue removing nodes one by one till no increase in IG
            doshrink = False
            for node in FinalParams['Pat'].NL:
                curParams = dict()
                curParams['Pat'] = FinalParams['Pat'].copy()
                curParams['Pat'].removeNode(node)
                curParams['codeLengthCprime'] = FinalParams['codeLengthCprime'] - self.computeCLgainRemoveNodeU(G, PD, curParams['Pat'].NL, node, [Lid])
                curParams['Pat'].setIC_dssg( baseParams['codeLengthC'] - curParams['codeLengthCprime'] )
                curParams['Pat'].setDL( computeDescriptionLength( case=6, C=len(PD.lprevUpdate), gtype=self.gtype, WS=baseParams['Pat'].NCount, W=curParams['Pat'].NCount, kw=curParams['Pat'].ECount ) )
                curParams['Pat'].setI( computeInterestingness( curParams['Pat'].IC_dssg, curParams['Pat'].DL, mode=2 ) )
                if curParams['Pat'].I > bestParams['Pat'].I:
                    bestParams = curParams
            if bestParams['Pat'].I > FinalParams['Pat'].I:
                FinalParams = bestParams
                count_remove_nodes += 1
                doshrink = True
        if count_remove_nodes > 0 or FinalParams['Pat'].NCount < baseParams['Pat'].NCount:
            FinalParams['codeLengthC'] = baseParams['codeLengthC']
            FinalParams['Pat'].setLambda( PD.updateDistribution( pat=FinalParams['Pat'].G, idx=None, val_retrun='return', case=3, dropLidx=[Lid]) ) #// Todo: computeNewLambda
            FinalParams['codeLengthCprime'] = computeCodeLengthShrinkU(self, G, PD, 3, baseParams, FinalParams, Lid, FinalParams['Pat'].la) #// Todo computeNewCodeLength
            FinalParams['Pat'].setIC_dssg( FinalParams['codeLengthC'] - FinalParams['codeLengthCprime'] )
            FinalParams['Pat'].setDL( computeDescriptionLength( case=6, C=len(PD.lprevUpdate), gtype=self.gtype, WS=baseParams['Pat'].NCount, W=FinalParams['Pat'].NCount, kw=FinalParams['Pat'].ECount, excActionType=False, l=6 ) )
            FinalParams['Pat'].setI( computeInterestingness( FinalParams['Pat'].IC_dssg, FinalParams['Pat'].DL, mode=2 ) )
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

        if len(fcomponents) == 1: # * if valid components is more than 1 than split shall be performed
            curPat = fcomponents[0]
            baseParams = dict()
            baseParams['Pat'] = Pattern(H)
            baseParams['codeLengthC'] = getCodeLengthParallel( H, PD, gtype=self.gtype, case=2, isSimple=self.isSimple, inNL=baseParams['Pat'].inNL, outNL=baseParams['Pat'].outNL )
            baseParams['codeLengthCprime'] = baseParams['codeLengthC']
            baseParams['Pat'].setIC_dssg( baseParams['codeLengthC'] - baseParams['codeLengthCprime'] )
            baseParams['Pat'].setDL( computeDescriptionLength( case=6, C=len(PD.lprevUpdate), gtype=self.gtype, WIS=baseParams['Pat'].InNCount, WOS=baseParams['Pat'].OutNCount, WI=baseParams['Pat'].InNL, WO=baseParams['Pat'].OutNL, kw=baseParams['Pat'].ECount ))
            baseParams['Pat'].setI( computeInterestingness( baseParams['Pat'].IC_dssg, baseParams['Pat'].DL, mode=2 ) )

            bestParams = None
            if curPat.number_of_nodes < baseParams['Pat'].NCount:
                bestParams = dict()
                bestParams['Pat'] = Pattern(curPat)
                bestParams['codeLengthCprime'] = self.computeCodeLengthShrinkD( G, PD, 2, baseParams, bestParams, i )
                bestParams['Pat'].setIC_dssg( baseParams['codeLengthC'] - bestParams['codeLengthCprime'] )
                bestParams['Pat'].setDL( computeDescriptionLength( case=6, C=len(PD.lprevUpdate), gtype=self.gtype, WIS=baseParams['Pat'].InNCount, WOS=baseParams['Pat'].OutNCount, WI=bestParams['Pat'].InNL, WO=bestParams['Pat'].OutNL, kw=bestParams['Pat'].ECount ) )
                bestParams['Pat'].setI( computeInterestingness( bestParams['Pat'].IC_dssg, bestParams['Pat'].DL, mode=2 ) )
            else:
                bestParams = baseParams

            # * Now reduce the only component in fcomponents
            FinalParams = self.getReducedSubgraphD(G, PD, baseParams, bestParams, i)
            if bestParams['Pat'].I > FinalParams['Pat'].I:
                self.Data[i] = FinalParams['Pat']
        return

    def getReducedSubgraphD(self, G, PD, baseParams, bestParams, Lid):
        """[summary]

        Args:
            G (networkx graph): [description]
            PD (PDClass): [description]
            NL (list): [description]
            baseParams (dict): [description]
            bestParams (dict): [description]
        """
        doshrink = True
        count_remove_nodes = 0
        FinalParams = bestParams
        while doshrink: #continue removing nodes one by one till no increase in IG
            doshrink = False
            for node in FinalParams['Pat'].inNL:
                curParams = dict()
                curParams['Pat'] = FinalParams['Pat'].copy()
                curParams['Pat'].removeInNode(node)
                curParams['codeLengthCprime'] = FinalParams['codeLengthCprime'] - self.computeCLgainRemoveNodeD(G, PD, curParams['Pat'].outNL, node, [Lid], 1)
                curParams['Pat'].setIC_dssg( baseParams['codeLengthC'] - curParams['codeLengthCprime'] )
                curParams['Pat'].setDL( computeDescriptionLength( case=6, C=len(PD.lprevUpdate), gtype=self.gtype, WIS=baseParams['Pat'].InNCount, WOS=baseParams['Pat'].OutNCount, WI=curParams['Pat'].InNL, WO=curParams['Pat'].OutNL, kw=curParams['Pat'].ECount ) )
                curParams['Pat'].setI( computeInterestingness( curParams['Pat'].IC_dssg, curParams['Pat'].DL, mode=2 ) )
                if curParams['Pat'].I > bestParams['Pat'].I:
                    bestParams = curParams
            for node in FinalParams['Pat'].outNL:
                curParams = dict()
                curParams['Pat'] = FinalParams['Pat'].copy()
                curParams['Pat'].removeOutNode(node)
                curParams['codeLengthCprime'] = FinalParams['codeLengthCprime'] - self.computeCLgainRemoveNodeD(G, PD, curParams['Pat'].inNL, node, [Lid], 2)
                curParams['Pat'].setIC_dssg( baseParams['codeLengthC'] - curParams['codeLengthCprime'] )
                curParams['Pat'].setDL( computeDescriptionLength( case=6, C=len(PD.lprevUpdate), gtype=self.gtype, WIS=baseParams['Pat'].InNCount, WOS=baseParams['Pat'].OutNCount, WI=curParams['Pat'].InNL, WO=curParams['Pat'].OutNL, kw=curParams['Pat'].ECount ) )
                curParams['Pat'].setI( computeInterestingness( curParams['Pat'].IC_dssg, curParams['Pat'].DL, mode=2 ) )
                if curParams['Pat'].I > bestParams['Pat'].I:
                    bestParams = curParams
            if bestParams['Pat'].I > FinalParams['Pat'].I:
                FinalParams = bestParams
                count_remove_nodes += 1
                doshrink = True
        if count_remove_nodes > 0 or (FinalParams['Pat'].InNCount < baseParams['Pat'].InNCount and FinalParams['Pat'].OutNCount < baseParams['Pat'].OutNCount) :
            FinalParams['codeLengthC'] = baseParams['codeLengthC']
            FinalParams['Pat'].setLambda( PD.updateDistribution( FinalParams['Pat'].G, idx=None, val_retrun='return', case=3, dropLidx=[Lid]) ) #// Todo: computeNewLambda
            FinalParams['codeLengthCprime'] = self.computeCodeLengthShrinkD( G, PD, 3, baseParams, FinalParams, Lid, FinalParams['Pat'].la) #// Todo computeNewCodeLength
            FinalParams['Pat'].setIC_dssg( FinalParams['codeLengthC'] - FinalParams['codeLengthCprime'] )
            FinalParams['Pat'].setDL( computeDescriptionLength( case=6, C=len(PD.lprevUpdate), gtype=self.gtype, WS=baseParams['Pat'].NCount, W=FinalParams['Pat'].NCount, kw=FinalParams['Pat'].ECount, excActionType=False, l=6 ) )
            FinalParams['Pat'].setI( computeInterestingness( FinalParams['Pat'].IC_dssg, FinalParams['Pat'].DL, mode=2 ) )
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

    def computeCodeLengthShrinkU(self, G, PD, condition, baseParams, curParams=None, Lidx=None, nlambda=None):
        codeLength = 0.0
        if condition == 1:
            codelength = getCodeLengthParallel( G, PD, gtype=self.gtype, case=2, isSimple=self.isSimple, NL=baseParams['Pat'].NL )
            return codelength
        elif condition == 2:
            codelength += getCodeLengthParallel( G, PD, gtype=self.gtype, case=2, isSimple=self.isSimple, NL=curParams['Pat'].NL )
            nodesDropped = list( set( baseParams['Pat'].NL ) - set( curParams['Pat'].NL ) )
            codelength += getCodeLengthParallel( G, PD, gtype=self.gtype, case=4, isSimple=self.isSimple, NL=nodesDropped, dropLidx=[Lidx] )
            codelength += getCodeLengthParallel( G, PD, gtype='D', case=4, isSimple=self.isSimple, inNL=curParams['Pat'].NL, outNL=nodesDropped, dropLidx=[Lidx] )
        elif condition == 3:
            codelength += getCodeLengthParallel( G, PD, gtype=self.gtype, case=5, isSimple=self.isSimple, NL=curParams['Pat'].NL, dropLidx=[Lidx], nlambda=nlambda )
            nodesDropped = list( set( baseParams['Pat'].NL ) - set( curParams['Pat'].NL ) )
            codelength += getCodeLengthParallel( G, PD, gtype=self.gtype, case=4, isSimple=self.isSimple, NL=nodesDropped, dropLidx=[Lidx] )
            codelength += getCodeLengthParallel( G, PD, gtype='D', case=4, isSimple=self.isSimple, inNL=curParams['Pat'].NL, outNL=nodesDropped, dropLidx=[Lidx] )
        return codelength

    def computeCodeLengthShrinkD(self, G, PD, condition, baseParams, curParams=None, Lidx=None, nlambda=None):
        codeLength = 0.0
        if condition == 1:
            codelength = getCodeLengthParallel( G, PD, gtype=self.gtype, case=2, isSimple=self.isSimple, inNL=baseParams['Pat'].inNL, outNL=baseParams['Pat'].outNL )
            return codelength
        elif condition == 2:
            codelength += getCodeLengthParallel( G, PD, gtype=self.gtype, case=2, isSimple=self.isSimple, inNL=curParams['Pat'].inNL, outNL=curParams['Pat'].outNL )
            inNodesDropped = list(set(baseParams['Pat'].inNL) - set(curParams['Pat'].inNL))
            outNodesDropped = list(set(baseParams['Pat'].outNL) - set(curParams['Pat'].outNL)) # * left here .. # Todo start code from here
            codelength += getCodeLengthParallel( G, PD, gtype=self.gtype, case=4, isSimple=self.isSimple, inNL=inNodesDropped, outNL=outNodesDropped, dropLidx=[Lidx] )
            codelength += getCodeLengthParallel( G, PD, gtype=self.gtype, case=4, isSimple=self.isSimple, inNL=curParams['Pat'].inNL, outNL=outNodesDropped, dropLidx=[Lidx] )
            codelength += getCodeLengthParallel( G, PD, gtype=self.gtype, case=4, isSimple=self.isSimple, inNL=inNodesDropped, outNL=curParams['Pat'].outNL, dropLidx=[Lidx] )
        elif condition == 3:
            codelength += getCodeLengthParallel( G, PD, gtype=self.gtype, case=5, isSimple=self.isSimple, inNL=curParams['Pat'].inNL, outNL=curParams['Pat'].outNL, dropLidx=[Lidx], nlambda=nlambda )
            inNodesDropped = list(set(baseParams['Pat'].inNL) - set(curParams['Pat'].inNL))
            outNodesDropped = list(set(baseParams['Pat'].outNL) - set(curParams['Pat'].outNL)) # * left here .. # Todo start code from here
            codelength += getCodeLengthParallel( G, PD, gtype=self.gtype, case=4, isSimple=self.isSimple, inNL=inNodesDropped, outNL=outNodesDropped, dropLidx=[Lidx] )
            codelength += getCodeLengthParallel( G, PD, gtype=self.gtype, case=4, isSimple=self.isSimple, inNL=curParams['Pat'].inNL, outNL=outNodesDropped, dropLidx=[Lidx] )
            codelength += getCodeLengthParallel( G, PD, gtype=self.gtype, case=4, isSimple=self.isSimple, inNL=inNodesDropped, outNL=curParams['Pat'].outNL, dropLidx=[Lidx] )
        return codelength





        # Todo:
            #// 1: Write code for compute code length
            #// 2: Write code to compute new lambda
            #// 3: Write code for split
            #* 4: Debug
            #// 5: Finalize Measure file (a) write encode edge for multigraphs, (b) check description length code for split action
            #* 6: Experiments for airline case study
            #* 7: In all Actions get best option code needs to written
            #* 8: When to update possibilities in each action needs to be coded