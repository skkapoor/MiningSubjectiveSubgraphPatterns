###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################
import os
import sys
path = os.getcwd().split('MiningSubjectiveSubgraphPatterns')[0]+'MiningSubjectiveSubgraphPatterns/'
if path not in sys.path:
    sys.path.append(path)
import math
import networkx as nx

from src.Utils.Measures import getCodeLength, getCodeLengthParallel, getDirectedSubgraph
from src.Utils.Measures import computeDescriptionLength, computeInterestingness
from src.Patterns.Pattern import Pattern
###################################################################################################################################################################
class EvaluateShrink:
    """
    This data structure shall contain all the possible shrinks
    along with pattern number as key and other parameters as value
    """
    def __init__(self, gtype='U', isSimple=True, l=6, imode=2, minsize=2):
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
        minsize : int, optional
            Minimum size of pattern, by default 2
        """
        self.Data = dict()
        self.gtype = gtype
        self.isSimple = isSimple
        self.l = l # possible types (give number) of action, default is 6
        self.minsize = minsize
        self.imode = imode
        print('initialized EvaluateShrink')
###################################################################################################################################################################
    def evaluateAllConstraints(self, G, PD):
        """
        function to evaluate all constraints and make a list of candidate constraints which are feasible to shrink

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
        function to evaluate if a constraint is a feasible candidate for shrink

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
            self.processAsU(G, PD, id)
        elif self.gtype == 'D':
            self.processAsD(G, PD, id)
        return
###################################################################################################################################################################
    def processAsU(self, G, PD, id):
        """
        Utility function for shrink action when the input graph is undirected.
        This function idenfies the final subgraph from a possible candidate shrink and compute the corresponding measures.

        Parameters
        ----------
        G : Networkx Graph
            Input Graph
        PD : PDClass
            Background Distribution
        id : int
            identifier of a constraint to be evaluated
        """
        NL = PD.lprevUpdate[id][1]
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
            baseParams['Pat'].setDL( computeDescriptionLength( dlmode=6, C=len(PD.lprevUpdate), gtype=self.gtype, WS=baseParams['Pat'].NCount, W=baseParams['Pat'].NCount, kw=baseParams['Pat'].ECount, isSimple=self.isSimple, kws=baseParams['Pat'].kws ))
            baseParams['Pat'].setI( computeInterestingness( baseParams['Pat'].IC_dssg, baseParams['Pat'].DL, mode=self.imode ) )

            curPat = fcomponents[0]

            bestParams = None
            if curPat.number_of_nodes() < baseParams['Pat'].NCount:
                bestParams = dict()
                bestParams['Pat'] = Pattern(curPat)
                bestParams['codeLengthCprime']  = self.computeCodeLengthShrinkU(G, PD, 2, baseParams, bestParams, id)
                bestParams['Pat'].setIC_dssg( baseParams['codeLengthC'] - bestParams['codeLengthCprime'] )
                bestParams['Pat'].setDL( computeDescriptionLength( dlmode=6, C=len(PD.lprevUpdate), gtype=self.gtype, WS=baseParams['Pat'].NCount, W=bestParams['Pat'].NCount, kw=bestParams['Pat'].ECount, isSimple=self.isSimple, kws=bestParams['Pat'].kws ) )
                bestParams['Pat'].setI( computeInterestingness( bestParams['Pat'].IC_dssg, bestParams['Pat'].DL, mode=self.imode ) )
            else:
                bestParams = baseParams

            # * Now reduce the only component in fcomponents
            FinalParams = self.getReducedSubgraphU(G, PD, baseParams, bestParams, id)
            FinalParams['SPat'] = FinalParams['Pat'].copy()
            FinalParams['Pat'] = baseParams['Pat'].copy()
            if bestParams['Pat'].I > FinalParams['SPat'].I:
                FinalParams['Pat'].setPrevOrder(id)
                FinalParams['Pat'].setPatType('shrink')
                FinalParams['SPat'].setPrevOrder(id)
                FinalParams['SPat'].setPatType('shrink')
                self.Data[id] = FinalParams
        return
###################################################################################################################################################################
    def getReducedSubgraphU(self, G, PD, baseParams, bestParams, Lid):
        """
        Utility function used when the input graph is undirected to remove nodes from subgrapht of the candidate sphrink

        Parameters
        ----------
        G : Networkx Graph
            Input Graph
        PD : PDClass
            Background Distribution
        baseParams : dict
            base value of prameters corresponding to the current shrink candidate, i.e., before shrink
        bestParams : dict
            current value of prameters corresponding to the current shrink candidate, i.e., after removing some disconnected nodes (if any)
        Lid : int
            identifier for the the candidate constraint for split

        Returns
        -------
        dict
            FinalParams: Updated after reducing a component
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
                curParams['Pat'].setDL( computeDescriptionLength( dlmode=6, C=len(PD.lprevUpdate), gtype=self.gtype, WS=baseParams['Pat'].NCount, W=curParams['Pat'].NCount, kw=curParams['Pat'].ECount, isSimple=self.isSimple, kws=curParams['Pat'].kws ) )
                curParams['Pat'].setI( computeInterestingness( curParams['Pat'].IC_dssg, curParams['Pat'].DL, mode=self.imode ) )
                if curParams['Pat'].I > bestParams['Pat'].I:
                    bestParams = curParams
            if bestParams['Pat'].I > FinalParams['Pat'].I:
                FinalParams = bestParams
                count_remove_nodes += 1
                doshrink = True
        if count_remove_nodes > 0 or FinalParams['Pat'].NCount < baseParams['Pat'].NCount:
            FinalParams['codeLengthC'] = baseParams['codeLengthC']
            FinalParams['Pat'].setLambda( PD.updateDistribution( pat=FinalParams['Pat'].G, idx=None, val_return='return', case=3, dropLidx=[Lid]) ) #// Todo: computeNewLambda
            FinalParams['codeLengthCprime'] = self.computeCodeLengthShrinkU(G, PD, 3, baseParams, FinalParams, Lid, FinalParams['Pat'].la) #// Todo computeNewCodeLength
            FinalParams['Pat'].setIC_dssg( FinalParams['codeLengthC'] - FinalParams['codeLengthCprime'] )
            FinalParams['Pat'].setDL( computeDescriptionLength( dlmode=6, C=len(PD.lprevUpdate), gtype=self.gtype, WS=baseParams['Pat'].NCount, W=FinalParams['Pat'].NCount, kw=FinalParams['Pat'].ECount, excActionType=False, l=self.l, isSimple=self.isSimple, kws=FinalParams['Pat'].kws ) )
            FinalParams['Pat'].setI( computeInterestingness( FinalParams['Pat'].IC_dssg, FinalParams['Pat'].DL, mode=self.imode ) )
        return FinalParams
###################################################################################################################################################################
    def computeCLgainRemoveNodeU(self, G, PD, nodes, node, dropLidx):
        """
        Utility function to compute the gain/change in codelength by removing a node from a pattern.
        This function is used if the input graph is undirected.

        Parameters
        ----------
        G : Networkx Graph
            Input Graph
        PD : PDClass
            Background Distribution
        nodes : list
            list of nodes
        node : int
            node to be removed
        dropLidx : int
            identifier of the constarint which is evaluated

        Returns
        -------
        float
            gain/change in codelength
        """
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
###################################################################################################################################################################
    def processAsD(self, G, PD, id):
        """
        Utility function for shrink action when the input graph is directed.
        This function idenfies the final subgraph from a possible candidate shrink and compute the corresponding measures.

        Parameters
        ----------
        G : Networkx Graph
            Input Graph
        PD : PDClass
            Background Distribution
        id : int
            identifier of a constraint to be evaluated
        """
        inNL = PD.lprevUpdate[id][1]
        outNL = PD.lprevUpdate[id][2]
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
            baseParams['Pat'].setDL( computeDescriptionLength( dlmode=6, C=len(PD.lprevUpdate), gtype=self.gtype, WIS=baseParams['Pat'].InNCount, WOS=baseParams['Pat'].OutNCount, WI=baseParams['Pat'].InNL, WO=baseParams['Pat'].OutNL, kw=baseParams['Pat'].ECount, isSimple=self.isSimple, kws=baseParams['Pat'].kws ))
            baseParams['Pat'].setI( computeInterestingness( baseParams['Pat'].IC_dssg, baseParams['Pat'].DL, mode=self.imode ) )

            bestParams = None
            if curPat.number_of_nodes() < baseParams['Pat'].NCount:
                bestParams = dict()
                bestParams['Pat'] = Pattern(curPat)
                bestParams['codeLengthCprime'] = self.computeCodeLengthShrinkD( G, PD, 2, baseParams, bestParams, id )
                bestParams['Pat'].setIC_dssg( baseParams['codeLengthC'] - bestParams['codeLengthCprime'] )
                bestParams['Pat'].setDL( computeDescriptionLength( dlmode=6, C=len(PD.lprevUpdate), gtype=self.gtype, WIS=baseParams['Pat'].InNCount, WOS=baseParams['Pat'].OutNCount, WI=bestParams['Pat'].InNL, WO=bestParams['Pat'].OutNL, kw=bestParams['Pat'].ECount, isSimple=self.isSimple, kws=bestParams['Pat'].kws ) )
                bestParams['Pat'].setI( computeInterestingness( bestParams['Pat'].IC_dssg, bestParams['Pat'].DL, mode=self.imode ) )
            else:
                bestParams = baseParams

            # * Now reduce the only component in fcomponents
            FinalParams = self.getReducedSubgraphD(G, PD, baseParams, bestParams, id)
            FinalParams['SPat'] = FinalParams['Pat'].copy()
            FinalParams['Pat'] = baseParams['Pat'].copy()
            if bestParams['Pat'].I > FinalParams['SPat'].I:
                FinalParams['Pat'].setPrevOrder(id)
                FinalParams['Pat'].setPatType('shrink')
                FinalParams['SPat'].setPrevOrder(id)
                FinalParams['SPat'].setPatType('shrink')
                self.Data[id] = FinalParams
        return
###################################################################################################################################################################
    def getReducedSubgraphD(self, G, PD, baseParams, bestParams, Lid):
        """
        Utility function used when the input graph is directed to remove nodes from subgraph of the candidate shrink

        Parameters
        ----------
        G : Networkx Graph
            Input Graph
        PD : PDClass
            Background Distribution
        baseParams : dict
            base value of prameters corresponding to the current shrink candidate, i.e., before shrink
        bestParams : dict
            current value of prameters corresponding to the current shrink candidate, i.e., after removing some disconnected nodes (if any)
        Lid : int
            identifier for the the candidate constraint for split

        Returns
        -------
        dict
            FinalParams: Updated after reducing a subgraph
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
                curParams['Pat'].setDL( computeDescriptionLength( dlmode=6, C=len(PD.lprevUpdate), gtype=self.gtype, WIS=baseParams['Pat'].InNCount, WOS=baseParams['Pat'].OutNCount, WI=curParams['Pat'].InNL, WO=curParams['Pat'].OutNL, kw=curParams['Pat'].ECount, isSimple=self.isSimple, kws=curParams['Pat'].kws ) )
                curParams['Pat'].setI( computeInterestingness( curParams['Pat'].IC_dssg, curParams['Pat'].DL, mode=self.imode ) )
                if curParams['Pat'].I > bestParams['Pat'].I:
                    bestParams = curParams
            for node in FinalParams['Pat'].outNL:
                curParams = dict()
                curParams['Pat'] = FinalParams['Pat'].copy()
                curParams['Pat'].removeOutNode(node)
                curParams['codeLengthCprime'] = FinalParams['codeLengthCprime'] - self.computeCLgainRemoveNodeD(G, PD, curParams['Pat'].inNL, node, [Lid], 2)
                curParams['Pat'].setIC_dssg( baseParams['codeLengthC'] - curParams['codeLengthCprime'] )
                curParams['Pat'].setDL( computeDescriptionLength( dlmode=6, C=len(PD.lprevUpdate), gtype=self.gtype, WIS=baseParams['Pat'].InNCount, WOS=baseParams['Pat'].OutNCount, WI=curParams['Pat'].InNL, WO=curParams['Pat'].OutNL, kw=curParams['Pat'].ECount, isSimple=self.isSimple, kws=curParams['Pat'].kws ) )
                curParams['Pat'].setI( computeInterestingness( curParams['Pat'].IC_dssg, curParams['Pat'].DL, mode=self.imode ) )
                if curParams['Pat'].I > bestParams['Pat'].I:
                    bestParams = curParams
            if bestParams['Pat'].I > FinalParams['Pat'].I:
                FinalParams = bestParams
                count_remove_nodes += 1
                doshrink = True
        if count_remove_nodes > 0 or (FinalParams['Pat'].InNCount < baseParams['Pat'].InNCount and FinalParams['Pat'].OutNCount < baseParams['Pat'].OutNCount) :
            FinalParams['codeLengthC'] = baseParams['codeLengthC']
            FinalParams['Pat'].setLambda( PD.updateDistribution( FinalParams['Pat'].G, idx=None, val_return='return', case=3, dropLidx=[Lid]) ) #// Todo: computeNewLambda
            FinalParams['codeLengthCprime'] = self.computeCodeLengthShrinkD( G, PD, 3, baseParams, FinalParams, Lid, FinalParams['Pat'].la) #// Todo computeNewCodeLength
            FinalParams['Pat'].setIC_dssg( FinalParams['codeLengthC'] - FinalParams['codeLengthCprime'] )
            FinalParams['Pat'].setDL( computeDescriptionLength( dlmode=6, C=len(PD.lprevUpdate), gtype=self.gtype, WIS=baseParams['Pat'].InNCount, WOS=baseParams['Pat'].OutNCount, WI=FinalParams['Pat'].InNL, WO=FinalParams['Pat'].OutNL, kw=FinalParams['Pat'].ECount, excActionType=False, l=self.l, isSimple=self.isSimple, kws=FinalParams['Pat'].kws ) )
            FinalParams['Pat'].setI( computeInterestingness( FinalParams['Pat'].IC_dssg, FinalParams['Pat'].DL, mode=self.imode ) )
        return FinalParams
###################################################################################################################################################################
    def computeCLgainRemoveNodeD(self, G, PD, nodes, node, dropLidx, dir):
        """
        Utility function to compute the gain/change in codelength by removing a node from a pattern.
        This function is used if the input graph is directed.

        Parameters
        ----------
        G : Networkx Graph
            Input Graph
        PD : PDClass
            Background Distribution
        nodes : list
            list of nodes
        node : int
            node to be removed
        dropLidx : int
            identifier of the constarint which is evaluated
        dir : int
            dir direction, 1: list to node, i.e., node is inNode
                           2: node to list, i.e., node is outNode

        Returns
        -------
        float
            gain/change in codelength
        """
        CL_I = 0.0
        CL_F = 0.0
        if dir==2:
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
        elif dir==1:
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
###################################################################################################################################################################
    def computeCodeLengthShrinkU(self, G, PD, condition, baseParams, curParams=None, Lidx=None, nlambda=None):
        """
        function to compute codelength, if input graph is undirected

        Parameters
        ----------
        G : Networkx Graph
            Input Graph
        PD : PDClass
            Background Distribution
        condition : int
            condition to compute codelength
            1: Codelength of initial pattern or a single component
            2: Condition 1 + removing some nodes for codelength computation
            3: Condition 2 but with a new lambda for reduced pattern/constraint
        baseParams : dict
            value of prameters corresponding to the initial pattern before shrink
        curParams : dict, optional
            value of prameters corresponding to the initial pattern after shrink, by default None
        Lidx : int, optional
            identifier of the constarint which is evaluated and dropped in some cases, by default None
        nlambda : float, optional
            new lambda if condition is 3, by default None

        Returns
        -------
        float
            commputed codelength
        """
        codelength = 0.0
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
###################################################################################################################################################################
    def computeCodeLengthShrinkD(self, G, PD, condition, baseParams, curParams=None, Lidx=None, nlambda=None):
        """
        function to compute codelength, if input graph is directed

        Parameters
        ----------
        G : Networkx Graph
            Input Graph
        PD : PDClass
            Background Distribution
        condition : int
            condition to compute codelength
            1: Codelength of initial pattern or a single component
            2: Condition 1 + removing some nodes for codelength computation
            3: Condition 2 but with a new lambda for reduced pattern/constraint
        baseParams : dict
            value of prameters corresponding to the initial pattern before shrink
        curParams : dict, optional
            value of prameters corresponding to the initial pattern after shrink, by default None
        Lidx : int, optional
            identifier of the constarint which is evaluated and dropped in some cases, by default None
        nlambda : float, optional
            new lambda if condition is 3, by default None

        Returns
        -------
        float
            commputed codelength
        """
        codelength = 0.0
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
        *Condition 1: Only update the description length
        *Condition 2: Codelength with desc. length is updated
        """
        if condition == 1:
            if self.gtype == 'U':
                DL = computeDescriptionLength( dlmode=6, C=len(PD.lprevUpdate), gtype=self.gtype, WS=self.Data[id]['Pat'].NCount, W=self.Data[id]['SPat'].NCount, kw=self.Data[id]['SPat'].ECount, excActionType=False, l=self.l, isSimple=self.isSimple, kws=self.Data[id]['SPat'].kws )
                IG = computeInterestingness( self.Data[id]['SPat'].IC_dssg, DL, mode=self.imode )
                self.Data[id]['Pat'].setDL(DL)
                self.Data[id]['Pat'].setI(IG)
                self.Data[id]['SPat'].setDL(DL)
                self.Data[id]['SPat'].setI(IG)
            else:
                DL = computeDescriptionLength( dlmode=6, C=len(PD.lprevUpdate), gtype=self.gtype, WIS=self.Data[id]['Pat'].InNCount, WOS=self.Data[id]['Pat'].OutNCount, WI=self.Data[id]['SPat'].InNL, WO=self.Data[id]['SPat'].OutNL, kw=self.Data[id]['SPat'].ECount, excActionType=False, l=self.l, isSimple=self.isSimple, kws=self.Data[id]['SPat'].kws )
                IG = computeInterestingness( self.Data[id]['SPat'].IC_dssg, DL, mode=self.imode )
                self.Data[id]['Pat'].setDL(DL)
                self.Data[id]['Pat'].setI(IG)
                self.Data[id]['SPat'].setDL(DL)
                self.Data[id]['SPat'].setI(IG)
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
        if prevPat.pat_type is not 'shrink':
            if prevPat.pat_type in ['merge']:
                for p in prevPat.prev_order:
                    if p in self.Data:
                        del self.Data[p]
            elif prevPat.pat_type in ['update', 'split', 'remove']:
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
        function to return the best candidate to shrink

        Returns
        -------
        dict
            dictionary containing a Pattern, and the two corresponding codelength associated to the pattern, i.e., prior and posterior to performing split action.
        """
        if len(self.Data) < 1:
            return None
        else:
            bestR = max(self.Data.items(), key=lambda x: x[1]['SPat'].I)
            return bestR[1]
###################################################################################################################################################################
    def updateDistribution(self, PD, bestSh):
        """
        function to update background distribution.
        * Now here we remove the knowledge of pervious pattern which is now updated and add the knowledge of pattern which is the result of shrink
        * hence we remove the previous lambda associated with the pattern and add a new lambda for skrinked pattern

        Parameters
        ----------
        PD : PDClass
            Background distribution
        bestSh : dict
            last shrink action details
        """
        #* Here bestSh is a dictionary as saved in self.Data
        del self.Data[bestSh['Pat'].prev_order]
        out = PD.lprevUpdate.pop(bestSh['Pat'].prev_order, None)
        if out is None:
            print('Something is fishy')
        else:
            la = PD.updateDistribution( bestSh['SPat'].G, idx=bestSh['SPat'].cur_order, val_return='save', case=2 )
            bestSh['SPat'].setLambda(la)
        return
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################