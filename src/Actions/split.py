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
from src.Utils.Measures import LN, DL_Edges, ncr, NW, NW_D
from src.Patterns.Pattern import Pattern

class EvaluateSplit:
    """
    This data structure shall contain all the possible splits
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
        """
        self.Data = dict()
        self.gtype = gtype
        self.isSimple = isSimple
        self.l = l # possible types (give number) of action, default is 6
        self.imode = imode
        self.minsize = minsize
        print('initialized EvaluateSplit')

    def evaluateAllConstraints(self, G, PD):
        """
        function to evaluate all constraints and make a list of candidate constraints which are feasible to split

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

    def evaluateConstraint(self, G, PD, id):
        """
        function to evaluate if a constraint is a feasible candidate for split

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

    def processAsU(self, G, PD, id):
        """
        Utility function for split action when the input graph is undirected.
        This function idenfies the final components from each possible candidate split and compute the corresponding measures.

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
            baseParams['codeLengthCprime'] = self.computeCodeLengthSplitU(G, PD, 2, baseParams, id) #// Todo : write code for this part
            baseParams['Pat'].setIC_dssg( baseParams['codeLengthC'] - baseParams['codeLengthCprime'] )
            baseParams['Pat'].setDL( computeDescriptionLength( case=7, C=len(PD.lprevUpdate), gtype=self.gtype, WS=baseParams['Pat'].NCount, compos=baseParams['compos'], isSimple=self.isSimple ) )
            baseParams['Pat'].setI( computeInterestingness( baseParams['Pat'].IC_dssg, baseParams['Pat'].DL, mode=2 ) )
            baseParams['Pat'].setPatType('split')

            #now try reducing each component
            FinalParams = baseParams
            for k in baseParams['compos'].keys():
                FinalParams = self.getReducedComponentU(G, PD, FinalParams, id, k)

            #compute new lambdas for each new pattern/component
            #// Todo: write code
            for k,v in FinalParams['compos'].items():
                v.setLambda( PD.updateDistribution( pat=v.G, idx=None, val_retrun='return', case=3, dropLidx=[id]) )
            FinalParams['codeLengthCprime'] = self.computeCodeLengthSplitU(G, PD, 3, FinalParams, id) #// Todo : write code for this part
            FinalParams['Pat'].setIC_dssg( FinalParams['codeLengthC'] - FinalParams['codeLengthCprime'] )
            FinalParams['Pat'].setDL( computeDescriptionLength( case=7, C=len(PD.lprevUpdate), gtype=self.gtype, WS=FinalParams['Pat'].NCount, compos=FinalParams['compos'], excActionType=False, l=self.l, isSimple=self.isSimple ) )
            FinalParams['Pat'].setI( computeInterestingness( FinalParams['Pat'].IC_dssg, FinalParams['Pat'].DL, mode=2 ) )
            # Now set these values to all component patterns
            #// Todo: Write Code
            for k,v in FinalParams['compos'].items():
                v.setIC_dssg( FinalParams['Pat'].IC_dssg )
                v.setDL( FinalParams['Pat'].DL )
                v.setI( FinalParams['Pat'].I )
                v.setPrevOrder(id)
                v.setPatType('split')
            self.Data[id] = FinalParams
        return

    def getReducedComponentU(self, G, PD, FinalParams, Lid, k):
        """
        Utility function used when the input graph is undirected to remove nodes from each component of the candidate split

        Parameters
        ----------
        G : Networkx Graph
            Input Graph
        PD : PDClass
            Background Distribution
        FinalParams : dict
            current value of prameters corresponding to the current split of the candidate constraint
        Lid : int
            identifier for the the candidate constraint for split
        k : int
            identifier for the component number of the candidate after split

        Returns
        -------
        dict
            FinalParams: Updated after reducing a component
        """
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
                FinalParams['Pat'].setDL( computeDescriptionLength( case=7, C=len(PD.lprevUpdate), gtype=self.gtype, WS=FinalParams['Pat'].NCount, compos=FinalParams['compos'], excActionType=False, l=self.l, isSimple=self.isSimple ) )
                FinalParams['Pat'].setI( computeInterestingness( FinalParams['Pat'].IC_dssg, FinalParams['Pat'].DL, mode=2 ) )
                FinalParams['NodesInc'] -= 1
                FinalParams['excludedNL'].append(bestRNode)
                doshrink = True

        return FinalParams

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

    def getDescriptionLengthChangeU(self, Pat, node, WI, W):
        """
        Utility function which computes the change in description length on removing a node.
        This function is used when the input graph is undirected

        Parameters
        ----------
        Pat : Pattern
            Input Pattern from which a node is removed
        node : int
            Node to be removed
        WI : int
            Initial number of nodes in the current candidate constraint/pattern
        W : int
            number of nodes included in any one component of the candidate constraint after split

        Returns
        -------
        float
            change in description length
        """
        change = 0.0
        change += LN(Pat.NCount) - LN(Pat.NCount - 1)
        change += ncr(WI, W) - ncr(WI, W-1)
        pDLEdge = DL_Edges(Pat.nw, Pat.ECount, self.isSimple, Pat.kws)
        nECount = Pat.ECount
        nNW = Pat.nw
        nkws = Pat.kws
        for k in Pat.NL:
            nECount -= Pat.G.number_of_edges(k, node)
            nkws -= ( Pat.G.number_of_edges(k, node) > 0 )
        nNW -= (Pat.NCount - 1)
        nDLEdge = DL_Edges(nNW, nECount, self.isSimple, nkws)
        change += pDLEdge - nDLEdge
        return change

    def computeCodeLengthSplitU(self, G, PD, condition, Params, Lidx=None):
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
            2: Condition 1 + intra-component codelength computation
            3: Condition 2 but with a new lambda for each component
        Params : dict
            value of prameters corresponding to the current split of the candidate constraint
        Lidx : int, optional
            identifier of the constarint which is evaluated and dropped in some cases, by default None

        Returns
        -------
        float
            commputed codelength
        """
        codelength = 0.0
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
        """
        Utility function for split action when the input graph is directed.
        This function idenfies the final components from each possible candidate split and compute the corresponding measures.

        Parameters
        ----------
        G : Networkx Graph
            Input Graph
        PD : PDClass
            Background Distribution
        id : int
            identifier of a constraint to be evaluated
        """
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
            out_nodes_union = set()
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
            baseParams['Pat'].setDL( computeDescriptionLength( case=7, C=len(PD.lprevUpdate), gtype=self.gtype, WIS=baseParams['Pat'].InNCount, WOS=baseParams['Pat'].OutNCount, compos=baseParams['compos'], isSimple=self.isSimple ) )
            baseParams['Pat'].setI( computeInterestingness( baseParams['Pat'].IC_dssg, baseParams['Pat'].DL, mode=2 ) )
            baseParams['Pat'].setPatType('split')

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
            FinalParams['Pat'].setDL( computeDescriptionLength( case=7, C=len(PD.lprevUpdate), gtype=self.gtype, WIS=FinalParams['Pat'].InNCount, WOS=FinalParams['Pat'].OutNCount, compos=FinalParams['compos'], excActionType=False, l=self.l, isSimple=self.isSimple ) )
            FinalParams['Pat'].setI( computeInterestingness( FinalParams['Pat'].IC_dssg, FinalParams['Pat'].DL, mode=2 ) )
            # Now set these values to all component patterns
            #// Todo: Write Code
            for k,v in FinalParams['compos'].items():
                v.setIC_dssg( FinalParams['Pat'].IC_dssg )
                v.setDL( FinalParams['Pat'].DL )
                v.setI( FinalParams['Pat'].I )
                v.setPrevOrder(i)
                v.setPatType('split')
            self.Data[i] = FinalParams
        return

    def getReducedComponentD(self, G, PD, FinalParams, Lid, k):
        """
        Utility function used when the input graph is directed to remove nodes from each component of the candidate split

        Parameters
        ----------
        G : Networkx Graph
            Input Graph
        PD : PDClass
            Background Distribution
        FinalParams : dict
            current value of prameters corresponding to the current split of the candidate constraint
        Lid : int
            identifier for the the candidate constraint for split
        k : int
            identifier for the component number of the candidate after split

        Returns
        -------
        dict
            FinalParams: Updated after reducing a component
        """
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
                FinalParams['Pat'].setDL( computeDescriptionLength( case=7, C=len(PD.lprevUpdate), gtype=self.gtype, WIS=FinalParams['Pat'].InNCount, WOS=FinalParams['Pat'].OutNCount, compos=FinalParams['compos'], excActionType=False, l=self.l, isSimple=self.isSimple ) )
                FinalParams['Pat'].setI( computeInterestingness( FinalParams['Pat'].IC_dssg, FinalParams['Pat'].DL, mode=2 ) )
                count_remove_nodes += 1
                doshrink = True

        return FinalParams

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

    def getDescriptionLengthChangeD(self, Pat, node, WI, W, tp):
        """
        Utility function which computes the change in description length on removing a node.
        This function is used when the input graph is directed

        Parameters
        ----------
        Pat : Pattern
            Input Pattern from which a node is removed
        node : int
            Node to be removed
        WI : int
            Initial number of nodes in the current candidate constraint/pattern
        W : int
            number of nodes included in any one component of the candidate constraint after split
        tp : int
            1 : if node is InNode
            2 : if node is OutNode

        Returns
        -------
        float
            change in description length
        """
        change = 0.0
        if tp == 1:
            change += LN(Pat.InNCount) - LN(Pat.InNCount - 1)
            change += ncr(WI, W) - ncr(WI, W-1)
            pDLEdge = DL_Edges(Pat.nw, Pat.ECount, self.isSimple, Pat.kws)
            nECount = Pat.ECount
            nNW = Pat.nw
            nkws = Pat.kws
            for k in Pat.outNL:
                nECount -= Pat.G.number_of_edges(k, node)
                nkws -= ( Pat.G.number_of_edges(k, node) > 0)
            nNW -= (Pat.OutNCount)
            nDLEdge = DL_Edges(nNW, nECount, self.isSimple, nkws)
            change += pDLEdge - nDLEdge
        else:
            change += LN(Pat.OutNCount) - LN(Pat.OutNCount - 1)
            change += ncr(WI, W) - ncr(WI, W-1)
            pDLEdge = DL_Edges(Pat.nw, Pat.ECount, self.isSimple, Pat.kws)
            nECount = Pat.ECount
            nNW = Pat.nw
            nkws = Pat.kws
            for k in Pat.inNL:
                nECount -= Pat.G.number_of_edges(node, k)
                nkws -= ( Pat.G.number_of_edges(node, k) > 0)
            nNW -= (Pat.InNCount)
            nDLEdge = DL_Edges(nNW, nECount, self.isSimple, nkws)
            change += pDLEdge - nDLEdge
        return change

    def computeCodeLengthSplitD(self, G, PD, condition, Params, Lidx=None):
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
            2: Condition 1 + intra-component codelength computation
            3: Condition 2 but with a new lambda for each component
        Params : dict
            value of prameters corresponding to the current split of the candidate constraint
        Lidx : int, optional
            identifier of the constarint which is evaluated and dropped in some cases, by default None

        Returns
        -------
        float
            commputed codelength
        """
        codelength = 0.0
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

        #? To understand when a candidate need to be updated or added in the potential list, we shall ealise thatonly the patternx from previous states can only be split.
        #? As any action performed will result in a connected pattern(s), thus the list of potential candidates is only updated at the start of the state
        #? Now with any other action performed there are only two possibilities for each candidate
        #? First either the Description Length changes (Condition==1)
        #? Or Second, both the codelength and description length changes (Condition==2)
        """
        if condition == 1:
            if self.gtype == 'U':
                DL = computeDescriptionLength( case=7, C=len(PD.lprevUpdate), gtype=self.gtype, WS=self.Data[id]['Pat'].NCount, compos=self.Data[id]['compos'], excActionType=False, l=self.l, isSimple=self.isSimple )
                IG = computeInterestingness( self.self.Data[id]['Pat'].IC_dssg, DL, mode=2 )
                self.Data[id]['Pat'].setDL(DL)
                self.Data[id]['Pat'].setI(IG)
            else:
                DL = computeDescriptionLength( case=7, C=len(PD.lprevUpdate), gtype=self.gtype, WIS=self.Data[id]['Pat'].InNCount, WOS=self.Data[id]['Pat'].OutNCount, compos=self.Data[id]['compos'], excActionType=False, l=self.l, isSimple=self.isSimple )
                IG = computeInterestingness( self.Data[id]['Pat'].IC_dssg, DL, mode=2 )
                self.Data[id]['Pat'].setDL(DL)
                self.Data[id]['Pat'].setI(IG)
            for k,v in self.Data[id]['compos'].items():
                v.setDL( self.Data[id]['Pat'].DL )
                v.setI( self.Data[id]['Pat'].I )
        elif condition == 2:
            if self.gtype == 'U':
                self.processAsU(G, PD, id)
            elif self.gtype == 'D':
                self.processAsD(G, PD, id)
        return

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
        if self.gtype == 'U':
            for k,v in self.Data.items():
                if len(set(v.NL).intersection(set(prevPat.NL))) > 1:
                    self.updateConstraintEvaluation(G, PD, k, 2)
                else:
                    self.updateConstraintEvaluation(G, PD, k, 1)
        else:
            for k,v in self.Data.items():
                inInt = len(set(v.inNL).intersection(set(prevPat.inNL)))
                outInt = len(set(v.outNL).intersection(set(prevPat.outNL)))
                if inInt > 1 and outInt > 1:
                    self.updateConstraintEvaluation(G, PD, k, 2)
                else:
                    self.updateConstraintEvaluation(G, PD, k, 1)
        return

    def getBestOption(self):
        """
        function to return the best candidate to split

        Returns
        -------
        dict
            dictionary containing a Pattern, and the two corresponding codelength associated to the pattern, i.e., prior and posterior to performing split action.
        """
        if len(self.Data) < 1:
            return None
        else:
            bestR = max(self.Data.items(), key=lambda x: x[1]['Pat'].I)
            return bestR[1]

    def updateDistribution(self, PD, bestSp):
        """
        function to update background distribution.
        * Now here we remove the knowledge of pervious pattern which is now updated and add the knowledge of pattern which is the result of split
        * hence we remove the previous lambda associated with the pattern and add a new lambdas for all new pattern

        Parameters
        ----------
        PD : PDClass
            Background distribution
        bestSp : Pattern
            last split pattern
        """
        #* Here bestSp is a dictionary as saved in self.Data
        if bestSp['Pat'].prev_order in self.Data: #? Removing the candidate from potential list
            del self.Data[bestSp['Pat'].prev_order]
        else:
            print('Trying to remove key:{} in merge Data but key not found'.format(bestSp['Pat'].prev_order))
        out = PD.lprevUpdate.pop(bestSp['Pat'].prev_order, None)
        if out is None:
            print('Something is fishy')
        else:
            for k,v in bestSp['compos'].items():
                PD.updateDistribution( v.G, idx=v.cur_order, val_retrun='save', case=2 )
        return