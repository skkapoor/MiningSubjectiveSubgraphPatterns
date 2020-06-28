###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################
import networkx as nx
import os
import sys
path = os.getcwd().split('MiningSubjectiveSubgraphPatterns')[0]+'MiningSubjectiveSubgraphPatterns/'
if path not in sys.path:
	sys.path.append(path)
from src.Utils.Measures import NW_D, NW
###################################################################################################################################################################
class Pattern:
    state_info = -1
    pat_type = 'UNKNOWN'
###################################################################################################################################################################
    def __init__(self, G, order = 0):
        self.G = G.copy()
        self.NL = []
        self.inNL = []
        self.outNL = []
        self.NCount = None
        self.InNCount = None
        self.OutNCount = None
        self.ECount = None
        self.nw = None
        self.kws = None
        self.updateGraphProperties()
        self.sumPOS = 0.0
        self.expectedEdges = 0.0
        self.minPOS = float('inf')
        self.IC_ssg = 0.0
        self.AD = 0.0
        self.IC_dssg = 0.0
        self.IC_dsimp = 0.0
        self.DL = 0.0
        self.I = -float('inf')
        self.cur_order = order
        self.prev_order = None
        self.la = None
###################################################################################################################################################################
    def updateGraph(self, H): # providing nodes and edges to be added inform of a subgraph
        if isinstance(self.G, nx.DiGraph) and isinstance(H, nx.DiGraph):
            self.G = nx.compose(self.G, H)
            inL = dict(self.G.in_degree())
            outL = dict(self.G.out_degree())
            self.inNL = []
            self.outNL = []
            for k,v in inL.items():
                if v!=0:
                    self.inNL.append(k)
            for k,v in outL.items():
                if v!=0:
                    self.outNL.append(k)
            # self.inNL = sorted(self.inNL)
            # self.outNL = sorted(self.outNL)
            self.InNCount = len(self.inNL)
            self.OutNCount = len(self.outNL)
            self.ECount = self.G.number_of_edges()
            self.nw = NW_D(self.inNL, self.outNL)
        elif isinstance(self.G, nx.Graph) and isinstance(H, nx.Graph):
            self.G = nx.compose(self.G, H)
            self.NL = list(self.G.nodes())
            self.NCount = self.G.number_of_nodes()
            self.ECount = self.G.number_of_edges()
            self.nw = NW(self.NCount)
        else:
            print('Graph type miss match cannot update Graph pattern')
        if isinstance(self.G, nx.MultiDiGraph):
            self.kws = nx.DiGraph(self.G).number_of_edges()
        elif isinstance(self.G, nx.MultiGraph):
            self.kws = nx.Graph(self.G).number_of_edges()
        else:
            self.kws = self.ECount
        return
###################################################################################################################################################################
    def updateGraphProperties(self):
        if isinstance(self.G, nx.DiGraph):
            inL = dict(self.G.in_degree())
            outL = dict(self.G.out_degree())
            self.inNL = []
            self.outNL = []
            for k,v in inL.items():
                if v!=0:
                    self.inNL.append(k)
            for k,v in outL.items():
                if v!=0:
                    self.outNL.append(k)
            self.inNL = sorted(self.inNL)
            self.outNL = sorted(self.outNL)
            self.InNCount = len(self.inNL)
            self.OutNCount = len(self.outNL)
            self.ECount = self.G.number_of_edges()
            self.nw = NW_D(self.inNL, self.outNL)
        elif isinstance(self.G, nx.Graph):
            self.NL = sorted(list(self.G.nodes()))
            self.NCount = self.G.number_of_nodes()
            self.ECount = self.G.number_of_edges()
            self.nw = NW(self.NCount)
        if isinstance(self.G, nx.MultiDiGraph):
            self.kws = nx.DiGraph(self.G).number_of_edges()
        elif isinstance(self.G, nx.MultiGraph):
            self.kws = nx.Graph(self.G).number_of_edges()
        else:
            self.kws = self.ECount
        return
###################################################################################################################################################################
    def setSumPOS(self, pos):
        self.sumPOS = pos
        return
    ##########################################
    def setExpectedEdges(self, value):
        self.expectedEdges = value
        return
    ##########################################
    def setMinPOS(self, pos):
        self.minPOS = pos
        return
    ##########################################
    def setIC_ssg(self, ic):
        self.IC_ssg = ic
        return
    ##########################################
    def setAD(self, ad):
        self.AD = ad
        return
    ##########################################
    def setIC_dssg(self, ic):
        self.IC_dssg = ic
        return
    ##########################################
    def setIC_dsimp(self, ic):
        self.IC_dsimp = ic
        return
    ##########################################
    def setDL(self, dl):
        self.DL = dl
        return
    ##########################################
    def setI(self, I):
        self.I = I
        return
    ##########################################
    def setCurOrder(self, order):
        self.cur_order = order
        return
    ##########################################
    def setPrevOrder(self, order):
        self.prev_order = order
        return
    ##########################################
    def setLambda(self, la):
        self.la = la
        return
    ##########################################
    def setNW(self, nw):
        self.nw = nw
        return
    ##########################################
    def setKWS(self, kws):
        self.kws = kws
        return
    ##########################################
    def setPatType(self, Ptype):
        self.pat_type = Ptype
        return
    ##########################################
    def setStateInfo(self, s):
        self.state_info = s
        return
###################################################################################################################################################################
    def removeNode(self, node):
        assert isinstance(self.G, nx.Graph) and not isinstance(self.G, nx.DiGraph), "function removeNode() is only for type nx.Graph or nx.MultiGraph"
        self.G.remove_node(node)
        self.NL.remove(node)
        self.NCount = len(self.NL)
        self.ECount = self.G.number_of_edges()
        self.nw = NW(self.NCount)
        if isinstance(self.G, nx.MultiDiGraph):
            self.kws = nx.DiGraph(self.G).number_of_edges()
        elif isinstance(self.G, nx.MultiGraph):
            self.kws = nx.Graph(self.G).number_of_edges()
        else:
            self.kws = self.ECount
        return
###################################################################################################################################################################
    def removeInNode(self, nodeIn):
        assert isinstance(self.G, nx.DiGraph), "function removeInNode() is only for type nx.DiGraph or nx.MultiDiGraph"
        predL = list(self.G.predecessors(nodeIn))
        for g in predL:
            self.G.remove_edges_from([(g, nodeIn)]*self.G.number_of_edges(g, nodeIn))
        if self.G.in_degree(nodeIn) == 0 and self.G.out_degree(nodeIn) == 0:
            self.G.remove_node(nodeIn)
            if nodeIn in self.outNL:
                self.outNL.remove(nodeIn)
                self.OutNCount = len(self.outNL)
        self.inNL.remove(nodeIn)
        self.InNCount = len(self.inNL)
        self.ECount = self.G.number_of_edges()
        self.nw = NW_D(self.inNL, self.outNL)
        if isinstance(self.G, nx.MultiDiGraph):
            self.kws = nx.DiGraph(self.G).number_of_edges()
        elif isinstance(self.G, nx.MultiGraph):
            self.kws = nx.Graph(self.G).number_of_edges()
        else:
            self.kws = self.ECount
        return
###################################################################################################################################################################
    def removeOutNode(self, nodeOut):
        assert isinstance(self.G, nx.DiGraph), "function removeOutNode() is only for type nx.DiGraph or nx.MultiDiGraph"
        succL = list(self.G.successors(nodeOut))
        for g in succL:
            self.G.remove_edges_from([(nodeOut, g)]*self.G.number_of_edges(nodeOut, g))
        if self.G.in_degree(nodeOut) == 0 and self.G.out_degree(nodeOut) == 0:
            self.G.remove_node(nodeOut)
            if nodeOut in self.inNL:
                self.inNL.remove(nodeOut)
                self.InNCount = len(self.inNL)
        self.outNL.remove(nodeOut)
        self.OutNCount -= 1
        self.ECount = self.G.number_of_edges()
        self.nw = NW_D(self.inNL, self.outNL)
        if isinstance(self.G, nx.MultiDiGraph):
            self.kws = nx.DiGraph(self.G).number_of_edges()
        elif isinstance(self.G, nx.MultiGraph):
            self.kws = nx.Graph(self.G).number_of_edges()
        else:
            self.kws = self.ECount
        return
###################################################################################################################################################################
    def copy(self):
        PC = Pattern(self.G)
        PC.setAD(self.AD)
        PC.setCurOrder(self.cur_order)
        PC.setDL(self.DL)
        PC.setExpectedEdges(self.expectedEdges)
        PC.setI(self.I)
        PC.setIC_dsimp(self.IC_dsimp)
        PC.setIC_dssg(self.IC_dssg)
        PC.setIC_ssg(self.IC_ssg)
        PC.setLambda(self.la)
        PC.setMinPOS(self.minPOS)
        PC.setPrevOrder(self.prev_order)
        PC.setStateInfo(self.state_info)
        PC.setSumPOS(self.sumPOS)
        PC.setNW(self.nw)
        PC.setKWS(self.kws)
        return PC
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################