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
###################################################################################################################################################################
    def __init__(self, G, order = 0):
        self.state_info = -1
        self.pat_type = 'UNKNOWN'
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
                if v!=0 or (v==0 and outL[k]==0):
                    self.inNL.append(k)
            for k,v in outL.items():
                if v!=0 or (v==0 and inL[k]==0):
                    self.outNL.append(k)

            for nd in H.nodes():
                if not self.G.has_node(nd):
                    print(nd)
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
        if self.G.is_directed():
            inL = dict(self.G.in_degree())
            outL = dict(self.G.out_degree())
            self.inNL = []
            self.outNL = []
            for k,v in inL.items():
                if v!=0 or (v==0 and outL[k]==0):
                    self.inNL.append(k)
            for k,v in outL.items():
                if v!=0 or (v==0 and inL[k]==0):
                    self.outNL.append(k)
            self.inNL = sorted(self.inNL)
            self.outNL = sorted(self.outNL)
            self.InNCount = len(self.inNL)
            self.OutNCount = len(self.outNL)
            self.ECount = self.G.number_of_edges()
            self.nw = NW_D(self.inNL, self.outNL)
        else:
            self.NL = sorted(list(self.G.nodes()))
            self.NCount = self.G.number_of_nodes()
            self.ECount = self.G.number_of_edges()
            self.nw = NW(self.NCount)
        if self.G.is_multigraph():
            if self.G.is_directed():
                self.kws = nx.DiGraph(self.G).number_of_edges()
            else:
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
        assert isinstance(self.G, nx.Graph) and not self.G.is_directed(), "function removeNode() is only for type nx.Graph or nx.MultiGraph"
        self.G.remove_node(node)
        self.NL.remove(node)
        self.NCount = len(self.NL)
        self.ECount = self.G.number_of_edges()
        self.nw = NW(self.NCount)
        if self.G.is_multigraph():
            self.kws = nx.Graph(self.G).number_of_edges()
        else:
            self.kws = self.ECount
        return
###################################################################################################################################################################
    def removeInNode(self, nodeIn):
        assert isinstance(self.G, nx.Graph) and self.G.is_directed(), "function removeInNode() is only for type nx.DiGraph or nx.MultiDiGraph"
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
        if self.G.is_multigraph():
            self.kws = nx.DiGraph(self.G).number_of_edges()
        else:
            self.kws = self.ECount
        return
###################################################################################################################################################################
    def removeOutNode(self, nodeOut):
        assert isinstance(self.G, nx.Graph) and self.G.is_directed(), "function removeOutNode() is only for type nx.DiGraph or nx.MultiDiGraph"
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
        if self.G.is_multigraph():
            self.kws = nx.DiGraph(self.G).number_of_edges()
        else:
            self.kws = self.ECount
        return
###################################################################################################################################################################
    def __repr__(self):
        st = "\t\tpat_type: {}\n".format(self.pat_type)
        st += "\t\tstate_info: {}\n".format(self.state_info)
        if self.G.is_directed():
            st += "\t\tInNCount: {}\tOutNCount: {}\n".format(self.InNCount, self.OutNCount)
        else:
            st += "\t\tNCount: {}\n".format(self.NCount)
        if self.G.is_multigraph():
            st += "\t\tECount: {}\tkws: {}\n".format(self.ECount, self.kws)
        else:
            st += "\t\tECount: {}\n".format(self.ECount)
        st += "\t\tDensity: {:.5f}\n".format(nx.density(self.G))
        st += "\t\tPrev_index: {}\tCur_index: {}\n".format(self.prev_order, self.cur_order)
        st += "\t\tI: {:.5f}\tDL: {:.5f}\n".format(self.I, self.DL)
        st += "\t\tIC_ssg: {:.5f}\tAD: {:.5f}\tIC_dssg: {:.5f}\tIC_dsimp: {:.5f}\n".format(self.IC_ssg, self.AD, self.IC_dssg, self.IC_dsimp)
        st += "\t\tla: {}\n".format(self.la)
        st += "\t\tsumPOS: {:.5f}\texpectedEdges: {:.5f}\n".format(self.sumPOS, self.expectedEdges)
        if self.G.is_directed():
            self.inNL = sorted(self.inNL)
            self.outNL = sorted(self.outNL)
            st += "\t\tinNL: "+", ".join(map(str, self.inNL))+"\n"
            st += "\t\toutNL: "+", ".join(map(str, self.outNL))+"\n"
        else:
            self.NL = sorted(self.NL)
            st += "\t\tNL: "+", ".join(map(str, self.NL))+"\n"
        return st
###################################################################################################################################################################
    def getDictForm(self):
        dt = dict()
        members = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
        members = set(members) - set(['G'])
        if self.G.is_directed():
            members = set(members) - set(['NCount', 'NL'])
        else:
            members = set(members) - set(['InNCount', 'OutNCount', 'inNL', 'outNL'])
        if not self.G.is_multigraph():
            members = set(members) - set('kws')
        for k in members:
            if isinstance(self.__dict__[k], (list, tuple, set)):
                # dt[k] = ", ".join(map(str, self.__dict__[k]))
                dt[k] = sorted(list(self.__dict__[k]))
            else:
                dt[k] = self.__dict__[k]
        dt['Density'] = nx.density(self.G)
        return dt
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