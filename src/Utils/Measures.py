###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################
import numpy as numpy
import networkx as nx
import math
import ray
##################################################################################################################################################################
def KL(q, p):
    """function to compute KL-divergence between two Bernoulli Distributions

    Args:
        q (float): probability of success of posterior distribution
        p ([type]): probability of success of prior distribution

    Returns:
        float: KL-divergence value
    """
    t1 = 0.0
    t2 = 0.0
    if p < 1e-10:
        p = 1e-10
    if p > 1 - 1e-10:
        p = 1-1e-10
    try:
        t1 = 0.0 if q == 0.0 else q * math.log(q / p)
        t2 = 0.0 if q == 1.0 else (1 - q) * math.log((1 - q) / (1 - p))
    except:
        print('Math Domain Error', p, q)
    return t1 + t2

def KL_g(q, p):
    """function to compute KL-divergence between two Bernoulli Distributions

    Args:
        q (float): probability of success of posterior distribution
        p ([type]): probability of success of prior distribution

    Returns:
        float: KL-divergence value
    """
    t1 = 0.0
    t2 = 0.0
    if p < 1e-10:
        p = 1e-10
    if p > 1 - 1e-10:
        p = 1-1e-10
    try:
        t1 = 0.0 if q == 0.0 else q * math.log(q / p)
        t2 = 0.0 if q == 1.0 else (1 - q)/q * math.log((1 - q) / (1 - p))
    except:
        print('Math Domain Error', p, q)
    return t1 + t2
##################################################################################################################################################################
def ncr(n, r):
    r1 = min(r, n-r)
    c = 1
    for i in range(n, n-r1, -1):
        c *= i
    for i in range(r1, 1, -1):
        c //= i
    return c
##################################################################################################################################################################
def NW(N, tp='U'):
    """function to compute number of edges or number of feasible vertex-pair

    Args:
        N (int): number of nodes
        tp (str, optional): Can be either 'U' for undirected and 'D' for directed graphs. Defaults to 'U'.

    Returns:
        int: number of edges or number of feasible vertex-pair between N nodes
    """
    assert tp=='U' or tp=='D', "Invalid type in NW, it shall be either 'U' or 'D'"
    if tp=='U':
        return N*(N-1)//2
    else:
        return N*(N-1)
##################################################################################################################################################################
def NW_D(inNL, outNL):
    """function to compute number of edges or number of feasible vertex-pair in a directed graph

    Args:
        inNL (list): list of nodes for a directed graph having non-zero indegree
        outNL (list): list of nodes for a directed graph having non-zero outdegree

    Returns:
        count (int): number of possible edges or combination of feasible vertex-pair
    """
    assert isinstance(inNL, list) and isinstance(outNL, list), "inNL and outNL shall be a list in NW_D"
    intersection = set(inNL).intersection(set(outNL))
    count = len(outNL)*len(inNL) - len(intersection)
    return count
##################################################################################################################################################################
def LN(n):
    """function to compute universal code for integer as given by Rissanen (1983)

    Args:
        n (int): input integer

    Returns:
        float: complexity or number of bits required to universally encode an integer
    """
    f = math.log2(2.865064)
    a = n
    if a>=1:
        f += math.log2(a)
        a = math.log2(a)
    return f
##################################################################################################################################################################
def IC_SSG(mode, **kwargs):
    """function to compute the information content of a simple graph pattern as proposed in SSG

    Args:
        mode (int): 1 if pw, kw, nw is provided
                    2 if supergraph G is provided along with a linst of nodes in subgraph WL
                    3 if a subgraph pattern W is provided as a networkx graph

    kwargs:
        gtype (str): 'U' for undirected and 'D' for directed, default is 'U'
        kw (int): Required if mode is 1, number of edges in a graph pattern
        nw (int): Required if mode is 1, number of feasible vertex-pair combination in a graph pattern
        pw (float): sum of probability of success (having an edge) between all feasible vertex-pair
        G (networkx graph): required if mode is 2, supergraph
        WL (list): required if mode is 2 and gtype is 'U', list of nodes in a pattern
        PD (list): required if mode is 2 and PD is not provided
        isSimple (boolean): True, if a pattern is a simple graph else false.
        W (networkx graph): Required if mode is 4, pattern as a networkx graph object


    Returns:
        ic (float): The information content (as per definition in SSG) of a (simple) graph pattern
    """
    gtype = 'U'
    isSimple = True
    if 'gtype' in kwargs:
        gtype = kwargs['gtype']
    if 'isSimple' in kwargs:
        isSimple = kwargs['isSimple']
    ic = 0.0
    if mode == 1:
        assert 'kw' in kwargs and 'nw' in kwargs and 'pw' in kwargs, "kw, nw and pw are required if mode is 1"
        ic = kwargs['nw'] * KL( kwargs['kw']/kwargs['nw'], kwargs['pw']/kwargs['nw'] )
    elif mode == 2:
        if gtype == 'U':
            assert 'G' in kwargs and 'WL' in kwargs, "'G' (supergraph) and 'WL' (list of nodes in pattern W) are required if mode is 2"
            H = kwargs['G'].subgraph(kwargs['WL'])
            nw = NW(H.number_of_nodes())
            kw = H.number_of_edges()
            pw = 0.0
            assert 'PD' in kwargs or 'pw' in kwargs, "if mode is 2, either PD or pw shall be provided"
            if 'pw' in kwargs:
                pw = kwargs['pw']
            else:
                pw = computeSumOfEdgeProbablity(kwargs['PD'], gtype=gtype, NL=kwargs['WL'], isSimple=isSimple)
            ic = nw * KL(kw/nw, pw/nw)
    elif mode == 3:
        if gtype == 'U':
            assert 'W' in kwargs, "'G' (subgraph pattern) as networkx graph is required if mode is 3"
            nw = NW(kwargs['W'].number_of_nodes())
            kw = kwargs['W'].number_of_edges()
            pw = 0.0
            assert 'PD' in kwargs or 'pw' in kwargs, "if mode is 3, either PD or pw shall be provided"
            if 'pw' in kwargs:
                pw = kwargs['pw']
            else:
                pw = computeSumOfEdgeProbablity(kwargs['PD'], gtype=gtype, NL=kwargs['WL'], isSimple=isSimple)
            ic = nw * KL(kw/nw, pw/nw)
    else:
        assert mode<4, "Invalid mode"
    return ic
##################################################################################################################################################################
def IC_DSIMP(kw, nw, mu, p_):
    """function to compute the information content of a multigraph graph pattern as proposed in DSIMP

    Args:
        kw (int): number of edges found in multigraph pattern
        nw (int): total number of feasible vertex-pair
        mu (float): expected number of edges in a multigraph pattern, i.e., sum of expected edges between all feasible vertex-pair
        p_ (float): minimum prabability of success among all geometric distribution between all feasible vertex pairs

    Returns:
        float: The information content (as per definition in DSIMP) of a multigraph pattern
    """
    ic = p_ * ( kw - mu ) + p_ * ( mu + nw ) * math.log ((mu + nw) / (kw + nw))
    # eps = kw/nw
    # ps = 1.0/(eps+1.0)
    # ic = KL_g(ps, p_)
    return ic
##################################################################################################################################################################
def IC_DSSG(CL_I, CL_F):
    """function to compute the information content of a pattern as proposed in DSSG, i.e., the gain in codelength of a model

    Args:
        CL_I (float): prior codelength or the number of bits required to encode the graph given the background distribution
        CL_F (float): posterior codelength or the number of bits required to encode the graph given the background distribution

    Returns:
        float: gain in codelength or information content
    """
    return CL_I - CL_F
##################################################################################################################################################################
def AD(kw, mu):
    """function to compute aggregate deviation of multigraph pattern as given in SIMP paper

    Args:
        kw (int): number of edges in a multigraph pattern
        mu (float): expected number of edges in a pattern as per the prior distribution

    Returns:
        float: the aggregate deviation
    """
    return kw - mu
##################################################################################################################################################################
def DL_Nodes(V, W, q):
    """function to compute the shannon-optimal code to describe the vertices in a graph pattern

    Args:
        V (int): number of nodes in a supergraph of original dataset
        W (int): number of nodes in a graph pattern
        q (float): expected size of a pattern, i.e., ratio of original dataset size, in range 0.0 to 1.0

    Returns:
        float: shannon-optimal code to describe W vertices
    """
    incCost = math.log((1.0 - q) / q)
    excCost = V * math.log(1.0 / (1.0 - q))
    dl = W * incCost + excCost
    return dl
##################################################################################################################################################################
def DL_Edges(nw, kw, isSimple=True, kws=None, delta=2):
    """
    function to encode the number of edges in a subgraph

    Parameters
    ----------
    nw : int
        Maximum possible edges in a simple graph or maximum number of feasible vertex-pair in simple or multigraph\\
    kw : int
        Number of edges in simple graph pattern or multigraph pattern\\
    isSimple : bool, optional
        True if Graph is a simple graph else false for multigraph, by default True\\
    kws : int, optional
        number of vertex-pair in a multigraph connected with each other by atleast one edge, by default None\\
    delta : int, optional
        accuracy to delta decimal points, by default 2

    Returns
    -------
    float
        number of bits required to communicate this information to the user
    """
    dl = 0
    if isSimple:
        remE = nw - kw
        dl = LN(remE)
    else:
        remE = nw - kws
        kappa = round(kw/kws, delta)
        dl += LN(remE)
        dl += math.log2(kappa+1)
        dl += math.log2(math.pow(10, delta))
    return dl
##################################################################################################################################################################
def computeSumOfEdgeProbablity(PD, **kwargs):
    """function to compute the sum of POS (Probability of success) of all possible random variables, i.e., all feasible vertex-pairs
    this fuction is mainly used in case of Bernoulli's distribution, i.e., simple graphs

    Args:
        PD (PDClass): Background distribution of the dataset, in this case it shall be a product of Bernoulli's distribution

    kwargs:
        gtype (str): 'D'-directed, 'U'-undirected\\
        NL (list): list of nodes, required if graph is orginally undirected\\
        inNL (list): list of inNodes, required if graph is originally directed\\
        outNL (list): list of outNodes, required if graph is originally directed\\
        case (int): Between 1-5 depending on the inclusion of Lagrangian multipliers to be counted while computing POS, default is 2\\
        dropLidx (int or list): index of lambda if required to be dropped, default is None\\
        nlambda (float): value of new lambda which shall be now included to compute POS, default is 0.0\\
        isSimple (Boolean): True, if graph is orginally a simple graph or False if it is a multigraph, default is True

    Returns:
        float: sum of POS of all distribution defined by different random variables, each representing a unique feasible vertex-pair defined by set of nodes NL(s)
    """
    assert 'gtype' in kwargs and (kwargs['gtype']=='U' or kwargs['gtype']=='D'), "gtype must be provided and it shall be either 'D'(Directed) or 'U' undirected"
    pw = 0.0
    if 'case' in kwargs:
        assert kwargs['case']>1 and 'dropLidx' in kwargs, "for case types 2-5, dropLidx shall be provided"
        assert kwargs['case']>1 and kwargs['case']%2==1 and 'dropLidx' in kwargs, "for case types 3 and 5, nlambda shall be provided"
    case = 2
    dropLidx = None
    nlambda = 0.0
    isSimple = True
    if 'case' in kwargs:
        case = kwargs['case']
    if 'dropLidx' in kwargs:
        dropLidx = kwargs['dropLidx']
    if 'nlambda' in kwargs:
        nlambda = kwargs['nlambda']
    if 'isSimple' in kwargs:
        isSimple = kwargs['isSimple']

    if kwargs['gtype'] == 'U':
        assert 'NL' in kwargs, "NL shall be provided if gtype is 'U'"
        NL = kwargs['NL']
        for i in range(len(NL)-1):
            for j in range(i+1, len(NL)):
                pw += PD.getPOS(NL[i], NL[j], case=case, dropLidx=dropLidx, nlambda=nlambda, isSimple=isSimple)
    else:
        assert 'inNL' in kwargs, "inNL shall be provided if gtype is 'D'"
        assert 'outNL' in kwargs, "outNL shall be provided if gtype is 'D'"
        inNL = kwargs['inNL']
        outNL = kwargs['outNL']
        for i in outNL:
            for j in inNL:
                if i != j:
                    pw += PD.getPOS(i, j, case=case, dropLidx=dropLidx, nlambda=nlambda, isSimple=isSimple)
    return pw
##################################################################################################################################################################
def computeSumOfExpectations(PD, **kwargs):
    """function to compute the sum of Expectations of all possible random variables, i.e., all feasible vertex-pairs
    this fuction is mainly used in case of Geometric's distribution, i.e., multigraphs. Note that, it can be used for
    simple graphs also, i.e, Bernoulli's Distribution

    Args:
        PD (PDClass): Background distribution of the dataset, in this case itcan be a product of Geometric's distribution

    kwargs:
        gtype (str): 'D'-directed, 'U'-undirected
        NL (list): list of nodes, required if graph is orginally undirected
        inNL (list): list of inNodes, required if graph is originally directed
        outNL (list): list of outNodes, required if graph is originally directed
        case (int): Between 1-5 depending on the inclusion of Lagrangian multipliers to be counted while computing POS, default is 2
        dropLidx (int or list): index of lambda if required to be dropped, default is None
        nlambda (float): value of new lambda which shall be now included to compute POS, default is 0.0
        isSimple (Boolean): True, if graph is orginally a simple graph or False if it is a multigraph, default is False

    Returns:
        float: sum of Expectations of all distribution defined by different random variables, each representing a unique feasible vertex-pair defined by set of nodes NL(s)
    """
    assert 'gtype' in kwargs and (kwargs['gtype']=='U' or kwargs['gtype']=='D'), "gtype must be provided and it shall be either 'D'(Directed) or 'U' undirected"
    SumExpect = 0.0
    if 'case' in kwargs:
        assert kwargs['case']>1 and 'dropLidx' in kwargs, "for case types 2-5, dropLidx shall be provided"
        assert kwargs['case']>1 and kwargs['case']%2==1 and 'dropLidx' in kwargs, "for case types 3 and 5, nlambda shall be provided"
    case = 2
    dropLidx = None
    nlambda = 0.0
    isSimple = False
    if 'case' in kwargs:
        case = kwargs['case']
    if 'dropLidx' in kwargs:
        dropLidx = kwargs['dropLidx']
    if 'nlambda' in kwargs:
        nlambda = kwargs['nlambda']
    if 'isSimple' in kwargs:
        isSimple = kwargs['isSimple']

    if kwargs['gtype'] == 'U':
        assert 'NL' in kwargs, "NL shall be provided if gtype is 'U'"
        NL = kwargs['NL']
        for i in range(len(NL)-1):
            for j in range(i+1, len(NL)):
                p = PD.getPOS(NL[i], NL[j], case=case, dropLidx=dropLidx, nlambda=nlambda, isSimple=isSimple)
                SumExpect += PD.getExpectation(NL[i], NL[j], case=case, dropLidx=dropLidx, nlambda=nlambda, isSimple=isSimple)
    else:
        assert 'inNL' in kwargs, "inNL shall be provided if gtype is 'D'"
        assert 'outNL' in kwargs, "outNL shall be provided if gtype is 'D'"
        inNL = kwargs['inNL']
        outNL = kwargs['outNL']
        for i in outNL:
            for j in inNL:
                if i != j:
                    SumExpect += PD.getExpectation(i, j, case=case, dropLidx=dropLidx, nlambda=nlambda, isSimple=isSimple)
    return SumExpect
##################################################################################################################################################################
def computeMinPOS(PD, **kwargs):
    """function to compute the minimum POS (Probability of success) among all possible random variables, i.e., all feasible vertex-pairs
    this fuction is mainly used in case of Geometric distribution, i.e., multigraphs

    Args:
        PD (PDClass): Background distribution of the dataset, in this case it shall be a product of Geometric distribution

    kwargs:
        gtype (str): 'D'-directed, 'U'-undirected
        NL (list): list of nodes, required if graph is orginally undirected
        inNL (list): list of inNodes, required if graph is originally directed
        outNL (list): list of outNodes, required if graph is originally directed
        case (int): Between 1-5 depending on the inclusion of Lagrangian multipliers to be counted while computing POS, default is 2
        dropLidx (int or list): index of lambda if required to be dropped, default is None
        nlambda (float): value of new lambda which shall be now included to compute POS, default is 0.0
        isSimple (Boolean): True, if graph is orginally a simple graph or False if it is a multigraph, default is True

    Returns:
        float: minimum POS among all distribution defined by different random variables, each representing a unique feasible vertex-pair defined by set of nodes NL(s)
    """
    assert 'gtype' in kwargs and (kwargs['gtype']=='U' or kwargs['gtype']=='D'), "gtype must be provided and it shall be either 'D'(Directed) or 'U' undirected"
    minp = float("inf")
    if 'case' in kwargs:
        assert kwargs['case']>1 and 'dropLidx' in kwargs, "for case types 2-5, dropLidx shall be provided"
        assert kwargs['case']>1 and kwargs['case']%2==1 and 'dropLidx' in kwargs, "for case types 3 and 5, nlambda shall be provided"
    case = 2
    dropLidx = None
    nlambda = 0.0
    isSimple = True
    if 'case' in kwargs:
        case = kwargs['case']
    if 'dropLidx' in kwargs:
        dropLidx = kwargs['dropLidx']
    if 'nlambda' in kwargs:
        nlambda = kwargs['nlambda']
    if 'isSimple' in kwargs:
        isSimple = kwargs['isSimple']

    if kwargs['gtype'] == 'U':
        assert 'NL' in kwargs, "NL shall be provided if gtype is 'U'"
        NL = kwargs['NL']
        for i in range(len(NL)-1):
            for j in range(i+1, len(NL)):
                minp = min(minp, PD.getPOS(NL[i], NL[j], case=case, dropLidx=dropLidx, nlambda=nlambda, isSimple=isSimple))
    else:
        assert 'inNL' in kwargs, "inNL shall be provided if gtype is 'D'"
        assert 'outNL' in kwargs, "outNL shall be provided if gtype is 'D'"
        inNL = kwargs['inNL']
        outNL = kwargs['outNL']
        for i in outNL:
            for j in inNL:
                if i != j:
                    minp = min(minp, PD.getPOS(i, j, case=case, dropLidx=dropLidx, nlambda=nlambda, isSimple=isSimple))
    return minp
##################################################################################################################################################################
def computePWparameters(PD, **kwargs):
    """function to compute the sum of Expectations and minPOS of all possible random variables, i.e., all feasible vertex-pairs
    this fuction is mainly used in case of Geometric's distribution, i.e., multigraphs. Note that, it can be used for
    simple graphs also, i.e, Bernoulli's Distribution

    Args:
        PD (PDClass): Background distribution of the dataset, in this case itcan be a product of Geometric's distribution

    kwargs:
        gtype (str): 'D'-directed, 'U'-undirected
        NL (list): list of nodes, required if graph is orginally undirected
        inNL (list): list of inNodes, required if graph is originally directed
        outNL (list): list of outNodes, required if graph is originally directed
        case (int): Between 1-5 depending on the inclusion of Lagrangian multipliers to be counted while computing POS, default is 2
        dropLidx (int or list): index of lambda if required to be dropped, default is None
        nlambda (float): value of new lambda which shall be now included to compute POS, default is 0.0
        isSimple (Boolean): True, if graph is orginally a simple graph or False if it is a multigraph, default is False

    Returns:
        float, float: sum of Expectations and minPOS of all distribution defined by different random variables, each representing a unique feasible vertex-pair defined by set of nodes NL(s)
    """
    assert 'gtype' in kwargs and (kwargs['gtype']=='U' or kwargs['gtype']=='D'), "gtype must be provided and it shall be either 'D'(Directed) or 'U' undirected"
    SumExpect = 0.0
    minp = float("inf")
    if 'case' in kwargs:
        assert kwargs['case']>1 and 'dropLidx' in kwargs, "for case types 2-5, dropLidx shall be provided"
        assert kwargs['case']>1 and kwargs['case']%2==1 and 'dropLidx' in kwargs, "for case types 3 and 5, nlambda shall be provided"
    case = 2
    dropLidx = None
    nlambda = 0.0
    isSimple = False
    if 'case' in kwargs:
        case = kwargs['case']
    if 'dropLidx' in kwargs:
        dropLidx = kwargs['dropLidx']
    if 'nlambda' in kwargs:
        nlambda = kwargs['nlambda']
    if 'isSimple' in kwargs:
        isSimple = kwargs['isSimple']

    if kwargs['gtype'] == 'U':
        assert 'NL' in kwargs, "NL shall be provided if gtype is 'U'"
        NL = kwargs['NL']
        for i in range(len(NL)-1):
            for j in range(i+1, len(NL)):
                p = PD.getPOS(NL[i], NL[j], case=case, dropLidx=dropLidx, nlambda=nlambda, isSimple=isSimple)
                minp = min(minp, p)
                SumExpect += PD.getExpectationFromPOS(p)
    else:
        assert 'inNL' in kwargs, "inNL shall be provided if gtype is 'D'"
        assert 'outNL' in kwargs, "outNL shall be provided if gtype is 'D'"
        inNL = kwargs['inNL']
        outNL = kwargs['outNL']
        for i in outNL:
            for j in inNL:
                if i != j:
                    p = PD.getPOS(i, j, case=case, dropLidx=dropLidx, nlambda=nlambda, isSimple=isSimple)
                    minp = min(minp, p)
                    SumExpect += PD.getExpectationFromPOS(p)
    return SumExpect, minp
##################################################################################################################################################################
def computeDescriptionLength(**kwargs):
    """function to compute description length of a pattern or an action
    --------------------
    Args:
        None
    --------------------
    kwargs:
        dlmode (int): 1 for encoding only nodes (used in SSG and SIMP), 2 for encoding nodes with #edges,
                    3 for encoding nodes, edges and action type (for add action),
                    4 for encoding lambda/pattern/constraint id and action type (for remove action),
                    5 for encoding action type, lambda/pattern/constraint id and #edges (for update action),
                    6 for encoding action type, lambda/pattern/constraint id, #nodes removed, nodes and updated edges (for shrink action)
                    7 for encoding action type, lambda/pattern/constraint id, #resulting patterns,
                        #nodes in each pattern, nodes in each pattern and #edges in each pattern (for split action)
                    8 for encoding action type, two lambda/pattern/constraint id, #edges in the resulting pattern (for merge action),
                    default is 1
        excActionType (boolean): if information to exclude action type in DL, only valid if dlmode is 3-8, default is true
        l (int): number of actions defined, required if dlmode is 3-8.
        gtype (str): 'U' for undirected and 'D' for directed. Default is 'U'.
        q (float): parameter of shannon-optimal code which is expected size of a pattern, i.e., ratio of original dataset size, in range 0.0 to 1.0. Required if dlmode is 1,2 or 3
        V (int): number of nodes in a dataset. Required if dlmode is 1,2 or 3
        W (int): number of nodes in a pattern. Required if gtype is 'U' and dlmode is 1-8 except 4 and 7
        WI (int or list): number of inNodes or list of inNodes in a pattern. Required if gtype is 'D' and dlmode is 1-8 except 4 and 7
        WO (int or list): number of outNodes or list of outNodes in a pattern. Required if gtype is 'D' and dlmode is 1-8 except 4 and 7
        kw (int): number of edges in a pattern. Required if dlmode is 2-8 except 4 and 7
        C (int): number of constraints, required if dlmode is 3-8
        WS (int): number of nodes initially present in a pattern before shrink (dlmode: 6) and split (dlmode: 7), required if gtype is 'U'
        WIS (int): number of inNodes initially present in a pattern before shrink (dlmode: 6) and split (dlmode: 7), required if gtype is 'D'
        WOS(int): number of outNodes initially present in a pattern before shrink (dlmode: 6) and split (dlmode: 7), required if gtype is 'D'
        compos (dict): dictiornary of connected components after split. Required if dlmode is 7. If gtype is 'U' compos should be a dictionary with keys as component identifier (int) and value as a connected component (networkx graph)
                        If gtype is 'D' it should be a dictionary with key (int) and value a 3 length tuple where first shall be networkx DiGraph, and rest shall be two list of IN and OUT Nodes
        isSimple (bool): True if the graph is a simple graph, false if multigraph, default is True
        kws (int): Number of edges in a simple graph equivalent of a multigraph, required if isSimple is False
    --------------------
    Returns:
        float: the required descroption length
    """
    dlmode = 1
    if 'dlmode' in kwargs:
        dlmode = kwargs['dlmode']

    gtype = 'U'
    if 'gtype' in kwargs:
        gtype = kwargs['gtype']

    excActionType = True
    if 'excActionType' in kwargs:
        assert 'l' in kwargs, "'l' (number of action types) is required"
        excActionType = kwargs['excActionType']
    isSimple = True
    if 'isSimple' in kwargs:
        isSimple = kwargs['isSimple']
        if not isSimple:
            assert 'kws' in kwargs, "'kws' is required to encode edges if isSimple is False"

    if dlmode == 1:
        DL = 0.0
        if gtype == 'U':
            assert 'V' in kwargs and 'W' in kwargs and 'q' in kwargs, "'V', 'W' and 'q' shall be provide if dlmode is 1 and gtype is 'U'"
            DL += DL_Nodes(kwargs['V'], kwargs['W'], kwargs['q']) ## encoding nodes in pattern
        else:
            assert 'V' in kwargs and 'WI' in kwargs and 'WO' in kwargs and 'q' in kwargs, "'V', 'WI', 'WO' and 'q' shall be provide if dlmode is 1 and gtype is 'D'"
            DL += DL_Nodes(kwargs['V'], kwargs['WI'], kwargs['q']) ## encoding inNodes in pattern
            DL += DL_Nodes(kwargs['V'], kwargs['WO'], kwargs['q']) ## encoding outNodes in pattern
        return DL
    elif dlmode == 2 or dlmode == 3: #add action
        DL = 0.0
        nw = 0
        assert 'kw' in kwargs, "kw is not provided"
        if gtype == 'U':
            assert 'V' in kwargs and 'W' in kwargs and 'q' in kwargs, "'V', 'W' and 'q' shall be provide if dlmode is 2 or 3 and gtype is 'U'"
            DL += DL_Nodes(kwargs['V'], kwargs['W'], kwargs['q']) ## encoding nodes in pattern
            nw = NW(kwargs['W'])
        else:
            assert 'V' in kwargs and 'WI' in kwargs and 'WO' in kwargs and 'q' in kwargs, "'V', 'WI', 'WO' and 'q' shall be provide if dlmode is 2 or 3 and gtype is 'D'"
            if isinstance(kwargs['WI'], list) and isinstance(kwargs['WO'], list):
                DL += DL_Nodes(kwargs['V'], len(kwargs['WI']), kwargs['q']) ## encoding inNodes in pattern
                DL += DL_Nodes(kwargs['V'], len(kwargs['WO']), kwargs['q']) ## encoding outNodes in pattern
                nw = NW_D(kwargs['WI'], kwargs['WO'])
            else:
                DL += DL_Nodes(kwargs['V'], kwargs['WI'], kwargs['q']) ## encoding inNodes in pattern
                DL += DL_Nodes(kwargs['V'], kwargs['WO'], kwargs['q']) ## encoding outNodes in pattern
                assert 'nw' in kwargs, "nw is required if WI and WO are length and not list"
                nw = kwargs['nw']
        if isSimple:
            DL += DL_Edges(nw, kwargs['kw']) ## encoding number of edges in a pattern
        else:
            DL += DL_Edges(nw, kwargs['kw'], isSimple, kwargs['kws'], 2)
        if dlmode == 3 and not excActionType:
            assert 'l' in kwargs,"'l' not provided"
            DL += math.log2(kwargs['l']) ## encoding action type
        return DL
    elif dlmode == 4: #remove action
        DL = 0.0
        assert 'C' in kwargs, "number of constraints 'C' is required"
        DL += math.log2(kwargs['C']) ## encoding constraint id
        if not excActionType:
            DL += math.log2(kwargs['l']) ## encoding action type
        return DL
    elif dlmode == 5: #update action
        DL = 0.0
        assert 'C' in kwargs, "number of constraints 'C' is required"
        DL += math.log2(kwargs['C']) ## encoding constraint id
        nw = 0
        if gtype == 'U':
            assert 'W' in kwargs and 'kw' in kwargs, "'W' and 'kw' are required if dlmode is 5 and gtype is 'U'"
            nw = NW(kwargs['W'])
        else:
            assert 'WI' in kwargs and 'WO' in kwargs and 'kw' in kwargs, "'WI', 'WO' and 'kw' are required if dlmode is 5 and gtype is 'D'"
            nw = NW_D(kwargs['WI'], kwargs['WO'])
        if isSimple:
            DL += DL_Edges(nw, kwargs['kw']) ## encoding number of edges in a pattern
        else:
            DL += DL_Edges(nw, kwargs['kw'], isSimple, kwargs['kws'], 2)
        if not excActionType:
            DL += math.log2(kwargs['l']) ## encoding action type
        return DL
    elif dlmode == 6: #shrink action
        DL = 0.0
        assert 'C' in kwargs, "number of constraints 'C' is required"
        DL += math.log2(kwargs['C']) ## encoding constraint id
        nw = 0
        if gtype == 'U':
            assert 'W' in kwargs and 'kw' in kwargs, "'W' and 'kw' are required if dlmode is 6 and gtype is 'U'"
            nw = NW(kwargs['W'])
        else:
            assert 'WI' in kwargs and 'WO' in kwargs and 'kw' in kwargs, "'WI', 'WO' and 'kw' are required if dlmode is 6 and gtype is 'D'"
            nw = NW_D(kwargs['WI'], kwargs['WO'])
        if isSimple:
            DL += DL_Edges(nw, kwargs['kw']) ## encoding number of edges in a pattern
        else:
            DL += DL_Edges(nw, kwargs['kw'], isSimple, kwargs['kws'], 2)
        if gtype == 'U':
            assert 'WS' in kwargs, "'WS' (initial #nodes is required if dlmode is 6 gtype is 'U'"
            DL += LN(kwargs['WS'] - kwargs['W']) ## encoding number of nodes removed in a pattern
            DL += math.log2(ncr(kwargs['WS'], kwargs['WS'] - kwargs['W'])) ## encoding nodes removed in a pattern
        else:
            assert 'WIS' in kwargs and 'WOS' in kwargs, "'WIS' (initial #inNodes) and 'WOS' (initial #outNodes) are required if dlmode is 6 gtype is 'D'"
            DL += LN(kwargs['WIS'] - len(kwargs['WI'])) ## encoding number of inNodes removed in a pattern
            DL += LN(kwargs['WOS'] - len(kwargs['WO'])) ## encoding number of outNodes removed in a pattern
            DL += math.log2(ncr(kwargs['WIS'], kwargs['WIS'] - len(kwargs['WI']))) ## encoding inNodes removed in a pattern
            DL += math.log2(ncr(kwargs['WOS'], kwargs['WOS'] - len(kwargs['WO']))) ## encoding outNodes removed in a pattern
        if not excActionType:
            DL += math.log2(kwargs['l']) ## encoding action type
        return DL
    elif dlmode == 7: #split action
        DL = 0.0
        # Todo check this code
        assert 'C' in kwargs, "number of constraints 'C' is required"
        DL += math.log2(kwargs['C']) ## encoding constraint id
        assert 'compos' in kwargs, "connected components ('comps') of a pattern is required"
        if gtype == 'U':
            assert isinstance(kwargs['compos'], dict) and isinstance(kwargs['compos'][0], nx.Graph), "'comps' shall be a dictionary with key (int) and value (networkx graph)"
            DL += LN(len(kwargs['compos'])) ### encoding number of components
            WST = 0 # to count sum of nodes in each component
            for k,v in kwargs['compos'].items():
                DL += LN(v.NCount) ## encoding number of nodes in each component
                if isSimple:
                    DL += DL_Edges(NW(v.NCount), v.ECount) ## encoding number of edges in a pattern
                else:
                    DL += DL_Edges(NW(v.NCount), v.ECount, isSimple, v.kws, 2)## encoding number of edges in each component
                WST += v.NCount
            assert 'WS' in kwargs, "'WS' (initial #nodes) is required if dlmode is 7 and gtype is 'U'"
            DL += math.log2(ncr(kwargs['WS'], WST)) ## encoding nodes in each component
        else:
            assert isinstance(kwargs['compos'], dict) and len(kwargs['compos'][0]) == 3 and isinstance(kwargs['compos'][0][0], nx.Graph) and isinstance(kwargs['compos'][0][1], list) and isinstance(kwargs['compos'][0][2], list), "'comps' shall be a dictionary with key (int) and value a 3 length tuple where first shall be networkx DiGraph, and rest shall be two list of IN and OUT Nodes"
            DL += LN(len(kwargs['compos'])) ### encoding number of components
            WST_i = 0 # to count sum of inNodes in each component
            WST_o = 0 # to count sum of outNodes in each component
            for k,v in kwargs['compos'].items():
                DL += LN(v.InNCount) ## encoding number of inNodes in each component
                DL += LN(v.OutNCount) ## encoding number of outNodes in each component
                if isSimple:
                    DL += DL_Edges(NW_D(v.inNL, v.outNL), v.ECount) ## encoding number of edges in a pattern
                else:
                    DL += DL_Edges(NW_D(v.inNL, v.outNL), v.ECount, isSimple, v.kws, 2) ## encoding number of edges in each component
                WST_i += LN(v.InNCount)
                WST_o += LN(v.OutNCount)
            assert 'WIS' in kwargs and 'WOS' in kwargs, "'WIS' (initial #inNodes) and 'WOS' (initial #outNodes) are required if dlmode is 7 gtype is 'D'"
            DL += math.log2(ncr(kwargs['WIS'], WST_i)) ## encoding inNodes in each component
            DL += math.log2(ncr(kwargs['WOS'], WST_o)) ## encoding outNodes in each component
        if not excActionType:
            DL += math.log2(kwargs['l']) ## encoding action type
        return DL
    elif dlmode == 8: #merge action
        DL = 0.0
        assert 'C' in kwargs, "number of constraints 'C' is required"
        DL += math.log2(ncr(kwargs['C'], 2)) ## encoding two constraint ids
        nw = 0
        if gtype == 'U':
            assert 'W' in kwargs and 'kw' in kwargs, "'W' and 'kw' are required if dlmode is 8 and gtype is 'U'"
            nw = NW(kwargs['W'])
        else:
            assert 'WI' in kwargs and 'WO' in kwargs and 'kw' in kwargs, "'WI', 'WO' and 'kw' are required if dlmode is 8 and gtype is 'D'"
            nw = NW_D(kwargs['WI'], kwargs['WO'])
        if isSimple:
            DL += DL_Edges(nw, kwargs['kw']) ## encoding number of edges in a pattern
        else:
            DL += DL_Edges(nw, kwargs['kw'], isSimple, kwargs['kws'], 2)
        if not excActionType:
            DL += math.log2(kwargs['l']) ## encoding action type
        return DL
    return
##################################################################################################################################################################
def computeInterestingness(IC, DL, **kwargs):
    """function to compute interestingness of a pattern

    Args:
        IC (float): Information Content
        DL (float): Description Length

    kwargs:
        mode (int): 1 if I = IC/DL, 2 if I = IC - DL, default is 1

    Returns:
        float: Interestingness score
    """
    mode = 1
    if 'mode' in kwargs:
        mode = kwargs['mode']
    I = 0.0
    if mode == 1:
        I = IC / DL
    elif mode == 2:
        I = IC - DL
    else:
        assert mode < 3, "Incorrect mode"
    return I
##################################################################################################################################################################
def computeSumOfEdgeProbablityBetweenNodeAndList(PD, node, NL, **kwargs):
    """function to compute the sum of POS (Probability of success) of all possible random variables,
        i.e., all feasible vertex-pairs with one vertex as provide node and second from a list of nodes (NL).

    Args:
        PD (PDClass): Background distribution
        node (int): node id
        NL (list): list of nodes

    kwargs:
        gtype (str): 'D'-directed, 'U'-undirected
        dir_mode (int): required id gtype is 'D"; 1 - from node to list and 2 from list to node
        case (int): Between 1-5 depending on the inclusion of Lagrangian multipliers to be counted while computing POS, default is 2
        dropLidx (int or list): index of lambda if required to be dropped, default is None
        nlambda (float): value of new lambda which shall be now included to compute POS, default is 0.0
        isSimple (Boolean): True, if graph is orginally a simple graph or False if it is a multigraph, default is True


    Returns:
        float: required sum of POS
    """
    assert 'gtype' in kwargs and (kwargs['gtype']=='U' or kwargs['gtype']=='D'), "gtype must be provided and it shall be either 'D'(Directed) or 'U' undirected"
    pw = 0.0
    if 'case' in kwargs:
        assert kwargs['case']>1 and 'dropLidx' in kwargs, "for case types 2-5, dropLidx shall be provided"
        assert kwargs['case']>1 and kwargs['case']%2==1 and 'dropLidx' in kwargs, "for case types 3 and 5, nlambda shall be provided"
    case = 2
    dropLidx = None
    nlambda = 0.0
    isSimple = True
    dir_mode = None

    if 'case' in kwargs:
        case = kwargs['case']
    if 'dropLidx' in kwargs:
        dropLidx = kwargs['dropLidx']
    if 'nlambda' in kwargs:
        nlambda = kwargs['nlambda']
    if 'isSimple' in kwargs:
        isSimple = kwargs['isSimple']
    if kwargs['gtype'] == 'D':
        assert 'dir_mode' in kwargs, 'dir_mode is required if gtype is \'D\', it can be either 1 or 2'
        dir_mode = kwargs['dir_mode']

    if kwargs['gtype'] == 'U' or (kwargs['gtype'] == 'D' and dir_mode == 1):
        for i in NL:
            pw += PD.getPOS(node, i, case=case, dropLidx=dropLidx, nlambda=nlambda, isSimple=isSimple)
    else:
        for i in NL:
            pw += PD.getPOS(i, node, case=case, dropLidx=dropLidx, nlambda=nlambda, isSimple=isSimple)
    return pw
##################################################################################################################################################################
def computeSumOfExpectationsBetweenNodeAndList(PD, node, NL, **kwargs):
    """function to compute the sum of Expectation of all possible random variables,
        i.e., all feasible vertex-pairs with one vertex as provide node and other from a list of nodes (NL).

    Args:
        PD (PDClass): Background distribution of the dataset
        node (int): node id
        NL (list): list of nodes

    kwargs:
        gtype (str): 'D'-directed, 'U'-undirected
        dir_mode (int): required id gtype is 'D"; 1 - from node to list and 2 from list to node
        case (int): Between 1-5 depending on the inclusion of Lagrangian multipliers to be counted while computing POS, default is 2
        dropLidx (int or list): index of lambda if required to be dropped, default is None
        nlambda (float): value of new lambda which shall be now included to compute POS, default is 0.0
        isSimple (Boolean): True, if graph is orginally a simple graph or False if it is a multigraph, default is False

    Returns:
        float: required sum of Expectations
    """
    assert 'gtype' in kwargs and (kwargs['gtype']=='U' or kwargs['gtype']=='D'), "gtype must be provided and it shall be either 'D'(Directed) or 'U' undirected"
    SumExpect = 0.0
    if 'case' in kwargs:
        assert kwargs['case']>1 and 'dropLidx' in kwargs, "for case types 2-5, dropLidx shall be provided"
        assert kwargs['case']>1 and kwargs['case']%2==1 and 'dropLidx' in kwargs, "for case types 3 and 5, nlambda shall be provided"
    case = 2
    dropLidx = None
    nlambda = 0.0
    isSimple = True
    dir_mode = None
    if 'case' in kwargs:
        case = kwargs['case']
    if 'dropLidx' in kwargs:
        dropLidx = kwargs['dropLidx']
    if 'nlambda' in kwargs:
        nlambda = kwargs['nlambda']
    if 'isSimple' in kwargs:
        isSimple = kwargs['isSimple']
    if kwargs['gtype'] == 'D':
        assert 'dir_mode' in kwargs, 'dir_mode is required if gtype is \'D\', it can be either 1 or 2'
        dir_mode = kwargs['dir_mode']

    if kwargs['gtype'] == 'U' or (kwargs['gtype'] == 'D' and dir_mode == 1):
        for i in NL:
            SumExpect += PD.getExpectation(node, i, case=case, dropLidx=dropLidx, nlambda=nlambda, isSimple=isSimple)
    else:
        for i in NL:
            SumExpect += PD.getExpectation(i, node, case=case, dropLidx=dropLidx, nlambda=nlambda, isSimple=isSimple)
    return SumExpect
##################################################################################################################################################################
def computeMinPOSBetweenNodeAndList(PD, node, NL, **kwargs):
    """function to compute the minPOS of all possible random variables,
        i.e., all feasible vertex-pairs with one vertex as provide node and other from a list of nodes (NL).

    Args:
        PD (PDClass): Background distribution of the dataset
        node (int): node id
        NL (list): list of nodes

    kwargs:
        gtype (str): 'D'-directed, 'U'-undirected
        dir_mode (int): required id gtype is 'D"; 1 - from node to list and 2 from list to node; default is 1
        case (int): Between 1-5 depending on the inclusion of Lagrangian multipliers to be counted while computing POS, default is 2
        dropLidx (int or list): index of lambda if required to be dropped, default is None
        nlambda (float): value of new lambda which shall be now included to compute POS, default is 0.0
        isSimple (Boolean): True, if graph is orginally a simple graph or False if it is a multigraph, default is False

    Returns:
        float: required sum of Expectations
    """
    assert 'gtype' in kwargs and (kwargs['gtype']=='U' or kwargs['gtype']=='D'), "gtype must be provided and it shall be either 'D'(Directed) or 'U' undirected"
    minp = float("inf")
    if 'case' in kwargs:
        assert kwargs['case']>1 and 'dropLidx' in kwargs, "for case types 2-5, dropLidx shall be provided"
        assert kwargs['case']>1 and kwargs['case']%2==1 and 'dropLidx' in kwargs, "for case types 3 and 5, nlambda shall be provided"
    case = 2
    dropLidx = None
    nlambda = 0.0
    isSimple = True
    dir_mode = None
    if 'case' in kwargs:
        case = kwargs['case']
    if 'dropLidx' in kwargs:
        dropLidx = kwargs['dropLidx']
    if 'nlambda' in kwargs:
        nlambda = kwargs['nlambda']
    if 'isSimple' in kwargs:
        isSimple = kwargs['isSimple']
    if kwargs['gtype'] == 'D':
        assert 'dir_mode' in kwargs, 'dir_mode is required if gtype is \'D\', it can be either 1 or 2'
        dir_mode = kwargs['dir_mode']

    if kwargs['gtype'] == 'U' or (kwargs['gtype'] == 'D' and dir_mode == 1):
        for i in NL:
            minp = min(minp, PD.getPOS(node, i, case=case, dropLidx=dropLidx, nlambda=nlambda, isSimple=isSimple))
    else:
        for i in NL:
            minp = min(minp, PD.getPOS(i, node, case=case, dropLidx=dropLidx, nlambda=nlambda, isSimple=isSimple))
    return minp
##################################################################################################################################################################
def computePWparametersBetweenNodeAndList(PD, node, NL, **kwargs):
    """function to compute the sum of Expectations and minPOS of all possible random variables, 
    i.e., all feasible vertex-pairs with one vertex as provide node and other from a list of nodes (NL).

    Args:
        PD (PDClass): Background distribution of the dataset
        node (int): node id
        NL (list): list of nodes

    kwargs:
        gtype (str): 'D'-directed, 'U'-undirected
        dir_mode (int): required id gtype is 'D"; 1 - from node to list and 2 from list to node
        case (int): Between 1-5 depending on the inclusion of Lagrangian multipliers to be counted while computing POS, default is 2
        dropLidx (int or list): index of lambda if required to be dropped, default is None
        nlambda (float): value of new lambda which shall be now included to compute POS, default is 0.0
        isSimple (Boolean): True, if graph is orginally a simple graph or False if it is a multigraph, default is False

    Returns:
        float, float: required sum of Expectations and minPOS
    """
    assert 'gtype' in kwargs and (kwargs['gtype']=='U' or kwargs['gtype']=='D'), "gtype must be provided and it shall be either 'D'(Directed) or 'U' undirected"
    SumExpect = 0.0
    minp = float("inf")
    if 'case' in kwargs:
        assert kwargs['case']>1 and 'dropLidx' in kwargs, "for case types 2-5, dropLidx shall be provided"
        assert kwargs['case']>1 and kwargs['case']%2==1 and 'dropLidx' in kwargs, "for case types 3 and 5, nlambda shall be provided"
    case = 2
    dropLidx = None
    nlambda = 0.0
    isSimple = True
    dir_mode = None
    if 'case' in kwargs:
        case = kwargs['case']
    if 'dropLidx' in kwargs:
        dropLidx = kwargs['dropLidx']
    if 'nlambda' in kwargs:
        nlambda = kwargs['nlambda']
    if 'isSimple' in kwargs:
        isSimple = kwargs['isSimple']
    if kwargs['gtype'] == 'D':
        assert 'dir_mode' in kwargs, 'dir_mode is required if gtype is \'D\', it can be either 1 or 2'
        dir_mode = kwargs['dir_mode']

    if kwargs['gtype'] == 'U' or (kwargs['gtype'] == 'D' and dir_mode == 1):
        for i in NL:
            if i != node:
                p = PD.getPOS(node, i, case=case, dropLidx=dropLidx, nlambda=nlambda, isSimple=isSimple)
                minp = min(minp, p)
                SumExpect += PD.getExpectationFromPOS(p)
    else:
        for i in NL:
            if i != node:
                p = PD.getPOS(i, node, case=case, dropLidx=dropLidx, nlambda=nlambda, isSimple=isSimple)
                minp = min(minp, p)
                SumExpect += PD.getExpectationFromPOS(p)
    return SumExpect, minp
##################################################################################################################################################################
def getDirectedSubgraph(G, WI, WO, isSimple):
    """function to compose a subgraph from given list of in-nodes and out-nodes

    Args:
        G (DiGraph or MultiDiGraph): SuperGraph
        WI (list): In-Nodes
        WO (list): Out-Nodes
        isSimple (bool): True if G is DiGraph, False if G is MultiDiGraph

    Returns:
        DiGraph or MultiDiGraph: resultant subgraph
    """
    H = None
    if isSimple:
        H = nx.DiGraph()
        for u in WO:
            for v in WI:
                if u != v and G.has_edges(u, v):
                    H.add_edge(u,v)
    else:
        H = nx.MultiDiGraph()
        for u in WO:
            for v in WI:
                if u !=v and G.has_edge(u, v):
                    H.add_edges_from([tuple([u, v])]*G.number_of_edges(u, v))
    return H
##################################################################################################################################################################
def getCodeLength(G, PD, **kwargs):
    assert 'gtype' in kwargs, "gtype is must to compute codelength"
    if 'case' in kwargs:
        assert kwargs['case']>1 and 'dropLidx' in kwargs, "for case types 2-5, dropLidx shall be provided"
        assert kwargs['case']>1 and kwargs['case']%2==1 and 'dropLidx' in kwargs, "for case types 3 and 5, nlambda shall be provided"
    case = 2
    dropLidx = None
    nlambda = 0.0
    isSimple = True
    if 'case' in kwargs:
        case = kwargs['case']
    if 'dropLidx' in kwargs:
        dropLidx = kwargs['dropLidx']
    if 'nlambda' in kwargs:
        nlambda = kwargs['nlambda']
    if 'isSimple' in kwargs:
        isSimple = kwargs['isSimple']
    codelength = 0.0
    if isSimple == True:
        if kwargs['gtype'] == 'U':
            assert 'NL' in kwargs, "NL is required if gtype is 'U'"
            NL = kwargs['NL']
            for i in range(len(NL)-1):
                    for j in range(i+1, len(NL)):
                        pos = PD.getPOS(NL[i], NL[j], case=case, dropLidx=dropLidx, nlambda=nlambda, isSimple=isSimple)
                        numE = G.number_of_edges(NL[i], NL[j])
                        codelength += math.log2(math.pow(1.0-pos, 1.0-numE)*math.pow(pos, numE))
        else:
            assert 'inNL' in kwargs and 'outNL' in kwargs, "inNL and outNL are required if gtype is 'D'"
            inNL = kwargs['inNL']
            outNL = kwargs['outNL']
            for v in inNL:
                for u in outNL:
                    if u != v:
                        pos = PD.getPOS(u, v, case=case, dropLidx=dropLidx, nlambda=nlambda, isSimple=isSimple)
                        numE = G.number_of_edges(u, v)
                        codelength += math.log2(math.pow(1.0-pos, 1.0-numE)*math.pow(pos, numE))
    else:
        if kwargs['gtype'] == 'U':
            assert 'NL' in kwargs, "NL is required if gtype is 'U'"
            NL = kwargs['NL']
            for i in range(len(NL)-1):
                    for j in range(i+1, len(NL)):
                        pos = PD.getPOS(NL[i], NL[j], case=case, dropLidx=dropLidx, nlambda=nlambda, isSimple=isSimple)
                        numE = G.number_of_edges(NL[i], NL[j])
                        codelength += math.log2(math.pow(1.0-pos, numE)*pos)
        else:
            assert 'inNL' in kwargs and 'outNL' in kwargs, "inNL and outNL are required if gtype is 'D'"
            inNL = kwargs['inNL']
            outNL = kwargs['outNL']
            for v in inNL:
                for u in outNL:
                    if u != v:
                        pos = PD.getPOS(u, v, case=case, dropLidx=dropLidx, nlambda=nlambda, isSimple=isSimple)
                        numE = G.number_of_edges(u, v)
                        codelength += math.log2(math.pow(1.0-pos, numE)*pos)
    return -codelength
##################################################################################################################################################################
def getCodeLengthParallel(G, PD, **kwargs):
    assert 'gtype' in kwargs, "gtype is must to compute codelength"
    if 'case' in kwargs:
        if kwargs['case']>3:
            assert 'dropLidx' in kwargs and isinstance(kwargs['dropLidx'], list), "for case types 4-5, dropLidx (list) shall be provided"
        if kwargs['case']>1 and kwargs['case']%2==1:
            assert 'nlambda' in kwargs, "for case types 3 and 5, nlambda shall be provided"
    case = 2
    dropLidx = None
    nlambda = 0.0
    isSimple = True
    if 'case' in kwargs:
        case = kwargs['case']
    if 'dropLidx' in kwargs:
        dropLidx = kwargs['dropLidx']
    if 'nlambda' in kwargs:
        nlambda = kwargs['nlambda']
    if 'isSimple' in kwargs:
        isSimple = kwargs['isSimple']
    codelength = 0.0
    Gid = ray.put(G)
    PDid = ray.put(PD)
    if kwargs['gtype'] == 'U':
        assert 'NL' in kwargs, "NL is required if gtype is 'U'"
        NL = kwargs['NL']
        codelength = sum(ray.get([getCodeLengthUtil.remote(Gid, PDid, NL[i], NL[i+1:], case=case, dropLidx=dropLidx, nlambda=nlambda, isSimple=isSimple) for i in range(len(NL)-1)]))
    else:
        assert 'inNL' in kwargs and 'outNL' in kwargs, "inNL and outNL are required if gtype is 'D'"
        inNL = kwargs['inNL']
        outNL = kwargs['outNL']
        codelength = sum(ray.get([getCodeLengthUtil.remote(Gid, PDid, u, inNL, case=case, dropLidx=dropLidx, nlambda=nlambda, isSimple=isSimple) for u in outNL]))
    return -codelength

@ray.remote
def getCodeLengthUtil(G, PD, node, lst, **kwargs):
    codelength = 0.0
    if kwargs['isSimple']:
        for v in lst:
            if node != v:
                pos = PD.getPOS(node, v, case=kwargs['case'], dropLidx=kwargs['dropLidx'], nlambda=kwargs['nlambda'], isSimple=kwargs['isSimple'])
                numE = G.number_of_edges(node, v)
                codelength += math.log2(math.pow(1.0-pos, 1.0-numE)*math.pow(pos, numE))
    else:
        for v in lst:
            if node != v:
                pos = PD.getPOS(node, v, case=kwargs['case'], dropLidx=kwargs['dropLidx'], nlambda=kwargs['nlambda'], isSimple=kwargs['isSimple'])
                numE = G.number_of_edges(node, v)
                codelength += math.log2(math.pow(1.0-pos, numE)*pos)
    return codelength
###################################################################################################################################################################
#*#################################################################################################################################################################
#?#################################################################################################################################################################
#*#################################################################################################################################################################
