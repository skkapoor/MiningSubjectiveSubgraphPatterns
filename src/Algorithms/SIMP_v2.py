###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################
import os
import shutil
import sys

import pandas as pd
path = os.getcwd().split('MiningSubjectiveSubgraphPatterns')[0]+'MiningSubjectiveSubgraphPatterns/'
if path not in sys.path:
	sys.path.append(path)
import networkx as nx
import time
import argparse
import configparser
import ast
import ray

from src.BackgroundDistributions.MaxEntMulti1 import MaxEntMulti1U as PDMEM1U
from src.BackgroundDistributions.MaxEntMulti1 import MaxEntMulti1D as PDMEM1D
from src.BackgroundDistributions.MaxEntMulti2 import MaxEntMulti2U as PDMEM2U
from src.BackgroundDistributions.MaxEntMulti2 import MaxEntMulti2D as PDMEM2D
from src.BackgroundDistributions.UniDistMulti import UniDistMulti as PDUDM

from src.HillClimbers.HC_v5 import HillClimber as HC
###################################################################################################################################################################
def parseStr(x):
    """
    function to parse a string

    Parameters
    ----------
    x : str
        input string

    Returns
    -------
    int or float or str
        parsed string
    """
    try:
        return int(x)
    except ValueError:
        try:
            return float(x)
        except ValueError:
            return x
###################################################################################################################################################################
def readConfFile(fname):
    """
    function to read the config file to run the experiment

    Parameters
    ----------
    fname : str
        input configuration file name

    Returns
    -------
    list, dict
        list of datasets and dictionary of required parameters

    Raises
    ------
    Exception
        if configuration file is not found
    """
    config = configparser.ConfigParser()
    if os.path.exists(path+'Confs/'+fname):
        config.read(path+'Confs/'+fname)
        DS = ast.literal_eval(config['Datasets']['DS'])
        Params = dict()
        for i in config['SIMP Params'].items():
            Params[i[0]] = parseStr(i[1])
        return DS, Params
    else:
        raise Exception('Configuration file does not exists:', path+'Confs/'+fname)
    return
###################################################################################################################################################################
def getGraphAndBD(d, Params):
    """
    function to read the graph and corresponding background distribution of the given type of prior belief

    Parameters
    ----------
    gname : str
        input filename of the graph
    Params : dict
        input parameters to run the experiment

    Returns
    -------
    networkx graph, PDClass, str
        Corresponding networkx graph, background distribution and graph type (undirected 'U' or directed 'D') respectively

    Raises
    ------
    Exception
        more than one file is found in the input directory of gml or gpickle type
    Exception
        no input file type is found
    Exception
        if a specific type of prior belief given as input which is not implemented yet
    """
    files = None
    if os.path.exists(path+'Data/SIMP/'+d):
        files = os.listdir(path+'Data/SIMP/'+d)
    Ffiles = dict()
    G = None
    PD = None
    print(files)
    print(Params)
    for f in files:
        if '.gml' in f:
            if '.gml' not in Ffiles:
                Ffiles['.gml'] = f
            else:
                raise Exception('There shall be only one .GML file in the data directory')
        elif '.gpickle' in f:
            if '.gpickle' not in Ffiles:
                Ffiles['.gpickle'] = f
            else:
                raise Exception('There shall be only one .gpickle file in the data directory')
    if '.gml' not in Ffiles and '.gpickle' not in Ffiles:
        raise Exception('No input graph file in the data directory. The directory shall contain a .gml or .gpickle file of input graph.')
    else:
        if '.gpickle' in Ffiles:
            G = nx.read_gpickle(path+'Data/SIMP/'+d+'/'+Ffiles['.gpickle'])
        else:
            G = nx.read_gml(path+'Data/SIMP/'+d+'/'+Ffiles['.gml'], destringizer=nx.readwrite.gml.literal_destringizer)

        if Params['priorbelief'] is 'c':
            PD = PDUDM(G)
        elif Params['priorbelief'] is 'i':
            if G.is_directed():
                PD = PDMEM1D(G)
            else:
                PD = PDMEM1U(G)
        elif Params['priorbelief'] is 'm':
            if G.is_directed():
                PD = PDMEM2D(G)
            else:
                PD = PDMEM2U(G)
        else:
            raise Exception('Specified type of Belief is not yet implemented')
    return G, PD
###################################################################################################################################################################
def RunSIMPUtil(d, Params):
    """
    function to run SIMP for one graph dataset

    Parameters
    ----------
    d : str
        dataset name
    Params : dict
        required parameters from config file to run the experiment

    Returns
    -------
    list, str
        list of found patterns and graph type (Undirected 'U' or Directed 'D')
    """
    G, PD = getGraphAndBD(d, Params)
    Patterns = list()
    gtype = 'D' if G.is_directed() else 'U'
    isSimple = False
    incEdges = True if Params['incedges'] == 1 else False
    flag = True
    mxpats = sys.maxsize if Params['maxpatterns'] == 0 else Params['maxpatterns']
    hc = HC(G, Params['q'], Params['seedmode'], Params['seedruns'], Params['icmode'], gtype, isSimple=isSimple, incEdge=incEdges)
    prevPat = None
    while flag:
        Pat = hc.getBestOption(PD, prevPat)
        if len(Patterns) < mxpats:
            prevPat = Pat.copy()
            if Pat is not None and Pat.NCount > Params['minsize'] and Pat.I > Params['mininterest']:
                la = PD.updateDistribution(Pat.G, len(Patterns), 'save')
                print('la=', la, PD.lprevUpdate)
                Pat.setLambda(la)
                Patterns.append(Pat)
            else:
                print('Founded all patterns with size and interestingness score greater than specified, finishing the task.....!!!!')
                flag = False
        if len(Patterns) >= mxpats:
            print('Founded maximum required patterns, finishing the task.....!!!!')
            flag = False
    return Patterns, gtype
###################################################################################################################################################################
def makeWritePath(ds):
    """
    utility function to make proper path to write the results

    Parameters
    ----------
    ds : str
        current dataset name

    Returns
    -------
    str
        path of the folder in which the results to be written
    """
    if not os.path.exists(path+'Results/'):
        os.mkdir(path+'Results/')
    if not os.path.exists(path+'Results/SIMP/'):
        os.mkdir(path+'Results/SIMP/')
    if not os.path.exists(path+'Results/SIMP/'+ds+'/'):
        os.mkdir(path+'Results/SIMP/'+ds+'/')
    wpath = path+'Results/SIMP/'+ds+'/'+'run_'+str(int(time.time()))
    os.mkdir(wpath)
    return wpath+'/'
###################################################################################################################################################################
def writeToCSV(df, dfname, wpath):
    """
    utility function of write a dataframe at the specified location and stored by the given name

    Parameters
    ----------
    df : pandas dataframe
        input dataframe to be written
    dfname : str
        name of the dataframe to be given in the final file
    wpath : str
        path to directory
    """
    df.to_csv(wpath+dfname+'.csv', index=False, sep=';')
    return
###################################################################################################################################################################
def writeResults(Patterns, df):
    """
    utility function to write patterns in a pandas dataframe

    Parameters
    ----------
    Patterns : list
        list of patterns
    df : pandas dadaframe
        input dataframe

    Returns
    -------
    pandas dadaframe
        updated dataframe
    """
    for pat in Patterns:
        df = df.append(pat.getDictForm(), ignore_index=True)
    return df
###################################################################################################################################################################
def RunSIMP(fname):
    """
    Function to run SIMP

    Parameters
    ----------
    fname : str
        filename of the configuration file found in "Confs" directory
    """
    DS, Params = readConfFile(fname)
    for d in DS:
        wpath = makeWritePath(d)
        log = open(wpath+'run.logs', 'a')
        sys.stdout = log
        stime = time.time()
        Patterns, gtype = RunSIMPUtil(d, Params)
        ftime = time.time()
        p_cols = None
        if gtype is 'U':
            p_cols = ['state_info', 'pat_type', 'prev_order', 'cur_order', 'NCount', 'ECount', 'Density',\
                'I', 'DL', 'IC_ssg', 'AD', 'IC_dssg', 'IC_dsimp', 'la', 'sumPOS', 'expectedEdges', 'NL',\
                'kws', 'nw', 'minPOS']
        else:
            p_cols = ['state_info', 'pat_type', 'prev_order', 'cur_order', 'InNCount', 'OutNCount', 'ECount', 'Density',\
                'I', 'DL', 'IC_ssg', 'AD', 'IC_dssg', 'IC_dsimp', 'la', 'sumPOS', 'expectedEdges', 'inNL',\
                'outNL', 'kws', 'nw', 'minPOS']
        df_patterns = pd.DataFrame(columns = p_cols)
        df_patterns = writeResults(Patterns, df_patterns)
        writeToCSV(df_patterns, 'patterns', wpath)
        shutil.copy(path+'Confs/'+fname, wpath+'conf.txt')
        print('\n\n\nTotal Time Taken: {:.4f} seconds'.format(ftime-stime))
        log.close()
    return
###################################################################################################################################################################
if __name__ == "__main__":
    if not ray.is_initialized():
        ray.init()
    parser = argparse.ArgumentParser(description='Running SIMP')
    parser.add_argument(dest ='filename', metavar ='filename', type=str, help='configuration filename to run SIMP.py')
    args = parser.parse_args()
    RunSIMP(args.filename)
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################