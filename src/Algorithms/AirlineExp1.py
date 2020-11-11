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

from src.HillClimbers.HC_v4 import findBestPattern
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
def getGraphAndBD(prefixA, prefixS, d, Params):
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
    if os.path.isfile(path+'Data/SIMP/'+prefixA+'/'+d+'.gpickle'):
        GA = None
        GS = None
        PD = None
        print(Params)
        GA = nx.read_gpickle(path+'Data/SIMP/'+prefixA+'/'+d+'.gpickle')
        GS = nx.read_gpickle(path+'Data/SIMP/'+prefixS+'/'+d+'.gpickle')

        if Params['priorbelief'] is 'c':
            PD = PDUDM(GS)
        elif Params['priorbelief'] is 'i':
            if GS.is_directed():
                PD = PDMEM1D(GS)
            else:
                PD = PDMEM1U(GS)
        elif Params['priorbelief'] is 'm':
            if GS.is_directed():
                PD = PDMEM2D(GS)
            else:
                PD = PDMEM2U(GS)
        else:
            raise Exception('Specified type of Belief is not yet implemented')
        PD.G = GA.copy()
        return GA, PD
    else:
        raise Exception('No file with Name: {} found in directory: {}'.format(d, path+'Data/SIMP/'+prefixS))
###################################################################################################################################################################
def RunSIMPUtil(prefixA, prefixS, d, Params):
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
    G, PD = getGraphAndBD(prefixA, prefixS, d, Params)
    Patterns = list()
    gtype = 'D' if G.is_directed() else 'U'
    isSimple = False
    incEdges = True if Params['incedges'] == 1 else False
    flag = True
    mxpats = sys.maxsize if Params['maxpatterns'] == 0 else Params['maxpatterns']
    while flag:
        print('new added lambdas', PD.lprevUpdate)
        Pat = findBestPattern(G, PD, Params['q'], Params['seedmode'], Params['seedruns'], Params['icmode'], gtype, isSimple=isSimple, incEdge=incEdges)
        if len(Patterns) < mxpats:
            if Pat is not None:
                if 'U' in gtype and (Pat.NCount > Params['minsize'] and Pat.I > Params['mininterest']):
                    la = PD.updateDistribution(Pat.G, len(Patterns), 'save')
                    Pat.setLambda(la)
                    Patterns.append(Pat)
                elif 'D' in gtype and (Pat.InNCount >= 1 and Pat.OutNCount >= 1 and Pat.InNCount+Pat.OutNCount > Params['minsize'] and Pat.I > Params['mininterest']):
                    la = PD.updateDistribution(Pat.G, len(Patterns), 'save')
                    Pat.setLambda(la)
                    Patterns.append(Pat)
                else:
                    print('Founded all patterns with size and interestingness score greater than specified, finishing the task.....!!!!')
                    flag = False
            else:
                print('Founded all patterns with size and interestingness score greater than specified, finishing the task.....!!!!')
                flag = False
        if len(Patterns) >= mxpats:
            print('Founded maximum required patterns, finishing the task.....!!!!')
            flag = False
    return Patterns, gtype
###################################################################################################################################################################
def makeWritePath(prefix, ds):
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
    pre_split = prefix.split('/')
    npath = path+'Results/SIMP/'
    for p in pre_split:
        if not os.path.exists(npath+p+'/'):
            os.mkdir(npath+p+'/')
        npath = npath+p+'/'
    if not os.path.exists(npath+ds+'/'):
        os.mkdir(npath+ds+'/')
    wpath = npath+ds+'/'
    return wpath
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
    _, Params = readConfFile(fname)
    for cyr in ['2016', '2017', '2018']:
        writePrefix = 'Airline-Belief-i-SI/HourWise/Scheduled/'+cyr
        prefixA = 'Airline/HourWise/Scheduled/'+cyr
        prefixS = 'Airline/HourWise/Scheduled/'+cyr
        DS = ['01_01_01']
        days_m = {1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}
        for i in range(1,13):
            d1 = ''
            if i < 10:
                d1 = '0'+str(i)
            else:
                d1 = str(i)
            for j in range(1, days_m[i]+1):
                d2 = ''
                if j < 10:
                    d2 = d1+'_0'+str(j)
                else:
                    d2 = d1+'_'+str(j)
                for k in range(1,23):
                    d = ''
                    if k < 10:
                        d = d2+'_0'+str(k)
                    else:
                        d = d2+'_'+str(k)

                    wpath = makeWritePath(writePrefix, d)
                    log = open(wpath+'run.logs', 'a')
                    sys.stdout = log
                    stime = time.time()
                    Patterns, gtype = RunSIMPUtil(prefixA, prefixS, d, Params)
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
    parser = argparse.ArgumentParser(description='Running AirlineExp1 using SIMP')
    parser.add_argument(dest ='filename', metavar ='filename', type=str, help='configuration filename to run AirlineExp1.py')
    args = parser.parse_args()
    RunSIMP(args.filename)
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################