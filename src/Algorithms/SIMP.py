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
from src.BackgroundDistributions.UniDistMulti import UniDistSimple as PDUDM

from src.HillClimbers.HC_v4 import findBestPattern

def parseStr(x):
    try:
        return int(x)
    except ValueError:
        try:
            return float(x)
        except ValueError:
            return x

def readConfFile(fname):
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

def getGraphAndBD(d, Params):
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

def RunSIMPUtil(d, Params):
    G, PD = getGraphAndBD(d, Params)
    Patterns = list()
    gtype = 'D' if G.is_directed() else 'U'
    isSimple = False
    incEdges = True if Params['incedges'] == 1 else False
    flag = True
    mxpats = sys.maxsize if Params['maxpatterns'] == 0 else Params['maxpatterns']
    while flag:
        Pat = findBestPattern(G, PD, Params['q'], Params['seedmode'], Params['seedruns'], Params['icmode'], gtype, isSimple=isSimple, incEdge=incEdges)
        if len(Patterns) < mxpats:
            if Pat is not None and Pat.NCount > Params['minsize'] and Pat.I > Params['mininterest']:
                Patterns.append(Pat)
                PD.updateDistribution(Pat.G, len(Patterns), 'save')
            else:
                print('Founded all patterns with size and interestingness score greater than specified, finishing the task.....!!!!')
                flag = False
        if len(Patterns) >= mxpats:
            print('Founded maximum required patterns, finishing the task.....!!!!')
            flag = False
    return Patterns, gtype

def makeWritePath(ds):
    if not os.path.exists(path+'Results/'):
        os.mkdir(path+'Results/')
    if not os.path.exists(path+'Results/SIMP/'):
        os.mkdir(path+'Results/SIMP/')
    if not os.path.exists(path+'Results/SIMP/'+ds+'/'):
        os.mkdir(path+'Results/SIMP/'+ds+'/')
    wpath = path+'Results/SIMP/'+ds+'/'+'run_'+str(int(time.time()))
    os.mkdir(wpath)
    return wpath+'/'

def writeToCSV(df, dfname, wpath):
    df.to_csv(wpath+dfname+'.csv', index=False, sep=';')
    return

def writeResults(Patterns, df):
    for pat in Patterns:
        df = df.append(pat.getDictForm(), ignore_index=True)
    return df

def RunSIMP(fname):
    DS, Params = readConfFile(fname)
    for d in DS:
        wpath = makeWritePath(d)
        log = open(wpath+'run.logs', 'a')
        sys.stdout = log
        Patterns, gtype = RunSIMPUtil(d, Params)
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
        log.close()
    return



if __name__ == "__main__":
    if not ray.is_initialized():
        ray.init()
    parser = argparse.ArgumentParser(description='Running SIMP')
    parser.add_argument(dest ='filename', metavar ='filename', type=str, help='configuration filename to run SIMP.py')
    args = parser.parse_args()
    RunSIMP(args.filename)