import os
import sys
path = os.getcwd().split('MiningSubjectiveSubgraphPatterns')[0]+'MiningSubjectiveSubgraphPatterns/'
if path not in sys.path:
	sys.path.append(path)
import math
import numpy as np
import networkx as nx
import time
import argparse
import configparser
import ast
import string
import ray

from src.Patterns.Pattern import Pattern

from src.BackgroundDistributions.MaxEntSimple import MaxEntSimpleU as PDMESU
from src.BackgroundDistributions.MaxEntSimple import MaxEntSimpleD as PDMESD
from src.BackgroundDistributions.UniDistSimple import UniDistSimple as PDUDS

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
        for i in config['SSG Params'].items():
            Params[i[0]] = parseStr(i[1])
        return DS, Params
    else:
        raise Exception('Configuration file does not exists:', path+'Confs/'+fname)
    return

def getGraphAndBD(d, Params):
    files = None
    if os.path.exists(path+'Data/SSG/'+d):
        files = os.listdir(path+'Data/SSG/'+d)
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
            G = nx.read_gpickle(path+'Data/SSG/'+d+'/'+Ffiles['.gpickle'])
        else:
            G = nx.read_gml(path+'Data/SSG/'+d+'/'+Ffiles['.gml'], destringizer=nx.readwrite.gml.literal_destringizer)

        if Params['priorbelief'] is 'c':
            PD = PDUDS(G)
        elif Params['priorbelief'] is 'i':
            if G.is_directed():
                PD = PDMESD(G)
            else:
                PD = PDMESU(G)
        else:
            raise Exception('Specified type of Belief is not yet implemented')
    return G, PD

def RunSSGUtil(d, Params):
    G, PD = getGraphAndBD(d, Params)
    Patterns = list()
    gtype = 'D' if G.is_directed() else 'U'
    isSimple = False if G.is_multigraph() else True
    incEdges = True if Params['incedges'] == 1 else False
    flag = True
    mxpats = sys.maxsize if Params['maxpatterns'] == 0 else Params['maxpatterns']
    while flag:
        Pat = findBestPattern(G, PD, Params['q'], Params['seedmode'], Params['seedruns'], 1, gtype, isSimple, incEdges)
        if len(Patterns) < mxpats:
            if Pat is not None and Pat.NCount > Params['minsize'] and Pat.I > Params['mininterest']:
                Patterns.append(Pat)
                PD.updateDistribution(Pat.G, len(Patterns), 'save')
            else:
                print('Found all patterns with size and interestingness score greater than specified, finishing the task.....!!!!')
                flag = False
        if len(Patterns) >= mxpats:
            print('Found maximum required patterns, finishing the task.....!!!!')
            flag = False
    return Patterns

def writeResults(Patterns, fname):
    if len(Patterns) == 0:
        print('no patterns found')
        return
    for p in Patterns:
        print(p.I, p.NCount, nx.density(p.G))
    return

def RunSSG(fname):
    if not ray.is_initialized():
        ray.init()
    DS, Params = readConfFile(fname)
    for d in DS:
        Patterns = RunSSGUtil(d, Params)
        writeResults(Patterns, fname)
    return



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Running SSG')
    parser.add_argument(dest ='filename', metavar ='filename', type=str, help='configuration filename to run SSG.py')
    args = parser.parse_args()
    RunSSG(args.filename)