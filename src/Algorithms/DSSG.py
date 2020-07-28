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

from src.Actions.update import EvaluateUpdate as EU
from src.Actions.split import EvaluateSplit as ESP
from src.Actions.merge import EvaluateMerge as EM
from src.Actions.shrink import EvaluateShrink as ESH
from src.Actions.remove import EvaluateRemove as ER
from src.Actions.add import EvaluateAdd as EA

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
        for i in config['DSSG Params'].items():
            Params[i[0]] = parseStr(i[1])
        return DS, Params
    else:
        raise Exception('Configuration file does not exists:', path+'Confs/'+fname)
    return

def getInitGraphAndBD(d, Params):
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
                Ffiles['.gml'] = [f]
        elif '.gpickle' in f:
            if '.gpickle' not in Ffiles:
                Ffiles['.gpickle'] = f
    if '.gml' not in Ffiles and '.gpickle' not in Ffiles:
        raise Exception('No input graph file in the data directory. The directory shall contain .gml or .gpickle files of input graph.')
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

def initializeActionObjects(gtype='U', isSimple=True, l=6, ic_mode=1, imode=2, minsize=2, seedType='interest', seedRuns=10, q=0.01, incEdges=True):
    EA_o = EA(gtype, isSimple, l, ic_mode, imode, minsize, seedType, seedRuns, q, incEdges)
    ER_o = ER(gtype, isSimple, l, imode)
    EU_o = EU(gtype, isSimple, l, imode)
    EM_o = EM(gtype, isSimple, l, imode)
    ESH_o = ESH(gtype, isSimple, l, imode, minsize)
    ESP_o = ESP(gtype, isSimple, l, imode, minsize)
    return EA_o, ER_o, EU_o, EM_o, ESH_o, ESP_o

def preProcessActionObjects(G_cur, PD, EA_o, ER_o, EU_o, EM_o, ESH_o, ESP_o):
    EA_o.evaluateNew(G_cur, PD)
    ER_o.evaluateAllConstraints(G_cur, PD)
    EU_o.evaluateAllConstraints(G_cur, PD)
    EM_o.evaluateAllConstraintsPairs(G_cur, PD)
    ESH_o.evaluateAllConstraints(G_cur, PD)
    ESP_o.evaluateAllConstraints(G_cur, PD)
    return

def getBestAction(G_cur, PD, EA_o, ER_o, EU_o, EM_o, ESH_o, ESP_o):
    EA_params = EA_o.getBestOption(G_cur, PD)
    ER_params = ER_o.getBestOption()
    EU_params = EU_o.getBestOption()
    EM_params = EM_o.getBestOption()
    ESH_params = ESH_o.getBestOption()
    ESP_params = ESP_o.getBestOption()
    bestAction = None
    bestI = 0.0
    if EA_params is not None and EA_params['Pat'].I > bestI:
        bestI = EA_params['Pat'].I
        bestAction = EA_params
    if ER_params is not None and ER_params['Pat'].I > bestI:
        bestI = ER_params['Pat'].I
        bestAction = ER_params
    if EM_params is not None and EM_params['Pat'].I > bestI:
        bestI = EM_params['Pat'].I
        bestAction = EM_params
    if EU_params is not None and EU_params['Pat'].I > bestI:
        bestI = EU_params['Pat'].I
        bestAction = EU_params
    if ESH_params is not None and ESH_params['SPat'].I > bestI:
        bestI = ESH_params['SPat'].I
        bestAction = ESH_params
    if ESP_params is not None and ESP_params['Pat'].I > bestI:
        bestI = ESP_params['Pat'].I
        bestAction = ESP_params
    return bestAction

def setNewDetails(bestParams, state_id, pat_ids):
    if bestParams['Pat'].pat_type is 'add' or bestParams['Pat'].pat_type == 'merge':
        bestParams['Pat'].setCurOrder(pat_ids)
        bestParams['Pat'].setStateInfo(state_id)
        pat_ids += 1
    elif bestParams['Pat'].pat_type in ['remove', 'shrink', 'update']:
        bestParams['Pat'].setCurOrder(bestParams['Pat'].prev_order)
        bestParams['Pat'].setStateInfo(state_id)
        if bestParams['Pat'].pat_type is 'shrink':
            bestParams['SPat'].setCurOrder(bestParams['Pat'].prev_order)
            bestParams['SPat'].setStateInfo(state_id)
    elif bestParams['Pat'].pat_type is 'split':
        bestParams['Pat'].setCurOrder(bestParams['Pat'].prev_order)
        bestParams['Pat'].setStateInfo(state_id)
        for k,v in bestParams['compos'].items():
            v.setCurOrder(pat_ids)
            v.setStateInfo(state_id)
            pat_ids += 1
    return pat_ids, bestParams

def postProssessActionObjects(bestParams, G_cur, PD, EA_o, ER_o, EU_o, EM_o, ESH_o, ESP_o):

    # Update Background Distribution
    if bestParams['Pat'].pat_type is 'add':
        EA_o.updateDistribution(PD, bestParams)
    elif bestParams['Pat'].pat_type is 'remove':
        ER_o.updateDistribution(PD, bestParams)
    elif bestParams['Pat'].pat_type is 'update':
        EU_o.updateDistribution(PD, bestParams)
    elif bestParams['Pat'].pat_type is 'merge':
        EM_o.updateDistribution(PD, bestParams)
    elif bestParams['Pat'].pat_type is 'shrink':
        ESH_o.updateDistribution(PD, bestParams)
    elif bestParams['Pat'].pat_type is 'split':
        ESP_o.updateDistribution(PD, bestParams)
    # Update candidate list of all actions
    EA_o.checkAndUpdateAllPossibilities(G_cur, PD, bestParams['Pat'])
    ER_o.checkAndUpdateAllPossibilities(G_cur, PD, bestParams['Pat'])
    EU_o.checkAndUpdateAllPossibilities(G_cur, PD, bestParams['Pat'])
    #EM_o.
    ESH_o.checkAndUpdateAllPossibilities(G_cur, PD, bestParams['Pat'])
    ESP_o.checkAndUpdateAllPossibilities(G_cur, PD, bestParams['Pat'])
    return

def RunDSSGUtil(gname, PD, state_id, pat_ids):
    flag = True
    G_cur = None
    if '.gpickle' in gname:
        G_cur = nx.read_gpickle(gname)
    elif '.gml' in gname:
        G_cur = nx.read_gml(gname, destringizer=nx.readwrite.gml.literal_destringizer)
    else:
        raise Exception("Invalid File Type, filetype shall either GML or gpickle")

    EA_o, ER_o, EU_o, EM_o, ESH_o, ESP_o = initializeActionObjects(gtype='U', isSimple=True, l=6, ic_mode=1, imode=2, minsize=2, seedType='interest', seedRuns=10, q=0.01, incEdges=True)

    #PreProcessing as the state at this point is changed
    preProcessActionObjects(G_cur, PD, EA_o, ER_o, EU_o, EM_o, ESH_o, ESP_o)

    while flag:
        bestParams = getBestAction(G_cur, PD, EA_o, ER_o, EU_o, EM_o, ESH_o, ESP_o)
        if bestParams is not None:
            pat_ids, bestParams = setNewDetails(bestParams, state_id, pat_ids)
            postProssessActionObjects(bestParams, PD, EA_o, ER_o, EU_o, EM_o, ESH_o, ESP_o)
        else:
            flag = False

    return