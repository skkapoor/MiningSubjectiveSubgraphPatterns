import os
from os import write
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

def getAllStates(d):
    files = None
    states = None
    if os.path.exists(path+'Data/DSSG/'+d):
        files = os.listdir(path+'Data/DSSG/'+d)
    Ffiles = dict()
    for f in files:
        if '.gml' in f:
            if '.gml' not in Ffiles:
                Ffiles['.gml'] = [f]
            else:
                Ffiles['.gml'].append(f)
        elif '.gpickle' in f:
            if '.gpickle' not in Ffiles:
                Ffiles['.gpickle'] = [f]
            else:
                Ffiles['.gpickle'].append(f)
    print(Ffiles['.gml'])
    if ('.gml' in Ffiles or '.gpickle' in Ffiles) and not ('.gml' in Ffiles and '.gpickle' in Ffiles):
        if '.gml' in Ffiles:
            states = sorted(Ffiles['.gml'])
        elif '.gpickle' in Ffiles:
            states = sorted(Ffiles['.gpickle'])
    else:
        raise Exception('The folder shall contain either gml or gpickle files of different graph snapshots with name in a lexiographical order')
    print('states:',states)
    return states

def getInitGraphAndBD(gname, Params):
    G = None
    gtype = 'U'
    if '.gpickle' in gname:
        G = nx.read_gpickle(gname)
    else:
        G = nx.read_gml(gname, destringizer=nx.readwrite.gml.literal_destringizer)
    if G.is_directed():
        gtype = 'D'
    if G.is_multigraph():
        raise Exception('********Encountered a multigraph********** Gname: {}'.format(gname))
    if Params['priorbelief'] is 'c':
        PD = PDUDS(G)
    elif Params['priorbelief'] is 'i':
        if G.is_directed():
            PD = PDMESD(G)
        else:
            PD = PDMESU(G)
    else:
        raise Exception('Specified type of Belief is not yet implemented')
    return G, PD, gtype

def processInitialState(G_cur, PD, state_id, pat_ids, df_patterns, gtype='U', isSimple=True, l=6, ic_mode=1, imode=2, minsize=2, seedType='interest', seedRuns=10, q=0.01, incEdges=True):
    EA_o = EA(gtype, isSimple, l, ic_mode, imode, minsize, seedType, seedRuns, q, incEdges)
    flag = True
    Summary = dict()
    action_id = 0
    while flag:
        EA_o.evaluateNew(G_cur, PD)
        EA_params = EA_o.getBestOption(G_cur, PD)
        # for k, v in EA_params.items():
        #     print('\t best Cand Add: {} ----- {}'.format(k,v))
        if EA_params is not None and EA_params['Pat'].I>0.0:
            print('\n\t**Action id: {}**'.format(action_id))
            EA_params['Pat'].setCurOrder(pat_ids)
            EA_params['Pat'].setStateInfo(state_id)
            pat_ids += 1
            Summary[action_id] = EA_params #? is this valid? Check
            action_id += 1
            EA_o.updateDistribution(PD, EA_params)
            if df_patterns is not None:
                df_patterns = writePattern(df_patterns, EA_params['Pat'])
            for k, v in EA_params.items():
                print('\t{} --- {}'.format(k, v))
        else:
            flag = False
    return Summary, PD, pat_ids, df_patterns

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
    EM_o.evaluateAllConstraintPairs(G_cur, PD)
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
    if bestParams['Pat'].pat_type in ['add', 'merge', 'shrink', 'update']:
        bestParams['Pat'].setCurOrder(pat_ids)
        bestParams['Pat'].setStateInfo(state_id)
        pat_ids += 1
        if bestParams['Pat'].pat_type is 'shrink':
            bestParams['SPat'].setCurOrder(bestParams['Pat'].prev_order)
            bestParams['SPat'].setStateInfo(state_id)
    elif bestParams['Pat'].pat_type in ['remove']:
        bestParams['Pat'].setCurOrder(bestParams['Pat'].prev_order)
        bestParams['Pat'].setStateInfo(state_id)
    elif bestParams['Pat'].pat_type is 'split':
        bestParams['Pat'].setStateInfo(state_id)
        lt = []
        for k,v in bestParams['compos'].items():
            v.setCurOrder(pat_ids)
            lt.append(pat_ids)
            v.setStateInfo(state_id)
            pat_ids += 1
        bestParams['Pat'].setCurOrder(lt)
    return pat_ids, bestParams

def postProssessActionObjects(bestParams, G_cur, PD, EA_o, ER_o, EU_o, EM_o, ESH_o, ESP_o):
    # Update Background Distribution
    # print("In postProcess prev lambda keys:", PD.lprevUpdate.keys())
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
    # print('bestParam.cur_id and prev-id', bestParams['Pat'].pat_type, bestParams['Pat'].cur_order, bestParams['Pat'].prev_order)
    # print("In postProcess prev lambda keys:", PD.lprevUpdate.keys())
    EA_o.checkAndUpdateAllPossibilities(G_cur, PD, bestParams['Pat'])
    ER_o.checkAndUpdateAllPossibilities(G_cur, PD, bestParams['Pat'])
    EU_o.checkAndUpdateAllPossibilities(G_cur, PD, bestParams['Pat'])
    EM_o.removeCandidates(bestParams['Pat'])
    EM_o.checkAndUpdateAllPossibilities(G_cur, PD, bestParams['Pat'])
    if 'split' is bestParams['Pat'].pat_type:
        EM_o.doProcessWithNewConstraint(G_cur, PD, list(bestParams['compos'].values()))
    elif 'shrink' is bestParams['Pat'].pat_type:
        EM_o.doProcessWithNewConstraint(G_cur, PD, bestParams['SPat'])
    else:
        EM_o.doProcessWithNewConstraint(G_cur, PD, bestParams['Pat'])
    ESH_o.checkAndUpdateAllPossibilities(G_cur, PD, bestParams['Pat'])
    ESP_o.checkAndUpdateAllPossibilities(G_cur, PD, bestParams['Pat'])
    return

def RunDSSGUtil(gname, PD, state_id, pat_ids, Params, df_patterns):
    print('State: {}'.format(state_id))
    flag = True
    Summary = dict()
    action_id = 0
    G_cur = None
    if '.gpickle' in gname:
        G_cur = nx.read_gpickle(gname)
    elif '.gml' in gname:
        G_cur = nx.read_gml(gname, destringizer=nx.readwrite.gml.literal_destringizer)
    else:
        raise Exception("Invalid File Type, filetype shall either GML or gpickle")

    if (G_cur.is_directed() and Params['gtype'] is 'U') or (not G_cur.is_directed() and Params['gtype'] is 'D'):
        raise Exception('************Mismatch graph type************ All states shall be either directed or undirected ******')
    if G_cur.is_multigraph():
        raise Exception('********Encountered a multigraph********** Gname: {}'.format(gname))

    EA_o, ER_o, EU_o, EM_o, ESH_o, ESP_o = initializeActionObjects(gtype=Params['gtype'], isSimple=True, l=Params['l'], ic_mode=Params['icmode'], imode=Params['interesttype'], minsize=Params['minsize'], seedType=Params['seedmode'], seedRuns=Params['seedruns'], q=Params['q'], incEdges=Params['incedges'])

    #PreProcessing as the state at this point is changed
    preProcessActionObjects(G_cur, PD, EA_o, ER_o, EU_o, EM_o, ESH_o, ESP_o)

    print('\t After PreProcessing the #candidates in each case are:')
    print('\t EA_o: {}'.format(len(EA_o.Data)))
    print('\t ER_o: {}'.format(len(ER_o.Data)))
    print('\t EU_o: {}'.format(len(EU_o.Data)))
    print('\t EM_o: {}'.format(len(EM_o.Data)))
    print('\t ESH_o: {}'.format(len(ESH_o.Data)))
    print('\t ESP_o: {}'.format(len(ESP_o.Data)))

    while flag:
        bestParams = getBestAction(G_cur, PD, EA_o, ER_o, EU_o, EM_o, ESH_o, ESP_o)
        if bestParams is not None:
            print('\n\t**Action id: {}**'.format(action_id))
            pat_ids, bestParams = setNewDetails(bestParams, state_id, pat_ids)
            Summary[action_id] = bestParams #? is this valid? Check
            action_id += 1
            postProssessActionObjects(bestParams, G_cur, PD, EA_o, ER_o, EU_o, EM_o, ESH_o, ESP_o)
            if df_patterns is not None:
                df_patterns = writePatternsToDF(df_patterns, bestParams)
            for k, v in bestParams.items():
                if k is 'compos':
                    print("\tCompos:")
                    for k1, v1 in v:
                        print('\t\t{} --- {}'.fomat(k1, v1))
                else:
                    print('\t{} --- {}'.format(k, v))
        else:
            flag = False

    return Summary, pat_ids, df_patterns

def writePattern(df, pat):
    df = df.append(pat.getDictForm(), ignore_index=True)
    return df

def writePatternsToDF(df, Params):
    if Params['Pat'].pat_type in ['add', 'merge', 'update']:
        df = writePattern(df, Params['Pat'])
    elif Params['Pat'].pat_type in ['shrink']:
        df = writePattern(df, Params['SPat'])
    elif Params['Pat'].pat_type in ['split']:
        for k, v in Params['compos']:
            df = writePattern(df, v)
    return df

def makeWritePath(ds):
    if not os.path.exists(path+'Results/'):
        os.mkdir(path+'Results/')
    if not os.path.exists(path+'Results/DSSG/'):
        os.mkdir(path+'Results/DSSG/')
    if not os.path.exists(path+'Results/DSSG/'+ds+'/'):
        os.mkdir(path+'Results/DSSG/'+ds+'/')
    wpath = path+'Results/DSSG/'+ds+'/'+'run_'+str(int(time.time()))
    os.mkdir(wpath)
    if not os.path.exists(wpath+'/constr/'):
        os.mkdir(wpath+'/constr/')
    return wpath+'/'

def writeToCSV(df, dfname, wpath):
    df.to_csv(wpath+dfname+'.csv', index=False, sep=';')
    return

def writeConstraints(PD, cols, dfname, wpath):
    df = pd.DataFrame(columns=cols)
    for k, v in PD.lprevUpdate.items():
        dt = dict()
        dt['order'] = k
        if PD.tp is 'U':
            dt['NL'] = v[1]
            dt['kw'] = v[2]
        else:
            dt['inNL'] = v[1]
            dt['outNL'] = v[2]
            dt['kw'] = v[3]
        dt['la'] = v[0]
        df = df.append(dt, ignore_index=True)
    writeToCSV(df, dfname, wpath+'/constr/')
    return

def writeActions(OSummary, wpath):
    df = pd.DataFrame(columns=['state_id', 'action_id', 'action', 'initial_pats', 'final_pats', 'CL_i', 'CL_f'])
    for k,v in OSummary.items():
        for k1, u in v.items():
            dt = dict()
            dt['state_id'] = k
            dt['action_id'] = k1
            dt['action'] = u['Pat'].pat_type
            if isinstance(u['Pat'].prev_order, (int, float)):
                dt['initial_pats'] = list([u['Pat'].prev_order])
            elif u['Pat'].prev_order is not None:
                dt['initial_pats'] = list(u['Pat'].prev_order)
            else:
                dt['initial_pats'] = list()
            if isinstance(u['Pat'].cur_order, (int, float)):
                dt['final_pats'] = list([u['Pat'].cur_order])
            elif u['Pat'].cur_order is not None:
                dt['final_pats'] = list(u['Pat'].cur_order)
            else:
                dt['final_pats'] = list()
            dt['CL_i'] = u['codeLengthC']
            dt['CL_f'] = u['codeLengthCprime']
            df = df.append(dt, ignore_index=True)
    writeToCSV(df, 'actions', wpath)
    return

# def writeResults(OSummary, fname):
#     return

def DSSGMain(fname):
    if not ray.is_initialized():
        ray.init()
    DS, Params = readConfFile(fname)
    print(Params)
    for d in DS:
        OSummary = dict()
        pat_ids = 0
        allStates = getAllStates(d)
        G_cur, PD, gtype = getInitGraphAndBD(path+'Data/DSSG/'+d+'/'+allStates[0], Params)
        Params['gtype'] = gtype
        p_cols = None
        c_cols = None
        if gtype is 'U':
            p_cols = ['state_info', 'pat_type', 'prev_order', 'cur_order', 'NCount', 'ECount', 'Density',\
                'I', 'DL', 'IC_ssg', 'AD', 'IC_dssg', 'IC_dsimp', 'la', 'sumPOS', 'expectedEdges', 'NL',\
                'kws', 'nw', 'minPOS']
            c_cols = ['order', 'NL', 'kw', 'la']
        else:
            p_cols = ['state_info', 'pat_type', 'prev_order', 'cur_order', 'InNCount', 'OutNCount', 'ECount', 'Density',\
                'I', 'DL', 'IC_ssg', 'AD', 'IC_dssg', 'IC_dsimp', 'la', 'sumPOS', 'expectedEdges', 'inNL',\
                'outNL', 'kws', 'nw', 'minPOS']
            c_cols = ['order', 'inNL', 'outNL', 'kw', 'la']
        wpath = makeWritePath(d)
        df_patterns = pd.DataFrame(columns = p_cols)

        OSummary[0], PD, pat_ids, df_patterns = processInitialState(G_cur, PD, 0, pat_ids, df_patterns=df_patterns, gtype=gtype, isSimple=True, l=Params['l'], ic_mode=Params['icmode'], imode=Params['interesttype'], minsize=Params['minsize'], seedType=Params['seedmode'], seedRuns=Params['seedruns'], q=Params['q'], incEdges=Params['incedges'])
        writeConstraints(PD, c_cols, 'S'+str(0)+'_constr', wpath)
        for state_id in range(1, len(allStates)):
            OSummary[state_id], pat_ids, df_patterns = RunDSSGUtil(path+'Data/DSSG/'+d+'/'+allStates[state_id], PD, state_id, pat_ids, Params, df_patterns)
            writeConstraints(PD, c_cols, 'S'+str(state_id)+'_constr', wpath)
        writeActions(OSummary, wpath)
        writeToCSV(df_patterns, 'patterns', wpath)
    return

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Running DSSG')
    parser.add_argument(dest ='filename', metavar ='filename', type=str, help='configuration filename to run DSSG.py')
    args = parser.parse_args()
    stime = time.time()
    DSSGMain(args.filename)
    ftime = time.time()
    print('\n\nTotal Time Taken: {:.4f} seconds'.format(ftime-stime))