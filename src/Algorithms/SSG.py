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

from src.Patterns.Pattern import Pattern

from src.BackgroundDistributions.MaxEntSimple import MaxEntSimpleU as PDMESU
from src.BackgroundDistributions.MaxEntSimple import MaxEntSimpleD as PDMESD
from src.BackgroundDistributions.UniDistSimple import UniDistSimple as PDUDSU

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
            print(i, type(i[1]), type(parseStr(i[1])))
    else:
        raise Exception('Configuration file does not exists:', path+'Confs/'+fname)
    return

def main(fname):
    readConfFile(fname)
    return



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Running SSG')
    parser.add_argument(dest ='filename', metavar ='filename', type=str, help='configuration filename to run SSG.py') 
    args = parser.parse_args()
    main(args.filename)