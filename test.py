# import os
# import sys
# path = os.getcwd().split('MiningSubjectiveSubgraphPatterns')[0]+'MiningSubjectiveSubgraphPatterns/'
# if path not in sys.path:
# 	sys.path.append(path)
# import math
# import numpy as np
# import copy
# import networkx as nx
# import time

# from src.HillClimbers.HC_v1 import HillClimber_v1 as HC1
# from src.HillClimbers.HC_v2 import HillClimber_v2 as HC2
# from src.HillClimbers.HC_v3 import HillClimber_v3 as HC3
# from src.HillClimbers.HC_v4 import findBestPattern

# import ray
# import psutil
# num_cpus = psutil.cpu_count(logical=False)
# ray.init(num_cpus=num_cpus)

# from src.Patterns.Pattern import Pattern

# from src.BackgroundDistributions.MaxEntSimple import MaxEntSimpleU as PDcl

# G = nx.read_gpickle(path+'testgraph3.gpickle')
# print(nx.density(G))
# PD = PDcl(G)

# st1 = time.time()
# H1 = HC1(G, 0.1, 'interest', 10)
# H1.setPD(PD)
# P1 = H1.hillclimber()
# ft1 = time.time()

# st2 = time.time()
# H2 = HC2(G, 0.1, 'interest', 10)
# H2.setPD(PD)
# P2 = H2.hillclimber()
# ft2 = time.time()

# st3 = time.time()
# H3 = HC3(G, 0.1, 'interest', 10)
# H3.setPD(PD)
# P3 = H3.hillclimber()
# ft3 = time.time()

# st4 = time.time()
# P4 = findBestPattern(G, PD, 0.1, 'interest', 10)
# ft4 = time.time()

# print('HC1', ft1-st1, P1.I)
# print('HC2', ft2-st2, P2.I)
# print('HC3', ft3-st3, P3.I)
# print('HC4', ft4-st4, P4.I)

# # G = nx.generators.gaussian_random_partition_graph(10000, 100, 5, 0.7, 0.1, False, 1)