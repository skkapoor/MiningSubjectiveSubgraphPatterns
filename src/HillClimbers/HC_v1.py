################ This file contains a basic version of hill climber with sequential implementation ################################################################
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################
import os
import sys
path = os.getcwd().split('MiningSubjectiveSubgraphPatterns')[0]+'MiningSubjectiveSubgraphPatterns/'
if path not in sys.path:
	sys.path.append(path)
import math
import numpy as np
import copy
import networkx as nx

from src.Patterns.Pattern import Pattern
from src.Utils.Measures import computeSumOfEdgeProbablity, IC_SSG, NW, NW_D
from src.Utils.Measures import computeDescriptionLength, computeInterestingness
from src.Utils.Measures import computeSumOfEdgeProbablityBetweenNodeAndList
from src.Utils.Measures import computeSumOfExpectations, computeSumOfExpectationsBetweenNodeAndList
from src.Utils.Measures import computePWparameters, computePWparametersBetweenNodeAndList
from src.Utils.Measures import AD, IC_DSIMP, getDirectedSubgraph
##################################################################################################################################################################
class HillClimber_v1:
	"""This is a Hill Climber Class Version 1; used for discovering new patterns in a graph dataset
	"""
	PD = None
##################################################################################################################################################################
	def __init__(self, G, q, seedMode, nKseed, mode = 1, gtype = 'U', isSimple=True, incEdge = False):
		"""
		Args:
			G (netwokx graph): input graph dataset
			q (float): expected pattern size in range between 0.0-1.0
			seedMode (str): type of seed mode to use in this hill climber
			nKseed (int): number of seed runs to be performed to find one pattern
			mode (int, optional): Information content mode: here mode can be 1 for SSG,
								2 for Aggregate deviation and 3 for measure in DSIMP.
								Defaults to 1.
			gtype (str, optional): input graph type: 'U' for undirected and 'D' for directed. Defaults to 'U'.
			isSimple (bool, optional): True is input graph is a simple graph else false. Defaults to True.
			incEdge (bool, optional): If edges shall be described in description of a pattern. Defaults to False.
		"""
		self.G = G.copy()
		self.seedMode = seedMode
		self.nKseed = nKseed
		self.allSeedCheck = []
		self.mode = mode # here mode can be 1 for SSG, 2 for Aggregate deviation and 3 for measure in DSIMP
		self.gtype = gtype
		self.isSimple = isSimple
		self.incEdge = incEdge
		self.q = q
##################################################################################################################################################################
	def setPD(self, PD):
		"""Function to set background distribution

		Args:
			PD (PDClass): Background distribution of input graph
		"""
		self.PD = PD
		return
##################################################################################################################################################################
	def getSeeds(self):
		"""Function to get seeds to run the hill climber

		Raises:
			Exception 1: if self.nKseed != mNumNodes:
				raise Exception("Number of seeds should be equal to number of nodes here.")
			Exception: raise Exception('no valid seed mode given')

		Returns:
			list: seed node's list
		"""
		mNumNodes = self.G.number_of_nodes()
		seedNodes = [None]*self.nKseed
		if 'all' in self.seedMode:
			if self.nKseed != mNumNodes:
				raise Exception("Number of seeds should be equal to number of nodes here.")

			for r in range(self.nKseed):
				seedNodes[r] = r

		elif 'uniform' in self.seedMode:
			randoml = list(self.G.nodes())
			np.random.shuffle(randoml)
			for r in range(self.nKseed):
				seedNodes[r] = randoml[r]

		elif 'degree' in self.seedMode:
			degreeList = sorted(dict(self.G.degree()).items(), key=lambda kv: kv[1], reverse=True)
			for r in range(self.nKseed):
				seedNodes[r] = degreeList[r][0]

		elif 'interest' in self.seedMode:
			ListNode = sorted(list(self.G.nodes()))
			interestList = []
			if self.gtype == 'U':
				for LNit in ListNode:
					print(LNit)
					curlist = list(set(self.G.neighbors(LNit)).union(set([LNit])))
					H = self.G.subgraph(curlist)
					if len(curlist)>1:
						ic = 0.0
						dl = 0.0
						if self.mode == 1:
							pw = computeSumOfEdgeProbablity(self.PD, gtype=self.gtype, NL=curlist, isSimple=self.isSimple)
							ic = IC_SSG(3, pw=pw, W=H)
						elif self.mode == 2:
							mu_w = computeSumOfExpectations(self.PD, gtype=self.gtype, NL=curlist, isSimple=self.isSimple)
							ic = AD(H.number_of_edges(), mu_w)
						elif self.mode == 3:
							mu_w, p0 = computePWparameters(self.PD, gtype=self.gtype, NL=curlist, isSimple=self.isSimple)
							ic = IC_DSIMP(H.number_of_edges(), NW(len(curlist)), mu_w, p0)
						dlmode = 1
						if self.incEdge:
							dlmode = 2
						dl = computeDescriptionLength(dlmode=dlmode, V=self.G.number_of_nodes(), W=H.number_of_nodes(), kw=H.number_of_edges(), q=self.q)
						interestValue = computeInterestingness(ic, dl)
						interestList.append(tuple([LNit, interestValue]))
			else:
				for LNit in ListNode:
					print(LNit)
					curlistOut = list(set(self.G.predecessors(LNit)).union(set([LNit])))
					curlistIn = list(set(self.G.successors(LNit)).union(set([LNit])))
					H = getDirectedSubgraph(self.G, curlistIn, curlistOut, self.isSimple)
					if len(curlistIn)>1 and len(curlistOut)>1:
						ic = 0.0
						dl = 0.0
						if self.mode == 1:
							pw = computeSumOfEdgeProbablity(self.PD, gtype=self.gtype, inNL=curlistIn, outNL=curlistOut, isSimple=self.isSimple)
							ic = IC_SSG(3, pw=pw, W=H)
						elif self.mode == 2:
							mu_w = computeSumOfExpectations(self.PD, gtype=self.gtype, inNL=curlistIn, outNL=curlistOut, isSimple=self.isSimple)
							ic = AD(H.number_of_edges(), mu_w)
						elif self.mode == 3:
							mu_w, p0 = computePWparameters(self.PD, gtype=self.gtype, inNL=curlistIn, outNL=curlistOut, isSimple=self.isSimple)
							ic = IC_DSIMP(H.number_of_edges(), NW_D(curlistIn, curlistOut), mu_w, p0)
						dlmode = 1
						if self.incEdge:
							dlmode = 2
						dl = computeDescriptionLength(dlmode=dlmode, V=self.G.number_of_nodes(), WI=curlistIn, WO=curlistOut, kw=H.number_of_edges(), q=self.q)
						interestValue = computeInterestingness(ic, dl)
						interestList.append(tuple([LNit, interestValue]))
			interestList = sorted(interestList, key=lambda kv: kv[1], reverse=True)
			mRange = min([self.nKseed, len(interestList)])
			seedNodes = [0]*mRange
			for r in range(mRange):
				# print(r, interestList[r][0])
				if interestList[r][0] is None:
					print(r, interestList[r][0])
				seedNodes[r] = interestList[r][0]

		else:
			raise Exception('no valid seed mode given')
		return seedNodes
##################################################################################################################################################################
	def hillclimber(self, minsize=0, mininterest = -100000.0):
		"""function to most interesting subgraph pattern

		Args:
			minsize (int, optional): minimum size of the pattern. Defaults to 0.
			mininterest (float, optional): minimum interestingness. Defaults to -100000.0.

		Returns:
			Pattern: most interesting subgraph pattern
		"""
		print('Already present patterns: ', len(self.PD.lprevUpdate.items()))

		seedNodes = self.getSeeds()

		# Do requested runs
		bestPattern = Pattern(nx.Graph())

		for k in range(len(seedNodes)):
			cP = self.searchPattern(seedNodes[k])
			if cP.I > bestPattern.I and cP.NCount > minsize and cP.I > mininterest:
				bestPattern = cP.copy()
			self.allSeedCheck.append(cP)

		return bestPattern
##################################################################################################################################################################
	def extendPatternUtil(self, pattern, nodeToCheck):
		"""Util function to check for the best candidate node to add

		Args:
			pattern (Pattern): input subgraph pattern
			nodeToCheck (int): node id of the vertex to check for addition

		Returns:
			dict: dictionary of parameters cmputed for the input node
		"""
		kw_surplus = 0
		for p in pattern.NL:
			kw_surplus += self.G.number_of_edges(p, nodeToCheck)
		curNL = pattern.NL

		params = dict()
		ic = 0.0
		dl = 0.0
		if self.mode == 1:
			params['pw_surplus'] = computeSumOfEdgeProbablityBetweenNodeAndList(self.PD, nodeToCheck, curNL, gtype=self.gtype, isSimple=self.isSimple)
			params['pw_new'] = pattern.sumPOS + params['pw_surplus']
			params['kw_new'] = pattern.ECount + kw_surplus
			params['nw_new'] = NW(len(curNL)+1)
			ic = IC_SSG(1, pw=params['pw_new'], kw=params['kw_new'], nw=params['nw_new'])
		elif self.mode == 2:
			params['mu_w_surplus'] = computeSumOfExpectationsBetweenNodeAndList(self.PD, nodeToCheck, curNL, gtype=self.gtype, isSimple=self.isSimple)
			params['mu_w_new'] = pattern.expectedEdges + params['mu_w_surplus']
			params['kw_new'] = pattern.ECount + kw_surplus
			ic = AD(params['kw_new'], params['mu_w_new'])
		else:
			params['mu_w_surplus'], params['p0_surplus'] = computePWparametersBetweenNodeAndList(self.PD, nodeToCheck, curNL, gtype=self.gtype, isSimple=self.isSimple)
			params['mu_w_new'] = pattern.expectedEdges + params['mu_w_surplus']
			params['p0_new'] = min(pattern.minPOS, params['p0_surplus'])
			params['kw_new'] = pattern.ECount + kw_surplus
			params['nw_new'] = NW(len(curNL)+1)
			ic = IC_DSIMP(params['kw_new'], params['nw_new'], params['mu_w_new'], params['p0_new'])

		dlmode = 1
		if self.incEdge:
			dlmode = 2
		dl = computeDescriptionLength(dlmode=dlmode, V=self.G.number_of_nodes(), W=len(curNL)+1, kw=params['kw_new'], q=self.q)
		I = computeInterestingness(ic, dl)

		params['ic'] = ic
		params['dl'] = dl
		params['I'] = I

		return params
##################################################################################################################################################################
	def extendPatternFinal(self, pattern, nodeToAdd, params):
		"""function to update the subgraph pattern while adding the node resultng in maximum increase of interestingness value

		Args:
			pattern (Pattern): input subgraph pattern
			nodeToAdd (int): node id of the vertex to be added
			params (dict): corresponding parameters of the node to be added

		Returns:
			Pattern: updated subgraph pattern
		"""
		H = nx.Graph()
		for p in pattern.NL:
			if self.G.has_edge(p, nodeToAdd):
				H.add_edge(p, nodeToAdd)
		pattern.updateGraph(H)
		if self.mode == 1:
			pattern.setSumPOS(params['pw_new'])
			pattern.setIC_ssg(params['ic'])
		elif self.mode == 2:
			pattern.setAD(params['ic'])
			pattern.setExpectedEdges(params['mu_w_new'])
		elif self.mode == 3:
			pattern.setIC_dsimp(params['ic'])
			pattern.setExpectedEdges(params['mu_w_new'])
			pattern.setMinPOS(params['p0_new'])
		pattern.setDL(params['dl'])
		pattern.setI(params['I'])
		return pattern
##################################################################################################################################################################
	def shrinkPatternUtil(self, pattern, nodeToCheck):
		"""Util function to check for the best candidate node to remove

		Args:
			pattern (Pattern): input subgraph pattern
			nodeToCheck (int): node id of the vertex to check for removal

		Returns:
			dict: dictionary of parameters cmputed for the input node
		"""
		kw_deficit  = 0
		for p in pattern.NL:
			kw_deficit  += self.G.number_of_edges(p, nodeToCheck)
		curNL = pattern.NL

		params = dict()
		ic = 0.0
		dl = 0.0
		if self.mode == 1:
			params['pw_deficit'] = computeSumOfEdgeProbablityBetweenNodeAndList(self.PD, nodeToCheck, curNL, gtype=self.gtype, isSimple=self.isSimple)
			params['pw_new'] = pattern.sumPOS - params['pw_deficit']
			params['kw_new'] = pattern.ECount - kw_deficit
			params['nw_new'] = NW(len(curNL)-1)
			ic = IC_SSG(1, pw=params['pw_new'], kw=params['kw_new'], nw=params['nw_new'])
		elif self.mode == 2:
			params['mu_w_deficit'] = computeSumOfExpectationsBetweenNodeAndList(self.PD, nodeToCheck, curNL, gtype=self.gtype, isSimple=self.isSimple)
			params['mu_w_new'] = pattern.expectedEdges - params['mu_w_deficit']
			params['kw_new'] = pattern.ECount - kw_deficit
			ic = AD(params['kw_new'], params['mu_w_new'])
		else:
			params['kw_new'] = pattern.ECount - kw_deficit
			params['nw_new'] = NW(len(curNL)-1)
			params['mu_w_deficit'], params['p0_deficit'] = computePWparametersBetweenNodeAndList(self.PD, nodeToCheck, curNL, gtype=self.gtype, isSimple=self.isSimple)
			if pattern.minPOS == params['p0_deficit']:
				curNL.remove(nodeToCheck)
				params['mu_w_new'], params['p0_new'] = computePWparameters(self.PD, gtype=self.gtype, NL=curNL, isSimple=self.isSimple)
			else:
				params['p0_new'] = pattern.minPOS
				params['mu_w_new'] = pattern.expectedEdges - params['mu_w_deficit']
			ic = IC_DSIMP(params['kw_new'], params['nw_new'], params['mu_w_new'], params['p0_new'])

		dlmode = 1
		if self.incEdge:
			dlmode = 2
		dl = computeDescriptionLength(dlmode=dlmode, V=self.G.number_of_nodes(), W=len(curNL)-1, kw=params['kw_new'], q=self.q)
		I = computeInterestingness(ic, dl)
		params['ic'] = ic
		params['dl'] = dl
		params['I'] = I
		return params
##################################################################################################################################################################
	def shrinkPatternFinal(self, pattern, nodeToRemove, params):
		"""function to update the subgraph pattern while removing the node resulting in maximum increase of interestingness value

		Args:
			pattern (Pattern): input subgraph pattern
			nodeToAdd (int): node id of the vertex to be removed
			params (dict): corresponding parameters of the node to be removed

		Returns:
			Pattern: updated subgraph pattern
		"""
		pattern.removeNode(nodeToRemove)
		if self.mode == 1:
			pattern.setSumPOS(params['pw_new'])
			pattern.setIC_ssg(params['ic'])
		elif self.mode == 2:
			pattern.setAD(params['ic'])
			pattern.setExpectedEdges(params['mu_w_new'])
		elif self.mode == 3:
			pattern.setIC_dsimp(params['ic'])
			pattern.setExpectedEdges(params['mu_w_new'])
			pattern.setMinPOS(params['p0_new'])
		pattern.setDL(params['dl'])
		pattern.setI(params['I'])
		return pattern
##################################################################################################################################################################
	def climbOneStep(self, pattern, candidates):
		"""function to climb one step at a time, i.e., either addition of node or removal from the subgraph pattern

		Args:
			pattern (Pattern): input subgraph pattern
			candidates (set): set of candidate node that can be added such that the subgraph remains connected

		Returns:
			Pattern, list, str: Updated subgraph pattern, updated candidate list and operation performed
		"""
		print('In climb one step')
		print('\tSize of candidates: ', len(candidates))
		print('\tBefore operation')
		print('\teP: ', pattern.sumPOS, 'interest: ', pattern.I)

		operation = 'none'
		nodeAddedFinal = None
		nodeRemovedFinal = None

		# Extend Pattern
		best_params = dict()
		best_params['I'] = pattern.I
		best_node = None
		bestPattern = None

		for cand in candidates:
			cur = self.extendPatternUtil(pattern, cand)
			if best_params['I'] < cur['I']:
				best_params = cur.copy()
				best_node = cand

		if best_params['I'] > pattern.I:
			print('Added', best_node)
			bestPattern = self.extendPatternFinal(pattern, best_node, best_params)
			nodeAddedFinal = best_node
			operation = 'addition'

		if 'none' in operation and pattern.NCount > 2:
			for cand in pattern.NL:
				cur = self.shrinkPatternUtil(pattern, cand)
				if best_params['I'] < cur['I']:
					best_params = cur.copy()
					best_node = cand

			if best_params['I'] > pattern.I:
				bestPattern = self.shrinkPatternFinal(pattern, best_node, best_params)
				nodeRemovedFinal = best_node
				operation = 'deletion'

		# update candidate list now
		# print(operation, "\t", bestPattern.NL)
		newcandidate = candidates[:]

		if 'addition' in operation:
			neighborsA = list(self.G.neighbors(nodeAddedFinal))
			newcandidate.remove(nodeAddedFinal)
			for nA in neighborsA:
				if nA not in newcandidate and nA not in bestPattern.NL:
					newcandidate.append(nA)

		elif 'deletion' in operation:
			neighborsR = list(self.G.neighbors(nodeRemovedFinal))
			for nR in neighborsR:
				nRneighbors = self.G.neighbors(nR)
				flg = True
				for nnR in nRneighbors:
					if nnR in bestPattern.NL:
						flg = False
				if flg and nR in newcandidate:
					newcandidate.remove(nR)

		# return relevant information
		print('\tAfter Operation ', operation)
		print('\tpw: ', pattern.sumPOS, 'interest: ', pattern.I)
		return pattern, newcandidate, operation
##################################################################################################################################################################
	def extendPatternUtilD(self, pattern, nodeToCheck, dir_mode):
		# add one node a time and check for best gain
		# count edges from nodeToCheck to pattern
		# dir_mode (int): required id gtype is 'D"; 1 - from node to list and 2 from list to node

		######### Check in-node and out-node addition ############
		kw_surplus = 0
		params = dict()
		ic = 0.0
		dl = 0.0
		curInNL = pattern.inNL[:]
		curOutNL = pattern.outNL[:]
		curFunc = None
		if dir_mode == 1:
			for p in pattern.inNL:
				if nodeToCheck!=p:
					kw_surplus += int(self.G.number_of_edges(nodeToCheck, p))
			curOutNL.append(nodeToCheck)
			curFunc = curInNL
		else:
			for p in pattern.outNL:
				if nodeToCheck!=p:
					kw_surplus += self.G.number_of_edges(p, nodeToCheck)
			curInNL.append(nodeToCheck)
			curFunc = curOutNL

		if self.mode == 1:
			params['pw_surplus'] = computeSumOfEdgeProbablityBetweenNodeAndList(self.PD, nodeToCheck, curFunc, dir_mode=dir_mode, gtype=self.gtype, isSimple=self.isSimple)
			params['pw_new'] = pattern.sumPOS + params['pw_surplus']
			params['kw_new'] = pattern.ECount + kw_surplus
			params['nw_new'] = NW_D(curInNL, curOutNL)
			ic = IC_SSG(1, pw=params['pw_new'], kw=params['kw_new'], nw=params['nw_new'])
		elif self.mode == 2:
			params['mu_w_surplus'] = computeSumOfExpectationsBetweenNodeAndList(self.PD, nodeToCheck, curFunc, dir_mode=dir_mode, gtype=self.gtype, isSimple=self.isSimple)
			params['mu_w_new'] = pattern.expectedEdges + params['mu_w_surplus']
			params['kw_new'] = pattern.ECount + kw_surplus
			ic = AD(params['kw_new'], params['mu_w_new'])
		else:
			params['mu_w_surplus'], params['p0_surplus'] = computePWparametersBetweenNodeAndList(self.PD, nodeToCheck, curFunc, dir_mode=dir_mode, gtype=self.gtype, isSimple=self.isSimple)
			params['mu_w_new'] = pattern.expectedEdges + params['mu_w_surplus']
			params['p0_new'] = min(pattern.minPOS, params['p0_surplus'])
			params['kw_new'] = pattern.ECount + kw_surplus
			params['nw_new'] = NW_D(curInNL, curOutNL)
			ic = IC_DSIMP(params['kw_new'], params['nw_new'], params['mu_w_new'], params['p0_new'])

		dlmode = 1
		if self.incEdge:
			dlmode = 2
		dl = computeDescriptionLength(dlmode=dlmode, V=self.G.number_of_nodes(), WI=curInNL, WO=curOutNL, kw=params['kw_new'], q=self.q)
		I = computeInterestingness(ic, dl)

		params['ic'] = ic
		params['dl'] = dl
		params['I'] = I

		return params
##################################################################################################################################################################
	def extendPatternFinalD(self, pattern, nodeToAdd, params, typeOfAddition):
		H = nx.DiGraph()
		if 'in' in typeOfAddition:
			for p in pattern.outNL:
				if self.G.has_edge(p, nodeToAdd):
					H.add_edge(p, nodeToAdd)
		else:
			for p in pattern.inNL:
				if self.G.has_edge(nodeToAdd, p):
					H.add_edge(nodeToAdd, p)
		pattern.updateGraph(H)
		if self.mode == 1:
			pattern.setSumPOS(params['pw_new'])
			pattern.setIC_ssg(params['ic'])
		elif self.mode == 2:
			pattern.setAD(params['ic'])
			pattern.setExpectedEdges(params['mu_w_new'])
		elif self.mode == 3:
			pattern.setIC_dsimp(params['ic'])
			pattern.setExpectedEdges(params['mu_w_new'])
			pattern.setMinPOS(params['p0_new'])
		pattern.setDL(params['dl'])
		pattern.setI(params['I'])
		return pattern
##################################################################################################################################################################
	def shrinkPatternUtilD(self, pattern, nodeToCheck, dir_mode):
		# remove node at a time, compute final interestingness and return the updated pattern
		# count edges from nodeToCheck to pattern

		########### Check in-node or out-node removal ############
		kw_deficit  = 0
		params = dict()
		ic = 0.0
		dl = 0.0
		curInNL = pattern.inNL[:]
		curOutNL = pattern.outNL[:]
		curFunc = None
		if dir_mode == 1:
			for p in pattern.inNL:
				if nodeToCheck!=p:
					kw_deficit += int(self.G.number_of_edges(nodeToCheck, p))
			curOutNL.remove(nodeToCheck)
			curFunc = curInNL
		else:
			for p in pattern.outNL:
				if nodeToCheck!=p:
					kw_deficit += self.G.number_of_edges(p, nodeToCheck)
			curInNL.remove(nodeToCheck)
			curFunc = curOutNL

		if self.mode == 1:
			params['pw_deficit'] = computeSumOfEdgeProbablityBetweenNodeAndList(self.PD, nodeToCheck, curFunc, dir_mode=dir_mode, gtype=self.gtype, isSimple=self.isSimple)
			params['pw_new'] = pattern.sumPOS - params['pw_deficit']
			params['kw_new'] = pattern.ECount - kw_deficit
			params['nw_new'] = NW_D(curInNL, curOutNL)
			ic = IC_SSG(1, pw=params['pw_new'], kw=params['kw_new'], nw=params['nw_new'])
		elif self.mode == 2:
			params['mu_w_deficit'] = computeSumOfExpectationsBetweenNodeAndList(self.PD, nodeToCheck, curFunc, dir_mode=dir_mode, gtype=self.gtype, isSimple=self.isSimple)
			params['mu_w_new'] = pattern.expectedEdges - params['mu_w_deficit']
			params['kw_new'] = pattern.ECount - kw_deficit
			ic = AD(params['kw_new'], params['mu_w_new'])
		else:
			params['kw_new'] = pattern.ECount - kw_deficit
			params['nw_new'] = NW_D(curInNL, curOutNL)
			params['mu_w_deficit'], params['p0_deficit'] = computePWparametersBetweenNodeAndList(self.PD, nodeToCheck, curFunc, dir_mode=dir_mode, gtype=self.gtype, isSimple=self.isSimple)
			if pattern.minPOS == params['p0_deficit']:
				params['mu_w_new'], params['p0_new'] = computePWparameters(self.PD, gtype=self.gtype, inNL = curInNL, outNL = curOutNL, isSimple=self.isSimple)
			else:
				params['p0_new'] = pattern.minPOS
				params['mu_w_new'] = pattern.expectedEdges - params['mu_w_deficit']
			ic = IC_DSIMP(params['kw_new'], params['nw_new'], params['mu_w_new'], params['p0_new'])

		dlmode = 1
		if self.incEdge:
			dlmode = 2
		dl = computeDescriptionLength(dlmode=dlmode, V=self.G.number_of_nodes(), WI=curInNL, WO=curOutNL, kw=params['kw_new'], q=self.q)
		I = computeInterestingness(ic, dl)
		params['ic'] = ic
		params['dl'] = dl
		params['I'] = I
		return params
##################################################################################################################################################################
	def shrinkPatternFinalD(self, pattern, nodeToRemove, params, typeOfDeletion):
		if 'in' in typeOfDeletion:
			pattern.removeInNode(nodeToRemove)
		else:
			pattern.removeOutNode(nodeToRemove)
		if self.mode == 1:
			pattern.setSumPOS(params['pw_new'])
			pattern.setIC_ssg(params['ic'])
		elif self.mode == 2:
			pattern.setAD(params['ic'])
			pattern.setExpectedEdges(params['mu_w_new'])
		elif self.mode == 3:
			pattern.setIC_dsimp(params['ic'])
			pattern.setExpectedEdges(params['mu_w_new'])
			pattern.setMinPOS(params['p0_new'])
		pattern.setDL(params['dl'])
		pattern.setI(params['I'])
		return pattern
##################################################################################################################################################################
	def climbOneStepD(self, pattern, candidatesIn, candidatesOut):
		operation = 'none'
		bestPattern = pattern.copy()
		nodeAddedFinal = None
		nodeRemovedFinal = None

		# Extend Pattern
		bestInParams = dict()
		bestInParams['I'] = pattern.I
		bestInNode = None

		bestOutParams = dict()
		bestOutParams['I'] = pattern.I
		bestOutNode = None

		#Check all possible in-node addition
		for cand in candidatesIn:
			curIn = self.extendPatternUtilD(pattern, cand, 2)
			if bestInParams['I'] < curIn['I']:
				bestInParams = curIn.copy()
				bestInNode = cand

		#Check all possible out-node addition
		for cand in candidatesOut:
			curOut = self.extendPatternUtilD(pattern, cand, 1)
			if bestOutParams['I'] < curOut['I']:
				bestOutParams = curOut.copy()
				bestOutNode = cand

		#Perform best addition
		if bestInParams['I'] > pattern.I or bestOutParams['I'] > pattern.I:
			if bestInParams['I'] > bestOutParams['I']:
				bestPattern = self.extendPatternFinal(pattern, bestInNode, bestInParams, 'in')
				nodeAddedFinal = bestInNode
				operation = 'inaddition'
			else:
				bestPattern = self.extendPatternFinal(pattern, bestOutNode, bestOutParams, 'out')
				nodeAddedFinal = bestOutNode
				operation = 'outaddition'


		# If no extension, shrink pattern
		if 'none' in operation:
			#Check all possible in-node removal
			if pattern.InNCount > 1:
				for node in pattern.inNL:
					curIn = self.shrinkPatternUtilD(pattern, node, 2)
					if bestInParams['I'] < curIn['I']:
						bestInParams = curIn.copy()
						bestInNode = node

			#Check all possible out-node removal
			if pattern.OutNCount>1:
				for node in pattern.outNL:
					curOut = self.shrinkPatternUtilD(pattern, node, 1)
					if bestOutParams['I'] < curOut['I']:
						bestOutParams = curOut.copy()
						bestOutNode = node

			#Perform best removal
			if bestInParams['I'] > pattern.I or bestOutParams['I'] > pattern.I:
				if bestInParams['I'] > bestOutParams['I']:
					bestPattern = self.shrinkPatternFinal(pattern, bestInNode, bestInParams, 'in')
					nodeRemovedFinal = bestInNode
					operation = 'indeletion'
				else:
					bestPattern = self.shrinkPatternFinal(pattern, bestOutNode, bestOutParams, 'out')
					nodeRemovedFinal = bestOutNode
					operation = 'outdeletion'

		# update candidate list now
		# print(operation, "\t", bestPattern.NL)
		newCandidateIn = candidatesIn[:]
		newCandidateOut = candidatesOut[:]

		if 'addition' in operation:
			if 'in' in operation:
				#remove the selected node from IN candidate list
				newCandidateIn.remove(nodeAddedFinal)
				#now an in-node is added thus potential list of OUT candidates in updated
				outCand = list(self.G.predecessors(nodeAddedFinal))
				for nC in outCand:
					if nC not in newCandidateOut and nC not in bestPattern.outNL:
						newCandidateOut.append(nC)
			else:
				#remove the selected node from Out candidate list
				newCandidateOut.remove(nodeAddedFinal)
				#now an out-node is added thus potential list of IN candidates in updated
				inCand = list(self.G.successors(nodeAddedFinal))
				for nC in inCand:
					if nC not in newCandidateIn and nC not in bestPattern.inNL:
						newCandidateIn.append(nC)
		elif 'deletion' in operation:
			if 'in' in operation:
				#since an IN node is removed thus OUT candidate list is updated
				outNeighbor = list(self.G.predecessors(nodeRemovedFinal))
				for oN in outNeighbor: #Iterative each predecessor of node which is removed
					if oN in newCandidateOut: #If a predecessor is present in out candidate then check if it should be removed
						flg = True
						oN_inNeighbor = list(self.G.successors(oN)) #Check the node which is to be removed does not have any other
						for oN_iN in oN_inNeighbor:					#successor in the bestPattern's in-node list
							if oN_iN in bestPattern.inNL:
								flg = False  						#if present then don't remove
						if flg:										#if not present then remove it
							newCandidateOut.remove(oN)
			else:
				#since an OUT node is removed thus IN candidate list is updated
				inNeighbor = list(self.G.successors(nodeRemovedFinal))
				for iN in inNeighbor: #Iterative each successor of node which is removed
					if iN in newCandidateIn: #If a successor is present in out candidate then check if it should be removed
						flg = True
						iN_onNeighbor = list(self.G.predecessors(iN)) #Check the node which is to be removed does not have any other
						for iN_oN in iN_onNeighbor:					#predecessor in the bestPattern's in-node list
							if iN_oN in bestPattern.outNL:
								flg = False  						#if present then don't remove
						if flg:										#if not present then remove it
							newCandidateIn.remove(iN)

		# return relevant information
		print('Best Pattern eP: ', bestPattern.sumPOS, 'interest: ', bestPattern.I)
		# print("inNL", bestPattern.inNL)
		# print("outNL", bestPattern.outNL)
		print("Operation", operation, bestInNode, bestOutNode)
		return bestPattern, newCandidateIn, newCandidateOut, operation
##################################################################################################################################################################
	def searchPattern(self, seed):
		"""function to search for a pattern starting from a given seed subgraph

		Args:
			seed (int): seed subgraph of one node

		Returns:
			Pattern: found pattern for given seed
		"""
		pattern = None
		if self.gtype == 'U':
			candidates = list(self.G.neighbors(seed))

			term = False

			pattern = Pattern(nx.Graph(self.G.subgraph(seed)))
			pattern.setPatType('Found')
			# while termination==false climb one step at a time
			while not term:
				pattern, candidates, operation = self.climbOneStep(pattern, candidates[:])
				if 'none' in operation:
					term = True
		else:
			candidatesIn = list(self.G.successors(seed))
			candidatesOut = list(self.G.predecessors(seed))

			term = False

			pattern = Pattern(nx.DiGraph(self.G.subgraph(seed)))
			pattern.setPatType('Found')
			# while termination==false climb one step at a time
			while not term:
				pattern, candidatesIn, candidatesOut, operation = self.climbOneStepD(pattern, candidatesIn[:], candidatesOut[:])
				if 'none' in operation:
					term = True


		#return pattern and other information
		return pattern
##################################################################################################################################################################
	def addNewPatternList(self, NL, count, Pattype):
		pattern = Pattern.Pattern(self.G.subgraph(NL).copy(), NL, 0, 0, self.PD.computeEdgeProbabilityListNodesIncLprev(NL), self.PD.computeInterestingnessListNodesIncLprev(NL), count)
		pattern.updateGraphProperties()
		self.PD.updateBackground(pattern, count)
		pattern.setPatType(Pattype)
		return pattern, self.PD.copy()
##################################################################################################################################################################
	def addNewPattern(self, pattern, count):
		self.PD.updateBackground(pattern, count)
		return self.PD.copy()
##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################