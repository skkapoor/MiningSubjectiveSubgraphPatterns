import pandas as pd
from ast import literal_eval

class Node:
    def __init__(self, pid, sid, aid, action, tid = 0):
        self.pat_id = pid
        self.state_id = sid
        self.action_id = aid
        self.action = action
        self.tid = tid
        self.child = list()
        self.parent = list()

class EPTree:
    def __init__(self, hd):
        self.head = hd
        self.otherh = set()
        self.cur = None
        # self.pats = set(hd.pat_id)

def changeTid(root, tid):
    root.tid = tid
    if len(root.child) == 0:
        return
    else:
        for c in root.child:
            return changeTid(c, tid)

def removeExtEP(eps):
    prosID = set()
    delks = list()
    for k,v in eps.items():
        if (v.head.state_id, v.head.action_id) in prosID:
            delks.append(k)
        else:
            prosID.add((v.head.state_id, v.head.action_id))
            for voh in v.otherh:
                prosID.add((voh.state_id, voh.action_id))
    for k in delks:
        eps.pop(k)
    return eps

def findAllEPTrees(adf):
    prosPat = dict()
    eps = dict()
    pit = 0
    for i in range(len(adf)):
        if 'add' in adf['action'][i]:
            ND = Node(adf['final_pats'][i][0], adf['state_id'][i], adf['action_id'][i], adf['action'][i], pit)
            eps[pit] = EPTree(ND)
            prosPat[adf['final_pats'][i][0]] = ND
            pit += 1

        elif 'remove' in adf['action'][i]:
            ND = Node(adf['final_pats'][i][0], adf['state_id'][i], adf['action_id'][i], adf['action'][i], prosPat[adf['initial_pats'][i][0]].tid)
            ND.parent.append(prosPat[adf['initial_pats'][i][0]])
            prosPat[adf['initial_pats'][i][0]].child.append(ND)

        elif 'merge' in adf['action'][i]:
            n1 = adf['initial_pats'][i][0]
            n2 = adf['initial_pats'][i][1]
            ND = Node(adf['final_pats'][i][0], adf['state_id'][i], adf['action_id'][i], adf['action'][i], min(prosPat[n1].tid, prosPat[n2].tid))
            ND.parent.append(prosPat[n1])
            ND.parent.append(prosPat[n2])
            prosPat[n1].child.append(ND)
            prosPat[n2].child.append(ND)
            prosPat[adf['final_pats'][i][0]] = ND

            eps[min(prosPat[n1].tid, prosPat[n2].tid)].otherh.add(eps[max(prosPat[n1].tid, prosPat[n2].tid)].head)
            for nd in eps[max(prosPat[n1].tid, prosPat[n2].tid)].otherh:
                eps[min(prosPat[n1].tid, prosPat[n2].tid)].otherh.add(nd)

            # recursively change the tid of parents and all the branches of tree with max(prosPat[n1].tid, prosPat[n2].tid)
            obhnode = eps[max(prosPat[n1].tid, prosPat[n2].tid)].head
            changeTid(obhnode, eps[min(prosPat[n1].tid, prosPat[n2].tid)].head.tid)

            for oboh in eps[max(prosPat[n1].tid, prosPat[n2].tid)].otherh:
                changeTid(oboh, eps[min(prosPat[n1].tid, prosPat[n2].tid)].head.tid)

            #pop the other tree now
            # eps.pop(max(prosPat[n1].tid, prosPat[n2].tid), None)

        elif 'update' in adf['action'][i]:
            ND = Node(adf['final_pats'][i][0], adf['state_id'][i], adf['action_id'][i], adf['action'][i], prosPat[adf['initial_pats'][i][0]].tid)
            ND.parent.append(prosPat[adf['initial_pats'][i][0]])
            prosPat[adf['initial_pats'][i][0]].child.append(ND)
            prosPat[adf['final_pats'][i][0]] = ND

        elif 'shrink' in adf['action'][i]:
            ND = Node(adf['final_pats'][i][0], adf['state_id'][i], adf['action_id'][i], adf['action'][i], prosPat[adf['initial_pats'][i][0]].tid)
            ND.parent.append(prosPat[adf['initial_pats'][i][0]])
            prosPat[adf['initial_pats'][i][0]].child.append(ND)
            prosPat[adf['final_pats'][i][0]] = ND

        elif 'split' in adf['action'][i]:
            fpts = adf['final_pats'][i]
            for f in fpts:
                ND = Node(f, adf['state_id'][i], adf['action_id'][i], adf['action'][i], prosPat[adf['initial_pats'][i][0]].tid)
                ND.parent.append(prosPat[adf['initial_pats'][i][0]])
                prosPat[adf['initial_pats'][i][0]].child.append(ND)
                prosPat[f] = ND
    print('Number of eps: {}, len of prosPat: {}'.format(len(eps), len(prosPat)))
    eps = removeExtEP(eps)
    print('Number of eps: {}, len of prosPat: {}'.format(len(eps), len(prosPat)))
    return eps

def writeEP(ep, f):
    curpts = [None]*(len(ep.otherh)+1)
    curpts[0] = ep.head
    epoh = list(ep.otherh)
    for i in range(len(ep.otherh)):
        curpts[i+1] = epoh[i]
    flag = True
    curst = curpts[0].state_id
    prosID = set()
    lp = False
    while flag:
        npts = []
        # print(len(curpts))
        printset = []
        for hit in curpts:
            if hit.state_id == curst and (hit.state_id, hit.action_id, hit.pat_id) not in prosID:# :
                parents = ''
                if len(hit.parent) > 0:
                    for p in hit.parent:
                        parents = parents+str(p.pat_id)+', '
                    parents = parents[:-2]
                else:
                    parents = 'NA'
                printset.append((hit.state_id, hit.action_id, hit.pat_id, hit.action, parents))
                for c in hit.child:
                    npts.append(c)
                prosID.add((hit.state_id, hit.action_id, hit.pat_id))
                lp = False
            elif (hit.state_id, hit.action_id, hit.pat_id) not in prosID:
                npts.append(hit)
        if len(printset) > 0:
            printsetS = sorted(printset, key=lambda x:x[1])
            for psS in printsetS:
                f.write('State id: {}, Action id: {}, Pat id: {}, action: {}, Parent Pat ids: {}\n'.format(psS[0], psS[1], psS[2], psS[3], psS[4]))
        else:
            curst += 1
            if not lp:
                f.write('********\n')
                lp = True
        if len(npts) == 0:
            flag = False
        else:
            curpts = npts
    f.write('---X---X---X---X---X---X---X---X---\n\n\n')
    return

def writeEPS(eps, path):
    f = open(path+'evolving_patterns.txt', 'a')
    for k,v in eps.items():
        f.write('E Pat {}: -------\n'.format(k))
        writeEP(v, f)

    f.write('---X---X---X---X---X---X---X---X---X---X---X---X---X---X---X---X---X---X---')
    f.close()
    return

def findAndSaveEPs(df, path):
    eps = findAllEPTrees(df)
    writeEPS(eps, path)
    return

def printEP(ep):
    curpts = [None]*(len(ep.otherh)+1)
    curpts[0] = ep.head
    epoh = list(ep.otherh)
    for i in range(len(ep.otherh)):
        curpts[i+1] = epoh[i]
    flag = True
    curst = curpts[0].state_id
    prosID = set()
    lp = False
    while flag:
        npts = []
        # print(len(curpts))
        printset = []
        for hit in curpts:
            if hit.state_id == curst and (hit.state_id, hit.action_id, hit.pat_id) not in prosID:# :
                parents = ''
                if len(hit.parent) > 0:
                    for p in hit.parent:
                        parents = parents+str(p.pat_id)+', '
                    parents = parents[:-2]
                else:
                    parents = 'NA'
                printset.append((hit.state_id, hit.action_id, hit.pat_id, hit.action, parents))
                for c in hit.child:
                    npts.append(c)
                prosID.add((hit.state_id, hit.action_id, hit.pat_id))
                lp = False
            elif (hit.state_id, hit.action_id, hit.pat_id) not in prosID:
                npts.append(hit)
        if len(printset) > 0:
            printsetS = sorted(printset, key=lambda x:x[1])
            for psS in printsetS:
                print('State id: {}, Action id: {}, Pat id: {}, action: {}, Parent Pat ids: {}'.format(psS[0], psS[1], psS[2], psS[3], psS[4]))
        else:
            curst += 1
            if not lp:
                print('********')
                lp = True
        if len(npts) == 0:
            flag = False
        else:
            curpts = npts
    print('---X---X---X---X---X---X---X---X---\n\n')
    return

def readCSV(pt):
    df = pd.read_csv(pt, converters={"initial_pats": literal_eval, "final_pats": literal_eval}, sep=';')
    return df


