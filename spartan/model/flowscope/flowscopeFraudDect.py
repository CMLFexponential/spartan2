import sys, os, time
import numpy as np
import scipy as sci
import scipy.stats as ss
import scipy.sparse.linalg as slin
import copy
from .mytools.MinTree import MinTree
from scipy.sparse import coo_matrix, csr_matrix, lil_matrix
from .mytools.ioutil import loadedge2sm
import math
from .._model import DMmodel
from ...util.basicutil import param_default
from ...backend import STensor

def fastGreedyDecreasing3(M1, M2, M3, alpha):
    print('start  greedy ')
    (row_length, mid_length1) = M1.shape
    (mid_length1, mid_length2) = M2.shape
    (mid_length2, col_length) = M3.shape

    M1 = M1.tolil()
    M2 = M2.tolil()
    M3 = M3.tolil()

    M_tran_1 = M1.transpose().tolil()
    M_tran_2 = M2.transpose().tolil()
    M_tran_3 = M3.transpose().tolil()

    rowSet = set(range(0, row_length))
    midSet1 = set(range(0, mid_length1))
    midSet2 = set(range(0, mid_length2))
    colSet = set(range(0, col_length))

    bestAveScore = -10000000000

    rowDeltas = np.squeeze(M1.sum(axis=1).A)  # sum of rows

    midDeltas1 = np.squeeze(M1.sum(axis=0).A)  # d_{1}-
    midDeltas2 = np.squeeze(M2.sum(axis=1).A)  # d_{1}+

    midDeltas3 = np.squeeze(M2.sum(axis=0).A)  # d_{2}-
    midDeltas4 = np.squeeze(M3.sum(axis=1).A)  # d_{2}+

    colDeltas = np.squeeze(M3.sum(axis=0).A)  # sum of cols
    mid_min1 = []
    mid_max1 = []
    for (m1, m2) in zip(midDeltas1, midDeltas2):
        temp = min(m1, m2)  # f1
        temp2 = max(m1, m2)  # q1
        mid_min1.append(temp)
        mid_max1.append(temp2)

    mid_min1 = np.array(mid_min1)
    mid_max1 = np.array(mid_max1)
    new_mid_priority1 = mid_min1 - alpha * mid_max1 + 0.5 * (1 - alpha) * midDeltas2  # weights of M1
    new_mid_tree1 = MinTree(new_mid_priority1)  # tree for M1

    mid_min2 = []
    mid_max2 = []

    for (m1, m2) in zip(midDeltas3, midDeltas4):
        temp = min(m1, m2)  # f2
        temp2 = max(m1, m2)  # q2
        mid_min2.append(temp)
        mid_max2.append(temp2)

    mid_min2 = np.array(mid_min2)
    mid_max2 = np.array(mid_max2)
    new_mid_priority2 = mid_min2 - alpha * mid_max2 + 0.5 * (1 - alpha) * midDeltas3  # weights of M2
    new_mid_tree2 = MinTree(new_mid_priority2)  # tree for M2

    rowTree = MinTree(rowDeltas)  # weights of A
    midTree1 = MinTree(midDeltas1)
    midTree2 = MinTree(midDeltas2)

    midTree3 = MinTree(midDeltas3)
    midTree4 = MinTree(midDeltas4)

    colTree = MinTree(colDeltas)  # weights of C

    numDeleted = 0
    deleted = []
    bestNumDeleted = 0

    f1_sum = sum(mid_min1)  # sum of f of M1 accounts
    f2_sum = sum(mid_min2)  # sum of f of M2 accounts
    q_minus_f_sum1 = sum(abs(midDeltas1 - midDeltas2))  # sum of all (q1 - f1) of M1 accounts
    q_minus_f_sum2 = sum(abs(midDeltas3 - midDeltas4))  # sum of all (q2 - f2) of M2 accounts

    while rowSet and colSet and midSet1 and midSet2:  # store A, M1, M2, C
        (nextRow, rowDelt) = rowTree.getMin()  # A min id and value
        (nextCol, colDelt) = colTree.getMin()  # C min id and value

        (nextmid1, midDelt1) = new_mid_tree1.getMin()  # M1 min id and value
        (nextmid2, midDelt2) = new_mid_tree2.getMin()  # M2 min id and value

        row_weight = rowDelt
        col_weight = colDelt
        mid_weight1 = midDelt1
        mid_weight2 = midDelt2

        # find min weight
        min_weight = min(row_weight, col_weight)
        min_weight = min(min_weight, mid_weight1)
        min_weight = min(min_weight, mid_weight2)

        if min_weight == row_weight:  # delete a node of A
            for j in M1.rows[nextRow]:  # connected M1 accounts
                new_mid_value = midTree1.changeVal(j, -M1[nextRow, j])  # changed d(j)-
                if new_mid_value == float('inf'):
                    continue
                temp_mid2 = midTree2.index_of(j)  # the value of d(j)+

                if (new_mid_value) < mid_min1[j]:  # d(j)- smaller than f(j)
                    f1_sum -= (mid_min1[j] - new_mid_value)
                    mid_min1[j] = new_mid_value  # f1(j) changes to new d(j)-
                    mid_max1[j] = temp_mid2

                mid_min_value = min(new_mid_value, temp_mid2)  # f: min(d(j)-, d(j)+)
                mid_delta_value = abs(new_mid_value - temp_mid2)  # new (q-f)
                new_mid1_w = mid_min_value - alpha * mid_delta_value + 0.5 * (1 - alpha) * temp_mid2  # new w(j)
                new_mid_tree1.setter(j, new_mid1_w)

                q_minus_f_sum1 -= abs(midDeltas1[j] - midDeltas2[j])  # d(j) - d+(j)
                q_minus_f_sum1 += abs(new_mid_value - midDeltas2[j])  # g(j) - d(j)+
                midDeltas1[j] = new_mid_value  # new d(j)-
            rowSet -= {nextRow}
            rowTree.changeVal(nextRow, float('inf'))
            deleted.append((0, nextRow))
        elif min_weight == col_weight:  # delete a node of C
            for i in M_tran_3.rows[nextCol]:  # connected M2 accounts
                new_mid_value = midTree4.changeVal(i, -M_tran_3[nextCol, i])  # new d(i)+
                if new_mid_value == float('inf'):
                    continue
                temp_mid3 = midTree3.index_of(i)  # d(i)-
                mid_min_value = min(new_mid_value, temp_mid3)  # new f
                mid_delta_value = abs(new_mid_value - temp_mid3)  # (q - f)
                new_mid2_w = mid_min_value - alpha * mid_delta_value + 0.5 * (1 - alpha) * temp_mid3
                new_mid_tree2.setter(i, new_mid2_w)
                if new_mid_value < mid_min2[i]:
                    f2_sum -= (mid_min2[i] - new_mid_value)
                    mid_min2[i] = new_mid_value  # update f of node i
                    mid_max2[i] = temp_mid3
                q_minus_f_sum2 -= abs(midDeltas3[i] - midDeltas4[i])
                q_minus_f_sum2 += abs(new_mid_value - midDeltas3[i])
                midDeltas4[i] = new_mid_value
            colSet -= {nextCol}
            colTree.changeVal(nextCol, float('inf'))
            deleted.append((1, nextCol))

        elif min_weight == mid_weight1:  # delete a node of M1
            mid_min1[nextmid1] = 0
            mid_max1[nextmid1] = 0
            midDeltas1[nextmid1] = 0
            midDeltas2[nextmid1] = 0
            for j in M2.rows[nextmid1]:  # connected M2 accounts of nextmid1
                new_mid_value = midTree3.changeVal(j, -M2[nextmid1, j])  # changed d(j)-
                if new_mid_value == float('inf'):
                    continue

                temp_mid4 = midTree4.index_of(j)  # d(j)+
                mid_min_value = min(new_mid_value, temp_mid4)
                mid_delta_value = abs(new_mid_value - temp_mid4)
                new_mid2_w = mid_min_value - alpha * mid_delta_value + 0.5 * (1 - alpha) * new_mid_value
                new_mid_tree2.setter(j, new_mid2_w)
                if new_mid_value < mid_min2[j]:
                    f2_sum -= (mid_min2[j] - new_mid_value)
                    mid_min2[j] = new_mid_value
                    mid_max2[j] = temp_mid4
                midDeltas3[j] = new_mid_value
            for j in M_tran_1.rows[nextmid1]:  # connected A accounts of nextmid1
                rowTree.changeVal(j, -M_tran_1[nextmid1, j])

            midSet1 -= {nextmid1}
            midTree1.changeVal(nextmid1, float('inf'))
            midTree2.changeVal(nextmid1, float('inf'))
            new_mid_tree1.changeVal(nextmid1, float('inf'))
            deleted.append((2, nextmid1))

        elif min_weight == mid_weight2:
            mid_min2[nextmid2] = 0
            mid_max2[nextmid2] = 0
            midDeltas3[nextmid2] = 0
            midDeltas4[nextmid2] = 0
            for j in M3.rows[nextmid2]:  # connected C accounts of nextmid2
                colTree.changeVal(j, -M3[nextmid2, j])
            for j in M_tran_2.rows[nextmid2]:  # connected M1 accounts of nextmid2
                new_mid_value = midTree2.changeVal(j, -M_tran_2[nextmid2, j])  # d(j)+
                if new_mid_value == float('inf'):
                    continue
                temp_mid1 = midTree1.index_of(j)
                mid_min_value = min(new_mid_value, temp_mid1)
                mid_delta_value = abs(new_mid_value - temp_mid1)
                new_mid1_w = mid_min_value - alpha * mid_delta_value + 0.5 * (1 - alpha) * new_mid_value  # j new weight
                new_mid_tree1.setter(j, new_mid1_w)

                if new_mid_value < mid_min1[j]:
                    f1_sum -= (mid_min1[j] - new_mid_value)
                    mid_min1[j] = new_mid_value
                    mid_max1[j] = temp_mid1
                midDeltas2[j] = new_mid_value
                q_minus_f_sum1 -= abs(midDeltas2[j] - midDeltas1[j])
                q_minus_f_sum1 += abs(new_mid_value - midDeltas1[j])

            midSet2 -= {nextmid2}
            midTree3.changeVal(nextmid2, float('inf'))
            midTree4.changeVal(nextmid2, float('inf'))
            new_mid_tree2.changeVal(nextmid2, float('inf'))
            deleted.append((3, nextmid2))

        numDeleted += 1

        S = len(rowSet) + len(midSet1) + len(midSet2) + len(colSet)
        curAveScore = (f1_sum + f2_sum - alpha * (q_minus_f_sum1 + q_minus_f_sum2))/S  # metric
        if curAveScore >= bestAveScore:
            bestNumDeleted = numDeleted
            bestAveScore = curAveScore
    # print('best number deleted : ', bestNumDeleted)

    # print('nodes number remain in rowSet, mid1Set, mid2Set, colSet',
    #       len(rowSet), len(midSet1), len(midSet2), len(colSet))
    # print('min value of each tree :  ', rowTree.getMin(),
    #       midTree1.getMin(), midTree2.getMin(), midTree3.getMin(), midTree4.getMin(), colTree.getMin())
    # print('best score: ', bestAveScore)

    finalRowSet = set(range(row_length))
    finalMidSet1 = set(range(mid_length1))
    finalMidSet2 = set(range(mid_length2))
    finalColSet = set(range(col_length))

    for i in range(bestNumDeleted):
        if deleted[i][0] == 0:
            finalRowSet.remove(deleted[i][1])
        elif deleted[i][0] == 1:
            finalColSet.remove(deleted[i][1])
        elif deleted[i][0] == 2:
            finalMidSet1.remove(deleted[i][1])
        elif deleted[i][0] == 3:
            finalMidSet2.remove(deleted[i][1])
    return ((finalRowSet, finalMidSet1, finalMidSet2, finalColSet), bestAveScore)

def fastGreedyDecreasing2(M1, M2, alpha):
    print('start  greedy')
    (row_length, mid_length) = M1.shape
    (mid_length, col_length) = M2.shape

    M1 = M1.tolil()
    M2 = M2.tolil()

    M_tran_1 = M1.transpose().tolil()
    M_tran_2 = M2.transpose().tolil()

    rowSet = set(range(0, row_length))
    midSet = set(range(0, mid_length))
    colSet = set(range(0, col_length))

    bestAveScore = float('-inf')

    # 这里应该就是计算总入除度的吧，以账户为单位
    # Deltas的shape 应为（n），待检查
    rowDeltas = np.squeeze(M1.sum(axis=1).A)   # sum of A
    midDeltas1 = np.squeeze(M1.sum(axis=0).A)
    midDeltas2 = np.squeeze(M2.sum(axis=1).A)
    midDeltas = midDeltas1 + midDeltas2  # sum of M

    colDeltas = np.squeeze(M2.sum(axis=0).A)  # sum of C

    #
    mid_min = []
    mid_max = []
    for (m1, m2) in zip(midDeltas1, midDeltas2):
        temp = min(m1, m2)      # fi
        temp2 = max(m1, m2)     # qi
        mid_min.append(temp)
        mid_max.append(temp2)

    mid_min = np.array(mid_min)
    mid_max = np.array(mid_max)
    new_mid_priority = (1 + alpha) * mid_min - alpha * mid_max
    new_mid_tree = MinTree(new_mid_priority)

    rowTree = MinTree(rowDeltas)
    midTree1 = MinTree(midDeltas1)
    midTree2 = MinTree(midDeltas2)
    midTree = MinTree(midDeltas)
    colTree = MinTree(colDeltas)

    numDeleted = 0
    deleted = []
    bestNumDeleted = 0

    row_sum = 0
    col_sum = 0
    mid_sum1 = 0
    mid_sum2 = 0

    temp1 = 0
    temp2 = 0

    curScore1 = sum(mid_min)
    curScore2 = sum(abs(midDeltas1 - midDeltas2))

    curAveScore1 = curScore1 / (len(rowSet) + len(midSet) + len(colSet))
    curAveScore2 = curScore2 / (len(rowSet) + len(midSet) + len(colSet))

    print('initial score', curScore1, curScore2 , curAveScore1, curAveScore2)
    while rowSet and colSet and midSet:  # repeat deleting until one node set in null
        (nextRow, rowDelt) = rowTree.getMin()  #  node of min weight in row
        (nextCol, colDelt) = colTree.getMin()  #  node of min weight in col
        (nextmid, midDelt) = new_mid_tree.getMin()  # node of min weight in mid
        row_weight = rowDelt * (1 + alpha)
        col_weight = colDelt * (1 + alpha)
        mid_weight = midDelt

        min_weight = min(row_weight, col_weight)
        min_weight = min(min_weight, mid_weight)
        if min_weight == row_weight:
            row_sum += rowDelt
            for j in M1.rows[nextRow]:  # update  the  weight of connected nodes
                new_mid_value = midTree1.changeVal(j, -M1[nextRow, j])

                if new_mid_value == float('inf'):
                    continue
                temp_mid2 = midTree2.index_of(j)
                mid_min_value = min(new_mid_value, temp_mid2)
                mid_delta_value = abs(new_mid_value - temp_mid2)
                new_mid_w = mid_min_value - alpha * mid_delta_value
                new_mid_tree.setter(j, new_mid_w)

                if (new_mid_value) < mid_min[j]:  # if new_mid_value  of node j < min_mid_value  of node j
                    curScore1 -= (mid_min[j] - new_mid_value)
                    mid_min[j] = new_mid_value
                    mid_max[j] = temp_mid2

                curScore2 = curScore2 - abs(midDeltas1[j] - midDeltas2[j])
                curScore2 = curScore2 + abs(new_mid_value - midDeltas2[j])
                midDeltas1[j] = new_mid_value

            rowSet -= {nextRow}
            rowTree.changeVal(nextRow, float('inf'))
            deleted.append((0, nextRow))

        elif min_weight == col_weight:

            col_sum += colDelt
            for i in M_tran_2.rows[nextCol]:
                new_mid_value = midTree2.changeVal(i, -M_tran_2[nextCol, i])
                if new_mid_value == float('inf'):
                    continue

                temp_mid1 = midTree1.index_of(i)
                mid_min_value = min(new_mid_value, temp_mid1)
                mid_delta_value = abs(new_mid_value - temp_mid1)
                new_mid_w = mid_min_value - alpha * mid_delta_value
                new_mid_tree.setter(i, new_mid_w)

                if (new_mid_value) < mid_min[i]:
                    curScore1 -= (mid_min[i] - new_mid_value)
                    mid_min[i] = new_mid_value
                    mid_max[i] = temp_mid1
                curScore2 = curScore2 - abs(midDeltas1[i] - midDeltas2[i])
                curScore2 = curScore2 + abs(new_mid_value - midDeltas1[i])
                midDeltas2[i] = new_mid_value

            colSet -= {nextCol}
            colTree.changeVal(nextCol, float('inf'))
            deleted.append((1, nextCol))
        elif min_weight == mid_weight:

            curScore1 -= mid_min[nextmid]
            curScore2 -= abs(midDeltas1[nextmid] - midDeltas2[nextmid])

            mid_min[nextmid] = 0
            midDeltas1[nextmid] = 0
            midDeltas2[nextmid] = 0

            mid_sum1 += midTree1.index_of(nextmid)
            mid_sum2 += midTree2.index_of(nextmid)

            for j in M2.rows[nextmid]:
                colTree.changeVal(j, -M2[nextmid, j])

            for j in M_tran_1.rows[nextmid]:
                rowTree.changeVal(j, -M_tran_1[nextmid, j])

            midSet -= {nextmid}
            midTree.changeVal(nextmid, float('inf'))
            midTree1.changeVal(nextmid, float('inf'))
            midTree2.changeVal(nextmid, float('inf'))
            new_mid_tree.changeVal(nextmid, float('inf'))
            deleted.append((2, nextmid))

        numDeleted += 1
        if (len(rowSet) + len(midSet) + len(colSet)) > 0:
            curAveScore1 = curScore1 / (len(rowSet) + len(midSet) + len(colSet))
        else:
            curAveScore1 = 0
        if (len(rowSet) + len(midSet) + len(colSet))> 0:
            curAveScore2 = curScore2 / (len(rowSet) + len(midSet) + len(colSet))
        else:
            curAveScore2 = 0

        curAveScore = curAveScore1 - alpha * curAveScore2

        if curAveScore >= bestAveScore:
            bestNumDeleted = numDeleted
            bestAveScore = curAveScore
            temp1 = curAveScore1
            temp2 = curAveScore2

    print('best delete number : ', bestNumDeleted)
    print('nodes number remaining', len(rowSet), len(midSet), len(colSet))
    print('matrix mass remaining:  ', curScore1, curScore2)
    print('min value of the tree :  ', '  row   ', rowTree.getMin(), ' mid  ', midTree1.getMin(), midTree2.getMin(),
          new_mid_tree.getMin(),  '  col ', colTree.getMin())


    print('best score : ', bestAveScore, temp1, temp2)
    finalRowSet = set(range(row_length))
    finalMidSet = set(range(mid_length))
    finalColSet = set(range(col_length))

    for i in range(bestNumDeleted):
        if deleted[i][0] == 0:
            finalRowSet.remove(deleted[i][1])
        elif deleted[i][0] == 1:
            finalColSet.remove(deleted[i][1])
        elif deleted[i][0] == 2:
            finalMidSet.remove(deleted[i][1])

    return (finalRowSet, finalMidSet, finalColSet), bestAveScore

def del_block(M, rowSet ,colSet):
    M = M.tolil()

    (rs, cs) = M.nonzero()
    for i in range(len(rs)):
        if rs[i] in rowSet or cs[i] in colSet:
            M[rs[i], cs[i]] = 0
    return M.tolil()

class FlowScope( DMmodel ):
    '''Anomaly detection base on contrastively dense subgraphs, considering
    topological, temporal, and categorical (e.g. rating scores) signals, or
    any supported combinations.

    Parameters
    ----------
    graph: Graph
        Graph instance contains adjency matrix, and possible multiple signals.
    '''
    def __init__(self, graphList: list, graphnum:int=2, **params):
        self.graphnum = graphnum
        if self.graphnum == 2:
            self.graph1 = graphList[0]
            self.graph2 = graphList[1]
            self.alpha = param_default(params, 'alpha', 4)
            self.run = self.runW2
        elif self.graphnum > 2:
            self.graph1 = graphList[0]
            self.graph2 = graphList[1]
            self.graph3 = graphList[2]
            self.alpha = param_default(params, 'alpha', 0.8)
            self.run = self.runW3
        else:
            print("graphnum must greater or equal to 2")
            
        # self.alg = param_default(params, 'alg', 'fastgreedy')

    def __str__(self):
        return str(vars(self))


    def runW3(self, k:int=2, level:int=2):
        '''run with how many blocks are output.
        Parameters:
        --------
        nblock: int
            The number of block we need from the algorithm
        '''
        # print("initialize graph...")
        print("you are running with 4 partite graph")
        self.level = level
        Mcur1 = self.graph1.graph_tensor._data.copy().tocsr().tolil()
        Mcur2 = self.graph2.graph_tensor._data.copy().tocsr().tolil()
        Mcur3 = self.graph3.graph_tensor._data.copy().tocsr().tolil()
        self.nres = []

        for i in range(k):
            if self.level == 2:
                ((rowSet, midSet1, midSet2, colSet), score) = fastGreedyDecreasing3(Mcur1, Mcur2, Mcur3, self.alpha)
            else:
                return print("No such level know")

            self.nres.append([(rowSet, midSet1, midSet2, colSet), score])
            Mcur1 = del_block(Mcur1, rowSet, midSet1)
            Mcur2 = del_block(Mcur2, midSet1, midSet2)
            Mcur3 = del_block(Mcur3, midSet2, colSet)

        return self.nres
    
    def runW2(self, k:int=30, level:int=0):
        '''run with how many blocks are output.
        Parameters:
        --------
        nblock: int
            The number of block we need from the algorithm
        '''
        # print("initialize graph...")
        print("you are running with 3 partite graph")
        self.level = level
        Mcur1 = self.graph1.graph_tensor._data.copy().tocsr().tolil()
        Mcur2 = self.graph2.graph_tensor._data.copy().tocsr().tolil()
        self.nres = []

        for i in range(k):
            if self.level == 0:
                ((rowSet, midSet, colSet), score) = fastGreedyDecreasing2(Mcur1, Mcur2, self.alpha)
            else:
                return print("No such level know")

            self.nres.append([(rowSet, midSet, colSet), score])

            Mcur1 = del_block(Mcur1, rowSet, midSet)
            Mcur2 = del_block(Mcur2, midSet, colSet)

        return self.nres
