import json
import sys
import numpy as np
from operator import itemgetter

with open("cost.json") as f:
    cost_all = json.load(f)

# k in [3, 8], k == 4 by default, #strokes <= 35
max_k = 4
w = 1/9

def get_cost_bi(cost_bi, i, j):
    if i == j:
        return 0
    else:
        ii = min(i, j)
        jj = max(i, j)
        return cost_bi[ii][jj - ii - 1]

graph = {}
for image, uids in cost_all.items():
    print(image)
    graph[image] = {}
    for uid, cost in uids.items():
        print("\t", uid)
        vertex_pair = []
        edge_weight = []
        n = len(cost["cost_uni"])
        for i in range(n):
            if n < max_k:
                continue
            all_pro_for_i = []
            for j in range(n):
                if j != i:
                    all_pro_for_i.append(get_cost_bi(cost["cost_bi_pro"], i, j))
                else:
                    all_pro_for_i.append(999999999)
            for k in range(max_k):
                # get the index of the k-th closest stroke
                j = sorted(enumerate(all_pro_for_i), key=itemgetter(1))[k][0]
                if [i, j] in cost["T_junctions"]:
                    # add j, i
                    vertex_pair.append((j, i))
                    edge_weight.append(w * get_cost_bi(cost["cost_bi_pro"], j, i) + (1-w) * get_cost_bi(cost["cost_bi_col"], j, i))
                elif [j, i] in cost["T_junctions"]:
                    # add i, j
                    vertex_pair.append((i, j))
                    edge_weight.append(w * get_cost_bi(cost["cost_bi_pro"], i, j) + (1-w) * get_cost_bi(cost["cost_bi_col"], i, j))
                else:
                    # add j, i
                    vertex_pair.append((j, i))
                    edge_weight.append(w * get_cost_bi(cost["cost_bi_pro"], j, i) + (1-w) * get_cost_bi(cost["cost_bi_col"], j, i))
                    # add i, j
                    vertex_pair.append((i, j))
                    edge_weight.append(w * get_cost_bi(cost["cost_bi_pro"], i, j) + (1-w) * get_cost_bi(cost["cost_bi_col"], i, j))
                ### NOTE didn't check conflicting directed cycles, could use toposort to check
        weight_binary = {}
        for i in range(len(vertex_pair)):
            weight_binary[str(vertex_pair[i][0]) + "," + str(vertex_pair[i][1])] = edge_weight[i]
        # print(n)
        # print(weight_binary)
        # print(cost["cost_uni"])
        graph[image][uid] = {"n": n, "weight_binary": weight_binary, "weight_unary": cost["cost_uni"]}

with open("graph.json", "w") as f:
    json.dump(graph, f)
