# Paper: The computationally expensive step involves finding Hamiltonian paths on k-NN graphs,
# whose running time ranges from a few seconds to 2-5 minutes, depending on the value of k,
# the number of significant lines, and the configuration of k-NN graphs.

# This version of branch and bound takes 50.5 minutes to process a graph with 26 nodes
# Maybe a C++ implementation could be faster?
import json
import sys
from queue import PriorityQueue

# with open("graph.json") as f:
#     graph = json.load(f)

# # graph_example = {
# #     "n": 4,
# #     "weight_binary": {
# #         "0,1": 1,
# #         "0,2": 0.1,
# #         "0,3": 0.1,
# #         "1,0": 0.1,
# #         "1,2": 1,
# #         "2,1": 0.1,
# #         "2,3": 1,
# #         "3,0": 1
# #     },
# #     "weight_unary": [0, 0, 0, 0]
# # }

# graph_example = graph["bunny"]["artist"]

class Node:
    def __init__(self, trace, curr, weight_unary, weight_binary, n):
        self.trace = trace
        self.curr = curr
        self.cost = 0
        self.n = n
        self.weight_unary = weight_unary
        self.weight_binary = weight_binary
        # compute history cost
        # unary cost
        for i in range(len(self.trace)):
            self.cost = self.cost + weight_unary[self.trace[i]] * (1 - i / n)
        self.cost = self.cost + weight_unary[self.curr] * (1 - len(self.trace) / n)
        # binary cost
        for i in range(len(self.trace) - 1):
            self.cost = self.cost + weight_binary[str(self.trace[i]) + "," + str(self.trace[i+1])]
        if len(self.trace) > 0:
            self.cost = self.cost + weight_binary[str(self.trace[-1]) + "," + str(self.curr)]

    def lower_bound(self):
        if len(self.trace) == self.n - 1:
            return [True, 0]
        # compute unary lower bound by sorting all untraced unary costs and multiplying them by descending scalars
        # e.g., untraced unary costs: 1, 10, 100, then lower bound is 1 * 3/n + 10 * 2/n + 100 * 1/n
        lower_bound_unary = 0
        uncomputed_unary = []
        # this doesn't include the unary cost of the current node
        for i in range(self.n):
            if i not in self.trace and i != self.curr:
                uncomputed_unary.append(self.weight_unary[i])
        uncomputed_unary = sorted(uncomputed_unary)
        assert len(uncomputed_unary) == self.n - len(self.trace) - 1
        for i in range(len(uncomputed_unary)):
            lower_bound_unary = lower_bound_unary + uncomputed_unary[i] * (len(uncomputed_unary) - i) / self.n

        # compute binary lower bound
        lower_bound_binary = 0
        untraced_nodes = []
        # this includes the current node
        for i in range(self.n):
            if i not in self.trace:
                untraced_nodes.append(i)
        # compute min outgoing and incoming weights for untraced nodes
        min_outgoing_weights = {}
        min_incoming_weights = {}
        for node in untraced_nodes:
            min_outgoing_weights[node] = []
            min_incoming_weights[node] = []
        for key in self.weight_binary.keys():
            if int(key.split(",")[0]) in untraced_nodes and\
                int(key.split(",")[1]) in untraced_nodes:
                min_outgoing_weights[int(key.split(",")[0])].append(self.weight_binary[key])
                min_incoming_weights[int(key.split(",")[1])].append(self.weight_binary[key])
        num_node_no_outgoing = 0
        node_no_outgoing = -1
        num_node_no_incoming = 0
        node_no_incoming = -1
        for node in untraced_nodes:
            if len(min_outgoing_weights[node]) == 0:
                num_node_no_outgoing = num_node_no_outgoing + 1
                node_no_outgoing = node
                min_outgoing_weights[node] = 999999999
            else:
                min_outgoing_weights[node] = min(min_outgoing_weights[node])
            if len(min_incoming_weights[node]) == 0:
                num_node_no_incoming = num_node_no_incoming + 1
                node_no_incoming = node
                min_incoming_weights[node] = 999999999
            else:
                min_incoming_weights[node] = min(min_incoming_weights[node])
        if num_node_no_outgoing > 1 or num_node_no_incoming > 1:
            return [False]
        if num_node_no_incoming > 0 and node_no_incoming != self.curr:
            return [False]
        if num_node_no_outgoing > 0 and node_no_outgoing == self.curr:
            return [False]
        # find the untraced node whose minimal outgoing weight is maximal, excluding the current node
        node_max_outgoing = -1        
        if num_node_no_outgoing > 0:
            node_max_outgoing = node_no_outgoing
        else:
            sorted_outgoing_weights = sorted(min_outgoing_weights.items(), key=lambda item: item[1], reverse=True)
            node_max_outgoing = sorted_outgoing_weights[0][0]
            if node_max_outgoing == self.curr:
                if len(sorted_outgoing_weights) == 1:
                    return [False] # dead end
                else:
                    node_max_outgoing = sorted_outgoing_weights[1][0]
        # for the current node, find the minimal outgoing weight, because the incoming weight is decided by self.trace[-1]
        lower_bound_binary = lower_bound_binary + min_outgoing_weights[self.curr]
        # for the node whose minimal outgoing weight is maximal, discard the outgoing weight and add the incoming weight
        lower_bound_binary = lower_bound_binary + min_incoming_weights[node_max_outgoing]
        # for the remaining nodes, compute 1/2 * (minimal outgoing weight + minimal incoming weight)
        for node in untraced_nodes:
            if node != self.curr and node != node_max_outgoing:
                lower_bound_binary = lower_bound_binary + (min_outgoing_weights[node] + min_incoming_weights[node]) / 2
        return [True, lower_bound_unary + lower_bound_binary]

# optim_node = None
# upper_bound = 999999999
# root = Node(trace=[], curr=0)
# q = PriorityQueue()
# q.put((root.cost + root.lower_bound()[1], root))

# print_count = 0

# while not q.empty():
#     this_node = q.get()[1]
#     if len(this_node.trace) == graph_example["n"] - 1:
#         if this_node.cost < upper_bound:
#             upper_bound = this_node.cost
#             optim_node = this_node
#     else:
#         next_nodes = []
#         for key in graph_example["weight_binary"].keys():
#             if int(key.split(",")[0]) == this_node.curr and \
#                 int(key.split(",")[1]) not in this_node.trace:
#                 next_nodes.append(int(key.split(",")[1]))
#         for next_node in next_nodes:
#             next_node_q = Node(trace=this_node.trace + [this_node.curr], curr=next_node)
#             next_lower_bound = next_node_q.lower_bound()
#             if next_lower_bound[0] and \
#                 next_node_q.cost + next_lower_bound[1] <= upper_bound:
#                 q.put((next_node_q.cost + next_lower_bound[1], next_node_q))
#     if print_count % 10000 == 0:
#         print(this_node.trace + [this_node.curr], upper_bound)
#     print_count = print_count + 1

# if optim_node is None:
#     print("Solution does not exist!")
# else:
#     print("Optimal solution:", optim_node.trace + [optim_node.curr], "cost:", optim_node.cost)

# result_json = {"solution": optim_node.trace + [optim_node.curr], "cost": optim_node.cost}
# with open("result.json", "w") as f:
#     json.dump(result_json, f)
