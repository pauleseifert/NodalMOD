import networkx as nx
import numpy as np
import pandas as pd


def cycles(lines):
    def nodes_to_edges(nodes, GG):
        elementcount = len(nodes)
        edgelist = []
        for i in range(elementcount):
            if i == elementcount-1:
                try:
                    edgelist.append(GG[nodes[i]][nodes[0]]["name"])
                except:
                    edgelist.append(-(GG[nodes[0]][nodes[i]]["name"]))
            elif i != elementcount-1:
                try:
                    edgelist.append(GG[nodes[i]][nodes[i+1]]["name"])
                except:
                    edgelist.append(-(GG[nodes[i+1]][nodes[i]]["name"]))
        return edgelist
    ## Create 2 graphs and add edges

    G = nx.Graph()
    GG = nx.DiGraph()
    lines.apply(lambda x: G.add_edge(int(x["from"]), int(x["to"]), name=x.name), axis = 1)
    lines.apply(lambda x: GG.add_edge(int(x["from"]), int(x["to"]), name=x.name), axis=1)


    #the cycle basis from the undirected graph
    cycles = nx.cycle_basis(G)
    #test_directed_cycles = nx.recursive_simple_cycles(G)
    #test = cycles[1]
    cycle_edges = []
    for nodes in cycles:
        cycle_edges.append(nodes_to_edges(nodes, GG))

    C_cl = np.zeros((len(cycle_edges), len(lines)), dtype=np.int8)
    for i, cycles in enumerate(cycle_edges):
        for nodes in cycles:
            if nodes >= 0:
                C_cl[i, abs(nodes)] = 1
            if nodes < 0:
                C_cl[i, abs(nodes)] = -1
    def check_function(G):
        testlist = []
        for node1, node2, i in G.edges(data=True):
            testlist.append(i["name"])
        test_frame = pd.DataFrame(testlist).sort_values(0).reset_index(drop=True)
        if test_frame.iloc[-1][0] == test_frame.iloc[-1].name:
            return True, ""
        else:
            #lets find the double lines and their positions
            double_entry = {}
            entry_number = 0
            for i, row in test_frame.iterrows():
                if i+1+entry_number == row[0]:
                    double_entry.update({entry_number:i})
                    entry_number += 1
                #test = test_frame.where(test_frame[0] != test_frame.index)
            return False, double_entry
    check_value, double_lines = check_function(G)
    if check_value:
        print("cycles checked")
        return C_cl
    else:
        raise Exception("There are parallel lines at the index:" + str(double_lines))
