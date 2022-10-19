import networkx as nx
import pandas as pd
import numpy as np

def cycles(lines):
    def nodes_to_edges(nodes):
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

    G = nx.Graph()
    GG = nx.DiGraph()
    for index, row in lines.iterrows():
        G.add_edge(row["from"], row["to"], name=index)
        GG.add_edge(row["from"], row["to"], name=index)

    cycles = nx.cycle_basis(G)
    #test_directed_cycles = nx.recursive_simple_cycles(G)
    #test = cycles[1]
    cycle_edges = []
    for nodes in cycles:
        cycle_edges.append(nodes_to_edges(nodes))

    C_cl = np.zeros((len(cycle_edges), len(lines)), dtype=np.int8)
    for i, cycles in enumerate(cycle_edges):
        for nodes in cycles:
            if nodes >= 0:
                C_cl[i, abs(nodes)] = 1
            if nodes < 0:
                C_cl[i, abs(nodes)] = -1
    def check_function(G):
        testlist = []
        test = G.edges(data=True)
        for node1, node2, i in test:
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
