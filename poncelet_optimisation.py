import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import os
from sys import platform
import sys

if platform == "linux" or platform == "linux2":
    directory = "/work/seifert/"
    N_repr = int(sys.argv[1])
    timelimit = int(sys.argv[2])
    number_of_nodes = int(sys.argv[3])
elif platform == "darwin":
    directory = ""
    N_repr = 21 #representative days we want to have
    timelimit = 15000
    #number_of_nodes = 523

location = "data/PyPSA_elec1024/"
#renewables_supply = pd.read_csv(directory+location + "renewables_for_poncelet.csv", index_col=0)
renewables_supply = pd.read_csv(directory+location + "/RES/renewables_full_0.csv", index_col=0)
test_ts = renewables_supply
#test_ts = renewables_supply
#test_ts = pd.read_csv(location + "test_csv.csv", index_col=0)
ts_numpy= test_ts.to_numpy()#[:,:number_of_nodes]
model = gp.Model("TS_optimisation")
#sets
D = range(365)  #representative days?
C = range(ts_numpy.shape[1])    #set of duration curves
B = range(20)   #number of bins

#Parameters
def find_L(ts_numpy, B, C):
    #sorted_ts = ts.sort_values(axis=0, ascending=False).reset_index(drop=True).to_numpy()
    sorted_ts = np.sort(ts_numpy, axis=0)[::-1,:]
    max_value = sorted_ts[0, :]
    min_value = sorted_ts[8759, :]
    span = max_value-min_value
    bin_dict = np.zeros((len(B), len(C)))
    L_b = np.zeros((len(B),len(C)), dtype=int)
    for c in C:
        for b in B:
            power_value_this_bin = max_value[c] -(b+1)*(span[c]/len(B))  # ideal power value
            difference_array = np.absolute(sorted_ts[:,c] - power_value_this_bin)
            L_b[b,c] = difference_array.argmin() # position in the 8760 timesteps
            bin_dict[b,c] = sorted_ts[L_b[b,c], c]  # power value at the position
    return L_b/8760*100, bin_dict

def find_A(ts_numpy, B, D, bin_dict, n_seq):
    A = np.zeros((len(B), len(D), len(C)), dtype=int)
    for c in C:
        for d in D:
            for b in B:
                daily_ts = ts_numpy[(d)*24:(d+1)*24, c]
                daily_ts_sort = np.flip(np.sort(daily_ts, axis=0))
                difference_array = np.absolute(daily_ts_sort - bin_dict[b,c])
                A[b, d, c] = difference_array.argmin()
    return A/24*100
N_total = 365       #Number of times a representative period has to be repeated to cover a whole year     #number of representative days I want to have
steps = N_repr* 24  #target representative days * resolution of the data I want to have
L, bin_dict = find_L(ts_numpy, B, C)
A = find_A(ts_numpy, B, D ,bin_dict, N_repr)

print("preprocessing done")

#variables
w = model.addVars(D, ub = GRB.INFINITY, name="w_d")       #weighting factors
u = model.addVars(D, vtype= GRB.BINARY, name="u_d")
error = model.addVars(B, C,  ub = GRB.INFINITY, name="bla")

model.setObjective(gp.quicksum(error[b,c] for b in B for c in C), GRB.MINIMIZE) # automatically positive defined
model.addConstrs(L[b,c] - gp.quicksum(((w[d])/N_total) * A[b,d,c] for d in D) == error[b,c] for b in B for c in C)
#model.addConstrs(L[b,c] - gp.quicksum((0.2) * A[b,d,c] for d in D) == error[b,c] for b in B for c in C)
model.addConstr(gp.quicksum(u[d] for d in D) == N_repr)
model.addConstrs((w[d] == u[d] * N_total/N_repr) for d in D)         #gleichgewichtung!
model.addConstr(gp.quicksum(w[d] for d in D) == N_total)

print("building model - done")


#model.feasRelaxS(0, True, False, True)
#DualReductions=0
#model.computeIIS()
#model.write('my_iis.ilp')
#model.Params.TuneTimeLimit = 20000
#model.tune()
#model.Params.Heuristics = 0
#model.Params.Cuts= 2
#model.Params.Presolve = 2
model.Params.MIPGap = 0.01
model.Params.TimeLimit = timelimit
#model.write('poncelet_problem.mps')
model.optimize()

u_result = pd.DataFrame({'value': u[d].X} for d in D)
os.makedirs(directory+location+"/poncelet/", exist_ok=True)
u_result.to_csv(directory+location+"/poncelet/u_result.csv")
#w_result = pd.DataFrame({'value': w[d].X} for d in D)

print("done")
poncelet_ts = np.empty((0, len(C)))
for d in D:
    if u_result.at[d,"value"] == 1.0:
        daily_ts = ts_numpy[(d) * 24:(d + 1) * 24]
        poncelet_ts = np.vstack([poncelet_ts, daily_ts])
    else: pass
#poncelet_ts = np.delete(A, (0), axis=0)
pd.DataFrame(poncelet_ts).to_csv(directory+location+"/poncelet/poncelet_ts.csv")
#poncelet_ts_sorted = np.sort(poncelet_ts, axis=0)[::-1,:]





