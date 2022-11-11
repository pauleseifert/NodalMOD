import pickle
import timeit

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB

from cyclefinding import cycles
from helper_functions import ren_helper2, demand_helper2, create_encyclopedia, hoesch, distance_line
from import_data_object import model_data, run_parameter

starttime = timeit.default_timer()

#load model parameters
run_parameter= run_parameter(scenario_name = "nordic_grid")
run_parameter.create_scenarios()

data = model_data(create_res = True ,reduced_ts = True, export_files= True, run_parameter = run_parameter)

# Create a new model
model = gp.Model("nordic_dispatch")

#Sets
T = range(run_parameter.timesteps)  # hours
T_extra = range(1, run_parameter.timesteps)
N = data.nodes.index
Y = range(run_parameter.years)
Y_extra = range(1, run_parameter.years)
G = data.dispatchable_generators[0].index  
R = data.res_series[0].columns
DAM = data.reservoir.index
S = data.storage.index
if run_parameter.scen != 1: E = run_parameter.electrolyser.index.tolist()
L = data.ac_lines.index
LDC = data.dc_lines.index
C = range(len(L)-len(N)+1) #C_cl_df.index
# separating the flexlines
I = data.dc_lines[data.dc_lines["EI"].isin(["BHEH", "NSEH1", "NSEH2", "CLUSTER"])].index  # BHEI
D = data.dc_lines[~data.dc_lines["EI"].isin(["BHEH", "NSEH1", "NSEH2", "CLUSTER"])].index # lines not to the EI's
Z = data.reservoir_zonal_limit.index


#Parameters

r = 0.04    #zinssatz
T_line = 40     #Lifetime line
T_elec = 30     #Lifetime electrolyser
factor_opex = 0.02       #share of capex for opex each year
cost_line = 1950         #/MW/km
dist_line = distance_line(nodes=data.nodes, dc_line_overview=data.dc_lines, index=I)
eff_elec = 0.68
storage_efficiency = 0.8
annuity_line = r/(1-(1/((1+r)**T_line)))
annuity_elec = r/(1-(1/((1+r)**T_elec)))
price_LL = 3000
storage_penalty = 0.1

if run_parameter.reduced_TS:
    full_ts = data.timesteps_reduced_ts
else:
    full_ts = 8760
delta = 8760/full_ts

k = hoesch(data.ac_lines,data.nodes)
C_cl = cycles(data.ac_lines)
C_cl_numpy = pd.DataFrame(C_cl).to_numpy()
k_dict = pd.DataFrame(k).to_dict()
x_numpy = pd.Series(data.ac_lines["x"]).to_numpy()
C_cl_x_multi = np.multiply(C_cl_numpy, x_numpy)
C_cl_x_multi_dict = pd.DataFrame(C_cl_x_multi).to_dict()

#here I do some dictionary reordering.
encyc_powerplants_bus = create_encyclopedia(data.dispatchable_generators[0]["bus"])
encyc_storage_bus = create_encyclopedia(data.storage["bus"])
encyc_DC_from = create_encyclopedia(data.dc_lines["from"])
encyc_DC_to = create_encyclopedia(data.dc_lines["to"])
encyc_dam = create_encyclopedia(data.reservoir["bus"])
encyc_dam_zones = create_encyclopedia(data.reservoir["bidding_zone"])
if run_parameter.scen != 1: encyc_elec = create_encyclopedia(run_parameter.electrolyser["bus"])

data.res_series_busses = dict()
for k in data.res_series[0].columns.to_list():
    data.res_series_busses[k] = False

ror_supply_busses = dict()
for k in data.ror_series.columns.to_list():
    ror_supply_busses[k] = False
ror_supply_dict = data.ror_series.to_dict()
#demand_col_list = data.demand.columns.to_list()
demand_dict = {}
for i in range(run_parameter.years):
    data.demand[i].reset_index(drop=True, inplace=True)
    demand_dict.update({i: data.demand[i].to_dict()})

print("Preparations done. The time difference is :", timeit.default_timer() - starttime)

# Variables

P_C = model.addVars(Y, T, G, lb=0.0, ub = GRB.INFINITY, name="P_C")
P_R = model.addVars(Y, T, R, lb=0.0, ub = GRB.INFINITY, name="P_R")
P_DAM = model.addVars(Y, T, DAM, lb=0.0, ub = GRB.INFINITY, name="P_DAM")
res_curtailment = model.addVars(Y, T, R, lb=0.0, ub = GRB.INFINITY, name="res_curtailment")
cap_BH = model.addVars(Y, I, lb=0.0, ub = GRB.INFINITY, name = "cap_BH")
F_AC = model.addVars(Y, T, L, lb =-GRB.INFINITY, ub=GRB.INFINITY, name="F_AC")
F_DC = model.addVars(Y, T, LDC, lb =-GRB.INFINITY,ub=GRB.INFINITY, name = "F_DC")
p_load_lost = model.addVars(Y, T,N, lb=0.0, ub = GRB.INFINITY, name = "p_load_lost")
if run_parameter.scen != 1:
    cap_E = model.addVars(Y, E, lb=0.0, ub = GRB.INFINITY, name = "cap_E")
    P_H = model.addVars(Y, T, E, lb=0.0, ub=GRB.INFINITY, name="P_H")
# storage variables
print("before Variables are made. The time difference is :", timeit.default_timer() - starttime)

P_S = model.addVars(Y, T, S, lb=0.0, ub = GRB.INFINITY, vtype=GRB.CONTINUOUS, name="P_S")  # power gen. by storage (depletion)
C_S = model.addVars(Y, T, S, lb=0.0, ub = GRB.INFINITY, vtype=GRB.CONTINUOUS, name="C_S")  # demand from storage (filling)
L_S = model.addVars(Y, T, S, lb=0.0, ub = GRB.INFINITY, vtype=GRB.CONTINUOUS, name="L_S")  # storage level
print("Variables made. The time difference is :", timeit.default_timer() - starttime)

# objective function
# Set objective
if run_parameter.scen in [1]:
    model.setObjective(
        gp.quicksum((
        delta * gp.quicksum(data.dispatchable_generators[y]["mc"][g] * P_C[y, t, g] for g in G for t in T )
        + delta * price_LL * gp.quicksum(p_load_lost[y, t, n] for n in N for t in T )
        + delta * storage_penalty * gp.quicksum(P_S[y, t, s] for s in S for t in T)                             #penalty for storage discharge
        + (gp.quicksum(cap_BH[y, i]* dist_line[i] for i in I) * cost_line * annuity_line)*(run_parameter.timesteps/full_ts)
        )/((1+r)**(5*y))
         for y in Y), GRB.MINIMIZE)
if run_parameter.scen in [2,3,4]:
    model.setObjective(
        gp.quicksum((
        delta * gp.quicksum(data.dispatchable_generators[y]["mc"][g] * P_C[y, t, g] for g in G for t in T)
        #+ delta * gp.quicksum(30 * P_R[y, t, r] for r in R for t in T)
        + delta * price_LL * gp.quicksum(p_load_lost[y, t, n] for n in N for t in T)
        + delta * storage_penalty * gp.quicksum(P_S[y, t, s] for s in S for t in T)                             #penalty for storage discharge
        - delta * gp.quicksum(P_H[y, t, e] * run_parameter.R_H[y] for e in E for t in T) * eff_elec
        + (gp.quicksum(cap_E[y, e] * run_parameter.electrolyser["cost"][e] for e in E)  * (annuity_elec+factor_opex))*(run_parameter.timesteps/full_ts)   #für unterjährig     #CAPEX electrolyser
        + (gp.quicksum(cap_BH[y, i]* dist_line[i] for i in I) * cost_line * annuity_line)*(run_parameter.timesteps/full_ts)
        )/((1+r)**(5*y))
         for y in Y), GRB.MINIMIZE)

model.addConstrs((P_C[y, t, g] <= data.dispatchable_generators[y]["P_inst"][g] for g in G for t in T for y in Y), name="GenerationLimitUp")
model.addConstrs((P_DAM[y, t, g] <= data.reservoir["P_inst"][g] for g in DAM for t in T for y in Y), name="DAMLimitUp")
model.addConstrs((gp.quicksum(P_DAM[y, t, g] for g in encyc_dam_zones[z] for t in T) <= data.reservoir_zonal_limit[z] for z in Z for y in Y), name="DAMSumUp")

model.addConstrs((P_R[y, t, r] <= data.res_series[y][r][t] for r in R for t in T for y in Y), name="ResGenerationLimitUp")
model.addConstrs((res_curtailment[y, t, r] == data.res_series[y][r][t] - P_R[y, t, r] for r in R for t in T for y in Y), name="Curtailment")
model.addConstrs((p_load_lost[y, t,j] <= demand_helper2(j,t, y,demand_dict) for j in N for t in T for y in Y), name= "limitLoadLoss")
model.addConstrs((P_S[y, t, s] <= data.storage["Pmax_out"][s] for s in S for t in T for y in Y), name="StoragePowerOutput")
model.addConstrs((C_S[y, t, s] <= data.storage["Pmax_in"][s] for s in S for t in T for y in Y), name="StoragePowerInput")
model.addConstrs((P_S[y, t, s] <= L_S[y, t, s] for s in S for t in T for y in Y), name="StorageLevelGen")
model.addConstrs((L_S[y, t, s] <= data.storage["capacity"][s] for s in S for t in T for y in Y), name="StorageLevelCap")

#storage
model.addConstrs((L_S[y, t, s] == 0.5 * data.storage["capacity"][s] - P_S[y, t, s] + storage_efficiency * C_S[y, t, s]  for s in S for t in [0] for y in Y), name="Storage_balance_init")
model.addConstrs((L_S[y, t, s] == L_S[y, t-1, s] - P_S[y, t, s] + storage_efficiency * C_S[y, t, s]  for s in S for t in T_extra for y in Y), name="Storage_balance")
model.addConstrs((L_S[y, T[-1], s] == 0.5 * data.storage["capacity"][s] for s in S for y in Y), name="Storage_end_level")

print("The time difference before flow lines :", timeit.default_timer() - starttime)
if run_parameter.scen in [1]:
    model.addConstrs((
        gp.quicksum(P_C[y, t, g] for g in encyc_powerplants_bus[n])
        + gp.quicksum(P_R[y, t, r] for r in ren_helper2(n, data.res_series_busses))
        + gp.quicksum(P_DAM[y, t, dam] for dam in encyc_dam[n])
        + gp.quicksum(ror_supply_dict[r][t] for r in ren_helper2(n, ror_supply_busses))
        + gp.quicksum(F_DC[y, t, d] for d in encyc_DC_from[n])
        - gp.quicksum(F_DC[y, t, d] for d in encyc_DC_to[n])
        + gp.quicksum(P_S[y, t, s] - C_S[y, t, s] for s in encyc_storage_bus[n])
        + gp.quicksum(k_dict[n][l] * F_AC[y, t, l] for l in L)
         == demand_helper2(n,t,y, demand_dict) - p_load_lost[y, t, n] for n in N for t in T for y in Y), name ="Injection_equality")
if run_parameter.scen in [2,3,4]:
    model.addConstrs((
        gp.quicksum(P_C[y, t, g] for g in encyc_powerplants_bus[n])
        + gp.quicksum(P_R[y, t, r] for r in ren_helper2(n, data.res_series_busses))
        + gp.quicksum(P_DAM[y, t, dam] for dam in encyc_dam[n])
        + gp.quicksum(ror_supply_dict[r][t] for r in ren_helper2(n, ror_supply_busses))
        + gp.quicksum(F_DC[y, t, d] for d in encyc_DC_from[n])
        - gp.quicksum(F_DC[y, t, d] for d in encyc_DC_to[n])
        + gp.quicksum(P_S[y, t, s] - C_S[y, t, s] for s in encyc_storage_bus[n])
        - gp.quicksum(P_H[y,t,e] for e in encyc_elec[n])
        + gp.quicksum(k_dict[n][l] * F_AC[y, t, l] for l in L)
         == demand_helper2(n,t,y, demand_dict) - p_load_lost[y, t, n] for n in N for t in T for y in Y), name ="Injection_equality")

print("The time difference after flow lines :", timeit.default_timer() - starttime)
model.addConstrs((gp.quicksum(C_cl_x_multi_dict[l][c] * F_AC[y, t, l] for l in L) == 0 for c in C for t in T for y in Y), name="Flow_lines")
print("The time difference after xbus:", timeit.default_timer() - starttime)
model.addConstrs((F_DC[y, t, d] <= data.dc_lines["max"][d] for d in D for t in T for y in Y), name="transfermax_up")
model.addConstrs((F_DC[y, t, d] >= -data.dc_lines["max"][d] for d in D for t in T for y in Y), name="transfermax_down")
model.addConstrs((F_DC[y, t, i] <= run_parameter.TRM * cap_BH[y,i] for i in I for t in T for y in Y), name="transfermaxflex")
model.addConstrs((F_DC[y, t, i] >= run_parameter.TRM *-cap_BH[y,i] for i in I for t in T for y in Y), name="transfermaxflex2")
model.addConstrs((F_AC[y, t, l] <= data.ac_lines["max"][l] for l in L for t in T for y in Y), name = "LinePowerFlowMax")
model.addConstrs((F_AC[y, t, l] >= -data.ac_lines["max"][l] for l in L for t in T for y in Y), name = "LinePowerFlowMmin")
if run_parameter.scen in [2,3,4]:
    model.addConstrs((P_H[y, t, e] <= cap_E[y,e] for e in E for t in T for y in Y), name="Capacity_Limit_Electrolyser")
    model.addConstrs((cap_E[y-1,e] <= cap_E[y,e] for e in E for y in Y_extra), name="Capacity_pre_year_electrolyser")
model.addConstrs((cap_BH[y-1,i] <= cap_BH[y, i] for i in I for y in Y_extra), name="Capacity_pre_year_lines")   #limiting construction
if run_parameter.scen in [4]:
    model.addConstrs(((gp.quicksum(cap_BH[y, i] for i in data.dc_lines[data.dc_lines["EI"].isin([0])].index) <= 3000) for y in Y), name="Capacity_Limit_flexlines")
    model.addConstrs(((gp.quicksum(cap_BH[y, i] for i in data.dc_lines[data.dc_lines["EI"].isin([1])].index) <= 10000) for y in Y),name="Capacity_Limit_flexlines")
    model.addConstrs(((gp.quicksum(cap_BH[y, i] for i in data.dc_lines[data.dc_lines["EI"].isin([2])].index) <= 10000) for y in Y),name="Capacity_Limit_flexlines")

try:
    model.write(run_parameter.export_model_formulation)
    print("The time difference after model writing:", timeit.default_timer() - starttime)
except:
    print("error while writing model data")
    pass
# necessary files: P_R_max, busses, data.dispatchable_generators, storage, lines, linesDC and ror
data.nodes.to_csv(run_parameter.export_folder + "busses.csv")
data.storage.to_csv(run_parameter.export_folder + "storage.csv")
data.ac_lines.to_csv(run_parameter.export_folder + "lines.csv")
data.dc_lines.to_csv(run_parameter.export_folder + "lines_DC.csv")
with open(run_parameter.export_folder+'share_wind.pkl', 'wb+') as f:
    pickle.dump(data.share_wind, f)
with open(run_parameter.export_folder+'share_solar.pkl', 'wb+') as f:
    pickle.dump(data.share_solar, f)
data.ror_series.to_csv(run_parameter.export_folder + "ror_supply.csv")
print("The time difference is :", timeit.default_timer() - starttime)
print("done")
