import timeit

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB

from cyclefinding import cycles
from helper_functions import ren_helper2, demand_helper2, create_encyclopedia, line_bus_matrix
from import_data_object import model_data, run_parameter

starttime = timeit.default_timer()

# load model parameters
run_parameter = run_parameter(scenario_name="nordic_grid")
run_parameter.create_scenarios()

####################################
####################################
###     Optimisation Model       ###
####################################
####################################


data = model_data(create_res=True, reduced_ts=False, export_files=True, run_parameter=run_parameter)

# def ptdf(bus_raw, lines, slack = 0):
#     b_matrix= np.diag(1/lines["x"])
#     b_df = pd.DataFrame(b_matrix, columns = lines.index, index = lines.index)# dimensions: LxL Lines and lines
#     a_matrix = pd.DataFrame(np.zeros((len(bus_raw.index), (len(lines.index))), dtype=int), columns=lines.index,index=bus_raw.index)  # dimensions: LxJ with nodes as index and lines as columns
#     for i in lines.index:
#         a_matrix[i][lines[lines.index == i]["from"]] = 1
#         a_matrix[i][lines[lines.index == i]["to"]] = -1
#     incidence_df = a_matrix.transpose()
#     incidence_matrix = incidence_df.to_numpy()
#     def create_B(lines, bus_raw):
#         fbus = lines["from"]
#         tbus = lines["to"]
#         lines_x = lines["x"]
#         ybus = pd.DataFrame(np.zeros((len(bus_raw), (len(bus_raw)))), columns=bus_raw.index,index=bus_raw.index)             #dimensions nodes*nodes
#         for i in lines.index:
#             ybus[fbus[i]][tbus[i]] = -1/lines_x[i]
#             ybus[tbus[i]][fbus[i]] = -1/lines_x[i]
#         rowsum_ybus = np.sum(ybus, axis=1)
#         for i in ybus.index:
#             ybus[i][i] = - rowsum_ybus[i]
#         return pd.DataFrame(ybus)
#
#     B = create_B(lines, bus_raw)# dimensions: JxJ nodes*nodes
#
#     B.loc[:,slack] = 0.0
#     B.loc[slack,:] = 0.0
#     B_inv = pd.DataFrame(np.linalg.pinv(B.values), B.index, B.index) #Dimension: bxb
#     ptdf_test = b_vector@incidence_matrix
#     ptdf_as_described =

#    return incidence_matrix, B
#ptdf(data.nodes, data.ac_lines)

def nodal_markets(data = data, run_parameter = run_parameter):

    # Create a new model
    model = gp.Model("nordic_da_dispatch")
    # Sets
    T = range(run_parameter.timesteps)  # hours
    T_extra = range(1, run_parameter.timesteps)
    N = data.nodes.index
    Y = run_parameter.years
    G = data.dispatchable_generators[run_parameter.years[0]].index
    R = data.res_series[run_parameter.years[0]].columns
    DAM = data.reservoir.index
    S = data.storage.index
    L = data.ac_lines.index
    #LDC = data.dc_lines.index
    C = range(len(L)-len(N)+1)  # C_cl_df.index
    #Z = run_parameter.bidding_zone_selection


    ####################################
    ###          Parameters          ###
    ####################################

    storage_efficiency = 0.8
    c_d = 3000
    storage_penalty = 0.001

    if run_parameter.reduced_TS:
        full_ts = data.timesteps_reduced_ts
    else:
        full_ts = 8760
    delta = 8760/full_ts

    k = line_bus_matrix(data.ac_lines, data.nodes)
    C_cl = cycles(data.ac_lines)
    C_cl_numpy = pd.DataFrame(C_cl).to_numpy()
    k_dict = pd.DataFrame(k).to_dict()
    x_numpy = pd.Series(data.ac_lines["x"]).to_numpy()
    C_cl_x_multi = np.multiply(C_cl_numpy, x_numpy)
    C_cl_x_multi_dict = pd.DataFrame(C_cl_x_multi).to_dict()

    # here I do some dictionary reordering.
    res_series_busses = {k: False for k in data.res_series[run_parameter.years[0]].columns.to_list()}
    ror_supply_busses = {k: False for k in data.ror_series.columns.to_list()}
    encyc_powerplants_in_node = create_encyclopedia(data.dispatchable_generators[run_parameter.years[0]]["node"])
    encyc_res_in_node = {n: ren_helper2(n, res_series_busses) for n in N}
    encyc_ror_in_node = {n: ren_helper2(n, ror_supply_busses) for n in N}

    ror_supply_dict = data.ror_series.to_dict()
    #demand_col_list = data.demand.columns.to_list()
    demand_dict = {year: data.demand[year].to_dict() for year in run_parameter.years}

    print("Preparations done. The time difference is :",
          timeit.default_timer() - starttime)

    # Variables
    print("before Variables are made. The time difference is :", timeit.default_timer() - starttime)

    P_G = model.addVars(Y, T, G, lb=0.0, ub = GRB.INFINITY, name="P_G")
    P_R = model.addVars(Y, T, R, lb=0.0, ub = GRB.INFINITY, name="P_R")
    res_curtailment = model.addVars(Y, T, R, lb=0.0, ub = GRB.INFINITY, name="res_curtailment")
    F_AC = model.addVars(Y, T, L, lb =-GRB.INFINITY, ub=GRB.INFINITY, name="F_AC")
    P_D = model.addVars(Y, T, N, lb=0.0, ub = GRB.INFINITY, name = "P_D")
    print("Variables made. The time difference is :", timeit.default_timer() - starttime)
    # objective function
    model.setObjective(
        gp.quicksum(
        gp.quicksum(data.dispatchable_generators[y]["mc"][g] * P_G[y, t, g] for g in G)
        - c_d * gp.quicksum(P_D[y, t, n] for n in N)
        for t in T for y in Y), GRB.MINIMIZE)

    #upper capacity limits
    model.addConstrs((P_G[y, t, g] <= data.dispatchable_generators[y]["P_inst"][g] for g in G for t in T for y in Y), name="GenerationLimitUp")
    model.addConstrs((P_R[y, t, r] <= data.res_series[y][r][t] for r in R for t in T for y in Y),name="ResGenerationLimitUp")
    model.addConstrs((P_D[y, t, n] <= demand_helper2(n, t, y, demand_dict) for n in N for t in T for y in Y),name="DemandUpperLimit")

    model.addConstrs((res_curtailment[y, t, r] == data.res_series[y][r][t] - P_R[y, t, r] for r in R for t in T for y in Y), name="Curtailment")
    # flow constraints
    model.addConstrs((gp.quicksum(C_cl_x_multi_dict[l][c] * F_AC[y, t, l] for l in L) == 0 for c in C for t in T for y in Y),name="Flow_lines")
    print("The time difference after xbus:", timeit.default_timer() - starttime)
    model.addConstrs((F_AC[y, t, l] <= data.ac_lines["max"][l] for l in L for t in T for y in Y),name="LinePowerFlowMax")
    model.addConstrs((F_AC[y, t, l] >= -data.ac_lines["max"][l] for l in L for t in T for y in Y),name="LinePowerFlowMmin")


    model.addConstrs((
        gp.quicksum(P_G[y, t, g] for g in encyc_powerplants_in_node[n])
        - P_D[y, t, n]
        + gp.quicksum(P_R[y, t, r] for r in encyc_res_in_node[n])
        + gp.quicksum(k_dict[n][l] * F_AC[y, t, l] for l in L)
        + gp.quicksum(ror_supply_dict[r][t] for r in encyc_ror_in_node[n])
        == 0
        for n in N for t in T for y in Y), name ="Energy_Balance")

    #take the prices and the nodal quantities and calculate
    #with ptdf -> calculate and not resolve it!
    # but with ptdfs you need to fix the DC net injections beforehand
    # PROBLEM: If the first stage solutaion is not restricted, the second stage - if an optimisation problem- finds more profitable trades -> not only redispatch to make it feasible
    try:
        model.write(run_parameter.export_model_formulation)
        print("The time difference after model writing:", timeit.default_timer() - starttime)
    except:
        print("error while writing model data")
        pass
    return model


def zonal_markets(data=data, run_parameter=run_parameter):
    # Create a new model
    model = gp.Model("nordic_da_dispatch")
    # Sets
    T = range(run_parameter.timesteps)  # hours
    T_extra = range(1, run_parameter.timesteps)
    N = data.nodes.index
    Y = run_parameter.years
    Y_extra = run_parameter.years[1:]
    G = data.dispatchable_generators[run_parameter.years[0]].index
    R = data.res_series[run_parameter.years[0]].columns
    DAM = data.reservoir.index
    S = data.storage.index
    L = data.ac_lines.index
    # LDC = data.dc_lines.index
    C = range(len(L) - len(N) + 1)  # C_cl_df.index
    Z = run_parameter.bidding_zone_selection

    ####################################
    ###          Parameters          ###
    ####################################

    storage_efficiency = 0.8
    price_LL = 3000
    storage_penalty = 0.001

    if run_parameter.reduced_TS:
        full_ts = data.timesteps_reduced_ts
    else:
        full_ts = 8760
    delta = 8760 / full_ts

    k = line_bus_matrix(data.ac_lines, data.nodes)
    C_cl = cycles(data.ac_lines)
    C_cl_numpy = pd.DataFrame(C_cl).to_numpy()
    k_dict = pd.DataFrame(k).to_dict()
    x_numpy = pd.Series(data.ac_lines["x"]).to_numpy()
    C_cl_x_multi = np.multiply(C_cl_numpy, x_numpy)
    C_cl_x_multi_dict = pd.DataFrame(C_cl_x_multi).to_dict()

    # here I do some dictionary reordering.
    encyc_nodes_in_zones = create_encyclopedia(data.nodes["bidding_zone"])
    encyc_powerplants_in_zone = create_encyclopedia(data.dispatchable_generators[run_parameter.years[0]]["bidding_zone"])
    encyc_storage_in_zone = create_encyclopedia(data.storage["bidding_zone"])
    encyc_dam_in_zones = create_encyclopedia(data.reservoir["bidding_zone"])
    encyc_res_in_zones = create_encyclopedia(
        data.res_series[run_parameter.years[0]].T.merge(data.nodes["bidding_zone"], left_index=True, right_index=True)[
            "bidding_zone"])
    encyc_ror_in_zones = create_encyclopedia(
        data.ror_series.T.merge(data.nodes["bidding_zone"], left_index=True, right_index=True)["bidding_zone"])

    encyc_powerplants_in_node = create_encyclopedia(data.dispatchable_generators[run_parameter.years[0]]["node"])
    encyc_storage_bus = create_encyclopedia(data.storage["node"])
    # encyc_DC_from = create_encyclopedia(data.dc_lines["from"])
    # encyc_DC_to = create_encyclopedia(data.dc_lines["to"])
    # encyc_dam = create_encyclopedia(data.reservoir["node"])
    # encyc_dam_zones = create_encyclopedia(data.reservoir["bidding_zone"])

    res_series_busses = {k: False for k in data.res_series[run_parameter.years[0]].columns.to_list()}
    ror_supply_busses = {k: False for k in data.ror_series.columns.to_list()}

    ror_supply_dict = data.ror_series.to_dict()
    # demand_col_list = data.demand.columns.to_list()
    demand_dict = {year: data.demand[year].to_dict() for year in run_parameter.years}

    print("Preparations done. The time difference is :",
          timeit.default_timer() - starttime)

    # Variables
    print("before Variables are made. The time difference is :", timeit.default_timer() - starttime)

    P_C = model.addVars(Y, T, G, lb=0.0, ub=GRB.INFINITY, name="P_C")
    P_R = model.addVars(Y, T, R, lb=0.0, ub=GRB.INFINITY, name="P_R")
    P_DAM = model.addVars(Y, T, DAM, lb=0.0, ub=GRB.INFINITY, name="P_DAM")
    res_curtailment = model.addVars(Y, T, R, lb=0.0, ub=GRB.INFINITY, name="res_curtailment")
    # cap_BH = model.addVars(Y, I, lb=0.0, ub = GRB.INFINITY, name = "cap_BH")
    F_AC = model.addVars(Y, T, L, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="F_AC")
    # F_DC = model.addVars(Y, T, LDC, lb =-GRB.INFINITY,ub=GRB.INFINITY, name = "F_DC")
    # ATC = model.addVars(Y, T, LDC, lb =-GRB.INFINITY,ub=GRB.INFINITY, name = "ATC")
    p_load_lost = model.addVars(Y, T, N, lb=0.0, ub=GRB.INFINITY, name="p_load_lost")
    # storage variables
    P_S = model.addVars(Y, T, S, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS,
                        name="P_S")  # power gen. by storage (depletion)
    C_S = model.addVars(Y, T, S, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS,
                        name="C_S")  # demand from storage (filling)
    L_S = model.addVars(Y, T, S, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="L_S")  # storage level
    print("Variables made. The time difference is :", timeit.default_timer() - starttime)

    # objective function
    # Set objective
    model.setObjective(
        gp.quicksum(
            delta * gp.quicksum(
                data.dispatchable_generators[y]["mc"][g] * P_C[y, t, g] for g in encyc_powerplants_in_zone[z] for t in
                T)
            + delta * price_LL * gp.quicksum(p_load_lost[y, t, n] for n in encyc_nodes_in_zones[z] for t in T)
            + delta * storage_penalty * gp.quicksum(P_S[y, t, s] for s in encyc_storage_in_zone[z] for t in T)
            # penalty for storage discharge
            for y in Y for z in Z), GRB.MINIMIZE)

    model.addConstrs((P_C[y, t, g] <= data.dispatchable_generators[y]["P_inst"][g] for g in G for t in T for y in Y),
                     name="GenerationLimitUp")
    model.addConstrs((P_DAM[y, t, g] <= data.reservoir["P_inst"][g] for g in DAM for t in T for y in Y),
                     name="DAMLimitUp")
    model.addConstrs(
        (gp.quicksum(P_DAM[y, t, dam] for dam in encyc_dam_in_zones[z] for t in T) <= data.reservoir_zonal_limit[z] for
         z in Z for y in Y), name="DAMSumUp")

    model.addConstrs((P_R[y, t, r] <= data.res_series[y][r][t] for r in R for t in T for y in Y),
                     name="ResGenerationLimitUp")
    model.addConstrs(
        (res_curtailment[y, t, r] == data.res_series[y][r][t] - P_R[y, t, r] for r in R for t in T for y in Y),
        name="Curtailment")
    model.addConstrs((p_load_lost[y, t, n] <= demand_helper2(n, t, y, demand_dict) for n in N for t in T for y in Y),name="limitLoadLoss")
    model.addConstrs((P_S[y, t, s] <= data.storage["P_inst"][s] for s in S for t in T for y in Y), name="StoragePowerOutput")
    model.addConstrs((C_S[y, t, s] <= data.storage["P_inst"][s] for s in S for t in T for y in Y),name="StoragePowerInput")
    model.addConstrs((P_S[y, t, s] <= L_S[y, t, s] for s in S for t in T for y in Y), name="StorageLevelGen")
    model.addConstrs((L_S[y, t, s] <= data.storage["capacity"][s] for s in S for t in T for y in Y),
                     name="StorageLevelCap")

    # storage
    model.addConstrs((L_S[y, t, s] == 0.5 * data.storage["capacity"][s] - P_S[y, t, s] + storage_efficiency * C_S[y, t, s] for s in S for t in [0] for y in Y), name="Storage_balance_init")
    model.addConstrs(
        (L_S[y, t, s] == L_S[y, t - 1, s] - P_S[y, t, s] + storage_efficiency * C_S[y, t, s] for s in S for t in T_extra
         for y in Y), name="Storage_balance")
    model.addConstrs((L_S[y, T[-1], s] == 0.5 * data.storage["capacity"][s] for s in S for y in Y), name="Storage_end_level")

    model.addConstrs((
        gp.quicksum(P_C[y, t, g] for g in encyc_powerplants_in_zone[z])
        + gp.quicksum(P_R[y, t, r] for r in encyc_res_in_zones[z])
        + gp.quicksum(P_DAM[y, t, dam] for dam in encyc_dam_in_zones[z])
        + gp.quicksum(ror_supply_dict[r][t] for r in encyc_ror_in_zones[z])
        + gp.quicksum(P_S[y, t, s] - C_S[y, t, s] for s in encyc_storage_in_zone[z])
        == gp.quicksum(demand_dict[y][n][t] - p_load_lost[y, t, n] for n in encyc_nodes_in_zones[z])
        for z in Z for t in T for y in Y), name="Injection_equality")

    # 70% of the physical capacity
    # take the prices and the nodal quantities and calculate
    # with ptdf -> calculate and not resolve it!
    # but with ptdfs you need to fix the DC net injections beforehand
    # PROBLEM: If the first stage solutaion is not restricted, the second stage - if an optimisation problem- finds more profitable trades -> not only redispatch to make it feasible
    try:
        model.write(run_parameter.export_model_formulation)
        print("The time difference after model writing:", timeit.default_timer() - starttime)
    except:
        print("error while writing model data")
        pass
    return model

model = nodal_markets(data=data, run_parameter=run_parameter)
model.optimize()
model.printStats()
#model.computeIIS()
#model.write("model.ilp")
model2 = zonal_markets(data=data, run_parameter=run_parameter)

test3 = {}
for y in Y:
    for t in T:
        for s in S:
            test3.update({y:{t: model.getVarByName("L_S["+str(y)+","+str(t)+","+str(s)+"]").X}})
test4 = {y:{t: {s: model.getVarByName("L_S["+str(y)+","+str(t)+","+str(s)+"]").X }}for y in Y for t in T for s in S}
test = model.getVarByName("P_G[2030,0,0]").X
test2 = model.getVars()
def convert_grb_to_dict(model, var_name, dimension, set, type = "variable"):
    if type == "variable":
        match dimension:
            case 1:
                pass
            case 2:
                pass
            case 3:
                dict = {y:{t: {x: model.getVarByName(var_name+"["+str(y)+","+str(t)+","+str(x)+"]").X  for x in set} for t in T} for y in Y}
    if type == "constraint":
        dict = {y:{t: {x: model.getConstrByName(var_name+"["+str(x)+","+str(t)+","+str(y)+"]").Pi for x in set} for t in T} for y in Y}
    return dict
storage_leve = convert_grb_to_dict(model,"L_S", 3, S)

model2 = gp.Model("nordic_da_redispatch")
#new Variables
P_C_up = model2.addVars(Y, T, G, lb=0.0, ub = GRB.INFINITY, name="P_C_up")
P_C_down = model2.addVars(Y, T, G, lb=0.0, ub = GRB.INFINITY, name="P_C_down")
# P_R_up = model2.addVars(Y, T, N, lb=0.0, ub = GRB.INFINITY, name="P_R_up")
# P_R_down = model2.addVars(Y, T, N, lb=0.0, ub = GRB.INFINITY, name="P_R_down")
# P_S_up = model2.addVars(Y, T, S, lb=0.0, ub = GRB.INFINITY, vtype=GRB.CONTINUOUS, name="P_S_up")  # power gen. by storage (depletion)
# P_S_down = model2.addVars(Y, T, S, lb=0.0, ub = GRB.INFINITY, vtype=GRB.CONTINUOUS, name="P_S_down")  # power gen. by storage (depletion)
# C_S_up = model2.addVars(Y, T, S, lb=0.0, ub = GRB.INFINITY, vtype=GRB.CONTINUOUS, name="C_S_up")
# C_S_down = model2.addVars(Y, T, S, lb=0.0, ub = GRB.INFINITY, vtype=GRB.CONTINUOUS, name="C_S_up")


zonal_price = model.getConstrByName("Injection_equality[DK1,0,2030]")
zonal_prices = convert_grb_to_dict(model = model , var_name = "Injection_equality", dimension= 3, set = Z, type = "constraint")
#test = model.getConstrs()


model2.setObjective(
    gp.quicksum(
    delta * gp.quicksum(data.dispatchable_generators[y]["mc"][g] * P_C_up + (zonal_prices[y][t][z] - data.dispatchable_generators[y]["mc"][g])* P_C_down for g in encyc_powerplants_in_zone[z] for t in T)
    #+ delta * gp.quicksum(zonal_price * P_R_down for t in T)
    + delta * price_LL * gp.quicksum(p_load_lost[y, t, n] for n in encyc_nodes_in_zones[z] for t in T )
    + delta * storage_penalty * gp.quicksum(P_S[y, t, s] for s in encyc_storage_in_zone[z] for t in T)                             #penalty for storage discharge
     for y in Y for z in Z), GRB.MINIMIZE)

model2.addConstrs((
    gp.quicksum(P_C[y, t, g].X + P_C_up[y, t, g] - P_C_down[y, t, g] for g in encyc_powerplants_in_node[n])
    + gp.quicksum(P_R[y, t, r] for r in ren_helper2(n, res_series_busses))
    + gp.quicksum(P_DAM[y, t, dam] for dam in encyc_dam_in_nodes[n])
    + gp.quicksum(ror_supply_dict[r][t] for r in ren_helper2(n, ror_supply_busses))
    + gp.quicksum(P_S[y, t, s] - C_S[y, t, s] for s in encyc_storage_bus[n])
    + gp.quicksum(k_dict[n][l] * F_AC[y, t, l] for l in L)
     == demand_helper2(n,t,y, demand_dict) - p_load_lost[y, t, n] for n in Zones for t in T for y in Y), name ="Injection_equality")

print("The time difference after flow lines :", timeit.default_timer() - starttime)
#model.addConstrs((gp.quicksum(C_cl_x_multi_dict[l][c] * F_AC[y, t, l] for l in L) == 0 for c in C for t in T for y in Y), name="Flow_lines")
print("The time difference after xbus:", timeit.default_timer() - starttime)
model.addConstrs((F_AC[y, t, l] <= data.ac_lines["max"][l] for l in L for t in T for y in Y), name = "LinePowerFlowMax")
model.addConstrs((F_AC[y, t, l] >= -data.ac_lines["max"][l] for l in L for t in T for y in Y), name = "LinePowerFlowMmin")
try:
    model.write(run_parameter.export_model_formulation)
    print("The time difference after model writing:", timeit.default_timer() - starttime)
except:
    print("error while writing model data")
    pass
print("The time difference is :", timeit.default_timer() - starttime)
print("done")
