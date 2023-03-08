import timeit

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB

from cyclefinding import cycles
from helper_functions import ren_helper2, demand_helper2, create_encyclopedia, line_bus_matrix
#from import_data_object import model_data, run_parameter

from import_data_object_modified import model_data, run_parameter

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


def nodal_FB(data = data, run_parameter = run_parameter):
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
    # LDC = data.dc_lines.index
    C = range(len(L) - len(N) + 1)  # C_cl_df.index
    # Z = run_parameter.bidding_zone_selection

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
    delta = 8760 / full_ts

    k = line_bus_matrix(data.ac_lines, data.nodes)

    # C_cl = cycles(data.ac_lines)
    # C_cl_numpy = pd.DataFrame(C_cl).to_numpy()
    k_dict = pd.DataFrame(k).to_dict()
    x_numpy = pd.Series(data.ac_lines["x"]).to_numpy()
    # C_cl_x_multi = np.multiply(C_cl_numpy, x_numpy)
    # C_cl_x_multi_dict = pd.DataFrame(C_cl_x_multi).to_dict()

    res_series_busses = {k: False for k in data.res_series[run_parameter.years[0]].columns.to_list()}
    ror_supply_busses = {k: False for k in data.ror_series.columns.to_list()}
    encyc_powerplants_in_node = create_encyclopedia(data.dispatchable_generators[run_parameter.years[0]]["node"])
    encyc_res_in_node = {n: ren_helper2(n, res_series_busses) for n in N}
    encyc_ror_in_node = {n: ren_helper2(n, ror_supply_busses) for n in N}

    ror_supply_dict = data.ror_series.to_dict()
    # demand_col_list = data.demand.columns.to_list()
    demand_dict = {year: data.demand[year].to_dict() for year in run_parameter.years}

    print("Preparations done. The time difference is :",
          timeit.default_timer() - starttime)

    # Variables
    print("before Variables are made. The time difference is :", timeit.default_timer() - starttime)

    P_G = model.addVars(Y, T, G, lb=0.0, ub=GRB.INFINITY, name="P_G")
    P_R = model.addVars(Y, T, R, lb=0.0, ub=GRB.INFINITY, name="P_R")
    res_curtailment = model.addVars(Y, T, R, lb=0.0, ub=GRB.INFINITY, name="res_curtailment")
    F_AC = model.addVars(Y, T, L, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="F_AC")
    P_D = model.addVars(Y, T, N, lb=0.0, ub=GRB.INFINITY, name="P_D")
    P_inj = model.addVars(Y, T, N, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="P_inj")
    print("Variables made. The time difference is :", timeit.default_timer() - starttime)
    # objective function
    model.setObjective(
        gp.quicksum(
            gp.quicksum(data.dispatchable_generators[y]["mc"][g] * P_G[y, t, g] for g in G)
            - c_d * gp.quicksum(P_D[y, t, n] for n in N)
            for t in T for y in Y), GRB.MINIMIZE)

    # upper capacity limits
    model.addConstrs((P_G[y, t, g] <= data.dispatchable_generators[y]["P_inst"][g] for g in G for t in T for y in Y),
                     name="GenerationLimitUp")
    model.addConstrs((P_R[y, t, r] <= data.res_series[y][r][t] for r in R for t in T for y in Y),
                     name="ResGenerationLimitUp")
    model.addConstrs((P_D[y, t, n] <= demand_helper2(n, t, y, demand_dict) for n in N for t in T for y in Y),
                     name="DemandUpperLimit")

    model.addConstrs(
        (res_curtailment[y, t, r] == data.res_series[y][r][t] - P_R[y, t, r] for r in R for t in T for y in Y),
        name="Curtailment")
    # flow constraints
    # model.addConstrs((gp.quicksum(C_cl_x_multi_dict[l][c] * F_AC[y, t, l] for l in L) == 0 for c in C for t in T for y in Y),name="Flow_lines")
    print("The time difference after xbus:", timeit.default_timer() - starttime)
    model.addConstrs((F_AC[y, t, l] <= data.ac_lines["max"][l] for l in L for t in T for y in Y),
                     name="LinePowerFlowMax")
    model.addConstrs((F_AC[y, t, l] >= -data.ac_lines["max"][l] for l in L for t in T for y in Y),
                     name="LinePowerFlowMmin")

    model.addConstrs((
        gp.quicksum(P_G[y, t, g] for g in encyc_powerplants_in_node[n])
        - P_D[y, t, n]
        + gp.quicksum(P_R[y, t, r] for r in encyc_res_in_node[n])
        + gp.quicksum(ror_supply_dict[r][t] for r in encyc_ror_in_node[n])
        == P_inj[y, t, n]
        for n in N for t in T for y in Y), name="NodalNetInjection")

    model.addConstrs((
        gp.quicksum(P_inj[y, t, n] * data.PTDF[l][n] for n in N) == F_AC[y, t, l]
        for l in L for t in T for y in Y), name="FlowBased")

    #model.update()
    #print(model.display())
    return model

modelFB = nodal_FB(data = data, run_parameter = run_parameter)
modelFB.optimize()