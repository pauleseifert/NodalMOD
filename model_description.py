import pickle
import timeit


#import geopandas as gpd #FEHLER?
import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB, Model

from cyclefinding import cycles
from helper_functions import ren_helper2, demand_helper2, create_encyclopedia

import geopandas as gpd
import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB

from cyclefinding import cycles
from helper_functions import ren_helper2, demand_helper2, create_encyclopedia, hoesch, distance_line

from import_data_object import model_data, run_parameter

starttime = timeit.default_timer()

#load model parameters
run_parameter= run_parameter(scenario_name = "Offshore_Bidding_Zone_Scenario")
run_parameter.create_scenarios()

data = model_data(create_res = False,reduced_ts = True, export_files= True, run_parameter = run_parameter)

data.nodes

#ToDo
#c = xxxxx
#zones festlegen, als set und zuordnung zu den nodes
#shapes = gpd.read_file("data/shapes/NUTS_RG_10M_2021_4326.geojson")
#shapes_filtered = shapes.query("LEVL_CODE ==1 and CNTR_CODE == 'DE'")
#shapes_filtered.plot()
#Ablauf:
#-nodes -> zeile -> [["LAT", "LON"]] -> for (loop over shapes_filtered) -> list -> entry with true (namen zurückgeben)
#def lockup_state(row):

#    row[["LAT", "LON"]]
#    for state in shapes_filtered
 #   return

#Zonen einfügen!
zone = ["DE1","DE2", "DE3", "DE4", "NO1", "NO2", "NO3", "DK1", "DK2", "BALTIC", "NORTH"]



#ToDo
#c = xxxxx
#zones festlegen, als set und zuordnung zu den nodes
#shapes = gpd.read_file("data/shapes/NUTS_RG_10M_2021_4326.geojson")
#shapes_filtered = shapes.query("LEVL_CODE ==1 and CNTR_CODE == 'DE'")
#shapes_filtered.plot()
#Ablauf:
#-nodes -> zeile -> [["LAT", "LON"]] -> for (loop over shapes_filtered) -> list -> entry with true (namen zurückgeben)
#def lockup_state(row):

#    row[["LAT", "LON"]]
#    for state in shapes_filtered
#    return

# Create a new model
model: Model = gp.Model("Energy_Island_Investment_Dispatch")

#Sets
T = range(run_parameter.timesteps)  # hours
T_extra = range(1, run_parameter.timesteps)

Y = range(run_parameter.years)
Y_extra = range(1, run_parameter.years)
G = data.dispatchable_generators[0].index  
R = data.res_series[0].columns
DAM = data.reservoir.index
S = data.storage.index
LDC = data.dc_lines.index

I = data.dc_lines[data.dc_lines["EI"].isin([0,1,2,3])].index  # BHEI
D = data.dc_lines[~data.dc_lines["EI"].isin([0,1,2,3])].index # lines not to the EI's
Z = data.reservoir_zonal_limit.index




#Parameters

c = 0
penalty_curtailment = 100

eff_elec = 0.68
storage_efficiency = 0.8
price_LL = 3000
storage_penalty = 0.1

if run_parameter.reduced_TS:
    full_ts = data.timesteps_reduced_ts
else:
    full_ts = 8760
delta = 8760/full_ts


#here I do some dictionary reordering. I want to have all indices as a list given selected bus. If there is none, I want
# to have an empty list. I call this "encyclopedia"
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
#F_AC = model.addVars(Y, T, L, lb =-GRB.INFINITY, ub=GRB.INFINITY, name="F_AC")
F_DC = model.addVars(Y, T, LDC, lb =-GRB.INFINITY,ub=GRB.INFINITY, name = "F_DC")
p_load_lost = model.addVars(Y, T,zone, lb=0.0, ub = GRB.INFINITY, name = "p_load_lost")
#if run_parameter.scen != 1:
#    cap_E = model.addVars(Y, E, lb=0.0, ub = GRB.INFINITY, name = "cap_E")
#    P_H = model.addVars(Y, T, E, lb=0.0, ub=GRB.INFINITY, name="P_H")
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
        gp.quicksum(data.dispatchable_generators[y]["mc"][g] * P_C[y, t, g] for g in G for t in T )
        + gp.quicksum(P_S[y,t,s] * c for t in T for s in S) # hier c[s] einbringen?
        + gp.quicksum(res_curtailment[y, t, r] * penalty_curtailment for t in T for r in R)
        ) for y in Y for z in zone), GRB.MINIMIZE)

#Generation dispatchable

# if run_parameter.scen in [1]:
#     model.setObjective(
#         gp.quicksum((
#         delta * gp.quicksum(data.dispatchable_generators[y]["mc"][g] * P_C[y, t, g] for g in G for t in T )
#         + delta * price_LL * gp.quicksum(p_load_lost[y, t, n] for n in N for t in T )
#         + delta * storage_penalty * gp.quicksum(P_S[y, t, s] for s in S for t in T)                             #penalty for storage discharge
#         + (gp.quicksum(cap_BH[y, i]* dist_line[i] for i in I) * cost_line * annuity_line)*(run_parameter.timesteps/full_ts)
#         )/((1+r)**(5*y))
#          for y in Y), GRB.MINIMIZE)

if run_parameter.scen in [1]:
    model.setObjective(
        gp.quicksum((
        gp.quicksum(data.dispatchable_generators[y]["mc"][g] * P_C[y, t, g] for g in G for t in T )
        #+ price_LL * gp.quicksum(p_load_lost[y, t, n] for n in N for t in T )
        #+  storage_penalty * gp.quicksum(P_S[y, t, s] for s in S for t in T)
        + gp.quicksum(P_S[y,t,s] * c[s] for t in T for s in S)
        + gp.quicksum(res_curtailment[y, t, r] * penalty_curtailment[y,t,z] for t in Z for r in R)
        ) for y in Y for z in zone), GRB.MINIMIZE)


model.addConstrs((P_C[y, t, g] <= data.dispatchable_generators[y]["P_inst"][g] for g in G for t in T for y in Y), name="GenerationLimitUp")

#Renewable Generation
model.addConstrs((P_R[y, t, r] <= data.res_series[y][r][t] for r in R for t in T for y in Y), name="ResGenerationLimitUp")

#Curtailment
model.addConstrs((res_curtailment[y, t, r] == data.res_series[y][r][t] - P_R[y, t, r] for r in R for t in T for y in Y), name="RESCurtailment")
#Lost load
model.addConstrs((p_load_lost[y, t,j] <= demand_helper2(j,t, y,demand_dict) for j in zone for t in T for y in Y), name= "limitLoadLoss")

#storage DAM
model.addConstrs((P_DAM[y, t, g] <= data.reservoir["P_inst"][g] for g in DAM for t in T for y in Y), name="DAMLimitUp")
model.addConstrs((gp.quicksum(P_DAM[y, t, g] for g in encyc_dam_zones[z] for t in T) <= data.reservoir_zonal_limit[z] for z in zone for y in Y), name="DAMSumUp")

#Storage

model.addConstrs((res_curtailment[y, t, r] == data.res_series[y][r][t] - P_R[y, t, r] for r in R for t in T for y in Y), name="RESCurtailment")
model.addConstrs((p_load_lost[y, t,j] <= demand_helper2(j,t, y,demand_dict) for j in N for t in T for y in Y), name= "limitLoadLoss")

model.addConstrs((P_S[y, t, s] <= data.storage["Pmax_out"][s] for s in S for t in T for y in Y), name="StoragePowerOutput")
model.addConstrs((C_S[y, t, s] <= data.storage["Pmax_in"][s] for s in S for t in T for y in Y), name="StoragePowerInput")
model.addConstrs((P_S[y, t, s] <= L_S[y, t, s] for s in S for t in T for y in Y), name="StorageLevelGen")
model.addConstrs((L_S[y, t, s] <= data.storage["capacity"][s] for s in S for t in T for y in Y), name="StorageLevelCap")

#warum hier bei t[0] einspeisen und ausspeisen des speichers aber in den constraints in der MA steht das gleich 0?
model.addConstrs((L_S[y, t, s] == L_S[y, t-1, s] - P_S[y, t, s] + storage_efficiency * C_S[y, t, s]  for s in S for t in T_extra for y in Y), name="Storage_balance")
model.addConstrs((L_S[y, T[-1], s] >= 0.5 * data.storage["capacity"][s] for s in S for y in Y), name="Storage_end_level")
model.addConstrs((L_S[y, t, s] >= 0.5 * data.storage["capacity"][s] - P_S[y, t, s] + storage_efficiency * C_S[y, t, s]  for s in S for t in [0] for y in Y), name="Storage_balance_init")

print("The time difference before flow lines :", timeit.default_timer() - starttime)

#Energy Mass Balance
model.addConstrs((
        gp.quicksum(P_C[y, t, g] for g in encyc_powerplants_bus[z])
        + gp.quicksum(P_R[y, t, r] for r in ren_helper2(z, data.res_series_busses))
        + gp.quicksum(P_DAM[y, t, dam] for dam in encyc_dam[z])
        + gp.quicksum(ror_supply_dict[r][t] for r in ren_helper2(z, ror_supply_busses))
        + gp.quicksum(F_DC[y, t, d] for d in encyc_DC_from[z])
        - gp.quicksum(F_DC[y, t, d] for d in encyc_DC_to[z])
        + gp.quicksum(P_S[y, t, s] - C_S[y, t, s] for s in encyc_storage_bus[z])
         == demand_helper2(z,t,y, demand_dict) - p_load_lost[y, t, z] for z in zone for t in T for y in Y), name ="Injection_equality")
#BIS HIER HER
print("The time difference after flow lines :", timeit.default_timer() - starttime)


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
