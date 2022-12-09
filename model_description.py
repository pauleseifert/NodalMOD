import pickle
import timeit
import geopandas as gpd
import gurobipy as gp
import pandas as pd
from gurobipy import GRB
from gurobipy import Model

from helper_functions import ren_helper2, demand_helper2, create_encyclopedia
from import_object_data_Zonal_Configuration import model_data, run_parameter

starttime = timeit.default_timer()

#load model parameters

run_parameter = run_parameter(scenario_name = "Offshore_Bidding_Zone_Scenario")

run_parameter.create_scenarios()
data = model_data(create_res = False,reduced_ts = True, export_files= True, run_parameter = run_parameter)
#self.demand.to_csv("demand.csv")


#data.demand[0].to_csv(r'C:\Users\marie\Documents\NTNU\Specialization Project\DataFrames Model\demand.csv', index=False)

#TODO: welche Daten fehlen? im Overleaf den Data Teil schreiben (structure)



#zones festlegen, als set und zuordnung zu den nodes
shapes = gpd.read_file('data/shapes/NUTS_RG_10M_2021_4326.geojson')
shapes_filtered = shapes.query("LEVL_CODE ==1")
#shapes_filtered = shapes.query("LEVL_CODE ==1 and CNTR_CODE == 'DE'")
#shape from the nodes longitude and latitude a geometry to sjoin them
df_buses = pd.read_csv("data/PyPSA_elec1024/buses.csv", index_col=0)
#df_buses['geometry'] = [Point(xy) for xy in zip(df_buses.x, df_buses.y)]
gdf_buses = gpd.GeoDataFrame(df_buses, geometry=gpd.points_from_xy(df_buses.x, df_buses.y), crs="EPSG:4326")


#TODO aufsummieren pro Zonen
#Dispatchable summe
data.dispatchable['Total Capacity'] = data.dispatchable.groupby([run_parameter.scen, 'type'])['P_inst'].transform('sum')
final_dispatchable = data.dispatchable.drop_duplicates(subset=[run_parameter.scen, 'type'])
final_dispatchable= final_dispatchable.sort_values(run_parameter.scen)


# reservoir
data.reservoir['Total Capacity'] = data.reservoir.groupby([run_parameter.scen])['P_inst'].transform('sum')
final_reservoirs = data.reservoir.drop_duplicates(subset=[run_parameter.scen])
final_reservoirs = final_reservoirs.sort_values(run_parameter.scen)

#Demand summe
final_demand = data.demand.transpose()
final_demand['index'] = final_demand.index
final_demand = final_demand.merge(data.nodes[['index',run_parameter.scen]], on="index",how='left')
del final_demand['index']
final_demand = final_demand.groupby([run_parameter.scen]).sum()
final_demand = final_demand.transpose()

#Res series
final_res_series = data.res_series.transpose()
final_res_series['index'] = final_res_series.index
final_res_series = final_res_series.merge(data.nodes[['index',run_parameter.scen]], on="index",how='left')
del final_res_series['index']
final_res_series = final_res_series.groupby([run_parameter.scen]).sum()
final_res_series = final_res_series.transpose()

#Ror series
final_ror_series = data.ror_series.transpose()
final_ror_series['index'] = final_ror_series.index
final_ror_series = final_ror_series.merge(data.nodes[['index',run_parameter.scen]], on="index",how='left')
del final_ror_series['index']
final_ror_series = final_ror_series.groupby([run_parameter.scen]).sum()
final_ror_series = final_ror_series.transpose()


#Storage
final_storage = data.storage.groupby([run_parameter.scen])[["Pmax_out", "Pmax_in", "capacity"]].sum()


# Create a new model
model: Model = gp.Model("Offshore_Bidding_Zones")

#Sets
T = range(run_parameter.timesteps)  # hours
T_extra = range(1, run_parameter.timesteps)

Y = range(run_parameter.years)
Y_extra = range(1, run_parameter.years)
#G = data.dispatchable_generators[0].index
G = final_dispatchable.type
#R = data.res_series[0].columns
R = final_res_series.columns
#DAM = data.reservoir.index
DAM = final_reservoirs[run_parameter.scen]
#S = data.storage.index
S = final_storage.index
#L = data.ac_lines.index
#N = data.nodes.index
Zone = run_parameter.lookup_dict

#LDC = data.dc_lines.index
LDC = data.ntc_BZ5.index
#C = range(len(L)-len(N)+1) #C_cl_df.index
# separating the flexlines
#I = data.dc_lines[data.dc_lines["EI"].isin(["BHEH", "NSEH1", "NSEH2", "CLUSTER"])].index  # BHEI
#D = data.dc_lines[~data.dc_lines["EI"].isin(["BHEH", "NSEH1", "NSEH2", "CLUSTER"])].index # lines not to the EI's

#Z = data.reservoir_zonal_limit.index



#Parameters
storage_efficiency = 0.8
price_LL = 3000
penalty_curtailment = 100

c = 0

#r = 0.04    #zinssatz
#T_line = 40     #Lifetime line
#T_elec = 30     #Lifetime electrolyser
#factor_opex = 0.02       #share of capex for opex each year
#cost_line = 1950         #/MW/km
#dist_line = distance_line(nodes=data.nodes, dc_line_overview=data.dc_lines, index=I)

eff_elec = 0.68
storage_penalty = 0.1

if run_parameter.reduced_TS:
    full_ts = data.timesteps_reduced_ts
else:
    full_ts = 8760
delta = 8760/full_ts


#here I do some dictionary reordering.
#encyc_powerplants_bus = create_encyclopedia(data.dispatchable_generators[0]["bus"])
#encyc_storage_bus = create_encyclopedia(data.storage["bus"])
#encyc_DC_from = create_encyclopedia(data.dc_lines["from"])
#encyc_DC_to = create_encyclopedia(data.dc_lines["to"])
#encyc_dam = create_encyclopedia(data.reservoir["bus"])
#encyc_dam_zones = create_encyclopedia(data.reservoir["bidding_zone"])
#if run_parameter.scen != 1: encyc_elec = create_encyclopedia(run_parameter.electrolyser["bus"])

#Todo was hat er hier gemacht:
#data.res_series_busses = dict()
#for k in data.res_series[0].columns.to_list():
#   data.res_series_busses[k] = False

ror_supply_busses = dict()
for k in final_ror_series.columns.to_list():
    ror_supply_busses[k] = False
ror_supply_dict = final_ror_series.to_dict()
#demand_col_list = data.demand.columns.to_list()
demand_dict = {}
for i in range(run_parameter.years):
    data.demand[i].reset_index(drop=True, inplace=True)
    demand_dict.update({i: data.demand[i].to_dict()})

print("Preparations done. The time difference is :", timeit.default_timer() - starttime)

# Variables
#NTC = model.addVars(T, G, lb=0.0, ub = GRB.INFINITY, name="NTC")
P_C = model.addVars(T, G, Zone, lb=0.0, ub = GRB.INFINITY, name="P_C")
P_R = model.addVars(T, R, lb=0.0, ub = GRB.INFINITY, name="P_R")
P_DAM = model.addVars(T, DAM, lb=0.0, ub = GRB.INFINITY, name="P_DAM")
res_curtailment = model.addVars(T, Zone, lb=0.0, ub = GRB.INFINITY, name="res_curtailment")
#cap_BH = model.addVars(Y, I, lb=0.0, ub = GRB.INFINITY, name = "cap_BH")
#F_AC = model.addVars(T, L, lb =-GRB.INFINITY, ub=GRB.INFINITY, name="F_AC")
F_DC = model.addVars(T, LDC, lb =-GRB.INFINITY,ub=GRB.INFINITY, name = "F_DC")
p_load_lost = model.addVars(T, Zone, lb=0.0, ub = GRB.INFINITY, name = "p_load_lost")

#if run_parameter.scen != 1:
#    cap_E = model.addVars(Y, E, lb=0.0, ub = GRB.INFINITY, name = "cap_E")
#    P_H = model.addVars(Y, T, E, lb=0.0, ub=GRB.INFINITY, name="P_H")
# storage variables

print("before Variables are made. The time difference is :", timeit.default_timer() - starttime)

P_S = model.addVars(T, S, lb=0.0, ub = GRB.INFINITY, vtype=GRB.CONTINUOUS, name="P_S")  # power gen. by storage (depletion)
C_S = model.addVars(T, S, lb=0.0, ub = GRB.INFINITY, vtype=GRB.CONTINUOUS, name="C_S")  # demand from storage (filling)
L_S = model.addVars(T, S, lb=0.0, ub = GRB.INFINITY, vtype=GRB.CONTINUOUS, name="L_S")  # storage level

print("Variables made. The time difference is :", timeit.default_timer() - starttime)


#ToDo model ansatz fehlt:
model.setObjective(
    gp.quicksum(P_C[t, g] * final_dispatchable["mc"][g] for g in G for t in T)
    + gp.quicksum(P_S[t, s] * storage_penalty for s in S for t in T)
    + gp.quicksum(res_curtailment[t, z] * penalty_curtailment for t in T for z in Zone), GRB.MINIMIZE)

#Paul:
#model.setObjective(
#    gp.quicksum((
#    gp.quicksum(final_dispatchable["mc"][g] * P_C[t, g] for g in G for t in T)
#    +  storage_penalty * gp.quicksum(P_S[ t, s] for s in S for t in T)
#    + gp.quicksum(P_S[t, s] * c for t in T for s in S)
#    + gp.quicksum(res_curtailment[t, r] * penalty_curtailment[t, z] for t in T for r in R)) for y in Y for z in Zone), GRB.MINIMIZE)

#MassBalance
#gp.quicksum(P_C[t, g] for g in

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


#model.addConstrs((P_C[y, t, g] <= data.dispatchable_generators[y]["P_inst"][g] for g in G for t in T for y in Y), name="GenerationLimitUp")
#TODO GEHT NICHT ???:
#model.addConstrs((P_C[t, g, z] <= final_dispatchable["P_inst"][g,z] for g in G for t in T for z in Zone), name="GenerationLimitUp")

#helper funktion die zurück gibt in zone x alle renewables
#Renewable Generation
#model.addConstrs((P_R[y, t, r] <= data.res_series[y][r][t] for r in R for t in T for y in Y), name="ResGenerationLimitUp")
model.addConstrs((P_R[t, r] <= final_res_series[r][t] for r in R for t in T), name="ResGenerationLimitUp")
#Curtailment
model.addConstrs((res_curtailment[t, r] == final_res_series[r][t] - P_R[t, r] for r in R for t in T), name="RESCurtailment")

#Lost load TODO ? :
#model.addConstrs((p_load_lost[t,z] <= demand_helper2(z,t,demand_dict) for z in Zone for t in T), name= "limitLoadLoss")

#storage DAM
model.addConstrs((P_DAM[t, g] <= final_reservoirs["P_inst"][g] for g in DAM for t in T ), name="DAMLimitUp")
#Todo? :
#model.addConstrs((gp.quicksum(P_DAM[ t, g] for g in encyc_dam_zones[z] for t in T) <= data.reservoir_zonal_limit[z] for z in zone for y in Y), name="DAMSumUp")

#Storage

model.addConstrs((res_curtailment[ t, r] == final_res_series[r][t] - P_R[ t, r] for r in R for t in T), name="RESCurtailment")
#Todo? :
#model.addConstrs((p_load_lost[y, t,j] <= demand_helper2(j,t, y,demand_dict) for j in N for t in T for y in Y), name= "limitLoadLoss")

model.addConstrs((P_S[t, s] <= final_storage["Pmax_out"][s] for s in S for t in T), name="StoragePowerOutput")
model.addConstrs((C_S[t, s] <= final_storage["Pmax_in"][s] for s in S for t in T), name="StoragePowerInput")
model.addConstrs((P_S[t, s] <= L_S[t, s] for s in S for t in T), name="StorageLevelGen")
model.addConstrs((L_S[t, s] <= final_storage["capacity"][s] for s in S for t in T), name="StorageLevelCap")
model.addConstrs((L_S[t, s] == L_S[t-1, s] - P_S[t, s] + storage_efficiency * C_S[t, s]  for s in S for t in T_extra), name="Storage_balance")
#Todo: model.addConstrs((L_S[T[-1], s] >= 0.5 * data.storage["capacity"][s] for s in S), name="Storage_end_level")
#Todo: model.addConstrs((L_S[t, s] >= 0.5 * data.storage["capacity"][s] - P_S[t, s] + storage_efficiency * C_S[t, s]  for s in S for t in [0]), name="Storage_balance_init")

print("The time difference before flow lines :", timeit.default_timer() - starttime)

#Energy Mass Balance
model.addConstrs((
        gp.quicksum(P_C[t, g] for g in encyc_powerplants_bus[z])
        + gp.quicksum(P_R[t, r] for r in ren_helper2(z, data.res_series_busses))
        + gp.quicksum(P_DAM[t, dam] for dam in encyc_dam[z])
        + gp.quicksum(ror_supply_dict[r][t] for r in ren_helper2(z, ror_supply_busses))
        + gp.quicksum(F_DC[t, d] for d in encyc_DC_from[z])
        - gp.quicksum(F_DC[t, d] for d in encyc_DC_to[z])
        + gp.quicksum(P_S[t, s] - C_S[y, t, s] for s in encyc_storage_bus[z])
         == demand_helper2(z,t,y, demand_dict) - p_load_lost[y, t, z] for z in Zone for t in T), name ="Injection_equality")




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
