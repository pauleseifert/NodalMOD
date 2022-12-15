import pickle
import timeit
import geopandas as gpd
import gurobipy as gp
import pandas as pd
from gurobipy import GRB
from gurobipy import Model
from collections import ChainMap

from helper_functions import ren_helper2, demand_helper2, create_encyclopedia
from import_object_data_Zonal_Configuration import model_data, run_parameter

starttime = timeit.default_timer()
run_parameter = run_parameter(scenario_name = "Offshore_Bidding_Zone_Scenario")
run_parameter.create_scenarios()
data = model_data(create_res = False,reduced_ts = True, export_files= True, run_parameter = run_parameter)


#Dispatchable
data.dispatchable['Total_Capacity'] = data.dispatchable.groupby([run_parameter.scen, 'type'])['P_inst'].transform('sum')
final_dispatchable = data.dispatchable.drop_duplicates(subset=[run_parameter.scen, 'type'])
final_dispatchable= final_dispatchable.sort_values(run_parameter.scen)

#reservoir
data.reservoir['Total_Capacity'] = data.reservoir.groupby([run_parameter.scen])['P_inst'].transform('sum')
final_reservoirs = data.reservoir.drop_duplicates(subset=[run_parameter.scen])
final_reservoirs = final_reservoirs.sort_values(run_parameter.scen)

#Demand
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
model = gp.Model("Offshore_Bidding_Zones")

#Sets
T = range(run_parameter.timesteps)  # hours
T_extra = range(1, run_parameter.timesteps)
Y = range(run_parameter.years)
Y_extra = range(1, run_parameter.years)
Z = list(range(len(data.nodes[run_parameter.scen].unique())))
G = list(range(len(final_dispatchable['type'].unique())))
#x = data.nodes[run_parameter.scen].unique()
#für die weiterverarbeitung müssen allen zonen und conventionals zahlen zugeordnet werden:
Z_dict = {}
keys = range(len(data.nodes[run_parameter.scen].unique()))
values = data.nodes[run_parameter.scen].unique()
for i in keys:
        Z_dict[i] = values[i]


G_dict = {}
keys = range(len(final_dispatchable['type'].unique()))
values = final_dispatchable['type'].unique()
for i in keys:
        G_dict[i] = values[i]


R = final_res_series.columns
DAM = final_reservoirs[[run_parameter.scen, 'Total_Capacity']].set_index(run_parameter.scen)
S = final_storage.index
F = data.ntc.index


#Parameters
storage_efficiency = 0.8
price_LL = 3000
penalty_curtailment = 10
storage_penalty = 0.1
marginal_costs = final_dispatchable.set_index('type').to_dict()['mc']
#eff_elec = 0.68
l_s_max = final_storage.to_dict()['capacity']
p_g_max = final_dispatchable[['type', 'Total_Capacity', run_parameter.scen]] #.set_index(run_parameter.scen)

def dispatchable_help(zone,disp):
    result = p_g_max.loc[p_g_max[run_parameter.scen] == zone]
    result = result.set_index('type')
    del result[run_parameter.scen]
    if disp in result.index:
        x = result.at[disp, 'Total_Capacity']
        return x
    else:
        x = 0
    return x

def reservoir_help(generationZone):
    if generationZone in final_reservoirs.index:
        x = final_reservoirs.at[generationZone, 'Total_Capacity']
        return x
    else:
        x = 0
    return x

def demand_help(t,z):
    if z in final_demand.columns:
        x = final_demand.at[t,z]
        return x
    else:
        x = 0
    return x


def ror_help(t,z):
    if z in final_ror_series.columns:
        x = final_ror_series.at[t,z]
        return x
    else:
        x = 0
    return x
def storage_help_in(z):
    if z in final_storage.index:
        x = final_storage.at[z,'Pmax_in']
        return x
    else:
        x = 0
    return x

def storage_help_out(z):
    if z in final_storage.index:
        x = final_storage.at[z,'capacity']
        return x
    else:
        x = 0
    return x

def storage_capacity(z):
    if z in final_storage.index:
        x = final_storage.at[z,'capacity']
        return x
    else:
        x = 0
    return x

final_dispatchable['Min_Capacity'] = final_dispatchable['Total_Capacity'].mul(0.2)
p_g_min = final_dispatchable[['type', 'Min_Capacity', run_parameter.scen]].set_index([run_parameter.scen])
p_r_max = R
p_s_max_in = final_storage.to_dict()['Pmax_in']
p_s_max_out = final_storage.to_dict()['Pmax_out']


if run_parameter.reduced_TS:
    full_ts = data.timesteps_reduced_ts
else:
    full_ts = 8760
delta = 8760/full_ts

#here I do some dictionary reordering.
encyc_NTC_from = create_encyclopedia(data.ntc["from" + run_parameter.scen])
encyc_NTC_to = create_encyclopedia(data.ntc["to" + run_parameter.scen])

x = data.ntc

res_series_zones = dict()
for k in final_res_series.columns.to_list():
    res_series_zones[k] = False

ror_supply_zones = dict()
for k in final_ror_series.columns.to_list():
    ror_supply_zones[k] = False

ror_supply_dict = final_ror_series.to_dict()
res_series_dict = final_res_series.to_dict()
demand_dict = final_demand.to_dict()

print("Preparations done. The time difference is :", timeit.default_timer() - starttime)

# Variables
L_S = model.addVars(T, Z, lb=0.0, ub = GRB.INFINITY, vtype=GRB.CONTINUOUS, name="L_S")  # storage level
P_CONV = model.addVars(T, G, Z, lb=0.0, ub = GRB.INFINITY, name="P_C") #conventional generation
p_load_lost = model.addVars(T, Z, lb=0.0, ub = GRB.INFINITY, name = "p_load_lost") #loss of load
P_R = model.addVars(T, Z, lb=0.0, ub = GRB.INFINITY, name="P_R") #res generation in 10 timesteps
res_curtailment = model.addVars(T, Z, lb=0.0, ub = GRB.INFINITY, name="res_curtailment")
P_DAM = model.addVars(T, Z, lb=0.0, ub = GRB.INFINITY, name="P_DAM") #dam generation
S_inj = model.addVars(T, Z, lb=0.0, ub = GRB.INFINITY, vtype=GRB.CONTINUOUS, name="S_inj")  # demand from storage (filling)
S_ext = model.addVars(T, Z, lb=0.0, ub = GRB.INFINITY, vtype=GRB.CONTINUOUS, name="S_ext")  # power gen. by storage (depletion)
F_NTC = model.addVars(T, F, lb =-GRB.INFINITY, ub=GRB.INFINITY, name="F_NTC")

print("before Variables are made. The time difference is :", timeit.default_timer() - starttime)


#objective function
model.setObjective(
    gp.quicksum(
    gp.quicksum(P_CONV[t, g, z] * marginal_costs[G_dict[g]] for g in G for t in T)
    + price_LL * gp.quicksum(p_load_lost[t, z] for t in T)
    + storage_penalty * gp.quicksum(S_ext[t, z] for t in T)
    + penalty_curtailment * gp.quicksum(res_curtailment[t, z] for t in T) for z in Z), GRB.MINIMIZE)


#Mass Balance neu
model.addConstrs((
    gp.quicksum(P_CONV[t, g, z] for g in G)
    +P_R[t, z]
    +P_DAM[t, z]
    +S_ext[t, z]
    +ror_help(t,z)
    +gp.quicksum( F_NTC[t, f] for f in encyc_NTC_from[Z_dict[z]]) #for t in T)
    -gp.quicksum( F_NTC[t, f] for f in encyc_NTC_to[Z_dict[z]]) #for t in T)
     #+F_NTC[t, f] for f in encyc_NTC_from[Z_dict[z]]
     #-F_NTC[t, f] for f in encyc_NTC_to[Z_dict[z]]
    +p_load_lost[t, z]
     == demand_help(t,Z_dict[z]) + S_inj[t, z] for z in Z for t in T), name ="Injection_equality")

#MassBalance alt
#model.addConstrs((
#    gp.quicksum(P_CONV[t, g, z] for g in G)
#    +gp.quicksum(P_R[t, r] for r in R)
#    +gp.quicksum(P_DAM[t, dam] for dam in DAM)
#    +gp.quicksum(S_ext[t, s] for s in S)
#    +ror_help(t,z)
#    +gp.quicksum(F_NTC[t, f] for f in encyc_NTC_from[z] for t in T)
#    -gp.quicksum(F_NTC[t, f] for f in encyc_NTC_to[z] for t in T)
#    +p_load_lost[t, z]
#     == demand_help(t,z) + gp.quicksum(S_inj[t, s] for s in S)for z in Z for t in T), name ="Injection_equality")

#NTC flow
model.addConstrs((F_NTC[t, f] <= data.ntc["ac_dc_sum"][f] for f in F for t in T), name="NTC_max_cap_limit_in")
model.addConstrs((F_NTC[t, f] >= -data.ntc["ac_dc_sum"][f] for f in F for t in T), name="NTC_max_cap_limit_out")

#Limit CONV Generation
model.addConstrs((P_CONV[t, g, z] <= dispatchable_help(Z_dict[z],G_dict[g]) for t in T for g in G for z in Z ), name="GenerationLimitUp")

#Curtailment
model.addConstrs((res_curtailment[t, z] == final_res_series[Z_dict[z]][t] - P_R[t, z] for z in Z for t in T), name="RESCurtailment")

#Storage Limits (was ist mit p_s_max_in /out)
model.addConstrs((S_ext[t, z] <= storage_help_out(Z_dict[z]) for z in Z for t in T), name="StoragePowerOutput")
model.addConstrs((S_inj[t, z] <= storage_help_in(Z_dict[z]) for z in Z for t in T), name="StoragePowerInput")
model.addConstrs((S_ext[t, z] <= L_S[t, z] for z in Z for t in T), name="StorageLevelGen")
model.addConstrs((L_S[t, z] == L_S[t-1, z] - S_ext[t, z] + storage_efficiency * S_inj[t, z] for z in Z for t in T_extra), name="Storage_balance")
model.addConstrs((L_S[t, z] == 0.5 * storage_capacity(Z_dict[z]) - S_ext[t, z] + storage_efficiency * S_inj[t, z] for z in Z for t in [0]), name="Storage_balance_init")
model.addConstrs((L_S[T[-1], z] == 0.5 * storage_capacity(Z_dict[z]) for z in Z), name="Storage_end_level")

#Limit DAM
model.addConstrs((P_DAM[t, z] <= reservoir_help(Z_dict[z]) for z in Z for t in T), name="DAMLimitUp")
#Todo ich glaube die hier brauchen wir nicht, sagt das selbe aus wie die gleichung darüber?: model.addConstrs((gp.quicksum(P_DAM[y, t, g] for g in encyc_dam_zones[z] for t in T) <= data.reservoir_zonal_limit[z] for z in Z for y in Y), name="DAMSumUp")

#Limit RES
model.addConstrs((P_R[t,z] <= final_res_series[Z_dict[z]][t] for z in Z for t in T), name="ResGenerationLimitUp")

#Limit Load loss
model.addConstrs((p_load_lost[t,z] <= demand_help(t,Z_dict[z]) for z in Z for t in T), name= "limitLoadLoss")

print("The time difference after flow lines :", timeit.default_timer() - starttime)


try:
    model.write(run_parameter.export_model_formulation)
    print("The time difference after model writing:", timeit.default_timer() - starttime)

except:
    print("error while writing model data")
    pass
# necessary files: P_R_max, busses, data.dispatchable_generators, storage, lines, linesDC and ror
data.dispatchable.to_csv(run_parameter.export_folder + "zones.csv")
data.storage.to_csv(run_parameter.export_folder + "storage.csv")
final_res_series.to_csv(run_parameter.export_folder + "renewables.csv")
final_demand.to_csv(run_parameter.export_folder + "demand.csv")
#data.ac_lines.to_csv(run_parameter.export_folder + "lines.csv")
data.ntc.to_csv(run_parameter.export_folder + "lines_NTC.csv")
#todo ist das hier richtig? wofür:
with open(run_parameter.export_folder+'share_renewables.pkl', 'wb+') as f:pickle.dump(data.res_series, f)
#with open(run_parameter.export_folder+'share_wind.pkl', 'wb+') as f:pickle.dump(data.share_wind, f)
#with open(run_parameter.export_folder+'share_solar.pkl', 'wb+') as f:pickle.dump(data.share_solar, f)
final_ror_series.to_csv(run_parameter.export_folder + "ror_supply.csv")
final_ror_series.to_csv(run_parameter.export_folder + "ror_supply.csv")
final_ror_series.to_csv(run_parameter.export_folder + "ror_supply.csv")
print("The time difference is :", timeit.default_timer() - starttime)
print("done")
#model.optimize()
#print(model.ObjVal)