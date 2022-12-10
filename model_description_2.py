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
run_parameter = run_parameter(scenario_name = "Offshore_Bidding_Zone_Scenario")
run_parameter.create_scenarios()
data = model_data(create_res = False,reduced_ts = True, export_files= True, run_parameter = run_parameter)

###################################
###Sum up values for each zone####
###################################

#Dispatchable
data.dispatchable['Total Capacity'] = data.dispatchable.groupby([run_parameter.scen, 'type'])['P_inst'].transform('sum')
final_dispatchable = data.dispatchable.drop_duplicates(subset=[run_parameter.scen, 'type'])
final_dispatchable= final_dispatchable.sort_values(run_parameter.scen)

#reservoir
data.reservoir['Total Capacity'] = data.reservoir.groupby([run_parameter.scen])['P_inst'].transform('sum')
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
Z = data.nodes[run_parameter.scen].unique()
G = final_dispatchable['type'].unique()
R = final_res_series.columns
DAM = final_reservoirs[[run_parameter.scen, 'Total Capacity']].set_index(run_parameter.scen)
S = final_storage.index
F = data.ntc_BZ5.index


#Parameters
storage_efficiency = 0.8
price_LL = 3000
penalty_curtailment = 100
storage_penalty = 0.1
marginal_costs = final_dispatchable.set_index('type').to_dict()['mc']
#eff_elec = 0.68
l_s_max = final_storage.to_dict()['capacity']
p_g_max = final_dispatchable[['type', 'Total Capacity', run_parameter.scen]] #.set_index(run_parameter.scen)

def dispatchable_help(zone,disp):
    result = p_g_max.loc[p_g_max[run_parameter.scen] == zone]
    result = result.set_index('type')
    del result[run_parameter.scen]
    if disp in result.index:
        x = result.at[disp, 'Total Capacity']
        return x
    else:
        x = 0
    return x

def reservoir_help(generationZone):
    if generationZone in final_reservoirs.index:
        x = final_reservoirs.at[generationZone, 'Total Capacity']
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



final_dispatchable['Min Capacity'] = final_dispatchable['Total Capacity'].mul(0.2)
p_g_min = final_dispatchable[['type', 'Min Capacity', run_parameter.scen]].set_index([run_parameter.scen])
p_r_max = R
p_s_max_in = final_storage.to_dict()['Pmax_in']
p_s_max_out = final_storage.to_dict()['Pmax_out']


if run_parameter.reduced_TS:
    full_ts = data.timesteps_reduced_ts
else:
    full_ts = 8760
delta = 8760/full_ts

#here I do some dictionary reordering.
encyc_NTC_from = create_encyclopedia(data.ntc_BZ5["fromBZ5"])
encyc_NTC_to = create_encyclopedia(data.ntc_BZ5["toBZ5"])


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
L_S = model.addVars(T, S, lb=0.0, ub = GRB.INFINITY, vtype=GRB.CONTINUOUS, name="L_S")  # storage level
P_CONV = model.addVars(T, G, Z, lb=0.0, ub = GRB.INFINITY, name="P_C") #conventional generation
p_load_lost = model.addVars(T, Z, lb=0.0, ub = GRB.INFINITY, name = "p_load_lost") #loss of load
P_R = model.addVars(T, R, lb=0.0, ub = GRB.INFINITY, name="P_R") #res generation in 10 timesteps
res_curtailment = model.addVars(T, R, lb=0.0, ub = GRB.INFINITY, name="res_curtailment")
P_DAM = model.addVars(T, DAM, lb=0.0, ub = GRB.INFINITY, name="P_DAM") #dam generation
S_inj = model.addVars(T, S, lb=0.0, ub = GRB.INFINITY, vtype=GRB.CONTINUOUS, name="S_inj")  # demand from storage (filling)
S_ext = model.addVars(T, S, lb=0.0, ub = GRB.INFINITY, vtype=GRB.CONTINUOUS, name="S_ext")  # power gen. by storage (depletion)
F_NTC = model.addVars(T, F, lb =-GRB.INFINITY, ub=GRB.INFINITY, name="F_NTC")

print("before Variables are made. The time difference is :", timeit.default_timer() - starttime)


#objective function
model.setObjective(
    gp.quicksum(
    gp.quicksum(P_CONV[t, g, z] * marginal_costs[g] for g in G for t in T)
    + price_LL * gp.quicksum(p_load_lost[t, z] for t in T)
    + storage_penalty * gp.quicksum(S_ext[t, s] for t in T for s in S)
    + penalty_curtailment * gp.quicksum(res_curtailment[t, z] for t in T) for z in Z), GRB.MINIMIZE)


#Mass Balance neu
model.addConstrs((
    gp.quicksum(P_CONV[t, g, z] for g in G)
    +gp.quicksum(P_R[t, r] for r in R)
    +gp.quicksum(P_DAM[t, dam] for dam in DAM)
    +gp.quicksum(S_ext[t, s] for s in S)
    +ror_help(t,z)
    +gp.quicksum(F_NTC[t, f] for f in encyc_NTC_from[z] for t in T)
    -gp.quicksum(F_NTC[t, f] for f in encyc_NTC_to[z] for t in T)
    +p_load_lost[t, z]
     == demand_help(t,z) + gp.quicksum(S_inj[t, s] for s in S)for z in Z for t in T), name ="Injection_equality")

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
model.addConstrs((F_NTC[t, f] <= data.ntc_BZ5["Sum of max"][f] for f in F for t in T), name="NTC_max_cap_limit")


#Limit CONV Generation
model.addConstrs((P_CONV[t, g, z] <= dispatchable_help(z,g) for t in T for g in G for z in Z ), name="GenerationLimitUp")

#Curtailment
model.addConstrs((res_curtailment[t, r] == final_res_series[r][t] - P_R[t, r] for r in R for t in T), name="RESCurtailment")

#Storage Limits (was ist mit p_s_max_in /out)
model.addConstrs((S_ext[t, s] <= final_storage["Pmax_out"][s] for s in S for t in T), name="StoragePowerOutput")
model.addConstrs((S_inj[t, s] <= final_storage["Pmax_in"][s] for s in S for t in T), name="StoragePowerInput")
model.addConstrs((S_ext[t, s] <= L_S[t, s] for s in S for t in T), name="StorageLevelGen")
model.addConstrs((L_S[t, s] <= final_storage["capacity"][s] for s in S for t in T), name="StorageLevelCap")
model.addConstrs((L_S[t, s] == L_S[t-1, s] - S_ext[t, s] + storage_efficiency * S_inj[t, s]  for s in S for t in T_extra), name="Storage_balance")
model.addConstrs((L_S[t, s] == 0.5 * final_storage.at[s, "capacity"] - S_ext[t, s] + storage_efficiency * S_inj[t, s] for s in S for t in [0]), name="Storage_balance_init")
model.addConstrs((L_S[T[-1], s] == 0.5 * final_storage.at[s, "capacity"] for s in S), name="Storage_end_level")



#Limit DAM
model.addConstrs((P_DAM[t, g] <= reservoir_help(g) for g in DAM for t in T), name="DAMLimitUp")
#TODO ich glaube die hier brauchen wir nicht, sagt das selbe aus wie die gleichung darüber?: model.addConstrs((gp.quicksum(P_DAM[y, t, g] for g in encyc_dam_zones[z] for t in T) <= data.reservoir_zonal_limit[z] for z in Z for y in Y), name="DAMSumUp")

#Limit RES
model.addConstrs((P_R[t,r] <= final_res_series[r][t] for r in R for t in T), name="ResGenerationLimitUp")

#Limit Load loss
model.addConstrs((p_load_lost[t,z] <= demand_help(t,z) for z in Z for t in T), name= "limitLoadLoss")

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
#data.ac_lines.to_csv(run_parameter.export_folder + "lines.csv")
data.ntc_BZ5.to_csv(run_parameter.export_folder + "lines_NTC.csv")
#todo ist das hier richtig?:
with open(run_parameter.export_folder+'share_renewables.pkl', 'wb+') as f:
    pickle.dump(data.res_series, f)

#with open(run_parameter.export_folder+'share_wind.pkl', 'wb+') as f:
#    pickle.dump(data.share_wind, f)
#with open(run_parameter.export_folder+'share_solar.pkl', 'wb+') as f:
#    pickle.dump(data.share_solar, f)
data.ror_series.to_csv(run_parameter.export_folder + "ror_supply.csv")
print("The time difference is :", timeit.default_timer() - starttime)
print("done")
#model.optimize()
#print(model.ObjVal)