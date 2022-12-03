import pickle
import timeit
import geopandas as gpd
import gurobipy as gp
import pandas as pd
from gurobipy import GRB
from gurobipy import Model

from helper_functions import ren_helper2, demand_helper2, create_encyclopedia
from import_data_object import model_data, run_parameter

starttime = timeit.default_timer()

#load model parameters
run_parameter = run_parameter(scenario_name = "Offshore_Bidding_Zone_Scenario")
run_parameter.create_scenarios()
data = model_data(create_res = False,reduced_ts = True, export_files= True, run_parameter = run_parameter)
#self.demand.to_csv("demand.csv")


#data.demand[0].to_csv(r'C:\Users\marie\Documents\NTNU\Specialization Project\DataFrames Model\demand.csv', index=False)

#TODO: welche Daten fehlen? im Overleaf den Data Teil schreiben (structure)
#TODO: diskutieren ob es wirklich nötig ist die Nordics mit in das Model mit aufzunhemen (Paul meinte wegen Wasser Zeugs)
#TODO: Recherche Flexilines - was für Caps für die NTCs nehmen wir?



#zones festlegen, als set und zuordnung zu den nodes
shapes = gpd.read_file('data/shapes/NUTS_RG_10M_2021_4326.geojson')
shapes_filtered = shapes.query("LEVL_CODE ==1")
#shapes_filtered = shapes.query("LEVL_CODE ==1 and CNTR_CODE == 'DE'")
#shape from the nodes longitude and latitude a geometry to sjoin them
df_buses = pd.read_csv("data/PyPSA_elec1024/buses.csv", index_col=0)
#df_buses['geometry'] = [Point(xy) for xy in zip(df_buses.x, df_buses.y)]
gdf_buses = gpd.GeoDataFrame(df_buses, geometry=gpd.points_from_xy(df_buses.x, df_buses.y), crs="EPSG:4326")


#case 1=1 Zone, 2=2 zones, 3= 3 zones, 4 = 5 zones
bidding_zone_configuration = 1

#hier die funktionen rein, welche die nodes entsprechend den zonen zuordnen
match bidding_zone_configuration:
            case 1:
                df_capacity = pd.DataFrame(
                    {
                        "Bidding_zone": ["DE1", "DE2", "DE2","DE3", "DE1", "DE3", "FR", "DE1", "DE1"],
                        "Plant": ["nuklear", "gas", "gas", "coal", "coal", "gas", "gas", "nuklear", "nuklear"],
                        "Capacity_MW": [10, 15, 2.5, 20, 25, 25, 15, 10,7.5]
                    }
                )
                df_demand = pd.DataFrame(
                    {
                    "Bidding_zone": ["DE1", "DE2", "DE2","DE3", "DE1", "DE3", "FR", "DE1", "DE1"],
                            "Demand_MW": [10, 15, 2.5, 20, 25, 25, 15, 10, 7.5]
                    }
                )

                df_exchange = pd.DataFrame(
                    {
                        "From": ["DE1", "DE2", "DE2", "DE3", "DE1", "DE3", "FR", "DE1", "DE2"],
                        "to": ["DE2", "DE1", "DE3", "FR", "DE3", "DE1", "DE2", "DE2", "DE1"],
                        "AC_MW": [10, 10, 2.5, 20, 25, 25, 15, 10, 7.5],
                        "DC_MW": [15, 15, 2.5, 20, 25, 25, 15, 10, 7.5]
                    }
                )
                print("Germany as one bidding zone")
            case 2:
                df_final=[]#funktion
                print("Germany as two bidding zones")
            case 3:
                df_final=[]#funktion
                print("Germany as three bidding zones")
            case 4:
                df_final=[]#funktion
                print("Germany as five bidding zones")


#summiert die Kapazitäten für jede PowerPlant (Fuel) pro Zone auf

# groupby nimmt zwei colums und applied eine function -> transform(sum)
# transform(sum) -> summiert argument ['Capacity_MW']
# df['Total'] -> legt neue Spalte an "Total" und schreibt den Ausdruck df.groupby ein
df_capacity['Total Capacity'] = df_capacity.groupby(['Bidding_zone', 'Plant'])['Capacity_MW'].transform('sum')
print(df_capacity)
df_capacity2 = df_capacity.drop_duplicates(subset=['Bidding_zone', 'Plant'])
print("df_capacity2: \n", df_capacity2)
df_capacity3 = df_capacity2.sort_values("Bidding_zone")
print("df_capacity3: \n", df_capacity3)
del df_capacity3["Capacity_MW"]
print(df_capacity3)


#Berechnung der gesamten Austauschkapazität zwischen den Zonen
#df_exchange['Total Exchange Capacity'] = df_exchange.groupby(['From','to'])['DC_MW'].transform('sum')
#print(df_exchange)
#df_exchange2 = df_exchange.drop_duplicates(subset=['From'])
#print("df_exchange2: \n", df_exchange2)
#df_exchange3 = df_exchange.sort_values("From")
#print("df_exchange3: \n", df_exchange3)
#del df_exchange3["AC_MW"]
#del df_exchange3["DC_MW"]
#print(df_exchange3)



#Spatial Join
#sjoined_nodes_states = gdf_buses.sjoin(shapes_filtered[["NUTS_NAME","NUTS_ID","geometry"]], how ="left")


# coordinate systems are correct?
#df_buses_selected.crs == shapes_filtered.crs
#Spatial Join (sjoin from geopandas)
#sjoined_nodes_states = gpd.sjoin(df_buses["geometry"],shapes_filtered, op="within")
sjoined_nodes_states = gdf_buses.sjoin(shapes_filtered[["NUTS_NAME","NUTS_ID","geometry"]], how ="left")

#Filtern der Columns die wir brauchen für Zones DE
#df_zones_DE = sjoined_nodes_states.query("country == 'DE'")
#df_zones_DE_filtered = df_zones_DE.filter(['NUTS_NAME', 'country', 'NUTS_ID', 'geometry'])

#ToDo
#Szenario 1: Dt= 1 Zone
#Szenario 2: Dt= 2 Zonen: DE1: NI+BR+HH+SH+MV+B+BB+SA+S+TH DE2: NRW+HE+BY+RP+SL+BW
#Szenario 3: Dt= 3 Zonen: DE1: NI+BR+HH+SH, DE2:MV+B+BB+SA+S+TH, DE3: NRW+HE+BY+RP+SL+BW
#Szenario 4: Dt= 5 Zonen: DE1: SH, DE2: NI+HH+BR, DE3: MV+B+BB+SA+S+TH, DE4: NRW+ RP+SL, DE5: BY+HE+BW

#Szenario 1: Bidding Zone DE1 = DE1-DE9, DEA-DEG

#Szenario 2: Dt= 2 Zonen:
#Bidding Zone DE1 = DE9+DE5+DE6+DEF+DE8+DE3+DE4+DEE+DED+DEG
#Bidding Zone DE2 = DEA+DE7+DE2+DEB+DEC+DE1

#How many nodes are in each state bzw zone "state_Bayern" = "NUTS_ID":"DE2"?
#df_zones_DE_filtered.groupby("NUTS_NAME == 'Bayern'").count()

#df.columns = ["NUTS_ID", ‘listings_count’]

#Szenario 3: Dt= 3 Zonen:
#Bidding Zone DE1: DE9+DE5+DE6+DEF,
#Bidding Zone DE2: DE8+DE3+DE4+DEE+DED+DEG,
#Bidding Zone DE3: DEA+DE7+DE2+DEB+DEC+DE1


#Szenario 4: Dt= 5 Zonen:
#Bidding Zone DE1: DEF,
#Bidding Zone DE2: DE9+DE6+DE5,
#Bidding Zone DE3: DE8+DE3+DE4+DEE+DED+DEG,
#Bidding Zone DE4: DEA+ DEB+DEC,
#Bidding Zone DE5: DE2+DE7+DE1


# Beispiele
#    c_cap = [gen[0]+gen[1] for gen,p in zip(network.generators.index, network.generators.p_nom) if 'solar' in gen or 'wind' in gen if p!=0]
#    c_cap=list(set(c_cap))
#    countries = network.buses.country.unique()
#    countries = (list(set(countries)-set(c_cap)))


# Create a new model
model: Model = gp.Model("Offshore_Bidding_Zones")

#Sets
T = range(run_parameter.timesteps)  # hours
T_extra = range(1, run_parameter.timesteps)

Y = range(run_parameter.years)
Y_extra = range(1, run_parameter.years)
G = data.dispatchable_generators[0].index  
R = data.res_series[0].columns
DAM = data.reservoir.index
S = data.storage.index
L = data.ac_lines.index
N = data.nodes.index
LDC = data.dc_lines.index
C = range(len(L)-len(N)+1) #C_cl_df.index
# separating the flexlines
I = data.dc_lines[data.dc_lines["EI"].isin(["BHEH", "NSEH1", "NSEH2", "CLUSTER"])].index  # BHEI
D = data.dc_lines[~data.dc_lines["EI"].isin(["BHEH", "NSEH1", "NSEH2", "CLUSTER"])].index # lines not to the EI's

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
p_load_lost = model.addVars(Y, T, zone, lb=0.0, ub = GRB.INFINITY, name = "p_load_lost")
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
#helper funktion die zurück gibt in zone x alle renewables
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
