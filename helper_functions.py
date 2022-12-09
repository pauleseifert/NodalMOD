import collections
import os

import numpy as np
import pandas as pd
from numba import jit, float32


def merge_timeseries_demand(demand, timeseries):
    timeseries_T = timeseries.T
    timeseries_T.index = timeseries_T.index.astype(int)
    bus_ts_matrix = demand[["country"]].merge(timeseries_T, how="left", left_on="country", right_index=True).drop(
        ['country'], axis=1).T
    numpy_bus_ts_matrix = bus_ts_matrix.to_numpy()
    Pmax = demand["max"].to_numpy()
    bus_np = np.multiply(numpy_bus_ts_matrix, Pmax)
    bus = pd.DataFrame(bus_np, columns=demand["bus"])
    return bus
def merge_timeseries_demand_entsoe(demand, timeseries):
    demand["zone_elements"] = demand["bidding zone"].apply(lambda x: len(demand[demand["bidding zone"] == x]))
    timeseries_T = timeseries.T
    timeseries_T.index = timeseries_T.index.astype(int)
    bus_ts_matrix = demand[["bidding zone"]].merge(timeseries_T, how="left", left_on="bidding zone", right_index=True).drop(
        ['bidding zone'], axis=1).T
    numpy_bus_ts_matrix = bus_ts_matrix.to_numpy()
    element_count = demand["zone_elements"].to_numpy()
    bus_np = np.multiply(numpy_bus_ts_matrix, 1/element_count)
    bus = bus_np[:, ~np.isnan(bus_np).all(axis=0)]
    #bus = pd.DataFrame(bus_np, columns=demand["bus"])
    return pd.DataFrame(bus)
def append_BHEH(string):
    if string == "country":
        return pd.DataFrame([{"COUNTRY": "Bornholm Energy Hub", "CODE": "BHEH", "ID": "34"}])
    if string == "bus":
        return pd.DataFrame([{"name":"BHEH", "bus_i": 6127, "baseKV": 400, "zone": 34, "LAT": 55.13615337829421, "LON": 14.898639089359104}])
    if string == "wind":
        return pd.DataFrame([{"index": 479, "country": 34, "zone": "BHEH", "bus": 6127, "Pmax": 3000}])
    # if string == "wind_ts":
    #     test = wind_ts_bh["electricity"]/15000
    #     return test
    if string == "line":
        return pd.DataFrame({"name": ["BHEH-DE","BHEH-DK2", "BHEH-SE", "BHEH-PL" ], "fbus": [509, 1583, 5497, 4952], "tbus": [6127,6127,6127, 6127], "rateA": [1000, 1000, 1000, 1000]})

def merge_timeseries_supply(supply, timeseries):
    supply= supply.groupby("bus").sum(numeric_only = True).reset_index()
    timeseries_T = timeseries.T
    timeseries_T.index = timeseries_T.index.astype(int)
    bus_ts_matrix = supply[["bus"]].merge(timeseries_T, how="left", left_on="bus", right_index=True).drop(['bus'], axis=1).T
    numpy_bus_ts_matrix = bus_ts_matrix.to_numpy()
    Pmax = supply["P_inst"].to_numpy()
    @jit(nopython=True)
    def multiplication(numpy_bus_ts_matrix, Pmax):
        bus_np = np.multiply(numpy_bus_ts_matrix, Pmax)
        return bus_np
    bus_np = multiplication(numpy_bus_ts_matrix, Pmax)
    bus = pd.DataFrame(bus_np, columns=supply["bus"])
    return bus



def read_write_excel_to_csv(import_path, export_path):
    countries_raw = pd.read_excel(import_path+"GRID_MODEL.xlsx", sheet_name="COUNTRY")
    countries_raw.to_csv(export_path + "countries_raw.csv")
    bus_raw = pd.read_excel(import_path+"GRID_MODEL.xlsx", sheet_name="BUS")
    bus_raw.to_csv(export_path + "bus_raw.csv")
    bus_dc_raw = pd.read_excel(import_path+"GRID_MODEL.xlsx", sheet_name="BUS_DC")
    bus_dc_raw.to_csv(export_path + "bus_dc_raw.csv")
    lines_raw = pd.read_excel(import_path+"GRID_MODEL.xlsx", sheet_name="BRANCH", na_filter=False)
    lines_raw[["rateA", "rateB", "rateC"]] = lines_raw[["rateA", "rateB", "rateC"]].round(0).astype(int)
    lines_raw[["r"]] = lines_raw[["r"]].astype(float)
    lines_raw.to_csv(export_path + "lines_raw.csv")
    lines_dc_raw = pd.read_excel(import_path+"GRID_MODEL.xlsx", sheet_name="BRANCH_DC")
    lines_dc_raw.to_csv(export_path + "lines_dc_raw.csv")
    lines_dc_mpc_raw = pd.read_excel(import_path+"GRID_MODEL.xlsx", sheet_name="BRANCHDC_MPC")
    lines_dc_mpc_raw.to_csv(export_path + "lines_dc_mpc_raw.csv")
    converter_raw = pd.read_excel(import_path+"GRID_MODEL.xlsx", sheet_name="CONVERTER")
    converter_raw.to_csv(export_path + "converter_raw.csv")
    generators = pd.read_excel(import_path+"GRID_MODEL.xlsx", sheet_name='GENERATOR')
    generators.to_csv(export_path + "generators.csv")
    generators_mpc = pd.read_excel(import_path+"GRID_MODEL.xlsx", sheet_name='GEN_MPC')
    generators_mpc.to_csv(export_path + "generators_mpc.csv")
    wind = pd.read_excel(import_path+"GRID_MODEL.xlsx", sheet_name='WIND', na_filter=False)
    wind = wind.loc[:, (wind != '').any(axis=0)]  # remove 0 entries
    wind.to_csv(export_path + "wind.csv")
    wind_ts = pd.read_excel(import_path+"GRID_MODEL.xlsx", sheet_name='WIND_TS', na_filter=False, index_col=0).T.reset_index(
        drop=True)
    wind_ts.to_csv(export_path + "wind_ts.csv")
    solar = pd.read_excel(import_path+"GRID_MODEL.xlsx", sheet_name='SOLAR', na_filter=False)
    solar = solar.loc[:, (solar != '').any(axis=0)]  # remove 0 entries
    solar.to_csv(export_path + "solar.csv")
    solar_ts = pd.read_excel(import_path+"GRID_MODEL.xlsx", sheet_name='SOLAR_TS', na_filter=False, index_col=0).T.reset_index(
        drop=True)
    solar_ts.to_csv(export_path + "solar_ts.csv")
    demand_raw = pd.read_excel(import_path+"GRID_MODEL.xlsx", sheet_name="DEMAND")
    demand_raw.to_csv(export_path + "demand_raw.csv")
    demand_raw_ts = pd.read_excel(import_path+"GRID_MODEL.xlsx", sheet_name="DEMAND_TS", index_col=0).T.reset_index(drop=True)
    demand_raw_ts.to_csv(export_path + "demand_raw_ts.csv")
    ntc_ac = pd.read_excel(import_path + "MARKET_MODEL.xlsx", sheet_name='INTERCONNECTORS_AC')
    ntc_ac.to_csv(export_path + "ntc_ac.csv")
    ntc_dc = pd.read_excel(import_path + "MARKET_MODEL.xlsx", sheet_name='INTERCONNECTORS_DC')
    ntc_dc.to_csv(export_path + "ntc_dc.csv")
    return
#read_write_excel_to_csv("data/north_sea_energy_islands/GRID_MODEL.xlsx", "data/north_sea_energy_islands/csv/")

def read_csv(import_path):
    countries_raw = pd.read_csv(import_path + "countries_raw.csv", index_col=0)
    bus_raw = pd.read_csv(import_path + "bus_raw.csv", index_col=0)
    bus_dc_raw = pd.read_csv(import_path + "bus_dc_raw.csv", index_col=0)
    lines_raw = pd.read_csv(import_path + "lines_raw.csv", index_col=0)
    lines_dc_raw = pd.read_csv(import_path + "lines_dc_raw.csv", index_col=0)
    lines_dc_mpc_raw = pd.read_csv(import_path + "lines_dc_mpc_raw.csv", index_col=0)
    converter_raw = pd.read_csv(import_path + "converter_raw.csv", index_col=0)
    generators = pd.read_csv(import_path + "generators.csv", index_col=0)
    generators_mpc = pd.read_csv(import_path + "generators_mpc.csv", index_col=0)
    wind = pd.read_csv(import_path + "wind.csv", index_col=0)
    wind_ts = pd.read_csv(import_path + "wind_ts.csv", index_col=0)
    solar = pd.read_csv(import_path + "solar.csv", index_col=0)
    solar_ts = pd.read_csv(import_path + "solar_ts.csv", index_col=0)
    demand_raw = pd.read_csv(import_path + "demand_raw.csv", index_col=0)
    demand_raw_ts = pd.read_csv(import_path + "demand_raw_ts.csv", index_col=0)
    ntc_ac = pd.read_csv(import_path + "ntc_ac.csv", index_col=0)
    ntc_dc = pd.read_csv(import_path + "ntc_dc.csv", index_col=0)
    #storage = pd.read_csv(import_path + "storage.csv")
    wind_ts_bornholm = pd.read_csv(import_path + "ninja_bornholm.csv", skiprows=3)
    hydro = pd.read_csv(import_path + "hydro_CM.csv")
    demand_entsoe= pd.read_csv(import_path + "zonal_demand.csv", index_col=0)
    return countries_raw, bus_raw, bus_dc_raw, lines_raw, lines_dc_raw, lines_dc_mpc_raw, converter_raw, generators, generators_mpc, wind, wind_ts, solar, solar_ts, demand_raw, demand_raw_ts, ntc_ac, ntc_dc, wind_ts_bornholm, hydro, demand_entsoe
@jit(float32(float32, float32, float32, float32), nopython=True)
def distance_calc(lat1, lon1, lat2, lon2):
    # approximate radius of earth in km
    R = 6373.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance #in km
def distance_line(nodes, dc_line_overview, index):
    distance_dict={}
    for i in index:
        entry = dc_line_overview.loc[i]
        distance = distance_calc_between_entries(entry_1=nodes.loc[entry["from"]], entry_2=nodes.loc[entry["to"]])
        distance_dict.update({i:distance})
    return distance_dict
def distance_calc_between_entries(entry_1, entry_2):
    # approximate radius of earth in km
    lat1 = entry_1["LAT"]
    lon1 = entry_1["LON"]
    lat2 = entry_2["LAT"]
    lon2 = entry_2["LON"]
    return distance_calc(lat1, lon1, lat2, lon2)

def map_distance(df):
    #return pd.Series(map(geopy.distance.distance().km,(df["LAT_from"], df["LON_from"]), (df["LAT_to"], df["LON_to"])))
    return pd.Series(map(distance_calc, df["LAT_from"].values, df["LON_from"].values, df["LAT_to"].values, df["LON_to"].values))
def create_ybus(lines, length):
    fbus = lines["fbus"]
    tbus = lines["tbus"]
    lines_x = lines["x"]
    ybus = np.zeros((length, length))
    for i in lines.index:
        ybus[fbus[i]-1][tbus[i]-1] = 1/lines_x[i]
        ybus[tbus[i]-1][fbus[i]-1] = 1/lines_x[i]
    xbus = -ybus
    rowsum_ybus = np.sum(ybus, axis=1)
    for i in range(ybus.shape[0]):
        ybus[i][i] = - rowsum_ybus[i]
    ybus = -ybus
    return pd.DataFrame(ybus), pd.DataFrame(xbus)

def create_ybus_for_index(lines, length):
    fbus = lines["from"]
    tbus = lines["to"]
    lines_x = lines["x"]
    ybus = np.zeros((length, length))
    for i in lines.index:
        ybus[fbus[i]-1][tbus[i]-1] = 1/lines_x[i]
        ybus[tbus[i]-1][fbus[i]-1] = 1/lines_x[i]
    xbus = -ybus
    rowsum_ybus = np.sum(ybus, axis=1)
    for i in range(ybus.shape[0]):
        ybus[i][i] = - rowsum_ybus[i]
    ybus = -ybus
    return pd.DataFrame(ybus), pd.DataFrame(xbus)

def create_ybus_for_index_comp(lines, length):
        fbus = lines["from"]
        tbus = lines["to"]
        lines_x = lines["x"]
        ybus = np.zeros((length, length))
        for i in lines.index:
            ybus[fbus[i]][tbus[i]] = 1 / lines_x[i]
            ybus[tbus[i]][fbus[i]] = 1 / lines_x[i]
        xbus = -ybus
        rowsum_ybus = np.sum(ybus, axis=1)
        for i in range(ybus.shape[0]):
            ybus[i][i] = - rowsum_ybus[i]
        ybus = -ybus
        return pd.DataFrame(ybus), pd.DataFrame(xbus)


def create_ybus_numpy(lines, length):
    fbus = lines["from"]
    tbus = lines["to"]
    lines_x = lines["x"]
    ybus = np.zeros((length, length))
    for i in lines.index:
        ybus[fbus[i]][tbus[i]] = 1 / lines_x[i]
        ybus[tbus[i]][fbus[i]] = 1 / lines_x[i]
    xbus = -ybus
    rowsum_ybus = np.sum(ybus, axis=1)
    for i in range(ybus.shape[0]):
        ybus[i][i] = - rowsum_ybus[i]
    ybus = -ybus
    return ybus, xbus

def create_ybus_df(lines, busses):
    fbus = lines["from"]
    tbus = lines["to"]
    lines_x = lines["x"]
    ybus = pd.DataFrame(np.zeros((len(busses.index), len(busses.index))), columns= busses.index, index=busses.index)
    for i in lines.index:
        ybus[fbus[i]][tbus[i]] = 1 / lines_x[i]
        ybus[tbus[i]][fbus[i]] = 1 / lines_x[i]
    xbus = -ybus
    rowsum_ybus = np.sum(ybus, axis=1)
    for i in ybus.index:
        ybus[i][i] = - rowsum_ybus[i]
    ybus = -ybus
    return ybus, xbus

def hoesch(lines,bus):
    k_il = np.zeros((len(lines.index), len(bus.index)), dtype=int)
    for i in lines.index:
        k_il[i, lines[lines.index == i]["from"]] = 1
        k_il[i, lines[lines.index == i]["to"]] = -1
    #b_vector = np.array(1 / lines["x"])
    #b_lk = np.diag(b_vector)
    return k_il#, b_lk

def ren_helper(n, renewables_list):
    if n in renewables_list:
        return [n]
    else:
        return []
def ren_helper2(n, renewables_dict):
    if n in renewables_dict:
        return [n]
    else:
        return []

def demand_helper(j,t, nodes_col, nodes_dict):
    if j in nodes_col:
        return nodes_dict[j][t]
    else:
        return 0


def demand_helper2(j,t, nodes_dict):
    if j in nodes_dict[y]:
        return nodes_dict[y][j][t]
    else:
        return 0


#def demand_helper2(j,t,y, nodes_dict):
#    if j in nodes_dict[y]:
#        return nodes_dict[y][j][t]
#    else:
#        return 0
def create_encyclopedia(data_series: object) -> object:
    mydict = data_series.to_dict()
    newd = collections.defaultdict(list)
    for k, vl in mydict.items():
            newd[vl].append(k)
    return newd
class Myobject:
    pass
def export(folder, scen, Y, P_C, P_R,P_DAM, res_curtailment ,P_H, cap_E,cap_BH,P_S, C_S, L_S, F_AC, F_DC, p_load_lost, additionals):
    os.makedirs(folder, exist_ok=True)
    if isinstance(P_C, np.ndarray):
        # cap_BH
        pd.DataFrame(cap_BH, columns=additionals["CAP_BH"]).to_csv(folder + "cap_BH.csv")
        for y in Y:
            #P_C
            pd.DataFrame(P_C[y, :, :]).to_csv(folder + str(y)+"_P_C.csv")
            # P_R
            pd.DataFrame(P_R[y, :, :], columns= additionals["P_R"]).to_csv(folder + str(y) + "_P_R.csv")
            # P_DAM
            pd.DataFrame(P_DAM[y, :, :]).to_csv(folder + str(y) + "_P_DAM.csv")
            if scen in [2, 3, 4]:
                    # cap_E
                pd.DataFrame(cap_E).to_csv(folder +"cap_E.csv")
                    # P_H
                pd.DataFrame(P_H[y, :, :]).to_csv(folder + str(y) + "_P_H.csv")
            # load lost
            pd.DataFrame(p_load_lost[y, :, :]).to_csv(folder + str(y) + "_p_load_lost.csv")
            # res_curtailment
            pd.DataFrame(res_curtailment[y, :, :]).to_csv(folder + str(y) + "_res_curtailment.csv")
            # storage
            pd.DataFrame(P_S[y, :, :]).to_csv(folder + str(y) + "_P_S.csv")
            pd.DataFrame(C_S[y, :, :]).to_csv(folder + str(y) + "_C_S.csv")
            pd.DataFrame(L_S[y, :, :]).to_csv(folder + str(y) + "_L_S.csv")
            #AC line flow
            pd.DataFrame(F_AC[y, :, :]).to_csv(folder + str(y) + "_F_AC.csv")
            #DC line flow
            pd.DataFrame(F_DC[y, :, :]).to_csv(folder + str(y) + "_F_DC.csv")
    else:
        for y in Y:
            #P_C
            pd.DataFrame({'T': t, "G": g, 'value': P_C[y, t, g].X} for (year, t, g) in P_C).pivot_table(index=["T"],columns='G', values='value').to_csv(folder + str(y)+"_P_C.csv")
            #P_R
            pd.DataFrame({'T': t, "R": r, 'value': P_R[y, t, r].X} for (year, t, r) in P_R).pivot_table(index=["T"],columns='R',values='value').to_csv(folder + str(y)+"_P_R.csv")
            # P_R
            pd.DataFrame({'T': t, "DAM": r, 'value': P_DAM[y, t, r].X} for (year, t, r) in P_DAM).pivot_table(index=["T"],columns='DAM',values='value').to_csv(folder + str(y) + "_P_DAM.csv")
            # res_curtailment
            pd.DataFrame({'T': t, "R": r, 'value': res_curtailment[y, t, r].X} for (year, t, r) in res_curtailment).pivot_table(index=["T"],columns='R',values='value').to_csv(folder + str(y) + "_res_curtailment.csv")
            #Load lost
            pd.DataFrame({'T': t, "Z": z, 'value': p_load_lost[y, t, z].X} for (year, t, z) in p_load_lost).pivot_table(index=["T"],columns='Z',values='value').to_csv(folder + str(y)+"_p_load_lost.csv")
            #Storage
            pd.DataFrame({'T': t, "S": s, 'value': P_S[y, t, s].X} for (year, t, s) in P_S).pivot_table(index=["T"],columns='S',values='value').to_csv(folder + str(y)+"_P_S.csv")
            pd.DataFrame({'T': t, "S": s, 'value': C_S[y, t, s].X} for (year, t, s) in C_S).pivot_table(index=["T"],columns='S',values='value').to_csv(folder + str(y)+"_C_S.csv")
            pd.DataFrame({'T': t, "S": s, 'value': L_S[y, t, s].X} for (year, t, s) in L_S).pivot_table(index=["T"],columns='S',values='value').to_csv(folder + str(y)+"_L_S.csv")
            #AC line flow
            pd.DataFrame({'T': t, "L": l, 'value': F_AC[y, t, l].X} for (year, t, l) in F_AC).pivot_table(index='T', columns='L', values='value').to_csv(folder + str(y)+"_F_AC.csv")
            #DC
            pd.DataFrame({'T': t, "L": l, 'value': F_DC[y, t, l].X} for (year, t, l) in F_DC).pivot_table(index='T', columns='L', values='value').to_csv(folder + str(y)+"_F_DC.csv")
            if scen in [2,3,4]:
                # P_H
                pd.DataFrame({'T': t, "E": e, 'value': P_H[y, t, e].X} for (year, t, e) in P_H).pivot_table(index=["T"],columns='E',values='value').to_csv(folder + str(y) + "_P_H.csv")
                # cap_E
                pd.DataFrame({'Y': year, "E": e, 'value': cap_E[year, e].X} for (year,e) in cap_E).pivot_table(index='Y',columns='E', values='value').to_csv(folder + "cap_E.csv")
        #CAP BH
        pd.DataFrame({'Y': year, "I": i, 'value': cap_BH[year, i].X} for (year,i) in cap_BH).pivot_table(index='Y', columns='I', values='value').to_csv(folder + "cap_BH.csv")

def check_ED(T, G, R, S, Z_ED, CB, P, P_R, P_S, C_S, demand_bus_ED, p_gen_lost_ED, p_load_lost_ED, I):
    hourly_sum = np.array(T)
    for t in T:
        hourly_sum[t]= sum(P[t, g].X for g in G)+ sum(P_R[t, r].X for r in R)+sum(P_S[t, s].X for s in S)- sum(C_S[t, s].X for s in S)-sum(p_gen_lost_ED[t, z].X for z in Z_ED) + sum(p_load_lost_ED[t, z].X for z in Z_ED)- demand_bus_ED.sum(axis=1).iloc[t] #+ sum(I[t, cb].X for cb in CB)
    if hourly_sum.all():
        print("checksum in ED is not 0 -> leaking!")
        exit()
    else:
        print("checksum in ED - pass")
def check_zonal_ED(T, G, R, S, Z_ED, CB, P, P_R, P_S, C_S, demand_bus_ED, p_gen_lost_ED, p_load_lost_ED, I, busses, powerplants_ED, storage_ED, get_demand, zonehelper, crossboarder_trade):
    checkzone_ED_T = []
    for t in T:
        checkzone_ED = []
        for z in Z_ED:
            dem = get_demand(z, t, busses)
            if type(dem) is int:
                dem = dem
            else:
                dem = dem.getValue()
            checkzone_ED_iter = - dem \
            + sum(P[t, g].X for g in zonehelper(z, powerplants_ED, "powerplants")) \
            + sum(P_R[t, r].X for r in zonehelper(z, busses, "renewables")) \
            + sum(P_S[t, s].X - C_S[t, s].X for s in zonehelper(z, storage_ED, "storage")) \
            - p_gen_lost_ED[t, z].X + p_load_lost_ED[t, z].X \
            - sum(I[t, cb].X for cb in crossboarder_trade.index[crossboarder_trade["from"] == z]) \
            + sum(I[t, cb].X for cb in crossboarder_trade.index[crossboarder_trade["from"] == z])
            checkzone_ED = np.append(checkzone_ED, checkzone_ED_iter)
        checkzone_ED_T = np.append(checkzone_ED_T,checkzone_ED.sum())
    if checkzone_ED_T.sum() <= 0.1:
        print("checkzone passed")
    else:
        print("checkzone failed")
        exit()
def check_CM_sum(T, G, P, delta_p_up, delta_p_down):
        sum_conv = sum(sum(P[t, g].X for g in G)+ sum(delta_p_up[t, g].X for g in G) - sum(delta_p_down[t, g].X for g in G) for t in T)
        return sum_conv

def renewables_scaling_country_specific(renewables_supply, scaling_factors, bus_CM, encyclopedia, scaling):
    renewables_supply_new = {}
    renewables_T = renewables_supply.T
    renewables_T.index = renewables_T.index.astype(int)
    merge_country = renewables_T.merge(bus_CM, left_index = True, right_index = True)[["country"]]
    #merge_country = merge_bz.merge(encyclopedia, left_on = "bidding zone", right_on="zone_number").set_index(merge_bz.index)[["country"]]
    merge_factor = merge_country.merge(scaling_factors, how ="outer", left_on="country", right_index=True).fillna(1).drop("country", axis=1)
    if scaling:
        renewables_supply_new.update({i:renewables_supply*merge_factor.iloc[:,i] for i in range(4)})
        return renewables_supply_new
    #means that the ts is already scaled for the first year and only needs the other ones
    if scaling == False:
        renewables_supply_new.update({i: renewables_supply *(merge_factor.iloc[:,i]/merge_factor.iloc[:,0])for i in range(4)})
        return renewables_supply_new

def conv_scaling_country_specific(generators, scaling_factors, bus_CM):
    conv_supply_new = {}
    conventional_h20 = generators[generators["type"].isin(["HROR", "HDAM"])]
    conventional_fossil = generators[~generators["type"].isin(["HROR", "HDAM"])]
    merge_country = conventional_fossil.merge(bus_CM, left_on = "bus", right_index = True)[["country"]]
    merge_factor = merge_country.merge(scaling_factors, how="outer", left_on="country", right_index=True).drop("country", axis=1)
    for i in range(4):
        yearly = conventional_fossil.copy()
        #yearly["pmin"] *= merge_factor.iloc[:,i]
        yearly["pmax"] *= merge_factor.iloc[:,i]
        yearly = yearly.append(conventional_h20)
        conv_supply_new.update({i: yearly})
    return conv_supply_new


def demand_columns(busses_filtered, load_raw, tyndp_demand):
    # def get_value(x):
    #     try:
    #         return busses_filtered[busses_filtered["old_index"] == x.strip()]["index"].values[0]
    #     except:
    #         pass

    #load_raw.columns = load_raw.columns.to_series().apply(get_value)
    demand_T = load_raw.T
    demand_T.index = demand_T.index.to_series().apply(lambda x: x.strip())
    demand_pypsa_merged = demand_T.merge(busses_filtered[["old_index", "bidding_zone"]], how= "inner", left_index = True, right_on= "old_index").drop(["old_index"], axis=1)
    def yearly_scaling(pypsa_demand_zones,tyndp_demand, i):
        pypsa_zones_sum = pypsa_demand_zones.groupby("bidding_zone").sum().sum(axis=1)
        #stupid aggregations because tyndp has NOS0 etc.
        pypsa_zones_sum["NO1"] = pypsa_zones_sum["NO1"] + pypsa_zones_sum["NO2"]+pypsa_zones_sum["NO5"]
        pypsa_zones_sum.drop(["NO2", "NO5"], inplace=True)
        tyndp_demand_sum = tyndp_demand[i].T.sum(axis=1)
        scaling_factor = tyndp_demand_sum/pypsa_zones_sum
        scaled_demand = pd.DataFrame()
        for zone in scaling_factor.index:
            if zone == "NO1":
                #print(pypsa_demand_zones[pypsa_demand_zones["bidding_zone"].isin(["NO1", "NO2", "NO5"])])
                scaled_demand = pd.concat([scaled_demand, (pypsa_demand_zones[pypsa_demand_zones["bidding_zone"].isin(["NO1", "NO2", "NO5"])].drop(["bidding_zone"], axis=1) * scaling_factor[zone]).T],ignore_index=False, axis=1)
            else:
                #print(pypsa_demand_zones[pypsa_demand_zones["bidding_zone"] == zone])
                scaled_demand = pd.concat([scaled_demand, (pypsa_demand_zones[pypsa_demand_zones["bidding_zone"] == zone].drop(["bidding_zone"], axis=1) * scaling_factor[zone]).T], ignore_index= False, axis=1)
            #print(scaled_demand.columns[-1])
            scaled_demand = scaled_demand.sort_index(axis=1)
        return scaled_demand
    demand_yearly = {i: yearly_scaling(pypsa_demand_zones=demand_pypsa_merged, tyndp_demand= tyndp_demand, i= i) for i in [0,1,2]}
    #TODO PL ist komisch -> zu niedrig
    return demand_yearly
def zones_busses_dam(bus_overview_limited_dam, limited_dam):
    busses_NO = bus_overview_limited_dam[bus_overview_limited_dam["country"] == "NO"]
    busses_SE = bus_overview_limited_dam[bus_overview_limited_dam["country"] == "SE"]
    busses_others = bus_overview_limited_dam[~bus_overview_limited_dam['country'].isin(["NO", "SE"])]
    zones_se = {476: 3,477: 1,478: 4,479: 3,480: 3,481: 2,482: 3,483: 3,484: 2,485: 3,486: 4,487: 2,488: 3,489: 4,490: 1,491: 3,492: 3,493: 3,494: 2,495: 3,496: 2,497: 3,498: 3,499: 2,500: 4,501: 2,502: 2,503: 3,504: 4,505: 1,506: 3,507: 2,508: 2,509: 3,510: 1,511: 3,512: 3,513: 4,514: 2,515: 2,516: 3,517: 2,518: 4,519: 1, 520: 2}
    zones_norge = {389: 1, 390: 4, 391: 5, 392: 3, 393: 4, 394: 1, 395: 3, 396: 3, 397: 1, 398: 2, 399: 3, 400: 5,
                   401: 5, 402: 1, 403: 2, 404: 5, 405: 3, 406: 2, 407: 4, 408: 1, 409: 2, 410: 1, 411: 2, 412: 2,
                   413: 2, 414: 1, 415: 3, 416: 1, 417: 1, 418: 3, 419: 2, 420: 3, 421: 5, 422: 4, 423: 4, 424: 3,
                   425: 1, 426: 2, 427: 5, 428: 3}
    busses_norge = busses_NO.merge(pd.DataFrame.from_dict(zones_norge, orient="index", columns=["zone"]), how="left",left_index=True, right_index=True)
    busses_norge["bidding_zone"] = busses_norge["country"] + busses_norge["zone"].astype(str)
    busses_SE = busses_SE.merge(pd.DataFrame.from_dict(zones_se, orient="index", columns=["zone"]), how="left",left_index=True, right_index=True)
    busses_SE["bidding_zone"] = busses_SE["country"] + busses_SE["zone"].astype(str)

    busses_others["bidding_zone"] = busses_others["country"]
    busses_all = pd.concat([busses_norge.drop(["zone"], axis=1),busses_SE.drop(["zone"], axis=1), busses_others])
    dam_with_zones = limited_dam.merge(busses_all["bidding_zone"], left_on = "bus", right_index = True)
    encyc_dam_zones = create_encyclopedia(dam_with_zones["bidding_zone"])


    return encyc_dam_zones
def give_nearest_bus_relative_position(bus_raw, hydro_numpy):
    bus_vector = np.zeros([hydro_numpy.shape[0]], dtype= np.int32)
    dict_key_list = list(bus_raw["LAT"].keys())
    for i in range(hydro_numpy.shape[0]):
        distance_vector = {}
        for j in range(len(dict_key_list)):
            distance_vector[j] = distance_calc(hydro_numpy[i,0], hydro_numpy[i,1], bus_raw["LAT"][dict_key_list[j]], bus_raw["LON"][dict_key_list[j]])
        bus_vector[i] = dict_key_list[min(distance_vector, key=distance_vector.get)]
    return bus_vector

def scaling_logic(df_scaled_capacity, tyndp_target, current_value, bz, type):
    if tyndp_target == 0:
        df_scaled_capacity.loc[
            (df_scaled_capacity['bidding_zone'].isin(bz)) & (df_scaled_capacity['type'] == type), "P_inst"] *= 0
    elif current_value == 0:
        number_entries = df_scaled_capacity.loc[(df_scaled_capacity['bidding_zone'].isin(bz)) & (df_scaled_capacity['type'] == type)].count()[0]
        df_scaled_capacity.loc[(df_scaled_capacity['bidding_zone'].isin(bz)) & (df_scaled_capacity['type'] == type), "P_inst"] = tyndp_target / number_entries
    else:
        factor = tyndp_target / current_value
        df_scaled_capacity.loc[(df_scaled_capacity['bidding_zone'].isin(bz)) & (df_scaled_capacity['type'] == type), "P_inst"] *= factor
    return df_scaled_capacity

def get_solar_yearly(tyndp_values, df_2020_capacity_bz, df_2020_capacity_bz_grouped, year, type):
    df_scaled_capacity = df_2020_capacity_bz.copy()
    bidding_zones = list(set(df_2020_capacity_bz["bidding_zone"].unique()) - {"NO2", "NO5"})
    for bz in bidding_zones:
        tyndp_zone = tyndp_values.query("node == @bz & generator == @type")[year].sum()
        if bz == "NO1":
            tyndp_zone = tyndp_values.query("node == @bz & generator == @type")[year].sum()
            bz = ["NO1", "NO2", "NO5"]
        else:
            bz = [bz]
        inst_capacity_bz = df_2020_capacity_bz_grouped.query("bidding_zone == @bz")["P_inst"].sum()
        df_scaled_capacity = scaling_logic(df_scaled_capacity=df_scaled_capacity, tyndp_target=tyndp_zone,current_value=inst_capacity_bz, type="solar", bz=bz)
        #print("Installed " + str(type) + " power in BZ " + str(bz) + " in " + str(year) + " = " + str(df_scaled_capacity.groupby(["bidding_zone"]).sum()["P_inst"].loc[bz].sum()) + " MW")
    df_scaled_capacity = df_scaled_capacity.sort_index()
    return df_scaled_capacity


def get_wind_yearly(tyndp_values, df_2020_capacity_bz, year, kinis_offshore_windfarms,df_2020_capacity_bz_type_grouped):

    df_scaled_capacity = df_2020_capacity_bz.copy()
    bidding_zones = list(set(df_2020_capacity_bz["bidding_zone"].unique()) - {"NO2", "NO5", "BHEH", "NSEH1", "NSEH2"})
    for bz in bidding_zones:
        if bz == "NO1":
            tyndp_zone_offshore = tyndp_values.query("node == @bz & generator == 'offwind'")[year].sum()
            tyndp_zone_onshore = tyndp_values.query("node == @bz & generator == 'onwind'")[year].sum()  # "offwind"][year]
            bz = ["NO1", "NO2", "NO5"]
            # inst_capacity_bz = df_2020_capacity_bz_type_grouped#.loc[bz].sum()
            #
        else:
            try:
                tyndp_zone_offshore = tyndp_values.loc[bz].query("generator == 'offwind'")[year].sum()
                if bz in kinis_offshore_windfarms.index:
                    tyndp_zone_offshore -= kinis_offshore_windfarms[bz]
                if bz in ["DK"]:
                    tyndp_zone_offshore -= 13000
                if bz in ["UK"]:
                    tyndp_zone_offshore -= 10000
            except:
                tyndp_zone_offshore = 0
            if tyndp_zone_offshore < 0:
                #print("bz " + str(bz) + " hat niedrigere TYNDP Werte als unsere Cluster im Jahr " + str(year))
                tyndp_zone_offshore = 0
            try:
                tyndp_zone_onshore = tyndp_values.loc[bz].query("generator == 'onwind'")[year].sum()
            except:
                tyndp_zone_onshore = 0
            bz = [bz]
        inst_onwind_capacity_bz = df_2020_capacity_bz_type_grouped.query("bidding_zone == @bz & type == 'onwind'")["P_inst"].sum()
        inst_offwind_capacity_bz = df_2020_capacity_bz_type_grouped.query("bidding_zone == @bz & type == 'offwind'")["P_inst"].sum()
        df_scaled_capacity = scaling_logic(df_scaled_capacity=df_scaled_capacity, tyndp_target=tyndp_zone_onshore,current_value=inst_onwind_capacity_bz, type="onwind", bz=bz)
        df_scaled_capacity = scaling_logic(df_scaled_capacity=df_scaled_capacity, tyndp_target=tyndp_zone_offshore,current_value=inst_offwind_capacity_bz, type="offwind", bz=bz)
        # df_scaled_capacity.replace([np.inf, -np.inf], 0, inplace=True)

        #print("Installed wind power in BZ " + str(bz) + " in " + str(year) + " = " + str(
        #    df_scaled_capacity.groupby(["bidding_zone", "type"]).sum().query("bidding_zone == @bz & type == 'offwind'")["P_inst"].sum()) + " MW offshore, " + str(
        #   df_scaled_capacity.groupby(["bidding_zone", "type"]).sum().query("bidding_zone == @bz & type == 'onwind'")["P_inst"].sum()) + " MW onshore")
    df_scaled_capacity = df_scaled_capacity.sort_index()
    return df_scaled_capacity

def flatten(l):
    return [item for sublist in l for item in sublist]

def res_normalisation(self, df, type):
    # grouped_pypsa.loc["NO1", "offwind"] = grouped_pypsa.loc["NO1", "offwind"]+ grouped_pypsa.loc["NO2", "offwind"] + grouped_pypsa.loc["NO5", "offwind"]
    # test_tyndp = self.tyndp_installed_capacity
    normalised = df.copy()
    if type == "wind":
        grouped_pypsa = df.groupby(["bidding_zone", "type"]).sum(numeric_only = True)["max"]
        # again the grouping of the NOS0
        grouped_pypsa.loc["NO1", "offwind"] = grouped_pypsa.loc["NO1", "offwind"] + grouped_pypsa.loc["NO2", "offwind"] + grouped_pypsa.loc["NO5", "offwind"]
        grouped_pypsa.loc["NO1", "onwind"] = grouped_pypsa.loc["NO1", "onwind"] + grouped_pypsa.loc["NO2", "onwind"] + grouped_pypsa.loc["NO5", "onwind"]
        grouped_pypsa.drop(["NO2", "NO5"], inplace=True)
        for subtype in grouped_pypsa.index.get_level_values(1).unique():
            for zone in grouped_pypsa.index.get_level_values(0).unique():
                if zone == "NO1":
                    try:
                        normalisation_factor = self.tyndp_installed_capacity.loc[zone, subtype][2020] /grouped_pypsa.loc[zone, subtype]
                        normalised.loc[(normalised["type"] == subtype) & (normalised["bidding_zone"].isin(["NO1", "NO2", "NO5"])), "max"] = normalised.loc[(normalised["type"] == subtype) & (normalised["bidding_zone"].isin(["NO1","NO2","NO5"])), "max"] * normalisation_factor
                    except:
                        normalised.loc[(normalised["type"] == subtype) & (normalised["bidding_zone"].isin(["NO1", "NO2", "NO5"])), "max"] = 0.00001
                else:
                    try:
                        normalisation_factor = self.tyndp_installed_capacity.loc[zone, subtype][2020] / grouped_pypsa.loc[zone, subtype]
                        normalised.loc[(normalised["type"] == subtype) & (normalised["bidding_zone"] == zone), "max"] = normalised.loc[(normalised["type"] == subtype) & (normalised["bidding_zone"] == zone), "max"] * normalisation_factor
                    except:
                        normalised.loc[(normalised["type"] == subtype) & (normalised["bidding_zone"] == zone), "max"] = 0.00001
                        # normalised = pd.concat([normalised, df[(df["type"] == subtype) & (df["bidding_zone"] == zone)] * normalisation_factor])
    if type == "solar":
        grouped_pypsa = df.groupby(["bidding_zone"]).sum(numeric_only = True)["max"]
        grouped_pypsa.loc["NO1"] = grouped_pypsa.loc["NO1"] + grouped_pypsa.loc["NO2"] + grouped_pypsa.loc["NO5"]
        grouped_pypsa.drop(["NO2", "NO5"], inplace=True)
        for zone in grouped_pypsa.index.unique():
            if zone == "NO1":
                try:
                    normalisation_factor = self.tyndp_installed_capacity.loc[zone, type][2020] / grouped_pypsa.loc[zone]
                    normalised.loc[normalised["bidding_zone"].isin(["NO1", "NO2", "NO5"]), "max"] = normalised.loc[normalised["bidding_zone"].isin(["NO1","NO2","NO5"]), "max"] * normalisation_factor
                except:
                    normalised.loc[(normalised["bidding_zone"].isin(["NO1", "NO2", "NO5"])), "max"] = 0.00001
            else:
                try:
                    normalisation_factor = self.tyndp_installed_capacity.loc[zone, type][2020] / grouped_pypsa.loc[
                        zone]
                    normalised.loc[(normalised["bidding_zone"] == zone), "max"] = normalised.loc[normalised["bidding_zone"] == zone, "max"] * normalisation_factor
                except:
                    normalised.loc[(normalised["bidding_zone"] == zone), "max"] = 0.00001
    normalised = normalised.rename(columns={"max": "P_inst"})
    return normalised