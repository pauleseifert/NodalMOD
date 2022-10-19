import pandas as pd
import time
from selenium import webdriver
import os
from colour import Color
from dataclasses import dataclass
import plotly.graph_objects as go
from sys import platform
import sys


#read in
if platform == "linux" or platform == "linux2":
    directory = "/work/seifert/"
    case_name = sys.argv[1]
    scen = int(sys.argv[2])
elif platform == "darwin":
    directory = ""
    case_name = "test"
    scen = 1
    years = [0]


@dataclass
class conventionals:
    hydro: float = 0.0
    oil: float = 0.0
    gas: float = 0.0
    coal: float = 0.0
    nuclear: float = 0.0
    other: float = 0.0

def export(path_and_name, html_name):
    driver = webdriver.Firefox()
    driver.set_window_size(1000, 2000)
    driver.get("file://" + os.getcwd()+ "/"+path_and_name+".html")
    time.sleep(2)
    driver.save_screenshot(path_and_name+".png")
    driver.quit()
def ad_cicular_data(data, color, radius, unit):
    for i, row in data.iterrows():
        folium.CircleMarker(location=(row["LAT"], row["LON"]), radius= row[0]/radius, popup = "index "+(str(i)+ " " + str(round(row[0]/radius, 2)) + " "+unit), weight = 0, fill_opacity = 0.4, color=color, fill=True).add_to(map)
def prepare_results_files_nodes(file, bus_raw, temporal):
    file_without_0 = file.loc[:, (file != 0).any(axis=0)]
    file_sum = file_without_0.dropna(axis=1, how='all').sum(axis=temporal)
    file_sum.index = file_sum.index.astype(int)
    if temporal == 0:
        file_sum = file_sum.to_frame()
        file_ready = file_sum.merge(bus_raw[["LAT", "LON"]], how="left", left_index=True, right_index=True)
        return file_ready
    return file_sum
def prepare_results_files_index(file, index_file, bus_raw):
    file_without_0 = file.loc[:, (file != 0).any(axis=0)]
    file_sum = file_without_0.dropna(axis=1, how='all').sum(axis=0).to_frame()
    file_sum.index = file_sum.index.astype(int)
    file_bus = file_sum.merge(index_file[["bus", "type","name"]], how="left", left_index=True, right_index=True)
    file_ready = file_bus.merge(bus_raw[["LAT", "LON"]], how = "left", left_on = "bus", right_index=True)
    return file_ready

def prepare_results_files_index_temporal(file, index_file, types):
    def get_sum_of_type(file, index_file, type):
        index_list = index_file[index_file["type"].isin(type)].index.values
        plants_in_group = file[file.columns.intersection(index_list)]
        sum_timestep = plants_in_group.sum(axis=1)
        return sum_timestep
    file_without_0 = file.loc[:, (file != 0).any(axis=0)]
    file_without_0.columns = file_without_0.columns.astype(int)

    conventionals_temporal = conventionals(hydro = get_sum_of_type(file_without_0, index_file, types[0]),
                  oil = get_sum_of_type(file_without_0, index_file, types[1]),
                  gas = get_sum_of_type(file_without_0, index_file, types[2]),
                  coal=get_sum_of_type(file_without_0, index_file, types[3]),
                  nuclear=get_sum_of_type(file_without_0, index_file, types[4]),
                  other=get_sum_of_type(file_without_0, index_file, types[5]))
    return conventionals_temporal

def groupby_bus_add_location_filter_for_type(file, filter, bus_raw):
    if filter != []:
        filtered = file[file["type"].isin(filter)]
        return filtered.groupby(['bus']).sum()[[0]].merge(bus_raw[["LAT", "LON"]], left_index = True, right_index = True)
    else:
        return file.groupby(['bus']).sum()[[0]].merge(bus_raw[["LAT", "LON"]], left_index=True, right_index=True)

def get_map():
    #map = folium.Map(location=[57.5236, 9.6750],tiles="Stamen Toner", zoom_start= 5)
    map = folium.Map(location=[57.5236, 9.6750], tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
                     attr = 'Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ' ,zoom_start=5)
    return map



for y in years:
    read_folder = directory + "results/" + case_name + "/" + str(scen) + "/"
    export_folder = directory + "results/" + case_name + "/"

    beta = [2.8289908676897135, 4.026974650921101, 5.619285222154198, 6.386270713539531]
    bus_raw = pd.read_csv(read_folder + "busses.csv", index_col=0)
    powerplants_raw = pd.read_csv(read_folder + "powerplants.csv", index_col=0)
    storage = pd.read_csv(read_folder + "storage.csv", index_col=0)
    lines_overview = pd.read_csv(read_folder + "lines.csv", index_col=0)
    lines_DC_overview = pd.read_csv(read_folder + "lines_DC.csv", index_col=0)
    share_solar_raw = pd.read_csv(read_folder + "share_solar.csv", index_col=0)
    share_wind_raw = pd.read_csv(read_folder + "share_wind.csv", index_col=0)
    wind_capacities_raw = pd.read_csv("data/north_sea_energy_islands/csv/wind_capacity.csv", index_col=0)
    solar_capacities_raw = pd.read_csv("data/north_sea_energy_islands/csv/wind_capacity.csv", index_col=0)
    wind_capacities = wind_capacities_raw.merge(bus_raw[["LAT", "LON"]], how="left", left_on="bus", right_index=True).rename({"Pmax":0}, axis=1)
    solar_capacities = solar_capacities_raw.merge(bus_raw[["LAT", "LON"]], how="left", left_on="bus", right_index=True)

    F_AC = pd.read_csv(read_folder + str(y)+ "_F_AC.csv", index_col=0)
    F_DC = pd.read_csv(read_folder +str(y)+ "_F_DC.csv", index_col=0)
    CAP_BH = pd.read_csv(read_folder + "CAP_BH.csv", index_col=0)
    P_R_max_raw = pd.read_csv(read_folder + "P_R_max.csv", index_col=0)*beta[y]
    P_R_raw = pd.read_csv(read_folder + str(y)+ "_P_R.csv", index_col=0)
    P_C_raw = pd.read_csv(read_folder + str(y)+ "_P_C.csv", index_col=0)
    P_S_raw = pd.read_csv(read_folder +str(y)+ "_P_S.csv", index_col=0)
    L_S_raw = pd.read_csv(read_folder +str(y)+ "_L_S.csv", index_col=0)
    D_S_raw = pd.read_csv(read_folder + str(y)+ "_D_S.csv", index_col=0)





    #calculate Conventional usage
    conventional_spatial = prepare_results_files_index(file = P_C_raw, index_file= powerplants_raw, bus_raw = bus_raw)
    conventional_temporal = prepare_results_files_index_temporal(file = P_C_raw, index_file= powerplants_raw, types =[['HROR', 'HDAM'],['HFO', 'Fossil Oil', 'Coal/Oil','Oil'], ['Fossil Gas', 'Distillate'], ['Fossil Hard Coal', 'Lignite', 'Fossil Peat','Fossil Hard coal', 'Peat', 'Coal'], ['Nuclear'], ['Waste', 'Biomass']])
    hydro_spatial = groupby_bus_add_location_filter_for_type(conventional_spatial,['HROR', 'HDAM'] , bus_raw)
    oil_spatial = groupby_bus_add_location_filter_for_type(conventional_spatial,['HFO', 'Fossil Oil', 'Coal/Oil','Oil'], bus_raw)
    gas_spatial = groupby_bus_add_location_filter_for_type(conventional_spatial,['Fossil Gas', 'Distillate'], bus_raw)
    coal_spatial = groupby_bus_add_location_filter_for_type(conventional_spatial,['Fossil Hard Coal', 'Lignite', 'Fossil Peat','Fossil Hard coal', 'Peat', 'Coal'], bus_raw)
    nuclear_spatial = groupby_bus_add_location_filter_for_type(conventional_spatial,['Nuclear'], bus_raw)
    other_spatial = groupby_bus_add_location_filter_for_type(conventional_spatial,['Waste', 'Biomass'], bus_raw)

    #calculate renewables infeed after redispatch
    timesteps = len(P_R_raw)
    P_R_CM = P_R_raw
    P_R = prepare_results_files_nodes(P_R_CM, bus_raw, temporal=0)
    P_R_solar_raw = (share_solar_raw.iloc[:timesteps,:] * P_R_CM)#.dropna(axis=1, how='all')
    P_R_wind_raw = (share_wind_raw.iloc[:timesteps,:] * P_R_CM).dropna(axis=1, how='all')
    P_R_solar_spatial = prepare_results_files_nodes(P_R_solar_raw, bus_raw, temporal=0)
    P_R_wind_spatial = prepare_results_files_nodes(P_R_wind_raw,bus_raw, temporal=0)
    P_R_solar_temporal = prepare_results_files_nodes(P_R_solar_raw, bus_raw, temporal=1)
    P_R_wind_temporal= prepare_results_files_nodes(P_R_wind_raw,bus_raw, temporal=1)


    #calculate curtailment
    P_R_curtailed_raw = P_R_max_raw- P_R_raw
    P_R_curtailed = prepare_results_files_nodes(file = P_R_curtailed_raw, bus_raw = bus_raw, temporal=0)
    KPI_curtailment = pd.DataFrame([{"RE Curtailment": P_R_curtailed.sum(axis = 0)[0], "% of P_R": round(P_R_curtailed.sum(axis = 0)[0]/P_R_max_raw.sum().sum()*100, 2)}])
    print(KPI_curtailment)

    #calculate lines
    P_flow = prepare_results_files_lines(file = F_AC, bus_raw = bus_raw, index_file = lines_overview, yearly = False, year = y, CAP_BH = "")
    P_flow_DC = prepare_results_files_lines(file = F_DC, bus_raw = bus_raw, index_file = lines_DC_overview, yearly=True, year = y, CAP_BH = CAP_BH)

    #
    # #method 1
    #
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x = list(range(timesteps)), y = conventional_temporal.other, name= 'Other', fill = 'tonexty', stackgroup= 'one', mode= 'none', fillcolor= "grey"))
    # fig.add_trace(go.Scatter(x = list(range(timesteps)), y = conventional_temporal.hydro, name= 'Hydro', fill = 'tonexty', stackgroup= 'one', mode= 'none', fillcolor= "blue"))
    # fig.add_trace(go.Scatter(x = list(range(timesteps)), y = conventional_temporal.nuclear, name= 'Nuclear', fill = 'tonexty', stackgroup= 'one', mode= 'none', fillcolor= "red"))
    # fig.add_trace(go.Scatter(x = list(range(timesteps)), y = conventional_temporal.coal, name= 'Coal', fill = 'tonexty', stackgroup= 'one', mode= 'none', fillcolor= "#855720"))
    # fig.add_trace(go.Scatter(x = list(range(timesteps)), y = conventional_temporal.oil, name= 'Oil', fill = 'tonexty', stackgroup= 'one', mode= 'none', fillcolor= "black"))
    # fig.add_trace(go.Scatter(x = list(range(timesteps)), y = conventional_temporal.gas, name= 'Gas', fill = 'tonexty', stackgroup= 'one', mode= 'none', fillcolor= "orange"))
    # fig.add_trace(go.Scatter(x = list(range(timesteps)), y = P_R_wind_temporal, name= 'Wind', fill = 'tonexty', stackgroup= 'one', mode= 'none', fillcolor= "#c6d2be"))
    # fig.add_trace(go.Scatter(x = list(range(timesteps)), y = P_R_solar_temporal, name= 'Solar', fill = 'tonexty', stackgroup= 'one', mode= 'none', fillcolor= "yellow"))
    # fig.update_layout(title={'text':'Modelled Electricity Generation by Source'},
    #     xaxis= dict(title='Timesteps', titlefont_size=12, tickfont_size=12),
    #     yaxis=dict(title='Power [MW]', titlefont_size=12, tickfont_size=12),
    #     legend=dict(x=0, y=-0.1, bgcolor='rgba(255, 255, 255, 0)', bordercolor='rgba(255, 255, 255, 0)'),
    #     legend_orientation="h"
    # )
    #
    #
    # #method 2
    # maps_folder = export_folder + "maps/"
    # os.makedirs(maps_folder, exist_ok=True)
    # fig.write_image(maps_folder+ str(scen)+ "_"+ str(y) + "_electricity_gen.png", width = 1000, height = 800, scale = 3)
    #
    # map = get_map()
    # ad_cicular_data(wind_capacities, "green",1000, "GW")
    # name = "wind_capacity"
    # path_and_name = maps_folder + str(scen) + "_" + str(y) + "_" + name
    # map.save(path_and_name + '.html')
    # export(path_and_name, name)
    #
    # map = get_map()
    # ad_cicular_data(solar_capacities, "orange", 1000, "GW")
    # name = "solar_capacity"
    # path_and_name = maps_folder + str(scen) + "_" + str(y) + "_" + name
    # map.save(path_and_name + '.html')
    # export(path_and_name, name)
    #
    #
    # colors = list(Color("green").range_to(Color("red"),71))
    # map = get_map()
    # for i, row in P_flow.iterrows():
    #     folium.PolyLine([[row["LAT_x"], row["LON_x"]], [row["LAT_y"], row["LON_y"]]], popup = "line number " + str(i), color = str(colors[int(round(100*row[0],0))])).add_to(map)
    # for i, row in P_flow_DC.iterrows():
    #     folium.PolyLine([[row["LAT_x"], row["LON_x"]], [row["LAT_y"], row["LON_y"]]], popup = "line number " + str(i) + "_DC", color = str(colors[int(round(100*row[0],0))])).add_to(map)
    # name = "grid"
    # path_and_name = maps_folder + str(scen) + "_" + str(y) + "_" + name
    # map.save(path_and_name + '.html')
    # export(path_and_name, name)
    #
    # map = get_map()
    # ad_cicular_data(P_R_curtailed, "green", 1000, "GWh")
    # name = "curtailed"
    # path_and_name = maps_folder + str(scen) + "_" + str(y) + "_" + name
    # map.save(path_and_name + '.html')
    # export(path_and_name, name)
    #
    #
    # #conventionals
    # map = get_map()
    # ad_cicular_data(hydro_spatial, "blue", 1000, "GWh")
    # ad_cicular_data(oil_spatial, "black", 1000, "GWh")
    # ad_cicular_data(nuclear_spatial, "red", 1000, "GWh")
    # ad_cicular_data(gas_spatial, "#F99A06", 1000, "GWh")
    # ad_cicular_data(coal_spatial, "#AD5507", 1000, "GWh")
    # ad_cicular_data(other_spatial, "grey", 1000, "GWh")
    # name = "conventional"
    # path_and_name = maps_folder + str(scen) + "_" + str(y) + "_" + name
    # map.save(path_and_name + '.html')
    # export(path_and_name, name)
    #
    # #renewable infeed
    # map = get_map()
    # ad_cicular_data(P_R_solar_spatial, "red", 1000, "GWh")
    # ad_cicular_data(P_R_wind_spatial, "blue", 1000, "GWh")
    # name = "RES infeed"
    # path_and_name = maps_folder + str(scen) + "_" + str(y) + "_" + name
    # map.save(path_and_name + '.html')
    # export(path_and_name, name)