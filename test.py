import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from import_object_data_Zonal_Configuration import model_data, run_parameter


#run_parameter = run_parameter(scenario_name = "Offshore_Bidding_Zone_Scenario")

#Scandic shapes
gdf_scandinavia = gpd.read_file("data//shapes//scandinavian_bidding_zones.geojson").reset_index()
test = gpd.read_file("data//shapes//scandinavian_bidding_zones.geojson")
    #scandinavian_bidding_zones.to_excel('scandic_BZ.xlsx')
    #gdf_scandinavian_bidding_zones = gdf_scandinavian_bidding_zones.drop(labels=[??], axis=0, inplace=False)
gdf_scandic = gdf_scandinavia.rename(columns = {'bidding_zone':'CNTR_CODE'})
plot_sc_res_curtailment = gdf_scandic.filter(items=['CNTR_CODE', 'geometry'], axis=1)

#NodalMod shapes
shapes = gpd.read_file('data/shapes/NUTS_RG_10M_2021_4326.geojson')
shapes_de_filtered = shapes.query("LEVL_CODE ==1 and CNTR_CODE == 'DE'")
shapes_rest_filtered = shapes.query("LEVL_CODE ==0").drop(labels=[ 2007, 2006, 1982, 1976, 1953, 1992, 1936, 1901, 1989, 1932, 1941], axis=0, inplace=False)
shapes_NodalMod = shapes_rest_filtered.loc[shapes_rest_filtered['id'].isin(['BE','DE', 'CZ', 'FI', 'NL', 'PL', 'UK'])]
plot_nm_res_curtailment = shapes_NodalMod.filter(items=['id', 'CNTR_CODE', 'geometry'], axis=1)

#german shapes BZ5
shapes = gpd.read_file('data/shapes/NUTS_RG_10M_2021_4326.geojson')
shapes_de_filtered = shapes.query("LEVL_CODE ==1 and CNTR_CODE == 'DE'")
nodes = pd.read_excel("data\\final_readin_data\\nodes.xlsx", index_col=0)
de_res_curtailment = pd.merge(shapes_de_filtered, nodes, on='NUTS_ID')
plot_de_res_curtailment = de_res_curtailment.filter(items=['CNTR_CODE', run_parameter.scen, 'geometry'], axis=1)


#Plotting RES CU
# concat all together
shapes_RES_CU_final = pd.concat([plot_de_res_curtailment, plot_nm_res_curtailment, plot_sc_res_curtailment])#ignore_index=True

# plot OffBZ points
res_cu_OffBZ = pd.read_excel("data//shapes//res_cu_offbz.xlsx").set_index("bidding_zone")
gdf_offbz = gpd.GeoDataFrame(res_cu_OffBZ, geometry=gpd.points_from_xy(res_cu_OffBZ.LON, res_cu_OffBZ.LAT), crs="EPSG:4326")
    #gdf_offbz.plot(marker='*', color='green', markersize=5)

#plot both together
fig, ax = plt.subplots()
ax.set_aspect('equal')
shapes_RES_CU_final.plot(ax=ax, color='white', edgecolor='black')
gdf_offbz.plot(ax=ax, marker='o', color='red', markersize=5)
plt.show()

#add res cu in dataframe
res_cu_BZ5 = pd.read_excel("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/_res_curtailment.xlsx")
res_cu_BZ5['Sum'] = res_cu_BZ5.sum(axis=1)
#ALT
    res_cu_BZ5 = pd.read_excel("data//shapes//res_cu_BZ5.xlsx").set_index("bidding_zone")
    gdf_res_cu_BZ5 = gpd.GeoDataFrame(res_cu_BZ5, geometry=res_cu_BZ5.geometry)
    res_cu_OffBZ = pd.read_excel("data//shapes//res_cu_offbz.xlsx", ).set_index("bidding_zone")

    gdf_offbz = gpd.GeoDataFrame(res_cu_OffBZ, geometry=gpd.points_from_xy(res_cu_OffBZ.LON, res_cu_OffBZ.LAT), crs="EPSG:4326")
    res_cu_OffBZ.plot()


    #zones festlegen, als set und zuordnung zu den nodes
    shapes = gpd.read_file('data/shapes/NUTS_RG_10M_2021_4326.geojson')
    shapes_de_filtered = shapes.query("LEVL_CODE ==1 and CNTR_CODE == 'DE'")
    shapes_rest_filtered = shapes.query("LEVL_CODE ==0").drop(labels=[ 2007, 2006, 1982, 1976, 1953, 1992, 1936, 1901, 1989, 1932, 1941], axis=0, inplace=False)

    shapes_NodalMod = shapes_rest_filtered.loc[shapes_rest_filtered['id'].isin(['BE', 'CZ', 'DE', 'DK', 'FI', 'NL', 'NO', 'PL', 'SE', 'UK'])]
    shapes_NodalMod.to_excel("shapes_NodalMod.xlsx")
    shapes_NodalMod.plot()



    shapes_BAU_final = shapes_rest_filtered
    shapes_BZ5_final = pd.concat([shapes_de_filtered, shapes_rest_filtered], ignore_index=True)
    shapes_BZ5_final.plot()

    nodes = pd.read_excel("data\\final_readin_data\\nodes.xlsx", index_col=0)
    shape_res_curtailment = pd.merge(shapes_BZ5_final, nodes, on='NUTS_ID')
    shape_res_curtailment_2 = pd.merge(shapes_rest_filtered, nodes, on=['id', 'country_y'])
    shapes_BZ5 = pd.concat([shape_res_curtailment, shape_res_curtailment], ignore_index=True)

    plot_res_curtailment = shape_res_curtailment.filter(items=['NUTS_ID', run_parameter.scen, 'geometry'], axis=1)
    plot_de_BZ5 = plot_res_curtailment.drop_duplicates(subset=['NUTS_ID'])
    plot_res_curtailment.plot(run_parameter.scen, cmap="Blues")
    plot_de_BZ5.plot(run_parameter.scen, cmap="Blues")
    plot_de_BZ5.to_excel("BZ5_plot_cu.xlsx")


    #read in shape NUTS EU Level 1 for DE
    states = gpd.read_file("data/shapes/NUTS_RG_10M_2021_4326.geojson")
    states_filtered = states.query("LEVL_CODE ==1")
    #read in data.nodes from paul
    df_data_nodes = pd.read_excel("data/shapes/data.nodes.xlsx")
    #define geometry of nodes
    nodes_geopandas = gpd.GeoDataFrame(df_data_nodes, geometry=gpd.points_from_xy(df_data_nodes.LON, df_data_nodes.LAT),crs="EPSG:4326")
    #Filter the nodes df in DE
    #nodes_DE_bidding_zones = nodes_geopandas.query('country in ["DE"]')
    #Spatial Join the nodes in BZ

    #zones festlegen, als set und zuordnung zu den nodes
    shapes = gpd.read_file('data/shapes/NUTS_RG_10M_2021_4326.geojson')
    shapes_filtered = shapes.query("LEVL_CODE ==1 and CNTR_CODE == 'DE'")
    shapes_level1 = shapes.query("LEVL_CODE ==1")


    df_buses = pd.read_csv("data/PyPSA_elec1024/buses.csv", index_col=0)
    #df_buses['geometry'] = [Point(xy) for xy in zip(df_buses.x, df_buses.y)]
    gdf_buses = gpd.GeoDataFrame(df_buses, geometry=gpd.points_from_xy(df_buses.x, df_buses.y), crs="EPSG:4326")
    #df_buses_selected = ['name', 'geometry']
    # coordinate systems are correct?
    #df_buses_selected.crs == shapes_filtered.crs
    #Spatial Join
    #sjoined_nodes_states = gpd.sjoin(df_buses["geometry"],shapes_filtered, op="within")
    sjoined_nodes_states = gdf_buses.sjoin(shapes_filtered[["NUTS_NAME","NUTS_ID","geometry"]], how ="left")

    #How many nodes are in each state bzw zone "state_Bayern" = "NUTS_ID":"DE2"
    # First grouping based on "NUTS_ID" - Within each team we are grouping based on "Position"
    df_nodes_Bayern = sjoined_nodes_states.groupby("NUTS_ID").count()

    #df_nodes_Bayern = grouped.to_frame().reset_index()
    #df.columns = ["NUTS_ID", ‘listings_count’]

    #nodes einlesen
    df_buses = pd.read_csv("data/PyPSA_elec1024/buses.csv", index_col=0)
    #aus long and lat von nodes eine geometry machen (points)
    gdf_buses = gpd.GeoDataFrame(df_buses, geometry=gpd.points_from_xy(df_buses.x, df_buses.y), crs="EPSG:4326")
    #Spatial Join (joinen der beiden tabellen anhand ihrer geometry)
    sjoined_nodes_states = gdf_buses.sjoin(shapes_filtered[["NUTS_NAME","NUTS_ID","geometry"]], how ="left")

    #Filtern der Columns die wir brauchen für Zones DE
    df_zones_DE = sjoined_nodes_states.query("country == 'DE'")
    df_zones_DE_filtered = df_zones_DE.filter(['NUTS_NAME', 'country', 'NUTS_ID', 'geometry'])
    #How many nodes are in each state bzw zone "state_Bayern" = "NUTS_ID":"DE2"?
    df_zones_DE_filtered.count("NUTS_NAME == 'Bayern'")

    sjoined_nodes_states2 = gdf_buses.sjoin(shapes_level1[["NUTS_NAME","NUTS_ID","geometry"]], how ="left",predicate='intersects').rename(columns={"index_right": "bidding_zone"})
    sjoined_nodes_states3 = sjoined_nodes_states2[['country','x','y', 'geometry','bidding_zone', 'NUTS_NAME', 'NUTS_ID']]
    sjoined_nodes_states3.to_csv("data_nodes_to_zones.csv")

    #Merge Pauls (old_index) and Pypsa dataframes (name) to one df_nodes_to_zones_merge
    #sjoined_nodes_states4 = sjoined_nodes_states3.rename(columns = {'name':'old_index'}, inplace = True)
    sjoined_nodes_states4 = sjoined_nodes_states3.reset_index()
    #sjoined_nodes_states4.plt()


    #Funktion zum groupen und aufsummeiren der generations and fuels
    #test = sjoined_nodes_states4.groupby(["NUTS_ID","Fuel"]).sum(numeric_only=True)[["bidding_zone"]]

    #nodes_geopandas2 = n.rename(columns = {'old_index':'name'}, inplace = True)
    #use Pauls Index for the nodes to find missing vaules?
    #df_nodes_to_zones_merge = pd.merge(sjoined_nodes_states4, nodes_geopandas, on='name')
    #Filter the df_nodes_to_zones_merge
    #df_nodes_to_zones_filtered = df_nodes_to_zones_merge[['index','name','country_y','LON','LAT', 'geometry_y','bidding_zone_y', 'NUTS_NAME', 'NUTS_ID']]
    #Save into csv
    #df_nodes_to_zones_filtered.to_csv("df_nodes_to_zones_filtered.csv")
    #group the df for the zonal configuration
    #test = df_nodes_to_zones_filtered.groupby(['country_y', 'NUTS_NAME']).groups
    #print(df_nodes_to_zones_filtered.groupby('NUTS_ID').filter())
    #test2 = df_nodes_to_zones_filtered.groupby('month')[['duration']].sum()
    df_nodes = pd.read_csv("data/import_data/df_nodes_to_zones_filtered_final.csv",sep=";", index_col=0)

    lookup_dictBZ2={"DEF":"DEII1", "DE6":"DEII1", "DE9":"DEII1", "DE3":"DEII1", "DE4":"DEII1", "DE8":"DEII1", "DED":"DEII1", "DEE":"DEII1", "DEG":"DEII1", "DEA":"DEII2", "DEB":"DEII2", "DEC":"DEII2", "DE1":"DEII2", "DE2":"DEII2", "DE7":"DEII2", "OffBZN":"OffBZN", "OffBZB":"OffBZB"}
    lookup_dictBZ3 = {"DEF":"DEII1", "DE6":"DEII1", "DE9":"DEII1", "DE3":"DEII2", "DE4":"DEII2", "DE8":"DEII2", "DED":"DEII2", "DEE":"DEII2", "DEG":"DEII2", "DEA":"DEII3", "DEB":"DEII3", "DEC":"DEII3", "DE1":"DEII3", "DE2":"DEII3", "DE7":"DEII3", "OffBZN":"OffBZN", "OffBZB":"OffBZB"}
    lookup_dictBZ5 = {"DEF":"DEII1", "DE6":"DEII2", "DE9":"DEII2", "DE3":"DEII3", "DE4":"DEII3", "DE8":"DEII3", "DED":"DEII3", "DEE":"DEII3", "DEG":"DEII3", "DEA":"DEII4", "DEB":"DEII4", "DEC":"DEII4", "DE1":"DEII5", "DE2":"DEII5", "DE7":"DEII5", "OffBZN":"OffBZN", "OffBZB":"OffBZB"}
    def lookup(row):
        try:
            value = lookup_dictBZ5[row["NUTS_ID"]]
        except:
            value = row["country_y"]
        return value

    df_nodes['BZ_2'] = df_nodes.apply(lambda row: lookup(row), axis=1)
    df_nodes['BZ_3'] = df_nodes.apply(lambda row: lookup(row), axis=1)
    df_nodes['BZ_5'] = df_nodes.apply(lambda row: lookup(row), axis=1)
    df_nodes_to_zones = df_nodes
    df_nodes.to_csv('df_nodes.csv')

    #plotting average marginal costs

    #nodes = pd.read_csv("data\\final_readin_data\\generators.csv", index_col=0)
    generation = pd.read_csv("data\\\PyPSA_elec1024\\generators.csv", index_col=0)
    dispatchables = pd.read_excel("data\\final_readin_data\\dispatchable.xlsx")
    dispatchables_mc = dispatchables.loc[dispatchables.groupby(["type"])]
    dispatchables_mc = dispatchables.groupby(['mc', 'type'])
    dispatchables_mc = dispatchables.drop_duplicates(subset=['mc', 'type'])
    dispatchables_mc= dispatchables_mc.sort_values('type')
    dispatchables_mc_final = dispatchables_mc.loc[:,["type","mc"]].reset_index(drop=True)
    #df = pd.read_excel("results\\self.case_name\\run_parameter.scen\\subscen"+ run_parameter.sensitivity_scen + ".csv")
    df_results_generation = pd.read_excel("results/"+ str(run_parameter.case_name) +"/"+ str(run_parameter.scen) +"/subscen"+ str(run_parameter.sensitivity_scen) + "/0_P_C.xlsx", index_col=0)
    conv_dict = {0: 'CCGT',
                 1: 'coal',
                 2: 'biomass',
                 3: 'HDAM',
                 4: 'OCGT',
                 5: 'nuclear',
                 6: 'lignite',
                 7: 'oil'}

    # rename columns in DataFrame using dictionary
    df_results_generation.rename(columns=conv_dict, inplace=True)
    df_results_generation.sum()
    mc_calc = df_results_generation.sum().mul(dispatchables_mc_final, axis = 0)
    mc_calc = dispatchables_mc_final.set_index('type').join(df_results_generation.set_index(columns))
    dispatchables_mc_final.to_excel("mc_conv.xlsx")
    df = df_results_generation.set_index().transpose()
    df_2 = df.reset_index(inplace=True)
    df = df.merge(dispatchables_mc_final, on='type')