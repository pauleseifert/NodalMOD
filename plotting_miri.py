import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from import_object_data_Zonal_Configuration import model_data, run_parameter


run_parameter = run_parameter(scenario_name = "Offshore_Bidding_Zone_Scenario")
#Scandic shapes
gdf_scandinavia = gpd.read_file("data//shapes//scandinavian_bidding_zones.geojson").reset_index()
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
