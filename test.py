import geopandas as gpd
import pandas as pd

#zones festlegen, als set und zuordnung zu den nodes
shapes = gpd.read_file('data/shapes/NUTS_RG_10M_2021_4326.geojson')
shapes_filtered = shapes.query("LEVL_CODE ==1 and CNTR_CODE == 'DE'")
shapes_filtered.plot()

df_buses = pd.read_csv("data/PyPSA_elec1024/buses.csv", index_col=0)
#df_buses['geometry'] = [Point(xy) for xy in zip(df_buses.x, df_buses.y)]
gdf_buses = gpd.GeoDataFrame(df_buses, geometry=gpd.points_from_xy(df_buses.x, df_buses.y), crs="EPSG:4326")
#df_buses_selected = ['name', 'geometry']
df_buses.head()
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