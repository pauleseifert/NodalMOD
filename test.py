import geopandas
import geopandas as gpd
import pandas as pd

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
sjoin2 = nodes_geopandas.sjoin(states_filtered[["geometry"]], how ="left", predicate='contains').rename(columns={"index_right": "bidding_zone"})

sjoined_nodes_DE_bidding_zones = nodes_geopandas.sjoin(states_filtered[["geometry"]],how="left", predicate='intersects').rename(columns={"index_right": "bidding_zone"})
#sjoined_nodes_states = gdf_buses.sjoin(shapes_filtered[["NUTS_NAME","id","geometry"]], how ="left")
#sjoined_nodes_states = gdf_buses.sjoin(shapes_filtered[["NUTS_NAME","NUTS_ID","geometry"]], how ="left")
sjoined_nodes_bidding_zones = nodes_geopandas.sjoin(states,how="left")
sjoin = gpd.sjoin(nodes_DE_bidding_zones,states_filtered,how='inner')

    nodes_scand_bidding_zones = nodes_geopandas.query('country in ["DK", "SE", "NO"]')
    nodes_scand_bidding_zones_resolved = nodes_scand_bidding_zones.sjoin(scandinavian_bidding_zones[["geometry"]],
                                                                         how="left", predicate='intersects').rename(
        columns={"index_right": "bidding_zone"})
    if nodes_scand_bidding_zones_resolved['bidding_zone'].isna().sum() >= 1:
        missing = nodes_scand_bidding_zones_resolved.loc[
                  pd.isna(nodes_scand_bidding_zones_resolved["bidding_zone"]), :].index
        print("not all nodes are matched! " + str(len(missing)) + " are missing")
        print(missing.values)

        # add the missing values
        nodes_scand_bidding_zones_resolved.at[217, "bidding_zone"] = "DK2"
        nodes_scand_bidding_zones_resolved.at[486, "bidding_zone"] = "SE4"

    nodes_other_bidding_zone = nodes_geopandas[
        ~nodes_geopandas["country"].isin(scandinavian_bidding_zones["country"])]
    nodes_other_bidding_zone["bidding_zone"] = nodes_other_bidding_zone["country"]
    self.nodes = pd.concat([nodes_scand_bidding_zones_resolved, nodes_other_bidding_zone]).drop(
        columns="geometry").sort_index()


#zones festlegen, als set und zuordnung zu den nodes
shapes = gpd.read_file('data/shapes/NUTS_RG_10M_2021_4326.geojson')
shapes_filtered = shapes.query("LEVL_CODE ==1 and CNTR_CODE == 'DE'")
shapes_level1 = shapes.query("LEVL_CODE ==1")
<<<<<<< Updated upstream
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
sjoined_nodes_states2 = gdf_buses.sjoin(shapes_level1[["NUTS_NAME","NUTS_ID","geometry"]], how ="left",predicate='intersects').rename(columns={"index_right": "bidding_zone"})
sjoined_nodes_states3 = sjoined_nodes_states2[['country','x','y', 'geometry','bidding_zone', 'NUTS_NAME', 'NUTS_ID']]
sjoined_nodes_states3.to_csv("data_nodes_to_zones.csv")
#How many nodes are in each state bzw zone "state_Bayern" = "NUTS_ID":"DE2"
# First grouping based on "NUTS_ID" - Within each team we are grouping based on "Position"
df_nodes_Bayern = sjoined_nodes_states.groupby("NUTS_ID").count()

#df_nodes_Bayern = grouped.to_frame().reset_index()
#df.columns = ["NUTS_ID", ‘listings_count’]
=======
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
>>>>>>> Stashed changes

#Merge Pauls (old_index) and Pypsa dataframes (name) to one df_nodes_to_zones_merge
#sjoined_nodes_states4 = sjoined_nodes_states3.rename(columns = {'name':'old_index'}, inplace = True)
sjoined_nodes_states4 = sjoined_nodes_states3.reset_index()
#nodes_geopandas2 = n.rename(columns = {'old_index':'name'}, inplace = True)
df_nodes_to_zones_merge = pd.merge(sjoined_nodes_states4, nodes_geopandas, on='name')
#Filter the df_nodes_to_zones_merge
df_nodes_to_zones_filtered = df_nodes_to_zones_merge[['index','name','country_y','LON','LAT', 'geometry_y','bidding_zone_y', 'NUTS_NAME', 'NUTS_ID']]
#Save into csv
df_nodes_to_zones_filtered.to_csv("df_nodes_to_zones_filtered.csv")
#group the df for the zonal configuration
test = df_nodes_to_zones_filtered.groupby(['country_y', 'NUTS_NAME']).groups
print(df_nodes_to_zones_filtered.groupby('NUTS_ID').filter())
#test2 = df_nodes_to_zones_filtered.groupby('month')[['duration']].sum()