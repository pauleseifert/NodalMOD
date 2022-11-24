import numpy as np
import pandas as pd
from shapely.geometry import Point
import geopandas as gpd
from geopandas import GeoDataFrame
import matplotlib.pyplot as plt
from printing_funct import plotly_maps_bubbles

#offshore bz filters and organizes offshore BZ for the df nodes to zones

#def add_future_windcluster(self, location):
    windfarms = pd.read_csv("data/additional_windfarm_cluster.csv", encoding="UTF-8").dropna(axis=1)
    windfarms["Market Zone"] = windfarms["Market Zone"].replace("DELU", "DE").replace("GB", "UK")
#plot windfarms and cluster to find baltic and north sea offshore BZ
    windfarms = windfarms.rename(columns={"Latitude (decimal units)": "LAT", "Longitude (decimal units)": "LON"})
    gdf_buses_windfarms = gpd.GeoDataFrame(windfarms, geometry=gpd.points_from_xy(windfarms.LON, windfarms.LAT), crs="EPSG:4326")
    shapes = gpd.read_file('data/shapes/NUTS_RG_10M_2021_4326.geojson')
    shapes_level0 = shapes.query("LEVL_CODE ==0 and CNTR_CODE == 'DE' and CNTR_CODE == 'PL'")
    germany = shapes_level0
    gdf_buses_windfarms.plot(ax=germany.plot(figsize=(10, 10)), marker='o', column='Wind_Farm_Name', legend=True)
    gdf_buses_windfarms.to_csv('gdf_windfarms.csv')
#add windfarm information to the df nodes to zones filtered
#df = pd.read_csv("data/shapes/df_nodes_to_zones_filtered.csv", sep = ';')
    plotly_maps_bubbles(df=windfarms, scen=9, maps_folder=location + "kini_locations",
                                  name="future_windfarms_locations_all", unit="GW", size_scale=100,
                                  title="findfarms", year=0)
    # Belgium
    additional_node = windfarms[windfarms["Market Zone"] == "BE"]
    if print: plotly_maps_bubbles(df=additional_node, scen=9, maps_folder=location + "kini_locations",
                                  name="future_windfarms_locations_BE", unit="GW", size_scale=100,
                                  title="findfarms", year=0)
    # new_nodes = pd.concat([self.nodes, additional_node])
    additional_dc_lines = pd.DataFrame()
    # attach every of the clusters to a number of onshore points
    # north sea
    for i in range(524, 525):
        # print(i)
        additional_dc_lines = pd.concat([additional_dc_lines, pd.DataFrame(
            {"from": (i, i, i, i, i), "to": (24, 366, 288, 523, 522),
             "EI": ("CLUSTER", "CLUSTER", "CLUSTER", "CLUSTER", "CLUSTER")})])

    # Deutschland -> hier nehme ich einfach alle
    additional_node = windfarms[windfarms["Market Zone"] == "DE"]
    # additional_node.index = np.arange(len(self.nodes), len(self.nodes) + len(additional_node))
    if print: plotly_maps_bubbles(df=additional_node, scen=9, maps_folder=location + "kini_locations",
                                  name="future_windfarms_locations_DE", unit="GW", size_scale=100,
                                  title="findfarms", year=0)
    # new_nodes = pd.concat([new_nodes, additional_node])
    # attach every of the clusters to a number of onshore points
    # north sea
    for i in range(525, 529):
        additional_dc_lines = pd.concat([additional_dc_lines, pd.DataFrame(
            {"from": (i, i, i, i, i), "to": (170, 212, 373, 523, 522),
             "EI": ("CLUSTER", "CLUSTER", "CLUSTER", "CLUSTER", "CLUSTER")})])
    # baltic
    for i in range(529, 531):
        additional_dc_lines = pd.concat([additional_dc_lines, pd.DataFrame(
            {"from": (i, i, i, i), "to": (218, 62, 513, 521), "EI": ("CLUSTER", "CLUSTER", "CLUSTER", "CLUSTER")})])

    # Dänemark
    additional_node = windfarms[windfarms["Market Zone"].isin(["DK1", "DK2"])]
    # additional_node.index = np.arange(len(new_nodes), len(new_nodes) + len(additional_node))
    if print: plotly_maps_bubbles(df=additional_node, scen=9, maps_folder=location + "kini_locations",
                                  name="future_windfarms_locations_DK", unit="GW", size_scale=100,
                                  title="findfarms", year=0)
    # nodes die ich haben möchte

    for i in range(543, 544):
        additional_dc_lines = pd.concat([additional_dc_lines, pd.DataFrame(
            {"from": (i, i, i, i, i), "to": (212, 426, 380, 522, 523),
             "EI": ("CLUSTER", "CLUSTER", "CLUSTER", "CLUSTER", "CLUSTER")})])

    # Netherlands
    additional_node = windfarms[windfarms["Market Zone"].isin(["NL"])]
    # additional_node.index = np.arange(len(new_nodes), len(new_nodes) + len(additional_node))
    if print: plotly_maps_bubbles(df=additional_node, scen=9, maps_folder=location + "kini_locations",
                                  name="future_windfarms_locations_NL", unit="GW", size_scale=100,
                                  title="findfarms")
    # nodes die ich haben möchte
    # additional_node = additional_node[additional_node.index.isin([547,550,551])]
    # additional_node.index = np.arange(len(new_nodes), len(new_nodes) + len(additional_node))
    # plotly_maps_bubbles(df=additional_node, scen=9, maps_folder= location+"kini_locations", name="future_windfarms_locations", unit="GW", size_scale=100,title="findfarms")

    # new_nodes = pd.concat([new_nodes, additional_node])
    for i in range(531, 534):
        additional_dc_lines = pd.concat([additional_dc_lines, pd.DataFrame(
            {"from": (i, i, i, i, i, i), "to": (376, 357, 265, 366, 522, 523),
             "EI": ("CLUSTER", "CLUSTER", "CLUSTER", "CLUSTER", "CLUSTER", "CLUSTER")})])

    # UK
    additional_node = windfarms[windfarms["Market Zone"].isin(["UK"])]
    # additional_node.index = np.arange(len(new_nodes), len(new_nodes) + len(additional_node))
    if print: plotly_maps_bubbles(df=additional_node, scen=9, maps_folder=location + "kini_locations",
                                  name="future_windfarms_locations_UK", unit="GW", size_scale=100,
                                  title="findfarms")
    # nodes die ich haben möchte
    # additional_node = additional_node[additional_node.index.isin([551,560,558, 550, 559, 568, 552, 548, 567, 557, 552, 548, 567, 557, 575, 574, 569, 565, 570, 563, 547, 564, 578, 576])]
    # additional_node.index = np.arange(len(new_nodes), len(new_nodes) + len(additional_node))
    # plotly_maps_bubbles(df=additional_node, scen=9, maps_folder= location+"kini_locations", name="future_windfarms_locations", unit="GW", size_scale=100,title="findfarms")

    for i in range(535, 538):
        additional_dc_lines = pd.concat([additional_dc_lines, pd.DataFrame(
            {"from": (i, i, i, i, i), "to": (300, 292, 307, 522, 523),
             "EI": ("CLUSTER", "CLUSTER", "CLUSTER", "CLUSTER", "CLUSTER")})])
    for i in range(538, 543):
        additional_dc_lines = pd.concat([additional_dc_lines, pd.DataFrame(
            {"from": (i, i, i, i, i, i), "to": (350, 265, 357, 24, 522, 523),
             "EI": ("CLUSTER", "CLUSTER", "CLUSTER", "CLUSTER", "CLUSTER", "CLUSTER")})])

    # Poland
    additional_node = windfarms[windfarms["Market Zone"].isin(["PL"])]
    # additional_node.index = np.arange(len(new_nodes), len(new_nodes) + len(additional_node))
    if print: plotly_maps_bubbles(df=additional_node, scen=9, maps_folder=location + "kini_locations",
                                  name="future_windfarms_locations_PL", unit="GW", size_scale=100,
                                  title="findfarms")

    # new_nodes = pd.concat([new_nodes, additional_node])
    for i in range(534, 535):
        additional_dc_lines = pd.concat([additional_dc_lines, pd.DataFrame(
            {"from": (i, i, i, i), "to": (470, 518, 62, 521), "EI": ("CLUSTER", "CLUSTER", "CLUSTER", "CLUSTER")})])

    new_dc_lines = pd.concat([self.dc_lines, additional_dc_lines])

    new_dc_lines = new_dc_lines.reset_index(drop=True)
    # new_nodes = new_nodes.reset_index()
    self.dc_lines = new_dc_lines

    # plotly_maps_lines_colorless(P_flow=test, P_flow_DC=test2, bus=self.nodes, scen=9, maps_folder=location+"grid_test")


self.add_future_windcluster = True
self.EI_bus = pd.DataFrame([
    {"country": "BHEH", "y": 55.13615337829421, "x": 14.898639089359104, "p_nom_max": 3000, "bus": "BHEH",
     "carrier": "offwind-dc"},
    {"country": "NSEH1", "y": 55.22300, "x": 3.78700, "p_nom_max": 10000, "bus": "NSEH1",
     "carrier": "offwind-dc"},
    {"country": "NSEH2", "y": 55.69354, "x": 3.97940, "p_nom_max": 10000, "bus": "NSEH2",
     "carrier": "offwind-dc"}], index=["BHEH", "NSEH1", "NSEH2"])
self.added_DC_lines = pd.DataFrame(
    {"p_nom": [1400, 2000, 2000, 700], "length": [720, 267, 400, 300], "index_x": [299, 198, 170, 513],
     "index_y": [419, 111, 93, 116], "tags": [
        "North Sea Link 2021: https://tyndp2020-project-platform.azurewebsites.net/projectsheets/transmission/110",
        "hvdc corridor norGer to WesGer 1034: https://tyndp2020-project-platform.azurewebsites.net/projectsheets/transmission/1034",
        "hvdc corridor norGer to WesGer 1034: https://tyndp2020-project-platform.azurewebsites.net/projectsheets/transmission/1034",
        "Hansa Power Bridge 1 https://tyndp2020-project-platform.azurewebsites.net/projectsheets/transmission/176"]})
self.added_AC_lines = pd.DataFrame(
    {"s_nom": [6000.0, 1000.0, 500.0, 1500.0, 300.0, 400.0, 500.0, 1500.0, 1000.0, 3200.0, 900.0],
     "length": [100.0, 40.0, 237.5, 182.0, 27.0, 46.0, 125.0, 95.0, 175.0, 60.0, 200.0],
     "x": [24.6, 9.84, 58.425, 44.772, 6.642, 11.316, 30.75, 23.37, 43.05, 14.76, 49.2],
     "index_x": [0, 26, 28, 85, 119, 142, 170, 180, 225, 303, 490],
     "index_y": [8, 138, 30, 119, 364, 217, 191, 198, 238, 327, 505]})
self.flexlines_EI = pd.DataFrame(
    {"from": [523, 523, 523, 523, 523, 523, 523, 522, 522, 522, 522, 522, 522, 521, 521, 521, 521],
     "to": [522, 403, 212, 209, 170, 376, 357, 279, 170, 103, 24, 357, 376, 62, 467, 218, 513],
     "EI": ["NSEH1", "NSEH1", "NSEH1", "NSEH1", "NSEH1", "NSEH1", "NSEH1", "NSEH2", "NSEH2", "NSEH2", "NSEH2",
            "NSEH2", "NSEH2", "BHEH", "BHEH", "BHEH", "BHEH"]})

self.TRM = 0.7
self.country_selection = ['BE', 'CZ', 'DE', 'DK', 'FI', 'NL', 'NO', 'PL', 'SE', 'UK', "NSEH1", "NSEH2", "BHEH"]
bidding_zones = ['AL', 'AT', 'BA', 'BE', 'BG', 'CH', 'CZ', 'DE', 'DK1', 'DK2', 'ES', 'FI', 'FR', 'GR', 'HR',
                 'HU', 'IE', 'IT1', 'IT2', 'IT3', 'IT4', 'IT5', 'ME', 'MK', 'NL', 'NO1', 'NO5', 'NO3', 'NO4',
                 'NO2', 'PL', 'PT', 'RO', 'RS', 'SE1', 'SE2', 'SE3', 'SE4', 'SI', 'SK', 'UK', 'CBN', 'TYNDP',
                 'NSEH', 'BHEH']
self.bidding_zones_overview = pd.DataFrame({"bidding zones": ['AL', 'AT', 'BA', 'BE', 'BG', 'CH', 'CZ', 'DE',
                                                              'DK1', 'DK2', 'ES', 'FI', 'FR', 'GR', 'HR', 'HU',
                                                              'IE', 'IT1', 'IT2', 'IT3', 'IT4', 'IT5', 'ME',
                                                              'MK', 'NL', 'NO1', 'NO5', 'NO3', 'NO4', 'NO2',
                                                              'PL', 'PT', 'RO', 'RS', 'SE1', 'SE2', 'SE3',
                                                              'SE4', 'SI', 'SK', 'UK', 'CBN', 'TYNDP', 'NSEH',
                                                              'BHEH'],
                                            "zone_number": [i for i, v in enumerate(bidding_zones)],
                                            "country": ["AL", "AT", "BA", "BE", "BG", "CH", "CZ", "DE", "DK",
                                                        "DK", "ES", "FI", "FR", "GR", "HR", "HU", "IE", "IT",
                                                        "IT", "IT", "IT", "IT", "ME", "MK", "NL", "NO", "NO",
                                                        "NO", "NO", "NO", "PL", "PT", "RO", "RS", "SE", "SE",
                                                        "SE", "SE", "SI", "SK", "UK", "CBN", "TYNDP", "NSEH",
                                                        "BHEH"]})