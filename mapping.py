import sys
import timeit

import numpy as np
import pandas as pd
from numba import jit

from helper_functions import merge_timeseries_supply, give_nearest_bus_relative_position, distance_calc, \
    get_wind_yearly, get_solar_yearly, res_normalisation


def hydro_mapping(bus_CM):
    location = "data/north_sea_energy_islands/"
    hydro_df = pd.read_csv(location+"jrc-hydro-power-plant-database.csv")
    # find the nearest bus for each hydro plant
    hydro_df = hydro_df.rename(columns = {"lat":"LAT", "lon": "LON"})
    country_list = ['BE', 'CZ', 'DE', 'DK', 'FI', 'NL','NO', 'PL','SE']
    hydro_df = hydro_df[hydro_df["country_code"].isin(country_list)]

    hydro_numpy = hydro_df[["LAT", "LON"]].to_numpy()
    dict_bus = bus_CM[["LAT", "LON"]].to_dict()
    bus_vector = give_nearest_bus_relative_position(bus_raw=dict_bus, hydro_numpy=hydro_numpy)
    hydro_df['bus'] = bus_vector
    hydro_df = hydro_df.drop(['id', 'dam_height_m','volume_Mm3', 'pypsa_id','GEO','WRI'], axis=1)
    hydro_df.to_csv(location+"csv/"+ 'hydro_CM_CBN.csv',index=True)
    return hydro_df
def new_res_mapping(self, old_solar, old_wind, create_res_mapping, location, query_ts):

    def create_renewables_supply(self, solar, wind, solar_ts, wind_ts):
        bus_overview = self.nodes
        solar_supply = merge_timeseries_supply(solar, solar_ts)
        wind_supply = merge_timeseries_supply(wind, wind_ts)
        renewables_full = pd.DataFrame(np.zeros((8760, len(bus_overview.index)), dtype=float), columns=bus_overview.index)
        share_solar = pd.DataFrame(np.zeros((8760, len(bus_overview.index)), dtype=float), columns=bus_overview.index)
        share_wind = pd.DataFrame(np.zeros((8760, len(bus_overview.index)), dtype=float), columns=bus_overview.index)
        def sum_if_multiple(wind, bus_nr):
            if isinstance(wind[bus_nr], pd.DataFrame):
                column = wind[bus_nr].sum(axis=1)
            else:
                column = wind[bus_nr]
            return column

        for bus in bus_overview.index:
            if bus in solar_supply.columns:
                renewables_full[bus] = renewables_full[bus] + solar_supply[bus]
                share_solar[bus] = share_solar[bus] + solar_supply[bus]
            if bus in wind_supply.columns:
                renewables_full[bus] = renewables_full[bus] + sum_if_multiple(wind_supply, bus)
                share_wind[bus] = share_wind[bus] + sum_if_multiple(wind_supply, bus)
        share_solar_relative = (share_solar.div(renewables_full)).fillna(0)
        share_wind_relative = (share_wind.div(renewables_full)).fillna(0)
        share_solar_relative = share_solar_relative[share_solar_relative.columns[(share_solar_relative != 0).any(axis=0)]]
        share_wind_relative = share_wind_relative[share_wind_relative.columns[(share_wind_relative != 0).any(axis=0)]]
        renewables_full = renewables_full[renewables_full.columns[(renewables_full != 0).any(axis=0)]]  # remove columns when only 0
        return renewables_full, share_solar_relative, share_wind_relative

    #efficient calculation of the nearest bus using numba
    @jit(nopython= True)
    def match_nearest_bus(search_lat, search_lon, bus_lat, bus_lon, bus_position_index):
        bus_vector = np.empty(search_lat.shape[0], dtype=np.int32)
        for i in range(search_lat.shape[0]):
            distance_vector = np.empty(bus_lat.shape[0], dtype=np.float_)
            for j in range(bus_lat.shape[0]):
                distance_vector[j] = distance_calc(search_lat[i], search_lon[i], bus_lat[j], bus_lon[j])
            bus_vector[i] = bus_position_index[np.argmin(distance_vector)]
        return bus_vector

    def kinis_windfarms_cluster(self, location):
        windfarms = pd.read_csv(location + "additional_windfarm_cluster.csv", encoding="latin1")
        windfarms.dropna(how = "all", axis=0,inplace=True)
        windfarms= windfarms.iloc[:,1:5]
        windfarms.columns = ["country", "LAT", "LON", "P_inst"]
        windfarms["type"] = "offwind"
        windfarms["bus"] = list(range(self.nodes.index[-1]+1, self.nodes.index[-1]+ len(windfarms)+1))
        windfarms["bidding_zone"] = windfarms["country"]
        windfarms.loc[windfarms["country"] == "DK","bidding_zone"] = "DK1"
        windfarms["old_index"] = "kinis_clusters"
        windfarms_bus = windfarms.rename(columns={"bus":"index"})
        bus_with_wind_clusters = pd.concat([self.nodes, windfarms_bus[["index", "LON", "LAT", "country", "bidding_zone"]]]).reset_index(drop=True)
        self.nodes = bus_with_wind_clusters
        return windfarms[["country", "bidding_zone", "P_inst", "type", "bus"]]


    if create_res_mapping == True:
        res_open_data = pd.read_csv(location+"renewable_power_plants_EU.csv", low_memory=False)
        res_od_our_region = res_open_data[res_open_data["country"].isin(["DE", "CZ", "DK", 'PL','SE','UK'])]
        res_od_our_region= res_od_our_region.rename(columns = {"electrical_capacity":"P_inst","technology": "type", "lat":"LAT", "lon": "LON"})
        res_od_our_region = res_od_our_region[["country", "type","P_inst","LAT", "LON"]]

        #wind
        # pypsa for CZ, DE, DK, SE, UK, "FI", "BE", "NL", "NO", "PL", "NSEH1", "NSEH2", "BHEH"
        pypsa_wind_filtered = old_wind[old_wind["country"].isin(["FI", "BE", "NL", "NO", "PL", "CZ", "DE", "DK", "SE", "UK"])]
        pypsa_wind_filtered = pypsa_wind_filtered[["type", "country", "bidding_zone", "bus", "max"]].reset_index(drop=True)

        # normalise the potentials to 2020 values
        pypsa_wind_normalised = res_normalisation(self=self, df = pypsa_wind_filtered, type = "wind")


        #attach the Offshore energy islands
        EI_wind = old_wind[old_wind["country"].isin(["NSEH1", "NSEH2", "BHEH"])]
        EI_wind.rename(columns={"max":"P_inst"}, inplace=True)
        offshore_windfarm_clustered= kinis_windfarms_cluster(self = self, location=location)
        wind_concat = pd.concat([pypsa_wind_normalised, EI_wind])
        wind_df = wind_concat.groupby(["type", "bus"]).sum(numeric_only = True)["P_inst"].reset_index()


        if query_ts == True:
            wind_query_set = pd.concat([wind_df, offshore_windfarm_clustered]).groupby("bus").sum().reset_index()
            wind_query_set = wind_query_set.merge(self.nodes[["LON", "LAT"]], how="left", left_on="bus",right_index=True)
            wind_query_set.to_csv(location +"timeseries/bus_wind.csv")

        #solar
        #DE, DK, UK
        solar_od = res_od_our_region[res_od_our_region["type"] == 'Photovoltaics']
        #solar_od = solar_od[:10000]
        od_solar_possible_nodes = self.nodes[self.nodes["country"].isin(solar_od["country"].unique())]
        starttime = timeit.default_timer()
        solar_od["bus"] = match_nearest_bus(search_lat=solar_od["LAT"].to_numpy(), search_lon=solar_od["LON"].to_numpy(), bus_lat=od_solar_possible_nodes["LAT"].to_numpy(), bus_lon=od_solar_possible_nodes["LON"].to_numpy(), bus_position_index=od_solar_possible_nodes.index.to_numpy() )
        solar_od = solar_od.merge(self.nodes["bidding_zone"], left_on = "bus", right_index = True)
        print("The time difference is :", timeit.default_timer() - starttime)

        # BE, CZ, FI, NL, NO, PL, SE
        pypsa_solar_filtered = old_solar[old_solar["country"].isin(["FI", "BE", "NL","SE", "DK", "PL", "NO", "CZ"])]
        pypsa_solar_filtered = pypsa_solar_filtered[["bus", "country", "bidding_zone","max"]].reset_index(drop=True)
        pypsa_solar_normalised = res_normalisation(self=self, df = pypsa_solar_filtered, type = "solar")


        #EI_solar = old_solar[old_solar["country"].isin(["NSEH1", "NSEH2", "BHEH"])]
        solar_df = pd.concat([solar_od, pypsa_solar_normalised])
        solar_df = solar_df.groupby(["bus"]).sum(numeric_only = True)["P_inst"].reset_index()
        solar_df["type"] = "solar"
        #solar_df.merge(bus_CM["country"], left_on = "bus", right_index = True).groupby("country").sum()["max"].to_csv(location+"aggregated_solar_capacity_country.csv")

        if query_ts == True:
            solar_query_set = solar_df.merge(self.nodes[["LON", "LAT"]], how="left", left_on="bus",right_index=True)
            solar_query_set.to_csv(location + "timeseries/bus_pv.csv")
            sys.exit("Please query the ts first")
        #solar_df.to_csv(location + "solar_capacity.csv")


        #read the ts
        wind_ts = pd.read_csv(location + "timeseries/res_ninja_wind_ts.csv", index_col=0)
        solar_ts = pd.read_csv(location + "timeseries/res_ninja_pv_ts.csv", index_col=0)
        #future scaling
        def scaling(self, df_2020_capacity, type, kinis_offshore_windfarms = ""):
            df_2020_capacity_bz = df_2020_capacity.merge(self.nodes["bidding_zone"], left_on="bus", right_index=True)
            df_2020_capacity_bz_grouped = df_2020_capacity_bz.groupby(["bidding_zone", "type"]).sum().reset_index()
            if type == "wind":
                df_2020_capacity_bz_type_grouped= df_2020_capacity_bz.groupby(["bidding_zone", "type"]).sum().reset_index()
                kinis_offshore_windfarms_grouped = kinis_offshore_windfarms.groupby(["bidding_zone"]).sum(numeric_only = True)["P_inst"]
                scaled = {y: get_wind_yearly(tyndp_values=self.tyndp_installed_capacity,df_2020_capacity_bz=df_2020_capacity_bz, year=year, kinis_offshore_windfarms = kinis_offshore_windfarms_grouped, df_2020_capacity_bz_type_grouped = df_2020_capacity_bz_type_grouped) for year, y in zip([2030, 2035, 2040], [0, 1, 2])}
            else: #solar
                scaled = {y: get_solar_yearly(tyndp_values = self.tyndp_installed_capacity,df_2020_capacity_bz = df_2020_capacity_bz, df_2020_capacity_bz_grouped = df_2020_capacity_bz_grouped,year = year, type = type) for year, y in zip([2030, 2035, 2040], [0, 1, 2])}
            return scaled

        ##solar
        solar_after_scaling =scaling(self = self, df_2020_capacity = solar_df, type = "solar")
        ##wind
        wind_after_scaling = scaling(self = self, df_2020_capacity = wind_df, type = "wind", kinis_offshore_windfarms= offshore_windfarm_clustered)

        #attach the windfarm clusters to the scaled dataset
        wind_after_merging = {i:pd.concat([wind_after_scaling[i], offshore_windfarm_clustered]).reset_index(drop=True) for i in [0,1,2]}
        for year in [0,1,2]:
            renewables_full, share_solar, share_wind = create_renewables_supply(self, solar_after_scaling[year], wind_after_merging[year], solar_ts, wind_ts)
            renewables_full.to_csv(location+ 'calculated_res/renewables_full_'+str(year)+'.csv',index=True)
            share_solar.to_csv(location +'calculated_res/share_solar_'+str(year)+'.csv', index=True)
            share_wind.to_csv(location +'calculated_res/share_wind_'+str(year)+'.csv', index=True)
    renewables_full = {year: pd.read_csv(location + 'calculated_res/renewables_full_'+str(year)+'.csv', index_col=0) for year in [0,1,2]}
    share_solar={year: pd.read_csv(location  + 'calculated_res/share_solar_'+str(year)+'.csv', index_col=0) for year in [0,1,2]}
    share_wind={year: pd.read_csv(location + 'calculated_res/share_wind_'+str(year)+'.csv', index_col=0) for year in [0,1,2]}
    for year in [0, 1, 2]:
        renewables_full[year].columns = renewables_full[year].columns.astype(int)

    return renewables_full, share_solar, share_wind