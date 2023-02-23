from dataclasses import dataclass, astuple

import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None
from helper_functions import Myobject, match_nearest_node, merge_timeseries_supply, fix_multiple_parallel_lines
from printing_funct import plotly_empty_map
import pickle
import requests
import json
from functools import reduce
import os
from sys import platform
import sys

class run_parameter:
    def __init__(self, scenario_name):
        #capture batch system specific parameters for running on a cluster computer
        if platform == "linux" or platform == "linux2":
            self.directory = "/work/seifert/powerinvest/"       #patch on the cluster computer
            self.case_name = sys.argv[1]                        #reading from the batch script
            self.years = int(sys.argv[2])
            self.timesteps = int(sys.argv[3])
            self.scen = int(sys.argv[4])
            self.sensitivity_scen = int(sys.argv[5])
        # local execution parameters
        elif (platform == "darwin") or (platform == "win32"):
            self.directory = ""
            self.case_name = scenario_name
            self.years = [2030]
            self.timesteps = 4000
            self.scen = 1
            self.sensitivity_scen = 0
        self.solving = False
        self.reduced_TS = False
        self.export_model_formulation = self.directory + "results/" + self.case_name + "/model_formulation_scen"+ str(self.scen) +"_subscen" + str(self.sensitivity_scen)+".mps"
        self.export_folder = self.directory + "results/" + self.case_name + "/" + str(self.scen) + "/" + "subscen" + str(self.sensitivity_scen) + "/"
        self.data_folder = self.directory + "data/"
        self.import_folder =  self.data_folder+"powergamma/"

        os.makedirs(self.export_folder, exist_ok=True)
        #
        self.hours = 504 #21 representative days
        self.scaling_factor = 8760 / self.hours


    def create_scenarios(self):
        match self.scen:
            case 1:
                self.electrolyser = []
                print("BASE case")
            case _:
                pass

        match self.sensitivity_scen:
            case 0:
                print("Base scenario sensitivity")
                self.CO2_price = {2030:80, 2035: 120, 2040:160}
                self.R_H = [108, 108, 108]
                self.grid_extension = False

        self.TRM = 0.7
        self.country_selection = ["NO", "SE", "FI", "DK"]
        bidding_zones = ['AL', 'AT', 'BA', 'BE', 'BG', 'CH', 'CZ', 'DE', 'DK1', 'DK2', 'ES', 'FI', 'FR', 'GR', 'HR','HU', 'IE', 'IT1', 'IT2', 'IT3', 'IT4', 'IT5', 'ME', 'MK', 'NL', 'NO1', 'NO5', 'NO3', 'NO4','NO2', 'PL', 'PT', 'RO', 'RS', 'SE1', 'SE2', 'SE3', 'SE4', 'SI', 'SK', 'UK', 'CBN', 'TYNDP','NSEH', 'BHEH']
        self.bidding_zones_overview = pd.DataFrame({"bidding zones": ['AL', 'AT', 'BA', 'BE', 'BG', 'CH', 'CZ', 'DE', 'DK1','DK2', 'ES', 'FI', 'FR', 'GR', 'HR', 'HU', 'IE', 'IT1','IT2', 'IT3', 'IT4', 'IT5', 'ME', 'MK', 'NL', 'NO1','NO5', 'NO3', 'NO4', 'NO2', 'PL', 'PT', 'RO', 'RS','SE1', 'SE2', 'SE3', 'SE4', 'SI', 'SK', 'UK', 'CBN','TYNDP', 'NSEH', 'BHEH'],
                                               "zone_number": [i for i, v in enumerate(bidding_zones)],
                                               "country": ["AL", "AT", "BA", "BE", "BG", "CH", "CZ", "DE", "DK", "DK","ES", "FI", "FR", "GR", "HR", "HU", "IE", "IT", "IT", "IT","IT", "IT", "ME", "MK", "NL", "NO", "NO", "NO", "NO", "NO","PL", "PT", "RO", "RS", "SE", "SE", "SE", "SE", "SI", "SK","UK", "CBN", "TYNDP", "NSEH", "BHEH"]})
        self.bidding_zone_selection=self.bidding_zones_overview.query('country in @self.country_selection')["bidding zones"].to_list()

class model_data:
    def __init__(self, create_res,reduced_ts, export_files, run_parameter):
        self.CO2_price = run_parameter.CO2_price
        #reading in the files
        if create_res:
            busses_raw =  pd.read_csv(run_parameter.import_folder + "grid_nordelNew_bus.csv", sep=";")
            generators_raw = pd.read_csv(run_parameter.import_folder + "grid_nordelNew_gen.csv", sep=";")
            coordinates= pd.read_csv(run_parameter.import_folder + "nordel_coordinates.csv", index_col=0)
            load_entsoe_ts = pd.read_csv(run_parameter.import_folder+ "entsoe_demand_2019.csv", index_col=0).reset_index(drop=True)
            lines_raw = pd.read_csv(run_parameter.import_folder + "grid_nordelNew_branch.csv", sep=";")
            dam_ts = pd.read_csv(run_parameter.import_folder + "/timeseries/hydro_dam_ts.csv", low_memory=False)
            ror_ts = pd.read_csv(run_parameter.import_folder + "/timeseries/hydro_ror_ts.csv", low_memory=False)
            open_hydro_database = pd.read_csv(run_parameter.import_folder + "jrc-hydro-power-plant-database.csv")

            # cleaning the nodes dataframe

            #attaching the coordinates
            busses_coordinates = busses_raw.merge(coordinates[["lat", "lon"]], how = "left", left_on = "bus_id", right_index=True)
            busses_filtered = busses_coordinates[["bus_id", "Pd","area", "lat", "lon"]]
            busses_filtered.columns = ["bus_id", "load_snapshot","country", "LAT", "LON"]
            self.nodes = busses_filtered

            # resolve bidding zones in NO and SE
            if (("NO") or ("SE") or ("DK")) in run_parameter.country_selection:
                self.resolve_bidding_zones()
            else:
                self.nodes["bidding_zone"] = self.nodes["country"]

            #cleaning the conventional plants
            generators_raw = generators_raw[["bus_id", "Pmax", "Pmin", "Gtype", "MC", "Start_up"]]
            generators_matched = generators_raw.merge(self.nodes.reset_index(), how="left", left_on="bus_id", right_on="bus_id")
            generators_matched = generators_matched[["index", "country", "Pmax", "Pmin", "Gtype", "MC", "Start_up", "bidding_zone"]]
            generators_matched.columns = ["node", "country", "P_inst", "P_min", "carrier", "mc","ramp_up", "bidding_zone"]
            conventionals_filtered = generators_matched[generators_matched["carrier"].isin(["gas", "hard_coal", "oil", "Nuclear", "not", "Renew_other_than_wind"])]
            conventionals = conventionals_filtered.reset_index(drop=True)
            conventionals["node"] = conventionals["node"].astype(int)

            lines_matched = lines_raw.merge(self.nodes.reset_index()[["index", "bus_id"]], how="left", left_on="bus_from",right_on="bus_id")
            lines_matched = lines_matched.merge(self.nodes.reset_index()[["index", "bus_id"]], how="left", left_on="bus_to",right_on="bus_id")
            lines_filtered = lines_matched[lines_matched['index_x'].notnull()]
            lines_filtered = lines_filtered[lines_filtered['index_y'].notnull()]

            lines = lines_filtered[["rate_b", "x", "index_x", "index_y"]].reset_index(drop=True)
            lines.columns = ["pmax", "x", "from", "to"]
            lines = lines[lines["pmax"]>0.1]
            lines["from"] = lines["from"].astype(int)
            lines["to"] = lines["to"].astype(int)

            lines["max"] = lines["pmax"] * run_parameter.TRM
            #lines_DC["max"] = lines_DC["pmax"] * run_parameter.TRM
            lines = self.find_duplicate_lines(lines)
            self.ac_lines = lines
            #self.dc_lines = lines_DC
            #plotly_empty_map(nodes=self.nodes, ac_lines=self.ac_lines, dc_lines=pd.DataFrame(), folder=run_parameter.import_folder)
            self.ATC_capacities = self.interzonal_lines(lines=self.ac_lines)

            # load TYNDP values
            self.tyndp_values(run_parameter=run_parameter)

            # new demand
            #self.demand = demand_columns(self.nodes, load_raw, self.tyndp_load)

            load_entsoe_ts.columns = load_entsoe_ts.columns.str.replace("_", "")
            self.scaling_demand(load_entsoe_ts = load_entsoe_ts, years = run_parameter.years)

            # get new renewables
            self.renewables_mapping(run_parameter = run_parameter, query_ts = False, export_files = export_files)
            #self.res_series, self.share_solar, self.share_wind = new_res_mapping(self, old_solar=solar_filtered, old_wind=wind_filtered, create_res_mapping=create_res, location = run_parameter.import_folder, query_ts=False)


            hydro_selection = open_hydro_database.query("country_code in @run_parameter.country_selection")[["installed_capacity_MW", "type", "country_code", "lat", "lon", "storage_capacity_MWh"]]
            hydro_selection.sort_values("country_code", inplace=True)
            hydro_nodes = np.empty(0, dtype=int)
            for country in hydro_selection["country_code"].unique():
                hydro_nodes = np.append(hydro_nodes, match_nearest_node(search_lat=hydro_selection.query("country_code == @country")["lat"].to_numpy(),
                                                          search_lon=hydro_selection.query("country_code == @country")["lon"].to_numpy(),
                                                          bus_lat=self.nodes.query("country == @country")["LAT"].to_numpy(),
                                                          bus_lon=self.nodes.query("country == @country")["LON"].to_numpy(),
                                                          bus_index=self.nodes.query("country == @country").index.to_numpy()))
            hydro_selection["node"] = hydro_nodes
            hydro_selection = hydro_selection.merge(self.nodes["bidding_zone"], left_on="node", right_index=True)

            # Hydro reservoir
            default_storage_capacity = 1000  # MWh
            dam = hydro_selection[hydro_selection["type"] == "HDAM"]
            dam_grouped = dam.groupby(["node"]).sum(numeric_only =True)[["installed_capacity_MW"]].reset_index()
            reservoir = dam_grouped.merge(self.nodes[["country", "bidding_zone"]], left_on = "node", right_index = True).rename(columns = {"installed_capacity_MW":"P_inst"})
            reservoir[["mc", "carrier", "P_min"]] = 3, "HDAM", 0
            self.reservoir = reservoir
            def clear_dam_ts(ts_raw, bz_selection):
                target_year = ts_raw[ts_raw["y"] == 2018.0]
                filtered = target_year.drop(["y", "t", "technology"], axis=1).reset_index(drop=True)
                filtered.columns = filtered.columns.map(lambda x: x.replace('00', '').replace("DKW1", "DK").replace('0', ''))
                droped_SE_DE = filtered.drop(columns=["DE", "SE"]).rename(columns={"DELU": "DE"})
                cleared_ts = droped_SE_DE[droped_SE_DE.columns.intersection(bz_selection)]
                cleared_ts[["DK1", "DK2", "FI"]] = 0
                return cleared_ts
            self.reservoir_zonal_limit = clear_dam_ts(dam_ts, run_parameter.bidding_zone_selection)


            # RoR
            ror = hydro_selection[hydro_selection["type"] == "HROR"]
            ror = ror.drop(["storage_capacity_MWh"], axis=1)
            ror_aggregated = ror.groupby("node").sum(numeric_only=True)[["installed_capacity_MW"]].merge(self.nodes[["country", "bidding_zone"]], left_index = True, right_index = True).rename(columns = {"installed_capacity_MW":"P_inst", "type":"carrier"})
            def clear_hydro_ts(ts_raw, countries):
                target_year = ts_raw[ts_raw["y"] == 2018.0]
                filtered = target_year.drop(["y", "t", "technology"], axis=1).reset_index(drop=True)
                filtered.columns = filtered.columns.map(lambda x: x.replace('00', '').replace("DELU", "DE").replace("DKW1", "DK"))
                cleared_ts = filtered[filtered.columns.intersection(countries)]
                norway = filtered[filtered.columns.intersection(["NO1", "NO2", "NO3", "NO4", "NO5"])]
                return pd.concat([cleared_ts, norway], axis=1)
            ror_ts = clear_hydro_ts(ror_ts, run_parameter.country_selection)
            ror_ts_T = ror_ts.T
            ror_bus_ts_matrix_nSE = ror_aggregated[~ror_aggregated["country"].isin(["SE"])].merge(ror_ts_T, how="left", left_on="bidding_zone", right_index=True).drop(["P_inst", 'bidding_zone', "country"], axis=1).T
            ror_bus_ts_matrix_SE = ror_aggregated[ror_aggregated["country"].isin(["SE"])].merge(ror_ts_T, how="left", left_on="country", right_index=True).drop(["P_inst", 'bidding_zone', "country"], axis=1).T
            ror_bus_ts_matrix = pd.concat([ror_bus_ts_matrix_nSE, ror_bus_ts_matrix_SE], axis=1)
            ror_bus_ts_matrix_np = ror_bus_ts_matrix.to_numpy()
            P_max = ror_aggregated["P_inst"].to_numpy()
            ror_bus_ts_np = np.multiply(ror_bus_ts_matrix_np, P_max)
            self.ror_series = pd.DataFrame(ror_bus_ts_np, columns=ror_aggregated.index)

            # PHS
            PHS = hydro_selection[hydro_selection['type'] == 'HPHS']
            PHS["storage_capacity_MWh"] = PHS["storage_capacity_MWh"].fillna(default_storage_capacity)
            self.storage = PHS.rename(columns={'installed_capacity_MW': 'P_inst', 'storage_capacity_MWh': 'capacity', "type": "carrier"}).reset_index(drop=True)

            self.dispatchable_generators = pd.concat([conventionals, reservoir], axis=0).reset_index(drop=True)[["carrier", "mc","P_inst","P_min", "node", "bidding_zone"]]
            self.reservoir_zonal_limit = self.reservoir_zonal_limit.sum()
            self.dispatchable_generators = self.conv_scaling_country_specific(run_parameter=run_parameter)


            if reduced_ts:
                try:
                    u = pd.read_csv(run_parameter.data_folder + "poncelet/u_result.csv", index_col=0)
                    u_index = u.index[u["value"] == 1.0].to_list()
                    self.timesteps_reduced_ts = 24*len(u_index)
                except:
                    sys.exit("need to run poncelet algorithm first!")
                self.res_series = {year: self.reduce_timeseries(self.res_series[year], u_index) for year in run_parameter.years}
                self.demand = {year:self.reduce_timeseries(self.demand[year], u_index) for year in run_parameter.years}
                self.share_solar = {year:self.reduce_timeseries(self.share_solar[year], u_index) for year in run_parameter.years}
                self.share_wind = {year:self.reduce_timeseries(self.share_wind[year], u_index)for year in run_parameter.years}
                self.ror_series = self.reduce_timeseries(self.ror_series, u_index)
                self.reservoir_zonal_limit  = self.reduce_timeseries(self.reservoir_zonal_limit, u_index)

            if export_files:
                self.nodes.to_csv(run_parameter.export_folder + "nodes.csv")
                self.storage.to_csv(run_parameter.export_folder + "storage.csv")
                self.ac_lines.to_csv(run_parameter.export_folder + "lines.csv")
                # self.dc_lines.to_csv(run_parameter.export_folder + "lines_DC.csv")
                self.ror_series.to_csv(run_parameter.export_folder + "ror_supply.csv")
                self.reservoir.to_csv(run_parameter.export_folder + "reservoir.csv")
                self.reservoir_zonal_limit.to_csv(run_parameter.export_folder + "reservoir_zonal_limit.csv")
                with open(run_parameter.export_folder + 'powerplants.pkl', 'wb+') as f:
                    pickle.dump(self.dispatchable_generators, f)
                with open(run_parameter.export_folder + 'demand.pkl', 'wb+') as f:
                    pickle.dump(self.demand, f)

        else:
            try:
                with open(run_parameter.export_folder + 'P_max.pkl', 'rb') as f:
                    self.res_series = pickle.load(f)
                with open(run_parameter.export_folder + 'share_solar.pkl', 'rb') as f:
                    self.share_solar = pickle.load(f)
                with open(run_parameter.export_folder + 'share_wind.pkl', 'rb') as f:
                    self.share_wind = pickle.load(f)
                self.nodes = pd.read_csv(run_parameter.export_folder + "nodes.csv")
                self.storage = pd.read_csv(run_parameter.export_folder + "storage.csv")
                self.ac_lines = pd.read_csv(run_parameter.export_folder + "lines.csv")
                # self.dc_lines.to_csv(run_parameter.export_folder + "lines_DC.csv")
                self.reservoir = pd.read_csv(run_parameter.export_folder + "reservoir.csv")
                self.reservoir_zonal_limit = pd.read_csv(run_parameter.export_folder + "reservoir_zonal_limit.csv")
                self.ror_series = pd.read_csv(run_parameter.export_folder + "ror_supply.csv")
                with open(run_parameter.export_folder + 'powerplants.pkl', 'rb') as f:
                    self.dispatchable_generators = pickle.load(f)
                with open(run_parameter.export_folder + 'demand.pkl', 'rb') as f:
                    self.demand = pickle.load(f)
            except:
                raise Exception("run the data generation first!")

        # Netzausbau
        if run_parameter.grid_extension:
            self.extend_overloaded_lines(type="AC", case_name = run_parameter.case_name)
            self.extend_overloaded_lines(type="DC", case_name = run_parameter.case_name)

    def interzonal_lines(self, lines):
        merged_lines = lines.merge(self.nodes["bidding_zone"], left_on="from", right_index=True).merge(self.nodes["bidding_zone"], left_on="to", right_index=True)
        filtered_lines = merged_lines.groupby(["bidding_zone_x", "bidding_zone_y"]).sum()[["pmax", "max"]].reset_index()
        single_entries = self.find_duplicate_lines(filtered_lines, ["bidding_zone_x", "bidding_zone_y"], False)
        filtered = single_entries.query('bidding_zone_x != bidding_zone_y')
    def find_duplicate_lines(self, lines, columns=["from", "to"], fix_reactance=True):
        #get rid of multiple lines in the same columns
        grouped_lines = lines.groupby(columns).size()
        grouped_lines = grouped_lines[grouped_lines>1]
        for index, count in grouped_lines.items():
                duplicate_index = lines[(lines[columns[0]] == index[0]) &  (lines[columns[1]] == index[1])].index
                lines = fix_multiple_parallel_lines(duplicate_index, columns, lines, fix_reactance)
        single_lines_same_order = lines.sort_index().reset_index(drop=True)

        #get rid of multiple lines in the other columns
        grouped_lines_oo = pd.concat([single_lines_same_order, single_lines_same_order.rename(columns= {columns[1]:columns[0], columns[0]:columns[1]})]).groupby([columns[0], columns[1]]).size()
        grouped_lines_oo = grouped_lines_oo[grouped_lines_oo>1]

        for index,count in grouped_lines_oo.items():
            duplicate_index = single_lines_same_order[((single_lines_same_order[columns[0]] == index[0]) & (single_lines_same_order[columns[1]] == index[1])) | ((single_lines_same_order[columns[1]] == index[0]) & (single_lines_same_order[columns[0]] == index[1]))].index
            single_lines_same_order = fix_multiple_parallel_lines(duplicate_index, columns, single_lines_same_order, fix_reactance)
            #grouped_lines_oo.drop([index[::-1]], inplace=True)
        single_lines_oo = single_lines_same_order.sort_index().reset_index(drop=True)
        return single_lines_oo


    def renewables_mapping(self, run_parameter, query_ts, export_files):
        def scaling_logic(df_to_scale, tyndp_target, current_value, bz, type, nodes):
            if tyndp_target == 0:
                df_to_scale.loc[(df_to_scale['bidding_zone'].isin(bz)) & (df_to_scale['type'] == type), "P_inst"] *= 0
            elif (current_value == 0) & (len(nodes[nodes["bidding_zone"].isin(bz)]) != 0):
                number_entries = nodes[nodes["bidding_zone"].isin(bz)].count()[
                    0]
                df_to_scale = pd.concat([df_to_scale, pd.DataFrame({"type": type, "node": nodes[nodes["bidding_zone"].isin(bz)].index, "P_inst": [tyndp_target/number_entries for i in range(0, number_entries)], "bidding_zone": [bz[0] for i in range(0, number_entries)]})])
            elif (len(nodes[nodes["bidding_zone"].isin(bz)]) == 0):
                pass
            else:
                factor = tyndp_target / current_value
                df_to_scale.loc[(df_to_scale['bidding_zone'].isin(bz)) & (df_to_scale['type'] == type), "P_inst"] *= factor
            return df_to_scale

        def scaling(df, type, year):
            df_bz = df.merge(self.nodes["bidding_zone"], left_on="node", right_index=True)
            df_scaled = df_bz.copy()
            try:
                df_bz_grouped = df_bz.groupby(["bidding_zone", "type"]).sum().reset_index()
            except:
                df_bz_grouped = 0
            if type == "wind":
                for bz in self.tyndp_installed_capacity.index.get_level_values(0).unique():
                    if bz == "NO1":
                        tyndp_zone_offshore = self.tyndp_installed_capacity.query("Node == @bz & Fuel == 'Offshore'")[year].sum()
                        tyndp_zone_onshore = self.tyndp_installed_capacity.query("Node == @bz & Fuel == 'Onshore'")[year].sum()
                        bz = ["NO1", "NO2", "NO5"]
                    else:
                        try:
                            tyndp_zone_offshore = self.tyndp_installed_capacity.loc[bz].query("Fuel == 'Offshore'")[year].sum()
                        except:
                            tyndp_zone_offshore = 0
                        if tyndp_zone_offshore < 0:
                            tyndp_zone_offshore = 0
                        try:
                            tyndp_zone_onshore = self.tyndp_installed_capacity.loc[bz].query("Fuel == 'Onshore'")[year].sum()
                        except:
                            tyndp_zone_onshore = 0
                        bz = [bz]
                    inst_onwind_capacity_bz = df_bz_grouped.query("bidding_zone == @bz & type == 'Onshore'")["P_inst"].sum()
                    inst_offwind_capacity_bz = df_bz_grouped.query("bidding_zone == @bz & type == 'Offshore'")["P_inst"].sum()
                    df_scaled = scaling_logic(df_to_scale=df_scaled, tyndp_target=tyndp_zone_onshore, current_value=inst_onwind_capacity_bz, type="Onshore", bz=bz, nodes=self.nodes)
                    df_scaled = scaling_logic(df_to_scale=df_scaled, tyndp_target=tyndp_zone_offshore, current_value=inst_offwind_capacity_bz, type="Offshore", bz=bz, nodes=self.nodes)
            if type == "solar":
                for bz in self.tyndp_installed_capacity.index.get_level_values(0).unique():
                    tyndp_zone = self.tyndp_installed_capacity.query("Node == @bz & Fuel == @type")[year].sum()
                    if bz == "NO1":
                        tyndp_zone = self.tyndp_installed_capacity.query("Node == @bz & Fuel == @type")[year].sum()
                        bz = ["NO1", "NO2", "NO5"]
                    else:
                        bz = [bz]
                    inst_capacity_bz = df_bz_grouped.query("bidding_zone == @bz")["P_inst"].sum()
                    df_scaled = scaling_logic(df_to_scale=df_scaled, tyndp_target=tyndp_zone,current_value=inst_capacity_bz, type="solar", bz=bz, nodes=self.nodes)
            df_scaled = df_scaled.sort_index()
            return df_scaled

        def res_multiplication(solar_capacity,wind_capacity, solar_ts, wind_ts):
            solar_supply = merge_timeseries_supply(solar_capacity, solar_ts)
            wind_supply = merge_timeseries_supply(wind_capacity, wind_ts)
            #if (solar_supply.isnull().values.any()) or (wind_supply.isnull().values.any()):
            #    raise Exception("Sorry, nas in the wind and solar supply df. Pls. fix!")
            renewables_full = wind_supply.add(solar_supply, fill_value = 0)
            share_solar = solar_supply.div(renewables_full, fill_value = 0)
            share_wind = wind_supply.div(renewables_full, fill_value=0)

            return renewables_full, share_solar, share_wind


        res_open_data = pd.read_csv(run_parameter.import_folder+ "renewable_power_plants_EU.csv", low_memory=False)
        res_od_filtered = res_open_data[res_open_data["country"].isin(run_parameter.country_selection)]
        res_od_filtered = res_od_filtered.rename( columns={"electrical_capacity": "P_inst", "technology": "type", "lat": "LAT", "lon": "LON"})
        res_od_filtered = res_od_filtered[["country", "type", "P_inst", "LAT", "LON"]]
        res_od_filtered["type"] = res_od_filtered["type"].replace({"Photovoltaics": "solar"})

        # wind
        wind_od = res_od_filtered[res_od_filtered["type"].isin(["Onshore", "Offshore"])]
        wind_od.dropna(axis="index", inplace=True)

        # solar
        solar_od = res_od_filtered[res_od_filtered["type"].isin(["Photovoltaics"])]
        solar_od.dropna(axis="index", inplace=True)

        #aggregation into nearest node
        wind_nodes = np.empty(0, dtype=int)
        solar_nodes = np.empty(0, dtype=int)
        for country in wind_od["country"].unique():
            wind_nodes = np.append(wind_nodes, match_nearest_node(search_lat=wind_od.query("country == @country")["LAT"].to_numpy(),search_lon=wind_od.query("country == @country")["LON"].to_numpy(),bus_lat=self.nodes.query("country == @country")["LAT"].to_numpy(), bus_lon=self.nodes.query("country == @country")["LON"].to_numpy(),bus_index=self.nodes.query("country == @country").index.to_numpy()))
        for country in solar_od["country"].unique():
            solar_nodes = np.append(solar_nodes, match_nearest_node(search_lat=solar_od.query("country == @country")["LAT"].to_numpy(),search_lon=solar_od.query("country == @country")["LON"].to_numpy(),bus_lat=self.nodes.query("country == @country")["LAT"].to_numpy(), bus_lon=self.nodes.query("country == @country")["LON"].to_numpy(),bus_index=self.nodes.query("country == @country").index.to_numpy()))

        wind_od["node"]=wind_nodes
        solar_od["node"]=solar_nodes
        wind_od_agg = wind_od.merge(self.nodes["bidding_zone"], left_on="node", right_index=True).groupby(["type", "node"]).sum(numeric_only=True)[["P_inst"]].reset_index()
        solar_od_agg = solar_od.merge(self.nodes["bidding_zone"], left_on = "node", right_index = True).groupby(["type", "node"]).sum(numeric_only=True)[["P_inst"]].reset_index()

        wind_scaled = {y:scaling(df = wind_od_agg, type ="wind", year =y) for y in run_parameter.years}
        solar_scaled = {y:scaling(df= solar_od_agg, type="solar", year=y) for y in run_parameter.years}

        if query_ts == True:
            os.makedirs(run_parameter.import_folder + "timeseries/", exist_ok=True)
            wind_query_set = wind_scaled[run_parameter.years[0]].groupby("node").sum(numeric_only= True).reset_index()
            wind_query_set = wind_query_set.merge(self.nodes[["LON", "LAT"]], how="left", left_on="node",right_index=True)
            wind_query_set.to_csv(run_parameter.import_folder + "timeseries/node_wind.csv")
            solar_query_set = solar_scaled[run_parameter.years[0]].groupby("node").sum(numeric_only=True).reset_index()
            solar_query_set = solar_query_set.merge(self.nodes[["LON", "LAT"]], how="left", left_on="node",right_index=True)
            solar_query_set.to_csv(run_parameter.import_folder + "timeseries/node_pv.csv")
        # read the ts
        wind_ts = pd.read_csv(run_parameter.import_folder + "timeseries/res_ninja_wind_ts.csv", index_col=0)
        solar_ts = pd.read_csv(run_parameter.import_folder  + "timeseries/res_ninja_pv_ts.csv", index_col=0)

        os.makedirs(run_parameter.import_folder + "calculated_res/", exist_ok=True)

        self.res_series = {}
        self.share_solar = {}
        self.share_wind={}
        for year in run_parameter.years:
            res_series, share_solar, share_wind = res_multiplication(solar_scaled[year],wind_scaled[year], solar_ts, wind_ts)
            self.res_series.update({year: res_series})
            self.share_solar.update({year: share_solar})
            self.share_wind.update({year: share_wind})
        if export_files:
            with open(run_parameter.export_folder + 'P_max.pkl', 'wb+') as f:
                pickle.dump(self.res_series, f)
            with open(run_parameter.export_folder + 'share_wind.pkl', 'wb+') as f:
                pickle.dump(self.share_wind, f)
            with open(run_parameter.export_folder + 'share_solar.pkl', 'wb+') as f:
                pickle.dump(self.share_solar, f)
            #self.res_series = {year: pd.read_csv(run_parameter.export_folder+ 'calculated_res/renewables_full_' + str(year) + '.csv', index_col=0) for year in run_parameter.years}
            #self.share_solar= {year: pd.read_csv(run_parameter.import_folder + 'calculated_res/share_solar_' + str(year) + '.csv', index_col=0) for year in run_parameter.years}
            #self.share_solar = {year: pd.read_csv(run_parameter.import_folder + 'calculated_res/share_solar_' + str(year) + '.csv', index_col=0) for year in run_parameter.years}



    def scaling_demand(self, load_entsoe_ts, years):
        #load = MWh
        #TYNDP = GWh
        def yearly_scaling(load_spatial, load_entsoe_ts, tyndp_demand, year):
            load_entsoe_ts["RU"]=0
            load_entsoe_sum = load_entsoe_ts.sum()
            spatial_sum = load_spatial.groupby("bidding_zone").sum(numeric_only=True)["load_snapshot"]
            def attach_factors(x, attachement, NO_aggregation=False):
                if NO_aggregation:
                    if x["bidding_zone"] in ["NO2", "NO5"]:
                        factor = attachement["NO1"]
                    elif x["bidding_zone"] in ["RU"]:
                        factor = 1
                    else: factor = attachement[x["bidding_zone"]]
                else:
                    factor = attachement[x["bidding_zone"]]
                return factor
            load_spatial["spatial_sum"] = load_spatial.apply(lambda x: attach_factors(x, spatial_sum), axis=1)
            load_spatial["spatial_factor"]= load_spatial["load_snapshot"]/load_spatial["spatial_sum"]
            try:
                load_entsoe_sum["NO1"] = load_entsoe_sum["NO1"] + load_entsoe_sum["NO2"] + load_entsoe_sum["NO5"]
                load_entsoe_sum.drop(["NO2", "NO5"], inplace=True)
            except: pass
            scaling_factor_ts = (tyndp_demand[year]*1000)/load_entsoe_sum
            load_spatial["temporal_factor"] = load_spatial.apply(lambda x: attach_factors(x, scaling_factor_ts, NO_aggregation=True), axis=1)
            load_spatial["factor_all"] = load_spatial["temporal_factor"].multiply(load_spatial["spatial_factor"], fill_value=0)

            scaled_demand = pd.DataFrame()
            for node in load_spatial.index:
                scaled_demand = pd.concat([scaled_demand, load_spatial.loc[node]["factor_all"]*load_entsoe_ts[load_spatial.loc[node]["bidding_zone"]]], ignore_index=False, axis=1)
            scaled_demand.columns = load_spatial.index
            return scaled_demand
        demand_yearly = {year: yearly_scaling(load_spatial= self.nodes, load_entsoe_ts = load_entsoe_ts, tyndp_demand=self.tyndp_demand, year=year) for year in years}
        self.demand = demand_yearly
        return

    def conv_scaling_country_specific(self, run_parameter):
        #dict to rename conventional carriers to fit the tyndp categories
        carrier_matching = {"gas":"Gas", "hard_coal":"Coal & Lignite", "Nuclear":"Nuclear", "oil":"Oil", "Renew_other_than_wind":"Other RES"}

        conventional_h20 = self.dispatchable_generators[self.dispatchable_generators["carrier"].isin(["HDAM"])]
        conventional_fossil = self.dispatchable_generators[~self.dispatchable_generators["carrier"].isin(["HDAM"])]
        conventional_fossil["carrier"].replace(carrier_matching, inplace = True)

        conventional_fossil_grouped = conventional_fossil.groupby(["bidding_zone", "carrier"]).sum()["P_inst"]
        tyndp_installed_capacity = self.tyndp_installed_capacity

        def get_conventional_yearly(tyndp_values, df_2020_capacity_bz, df_2020_capacity_bz_grouped, conventional_h20, year, CO2_price):
            df_scaled_capacity = df_2020_capacity_bz.copy()
            bidding_zones = list(set(df_2020_capacity_bz["bidding_zone"].unique()) - {"NO2", "NO5"})
            technology = df_scaled_capacity["carrier"].unique()
            for tech in technology:
                for bz in bidding_zones:
                    if bz == "NO1":
                        try:
                            tyndp_zone = tyndp_values.loc[bz, tech][year]
                        except:
                            tyndp_zone = 0
                        try:
                            inst_capacity_bz = df_2020_capacity_bz_grouped.loc[tech, ["NO1", "NO2", "NO5"]].sum()
                        except:
                            inst_capacity_bz = 0
                        bz = ["NO1", "NO2", "NO5"]
                    else:
                        try:
                            tyndp_zone = tyndp_values.loc[bz, tech][year]
                        except:
                            tyndp_zone = 0
                        try:
                            inst_capacity_bz = df_2020_capacity_bz_grouped.loc[tech, bz]["P_inst"]
                        except:
                            inst_capacity_bz = 0
                        bz = [bz]

                    if tyndp_zone == 0: # Gleichverteiltung über alle relavanten Nodes -> kann verbessert werden wieder mit den Potentialen
                        df_scaled_capacity.query('bidding_zone in @bz & carrier == @tech')["P_inst"] = 0
                    elif (tyndp_zone != 0) & (inst_capacity_bz != 0):
                        factor_total = tyndp_zone / inst_capacity_bz
                        df_scaled_capacity.query('bidding_zone in @bz & carrier == @tech')["P_inst"]*= factor_total
                        #df_scaled_capacity.loc[df_scaled_capacity['bidding_zone'].isin(bz), "P_inst"] *= factor_total
                    elif (tyndp_zone != 0) & (inst_capacity_bz == 0):
                        number_entries = df_scaled_capacity.query('bidding_zone in @bz').count()[1]
                        df_scaled_capacity.query('bidding_zone in @bz')["P_inst"]= tyndp_zone / number_entries
            #df_scaled_capacity["mc"] += df_scaled_capacity["co2_fac"] / df_scaled_capacity["efficiency"] * CO2_price[year]
            df_scaled_capacity = pd.concat([df_scaled_capacity, conventional_h20])
            df_scaled_capacity = df_scaled_capacity.reset_index()
            return df_scaled_capacity
        dispatchable_generators = {year:get_conventional_yearly(tyndp_values=tyndp_installed_capacity, df_2020_capacity_bz = conventional_fossil, df_2020_capacity_bz_grouped = conventional_fossil_grouped, conventional_h20= conventional_h20, year = year, CO2_price = self.CO2_price) for year in run_parameter.years}
        return dispatchable_generators

    def tyndp_values(self, run_parameter, filename_capacity = "Updated_Electricity_Modelling_Results.xlsx", filename_demand ="Updated_Electricity_Modelling_Results.xlsx", scenario = "Global Ambition"):
        df_installed_capacity = pd.read_excel(run_parameter.data_folder+ "/TYNDP/"+filename_capacity, sheet_name="Capacity & Dispatch")
        df_demand = pd.read_excel(run_parameter.data_folder+ "/TYNDP/"+filename_demand, sheet_name="Demand")
        ## Supply
        #data curation
        #include the market demand from Hydrogen
        df_installed_capacity["Hydrogen"] = df_installed_capacity["Node/Line"].str.contains("H2R4")
        #remove other scenarios
        df_installed_capacity["Special"] = df_installed_capacity["Node/Line"].str.contains("EV2W|H2C1|H2C2|H2C3|H2R1|H2C5|HER4|H2MT")
        df_installed_capacity["Node"] = df_installed_capacity["Node"].str.split("00", expand=True)[0]
        #replace the zones
        df_installed_capacity["Node"] = df_installed_capacity["Node"].replace({"DKE1": "DK1", "DKW1":"DK2", "NOM1":"NO3", "NON1":"NO4", "NOS0":"NO1", "SE01":"SE1", "SE02":"SE2", "SE03":"SE3", "SE04":"SE4"})
        df_installed_capacity["Fuel"] = df_installed_capacity["Fuel"].replace({"Wind Offshore": "Offshore", "Wind Onshore": "Onshore"})
        df_installed_capacity["Fuel"] = df_installed_capacity["Fuel"].replace({"Solar": "solar"})

        tyndp_installed_capacity= df_installed_capacity.query('Node in @run_parameter.bidding_zone_selection and Scenario == @scenario and `Climate Year` == "CY 2009" and Parameter == "Capacity (MW)" and Special == False')
        tyndp_installed_capacity = tyndp_installed_capacity.groupby(["Node", "Year", "Fuel"]).sum(numeric_only = False)["Value"].reset_index(["Year"])
        tyndp_installed_capacity = tyndp_installed_capacity.pivot(columns ="Year", values ="Value")

        tyndp_installed_capacity[2035] = (tyndp_installed_capacity[2030] + tyndp_installed_capacity[2040]) / 2
        self.tyndp_installed_capacity=tyndp_installed_capacity

        ##Demand
        # data curation
        # include the market demand from Hydrogen
        df_demand["Hydrogen"] = df_demand["Node/Line"].str.contains("H2R4")
        # remove other scenarios
        df_demand["Special"] = df_demand["Node/Line"].str.contains("EV2W|H2C1|H2C2|H2C3|H2R1|H2C5|HER4|H2MT")
        df_demand["Node"] = df_demand["Node"].str.split("00", expand=True)[0]
        # replace the zones
        df_demand["Node"] = df_demand["Node"].replace({"DKE1": "DK1", "DKW1": "DK2", "NOM1": "NO3", "NON1": "NO4", "NOS0": "NO1", "SE01": "SE1", "SE02": "SE2","SE03": "SE3", "SE04": "SE4"})
        tyndp_demand= df_demand.query('Node in @run_parameter.bidding_zone_selection and Scenario == @scenario and `Climate Year` == "CY 2009" and Special == False')
        tyndp_demand = tyndp_demand.groupby(["Node", "Year"]).sum(numeric_only = False)["Value"].reset_index(["Year"])
        tyndp_demand = tyndp_demand.pivot(columns="Year", values="Value")
        tyndp_demand[2035] = (tyndp_demand[2030] + tyndp_demand[2040])/2
        self.tyndp_demand = tyndp_demand

    def reduce_timeseries(self, long_ts, u_index):
        short_ts = pd.DataFrame()
        for index in u_index:
            current_day = long_ts.loc[index*24:index*24+23]
            short_ts = pd.concat([short_ts,current_day])
        return short_ts.reset_index(drop=True)

    def resolve_bidding_zones(self):
        try:
            import geopandas as gpd
            scandinavian_bidding_zones = gpd.read_file("data/shapes/scandinavian_bidding_zones.geojson").set_index("bidding_zone")
        except: sys.exit("Error loading bidding zone shape")
        nodes_geopandas = gpd.GeoDataFrame(self.nodes, geometry=gpd.points_from_xy(self.nodes.LON, self.nodes.LAT), crs="EPSG:4326")
        nodes_scand_bidding_zones = nodes_geopandas.query('country in ["DK", "SE", "NO"]')
        nodes_scand_bidding_zones_resolved = nodes_scand_bidding_zones.sjoin(scandinavian_bidding_zones[["geometry"]], how="left", predicate='intersects').rename(columns={"index_right":"bidding_zone"})
        if nodes_scand_bidding_zones_resolved['bidding_zone'].isna().sum()>=1:
            missing = nodes_scand_bidding_zones_resolved.loc[pd.isna(nodes_scand_bidding_zones_resolved ["bidding_zone"]), :].index
            print("not all nodes are matched! " + str(len(missing))+ " are missing")
            print(missing.values)

            # add the missing values
            nodes_scand_bidding_zones_resolved.at[8, "bidding_zone"] = "SE1"

        nodes_other_bidding_zone = nodes_geopandas[~nodes_geopandas["country"].isin(scandinavian_bidding_zones["country"])]
        nodes_other_bidding_zone["bidding_zone"] = nodes_other_bidding_zone["country"]
        self.nodes= pd.concat([nodes_scand_bidding_zones_resolved, nodes_other_bidding_zone]).drop(columns="geometry").sort_index()

    def extend_overloaded_lines(self, type, case_name):
        #{index, hours_with_overload_in_3_years}
        # base scenario, subscenario 3
        try:
            overloaded_AC_lines = pd.read_csv("results/"+case_name+"/1/subscen0/overloaded_lines_AC.csv", index_col=0)["full_load_h"].to_dict()
            overloaded_DC_lines = pd.read_csv("results/" + case_name + "/1/subscen0/overloaded_lines_DC.csv", index_col=0)["full_load_h"].to_dict()
        except: sys.exit("need to run scenario 1 first!")
        #overloaded_AC_lines = {51: 1121, 249: 1313, 315: 1342, 363: 1354, 397: 1341, 408: 1114, 488: 1319, 489: 1345, 494: 1473, 497: 1204, 530: 1182, 550: 1368, 563: 1175, 600: 1330, 624: 1503, 631: 1273, 646: 1393, 679: 1075, 782: 1165}
        #overloaded_DC_lines ={0: 1277, 1: 1188, 2: 1137, 3: 1466, 4: 1402, 6: 1113, 7: 1360, 8: 1285, 9: 1318, 10: 1320, 11: 1483, 12: 1401, 13: 1195, 14: 1461, 15: 1398, 16: 1183, 17: 1506, 18: 1397, 21: 1415, 22: 1250, 23: 1136, 24: 1188, 25: 1377, 26: 1330, 31: 1201, 35: 1369, 37: 1105, 38: 1316, 41: 1314, 42: 1343}
        #overloaded_AC_lines = {33: 1124, 51: 1095, 137: 1154, 249: 1324, 315: 1450, 363: 1364, 397: 1371, 408: 1067, 488: 1306, 489: 1259, 494: 1472, 497: 1316, 530: 1103, 550: 1372, 563: 1064, 600: 1469, 606: 1076, 624: 1512, 631: 1293, 646: 1263, 679: 1087, 782: 1174}
        #overloaded_DC_lines = {0: 1269, 1: 1227, 2: 1166, 3: 1403, 4: 1340, 6: 1153, 7: 1241, 9: 1330, 10: 1325, 11: 1468, 12: 1384, 13: 1196, 14: 1473, 15: 1384, 16: 1216, 17: 1510, 18: 1423, 21: 1393, 24: 1150, 25: 1450, 26: 1401, 29: 1406, 32: 1443, 36: 1164}
        match type:
            case "AC":
                for key in self.ac_lines.index:
                    if key in overloaded_AC_lines.keys():
                        self.ac_lines.loc[key,"max"] = self.ac_lines.loc[key,"max"]*(1+0.2*overloaded_AC_lines[key]/(self.number_years*self.timesteps_reduced_ts))
            case "DC":
                for key  in self.dc_lines.index:
                    if key in overloaded_DC_lines.keys():
                        self.dc_lines.loc[key, "max"] = self.dc_lines.loc[key, "max"] * (1+0.2 * overloaded_DC_lines[key]/(self.number_years * self.timesteps_reduced_ts))

class kpi_data:
    def __init__(self, run_parameter, scen):
        self.run_parameter = run_parameter
        self.run_parameter.years=range(0,run_parameter.years)
        years = self.run_parameter.years

        read_folder = run_parameter.read_folder = run_parameter.directory + "results/" + run_parameter.case_name + "/" + str(scen) + "/subscen" +str(run_parameter.sensitivity_scen) + "/"
        self.bus = pd.read_csv(read_folder + "busses.csv", index_col=0)

        #create empty objects
        self.load_factor = Myobject()
        self.P_R = Myobject()
        self.P_R.max = Myobject()
        self.curtailment = Myobject()
        self.line_loading= Myobject()
        self.line_balance = Myobject()
        with open(read_folder + 'powerplants.pkl', 'rb') as f:
            powerplants_raw = pickle.load(f)
            #powerplants_raw = self.change_column_to_int(powerplants_raw)
        with open(read_folder + 'P_max.pkl', 'rb') as f:
            self.P_R.max.raw = pickle.load(f)
            self.P_R.max.raw = self.change_column_to_int(self.P_R.max.raw)
        with open(read_folder + 'demand.pkl', 'rb') as f:
            demand_raw = pickle.load(f)
            demand_raw = self.change_column_to_int(demand_raw)
        with open(read_folder + 'share_solar.pkl', 'rb') as f:
            share_solar_raw = pickle.load(f)
            share_solar_raw = self.change_column_to_int(share_solar_raw)
        with open(read_folder + 'share_wind.pkl', 'rb') as f:
            share_wind_raw = pickle.load(f)
            share_wind_raw = self.change_column_to_int(share_wind_raw)

        bus_raw = self.read_in(y = "", string =  "busses.csv", int_convert=False)
        # overwrite the wind cluster bus country and bidding zone
        bus_raw.loc[524:, ["country", "bidding_zone"]] = bus_raw.loc[524:, ["country", "bidding_zone"]].apply(lambda x: x+"_wind_cluster")

        storage = self.read_in(y = "", string = "storage.csv", int_convert=False)
        lines_overview = self.read_in(y = "", string = "lines.csv", int_convert=False)
        lines_DC_overview = self.read_in(y = "", string = "lines_DC.csv", int_convert=False)
        ror_supply = self.read_in(y = "", string = "ror_supply.csv")
        CAP_lines = self.read_in(y = "", string = "CAP_BH.csv")
        self.CAP_lines = CAP_lines.T.merge(lines_DC_overview[["from", "to","EI"]],left_index=True, right_index=True).merge(bus_raw[["LON", "LAT"]], left_on ="from", right_index=True).merge(bus_raw[["LON", "LAT"]], left_on ="to", right_index=True).sort_index()
        #encyc_powerplants_bus = create_encyclopedia(powerplants_raw[0]["bus"])
        #encyc_storage_bus = create_encyclopedia(storage["bus"])
        if scen!= 1:
            self.CAP_E =self.read_in(y = "", string ="CAP_E.csv").transpose().merge(run_parameter.electrolyser[scen][["name", "bus"]], left_index=True, right_index=True).merge(bus_raw[["LON", "LAT"]], left_on ="bus", right_index=True).set_index("name")

        self.F_AC = {y: self.read_in(y = y, string = "_F_AC.csv") for y in years}
        self.timesteps = self.F_AC[0].shape[0]
        self.F_DC = {y: self.read_in(y = y, string =  "_F_DC.csv")for y in years}
        self.EI_trade = {y: self.EI_connections(lines_DC_overview = lines_DC_overview, bus_overview=bus_raw, year = y)for y in years}
        self.P_R.raw = {y: self.read_in(y = y, string =  "_P_R.csv") for y in years}
        self.P_DAM = {y: self.read_in(y = y, string = "_P_DAM.csv") for y in years}
        self.res_curtailment = {y: pd.read_csv(read_folder + str(y)+ "_res_curtailment.csv", index_col=0, names = self.P_R.raw[y].columns, header= 0) for y in years}
        self.P_C ={y: self.read_in(y = y, string =  "_P_C.csv") for y in years}
        self.P_S = {y: self.read_in(y = y, string =  "_P_S.csv")for y in years}
        self.L_S = {y: self.read_in(y = y, string =  "_L_S.csv")for y in years}
        self.C_S = {y: self.read_in(y = y, string = "_C_S.csv") for y in years}
        self.P_loss_load = {y: self.read_in(y = y, string = "_p_load_lost.csv") for y in years}
        if scen != 1:
            self.P_H = {y: self.read_in(y = y, string = "_P_H.csv").transpose().merge(run_parameter.electrolyser[scen]["name"], left_index=True, right_index=True).set_index("name").T for y in years}


        #calculations
        if scen != 1: self.load_factor.elect = pd.DataFrame({y: (self.P_H[y] / self.CAP_E[y]).mean() for y in years})
        self.P_R.bz = {y: self.prepare_results_files_bz(self.P_R.raw[y], bus_raw) for y in years}
        self.P_R.max.bz = {y: self.prepare_results_files_bz(self.P_R.max.raw[y], bus_raw) for y in years}
        self.P_R.solar = {y: share_solar_raw[y].multiply(self.P_R.raw[y]).dropna(axis=1, how = 'all') for y in years}
        self.P_R.wind = {y: (share_wind_raw[y] * self.P_R.raw[y]).dropna(axis=1, how='all') for y in years}
        self.zonal_trade_balance = {y: self.zonal_trade_balance_function(self.F_AC[y], self.F_DC[y], bus_raw, lines_overview, lines_DC_overview, self.run_parameter.scaling_factor) for y in years}


        ## curtailment
        self.curtailment.raw = {y: self.res_curtailment[y] for y in years}
        self.curtailment.bz = {y: self.prepare_results_files_bz(self.curtailment.raw[y], bus_raw) for y in years}
        self.curtailment.bz_relative = {y: pd.DataFrame(self.curtailment.bz[y][0] / (self.P_R.max.bz[y][0]))for y in years}

        ## electricity sources
        self.generation_temporal = {y: self.prepare_results_files_index_temporal(y = y, ror_supply = ror_supply, index_file = powerplants_raw[y], scen= scen) for y in years}

        ## line loading
        self.line_loading.AC = {y: self.prepare_results_files_lines(y = y, file=self.F_AC, bus_raw=bus_raw, index_file=lines_overview, yearly=False, full_load_tolerance=0.01) for y in years}
        self.line_loading.DC = {y: self.prepare_results_files_lines(y = y, file=self.F_DC, bus_raw=bus_raw,index_file=lines_DC_overview, yearly=True, CAP_BH=self.CAP_lines, full_load_tolerance=0.01) for y in years}
        self.line_loading.AC.update({"avg": (pd.concat([self.line_loading.AC[year][[0, "full_load_h"]] for year in run_parameter.years]).groupby(level=0).sum()/len(run_parameter.years)).merge(lines_overview[["from", "to"]], left_index = True, right_index = True).merge(bus_raw[["LAT","LON"]], left_on = "from", right_index =True).merge(bus_raw[["LAT","LON"]], left_on = "to", right_index =True).sort_index(ascending=True)})
        #self.line_loading.AC.update({"avg": (sum(self.line_loading.AC[year][0] for year in run_parameter.years)/len(run_parameter.years)).to_frame().merge(lines_overview[["from", "to"]], left_index = True, right_index = True).merge(bus_raw[["LAT","LON"]], left_on = "from", right_index =True).merge(bus_raw[["LAT","LON"]], left_on = "to", right_index =True).sort_index(ascending=True)})
        self.line_loading.DC.update({"avg": (pd.concat([self.line_loading.DC[year][[0, "full_load_h"]] for year in run_parameter.years]).groupby(level=0).sum()/len(run_parameter.years)).merge(lines_DC_overview[["from", "to"]], left_index = True, right_index = True).merge(bus_raw[["LAT","LON"]], left_on = "from", right_index =True).merge(bus_raw[["LAT","LON"]], left_on = "to", right_index =True).sort_index(ascending=True)})
        #self.line_loading.DC.update({"avg": (sum(self.line_loading.DC[year][0] for year in run_parameter.years)/len(run_parameter.years)).to_frame().merge(lines_DC_overview[["from", "to"]], left_index = True, right_index = True).merge(bus_raw[["LAT","LON"]], left_on = "from", right_index =True).merge(bus_raw[["LAT","LON"]], left_on = "to", right_index =True).sort_index(ascending=True)})


        self.line_balance.AC = {y: self.get_trade_balance_yearly(file=self.F_AC[y], bus_raw=bus_raw, index_file=lines_overview)for y in years}
        self.line_balance.DC = {y: self.get_trade_balance_yearly(file=self.F_DC[y], bus_raw=bus_raw, index_file=lines_DC_overview) for y in years}
        self.trade_balance_bz = {y: self.trade_balance(self.line_balance.AC[y], self.line_balance.DC[y]) for y in years}

        # P_R
        self.P_R.nodal_sum = reduce(lambda x, y: x.add(y), list(self.P_R.raw[y] for y in years))
        self.P_R.total_nodes = self.prepare_results_files_nodes(self.P_R.nodal_sum, bus_raw, temporal=0)
        self.P_R.total_bz = self.prepare_results_files_bz(self.P_R.nodal_sum, bus_raw)
        # total curtailments

        self.curtailment.bz_sum = reduce(lambda x, y: x.add(y), list(self.curtailment.bz[y] for y in years))
        self.P_R.max.bz_sum = reduce(lambda x, y: x.add(y), list(self.P_R.max.bz[y] for y in years))
        self.curtailment.sum = reduce(lambda x, y: x.add(y), list(self.curtailment.raw[y] for y in years))
        self.curtailment.bz_relative_sum = pd.DataFrame(self.curtailment.bz_sum[0] / self.P_R.max.bz_sum[0]).rename({0: "relative"}, axis=1)
        self.curtailment.location_sum = self.prepare_results_files_nodes(self.curtailment.sum, bus_raw, temporal=0)
        self.curtailment.location = self.dataframe_creator(run_parameter = run_parameter, dict = self.curtailment.raw, bus_raw = bus_raw)

        # further calculations
        ##overloaded lines -> > 70% load über die ganze periode, base case
        if (run_parameter.sensitivity_scen == 0) & (scen == 1):
            try:
                overloaded_AC = self.line_loading.AC["avg"][self.line_loading.AC["avg"]["full_load_h"] >= 0.7 * 504]["full_load_h"]
                overloaded_AC = overloaded_AC * 3
                overloaded_AC.to_csv(run_parameter.export_folder + str(1) +"/subscen" + str(run_parameter.sensitivity_scen) + "/overloaded_lines_AC.csv")
                overloaded_DC = self.line_loading.DC["avg"][self.line_loading.DC["avg"]["full_load_h"] >= 0.7 * 504]["full_load_h"]
                overloaded_DC = overloaded_DC * 3
                overloaded_DC.to_csv(run_parameter.export_folder + str(1) + "/subscen" + str(run_parameter.sensitivity_scen) + "/overloaded_lines_DC.csv")
            except:pass


    def dataframe_creator(self,run_parameter, dict, bus_raw):
        df = pd.DataFrame({year: dict[year].sum(axis=0) for year in run_parameter.years}).replace(0, np.nan).dropna(
            axis=0).merge(bus_raw[["LON", "LAT", "country", "bidding_zone"]], left_index=True, right_index=True)
        return df
    def read_in(self, y, string, int_convert = True):
        if isinstance(y, str):
            data = pd.read_csv(self.run_parameter.read_folder + string, index_col=0)
        else:
            data = pd.read_csv(self.run_parameter.read_folder + str(y) + string, index_col=0)
        if int_convert:
            data.columns = data.columns.astype(int)
        return data

    def prepare_results_files_nodes(self, file, bus_raw, temporal):
        file_w_na= file.dropna(axis = 1)
        file_without_0 = file_w_na.loc[(file_w_na != 0).any(axis=1)]
        file_sum = file_without_0.dropna(how='all').sum(axis=temporal)
        file_sum.index = file_sum.index.astype(int)
        if temporal == 0:
            file_frame = file_sum.to_frame()
            file_ready = file_frame.merge(bus_raw[["LAT", "LON"]], how="left", left_index=True, right_index=True)
            return file_ready
        return file_sum

    @dataclass
    class temporal_generation:
        hydro: float = 0.0
        oil: float = 0.0
        gas: float = 0.0
        coal: float = 0.0
        lignite: float= 0.0
        nuclear: float = 0.0
        biomass: float = 0.0
        other: float = 0.0
        wind: float = 0.0
        solar: float = 0.0
        P_S: float = 0.0
        C_S: float = 0.0
        curtailment: float = 0.0
        electrolyser: float = 0.0
        def __iter__(self):
            return iter(astuple(self))
        def ts_conventional(self):
            conventionals = self.oil+self.gas+ self.coal+self.lignite + self.nuclear+self.other
            return conventionals
        def to_df(self):
            df = pd.DataFrame()
            for object in self:
                df = pd.concat([df, object], axis=1)
            try:
                df.columns = ["hydro", "oil", "gas", "coal", "lignite","nuclear", "biomass", "other", "wind", "solar", "P_S", "C_S", "curtailment", "electrolyser"]
            except: df.columns = ["hydro", "oil", "gas", "coal", "lignite", "nuclear", "biomass", "other", "wind", "solar", "P_S", "C_S", "curtailment"]
            return df

    def prepare_results_files_index_temporal(self, y, ror_supply, index_file, scen):
        types = {"hydro": ['HDAM'], "oil": ['oil'], "gas": ['CCGT', 'OCGT'],"coal" : ['coal'],"lignite":["lignite"], "nuclear": ['nuclear'], "biomass": ['biomass'], "other": ["other"]}
        def get_sum_of_type(file, index_file, type):
            index_list = index_file[index_file["type"].isin(type)].index.values
            plants_in_group = file[file.columns.intersection(index_list)]
            sum_timestep = plants_in_group.sum(axis=1)
            return sum_timestep

        conventional_without_0 = self.P_C[y].loc[:, (self.P_C[y] != 0).any(axis=0)]
        conventional_without_0.columns = conventional_without_0.columns.astype(int)
        dam_sum = self.P_DAM[y].sum(axis=1)
        ror_sum = ror_supply.sum(axis=1)

        generation = self.temporal_generation(
            hydro=dam_sum + ror_sum + get_sum_of_type(conventional_without_0, index_file, types["hydro"]),
            oil=get_sum_of_type(conventional_without_0, index_file, types["oil"]),
            gas=get_sum_of_type(conventional_without_0, index_file, types["gas"]),
            coal=get_sum_of_type(conventional_without_0, index_file, types["coal"]),
            lignite = get_sum_of_type(conventional_without_0, index_file, types["lignite"]),
            nuclear=get_sum_of_type(conventional_without_0, index_file, types["nuclear"]),
            other=get_sum_of_type(conventional_without_0, index_file, types["other"]),
            biomass = get_sum_of_type(conventional_without_0, index_file, types["biomass"]),
            wind=self.P_R.wind[y].sum(axis=1),
            solar=self.P_R.solar[y].sum(axis=1),
            P_S=self.P_S[y].sum(axis=1),
            C_S=self.C_S[y].sum(axis=1),
            curtailment=self.res_curtailment[y].sum(axis=1),
            electrolyser=self.P_H[y].sum(axis=1) if scen != 1 else pd.DataFrame()
            )
        return generation
    def EI_connections(self, lines_DC_overview, bus_overview, year):
        #data_ei = self.F_DC[year].iloc[:, -self.run_parameter.number_flexlines:].T
        transposed = self.F_DC[year].T
        index_EI_lines = lines_DC_overview[lines_DC_overview["EI"].isin([0,1,2])].index
        data_ei = transposed[transposed.index.isin(index_EI_lines)]
        data_ei.index = data_ei.index.astype(int)
        data_ei_matched = data_ei.merge(lines_DC_overview[["EI"]], how="left", left_index=True, right_index=True)
        trade_to_bz = {}
        self.run_parameter.create_scenarios()
        for EI in self.run_parameter.EI_bus.index:
            data_ei_individual = data_ei_matched[data_ei_matched["EI"] == EI]
            data_ei_from_bus = data_ei_individual.merge(lines_DC_overview[["from"]], how="left", left_index=True,right_index=True).set_index("from")
            data_ei_from_country = data_ei_from_bus.merge(bus_overview["country"], how="left", left_index=True,right_index=True).set_index("country")
            aggregated_trade = data_ei_from_country.groupby("country", axis=0).sum(numeric_only = True)
            trade_to_bz.update({EI: aggregated_trade.iloc[:, :self.timesteps]})
        return trade_to_bz

    def change_column_to_int(self, item):
        for y in self.run_parameter.years:
            item[y].columns = item[y].columns.astype(int)
        return item
    def prepare_results_files_bz(self, file, bus_raw):
        file= file.dropna()
        file_without_0 = file.loc[(file != 0).any(axis=1)]
        file_sum = file_without_0.dropna(how='all').sum(axis=0)
        file_sum.index = file_sum.index.astype(int)
        file_frame = file_sum.to_frame()
        file_ready = file_frame.merge(bus_raw[["country"]], how="left", left_index=True, right_index=True)
        file_ready_bz = file_ready.groupby("country", sort = False).sum()#.reset_index()
        #file_ready_bz_resolved_names = file_ready_bz.merge(bidding_zones_encyclopedia, how="left", left_on="bidding zone",right_on="zone_number")[["bidding zones", 0]].set_index("bidding zones")
        return file_ready_bz
    def zonal_trade_balance_function(self, F_AC, F_DC, bus_raw, lines_overview, lines_DC_overview, scaling_factor):
        line_balance_total = self.get_trade_balance_yearly(file=F_AC, bus_raw=bus_raw, index_file=lines_overview)
        line_balance_DC_total = self.get_trade_balance_yearly(file=F_DC, bus_raw=bus_raw, index_file=lines_DC_overview)
        trade_balance_bz_total = self.trade_balance(line_balance_total, line_balance_DC_total)
        #trade_balance_bz_total  = trade_balance_bz_total.merge(bidding_zones_encyclopedia, how="left", left_on="bidding zone_from",right_on="zone_number")[["bidding zones", "bidding zone_to", 0]].rename(columns={"bidding zones": "From bidding zone"})
        trade_balance_bz_total = trade_balance_bz_total.sort_values("country_to", axis=0)
        trade_balance_bz_total[0] = trade_balance_bz_total[0] * scaling_factor
        #exports - imports -> yes that is correct in the equation -> from defines where it starts == what country exports
        zonal_trade_balance = trade_balance_bz_total.groupby("country_from").sum(numeric_only=True).sub(trade_balance_bz_total.groupby("country_to").sum(numeric_only = True), fill_value=0)
        return zonal_trade_balance
    def trade_balance(self, AC_balance, DC_balance):
        def change_direction(x):
            if x[0] <= 0:
                zwischenspeicher = x["country_x"]
                x["country_x"] = x["country_y"]
                x["country_y"] = zwischenspeicher
                x[0] = -x[0]
            return x
        #balance = AC_balance.append(DC_balance).drop(AC_balance.index[AC_balance[0] == 0].tolist()) # append DC and drop zero loadings
        balance = pd.concat([AC_balance, DC_balance]).drop(AC_balance.index[AC_balance[0] == 0].tolist())
        magnitude = balance.apply(lambda x: change_direction(x), axis=1)
        bz_balance = magnitude.groupby(["country_x", "country_y"], sort = True).sum()[0].reset_index()
        bz_balance_rename = bz_balance.rename(columns = {"country_x":"country_to", "country_y":"country_from"})
        interconnectors = bz_balance_rename[bz_balance_rename["country_to"] != bz_balance_rename["country_from"]]
        return interconnectors
    def get_trade_balance_yearly(self, file, bus_raw, index_file):
        #line loading comes in and is aggregated to the yearly node balance
        summed_bus = file.sum(axis=0).to_frame()
        summed_bus.index = summed_bus.index.astype(int)
        file_bus = summed_bus.merge(index_file[["from", "to"]], how="left", left_index=True, right_index=True)
        file_ready = file_bus.merge(bus_raw[["LAT", "LON", "country"]], how="left", left_on="from", right_index=True)
        file_ready = file_ready.merge(bus_raw[["LAT", "LON", "country"]], how="left", left_on="to", right_index=True)
        return file_ready

    def prepare_results_files_lines(self, y, file, index_file, bus_raw, yearly, full_load_tolerance, CAP_BH = ""):
        line_data = file[y].round(8)
        line_data.columns = line_data.columns.astype(int)
        if yearly:
            CAP_BH = CAP_BH.T
            for i in CAP_BH:
                index_file["max"].iat[int(i)] = CAP_BH[i][y]
        elif isinstance(CAP_BH, pd.DataFrame):
            CAP_BH.columns = CAP_BH.columns.astype(int)
            for i in CAP_BH.index:
                index_file["max"].iat[int(i)] = CAP_BH.loc[i][0]
        file_without_0 = line_data[line_data != 0].dropna(axis= 1, how="all").abs()
        max_power = index_file[index_file.index.isin(line_data.columns)]["max"]
        max_power_filtered = max_power[(max_power!=0.0)]
        relative = file_without_0.divide(max_power_filtered, axis=1)
        avg = relative.mean(axis=0).to_frame().fillna(0)
        avg["full_load_h"] = relative[relative > 1.0 - full_load_tolerance].count()
        # avg.index = avg.index.astype(int)
        file_bus = avg.merge(index_file[["pmax", "from", "to"]], how="left", left_index=True, right_index=True)
        file_ready = file_bus.merge(bus_raw[["LAT", "LON"]], how="left", left_on="from", right_index=True)
        file_ready = file_ready.merge(bus_raw[["LAT", "LON"]], how="left", left_on="to", right_index=True)
        return file_ready

class comparison_data():
    def __init__(self):
        response = requests.get("https://energy-charts.info/charts/power/data_unit/de/year_wind_offshore_unit_2019.json")
        text = response.text
        parsed = json.loads(text)
        self.data = {}
        #self.capacity =
        #test = parsed[17]["name"][0]["en"]
        #test2 = pd.Series(parsed[1]["data"])
        for i in parsed[:-3]:
            print(i)
            self.data.update({i["name"][0]["en"]: pd.Series(i["data"])})
        self.summing_of_non_nan()
    def summing_of_non_nan(self):
        status_na={}
        non_na = {}
        self.yearly_sum = {}
        for key, value in self.data.items():
            status_na.update({key: value.hasnans})
            if value.hasnans == False:
                non_na.update({key:value})
                self.yearly_sum.update({key: value.sum()/4})

class gurobi_variables:
    def __init__(self, solved_model):
        all_variables = solved_model.getVars()
        last_item = all_variables[-1].VarName.split(",")
        self.years = int(last_item[0].split("[")[1])+1
        self.timesteps = int(last_item[1])+1
        counter = len(all_variables)-1
        self.additional_columns = {}
        self.results = {}
        while counter > 0:
            current_variable = all_variables[counter].VarName
            variable_name = current_variable.split("[")[0]
            array, counter, irregular_columns, bus_column_irregular= self.get_variable_from_position(variables = all_variables, counter=counter)
            self.results.update({variable_name: array})
            if irregular_columns:
                bus_column_irregular.reverse()
                self.additional_columns.update({variable_name: bus_column_irregular})

    def get_variable_from_position(self, variables, counter):
        current_variable = variables[counter].VarName
        bus_column_irregular = []
        irregular_columns = False
        first_run = True
        if len(current_variable.split(",")) == 3:
            first_dimension = int(current_variable.split(",")[0].split("[")[1])+1
            second_dimension = int(current_variable.split(",")[1])+1
            last_dimension = int(current_variable.split(",")[-1].split("]")[0])+1
            dimension_counter = 1
            while dimension_counter < last_dimension:
                if int(variables[counter - dimension_counter].VarName.split(",")[-1].split("]")[0]) == int(variables[counter].VarName.split(",")[-1].split("]")[0]):
                    irregular_columns = True
                    break
                dimension_counter += 1
            array = np.zeros((first_dimension, second_dimension, dimension_counter))
            for first in reversed(range(first_dimension)):
                for second in reversed(range(second_dimension)):
                    for third in reversed(range(dimension_counter)):
                        array[first, second, third] = variables[counter].X
                        if first_run:
                            bus_column_irregular.append(int(variables[counter].VarName.split(",")[-1].split("]")[0]))
                        counter -= 1
                    first_run = False
        if len(current_variable.split(",")) == 2:
            first_dimension = int(current_variable.split(",")[0].split("[")[1]) + 1
            last_dimension = int(current_variable.split(",")[-1].split("]")[0]) + 1
            dimension_counter = 1
            while dimension_counter < last_dimension:
                if int(variables[counter - dimension_counter].VarName.split(",")[-1].split("]")[0]) == int(variables[counter].VarName.split(",")[-1].split("]")[0]):
                    irregular_columns = True
                    break
                dimension_counter += 1
            array = np.zeros((first_dimension, dimension_counter))
            for first in reversed(range(first_dimension)):
                for third in reversed(range(dimension_counter)):
                    array[first, third] = variables[counter].X
                    if first_run:
                        bus_column_irregular.append(int(variables[counter].VarName.split(",")[-1].split("]")[0]))
                    counter -= 1
                first_run = False
        if len(current_variable.split(",")) == 1:
            print("Error!")
        return array, counter, irregular_columns, bus_column_irregular

    def export_csv(self, folder, scen):
        os.makedirs(folder, exist_ok=True)
        # cap_BH
        pd.DataFrame(self.results["cap_BH"], columns=self.additional_columns["cap_BH"]).to_csv(folder + "cap_BH.csv")
        for y in range(self.years):
            # P_C
            pd.DataFrame(self.results["P_C"][y, :, :]).to_csv(folder + str(y) + "_P_C.csv")
            # P_R
            pd.DataFrame(self.results["P_R"][y, :, :], columns=self.additional_columns["P_R"]).to_csv(folder + str(y) + "_P_R.csv")
            # P_DAM
            pd.DataFrame(self.results["P_DAM"][y, :, :]).to_csv(folder + str(y) + "_P_DAM.csv")
            if scen in [2, 3, 4]:
                # cap_E
                pd.DataFrame(self.results["cap_E"]).to_csv(folder + "cap_E.csv")
                # P_H
                pd.DataFrame(self.results["P_H"][y, :, :]).to_csv(folder + str(y) + "_P_H.csv")
            # load lost
            pd.DataFrame(self.results["p_load_lost"][y, :, :]).to_csv(folder + str(y) + "_p_load_lost.csv")
            # res_curtailment
            pd.DataFrame(self.results["res_curtailment"][y, :, :], columns=self.additional_columns["res_curtailment"]).to_csv(folder + str(y) + "_res_curtailment.csv")
            # storage
            pd.DataFrame(self.results["P_S"][y, :, :]).to_csv(folder + str(y) + "_P_S.csv")
            pd.DataFrame(self.results["C_S"][y, :, :]).to_csv(folder + str(y) + "_C_S.csv")
            pd.DataFrame(self.results["L_S"][y, :, :]).to_csv(folder + str(y) + "_L_S.csv")
            # AC line flow
            pd.DataFrame(self.results["F_AC"][y, :, :]).to_csv(folder + str(y) + "_F_AC.csv")
            # DC line flow
            pd.DataFrame(self.results["F_DC"][y, :, :]).to_csv(folder + str(y) + "_F_DC.csv")
#class shapes:
#    def __init__(self):

