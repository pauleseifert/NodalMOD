from dataclasses import dataclass,astuple
from printing_funct import plotly_maps_bubbles, plotly_maps_lines_colorless

import pandas as pd
import numpy as np
import sys
pd.options.mode.chained_assignment = None
from helper_functions import demand_columns, give_nearest_bus_relative_position, Myobject
from mapping import new_res_mapping
import pickle
from helper_functions import create_encyclopedia
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
            self.sensitivit_scen = int(sys.argv[5])
        # local execution parameters
        elif (platform == "darwin") or (platform == "win32"):
            self.directory = ""
            self.case_name = scenario_name
            self.years = 3
            self.timesteps = 10
            self.scen = 1
            self.sensitivit_scen = 0
        self.solving = False
        self.reduced_TS = False
        self.export_model_formulation = self.directory + "results/" + self.case_name + "/model_formulation_scen"+ str(self.scen) +"_subscen" + str(self.sensitivit_scen)+".mps"
        self.export_folder = self.directory + "results/" + self.case_name + "/" + str(self.scen) + "/" + "subscen" + str(self.sensitivit_scen) + "/"
        self.import_folder = self.directory + "data/"
        os.makedirs(self.export_folder, exist_ok=True)

    def create_scenarios(self):
        match self.scen:
            case 1:
                self.electrolyser = []
                print("BASE case")
            case 2:
                self.electrolyser = pd.DataFrame({
                    "name": ["electrolyser Bornholm", "electrolyser_NS1", "electrolyser_NS2"],
                    "bus": [521, 522, 523],
                    "cost": [645000, 645000, 645000]})
                print("EI case")
            case 3:
                self.electrolyser = pd.DataFrame({
                    "name": ["electrolyser Bornholm", "electrolyser_NS1", "electrolyser_NS2", "e1", "e2", "e3", "e4", "e5",
                             "e6", "e7", "e8", "e9", "e10", "e11", "e12", "e13", "e14"],
                    "bus": [521, 522, 523, 403, 212, 209, 170, 376, 357, 279, 103, 24, 357, 62, 467, 218, 513],
                    "cost": [645000, 645000, 645000, 450000, 450000, 450000, 450000, 450000, 450000, 450000, 450000, 450000,
                             450000, 450000, 450000, 450000, 450000]
                })
                print("COMBI case")
            case 4:
                self.electrolyser = pd.DataFrame({
                    "name": ["electrolyser Bornholm", "electrolyser_NS1", "electrolyser_NS2", "e1", "e2", "e3", "e4", "e5",
                             "e6", "e7", "e8", "e9", "e10", "e11", "e12", "e13", "e14"],
                    "bus": [521, 522, 523, 403, 212, 209, 170, 376, 357, 279, 103, 24, 357, 62, 467, 218, 513],
                    "cost": [645000, 645000, 645000, 450000, 450000, 450000, 450000, 450000, 450000, 450000, 450000, 450000,
                             450000, 450000, 450000, 450000, 450000]})
                print("Stakeholder case")

        match self.sensitivit_scen:
            case 0:
                print("Base scenario sensitivity")
                self.CO2_price = [80, 120, 160]
                self.R_H = [108, 108, 108]
                self.grid_extension = False
            case 1:
                print("Low H2 price subscen")
                self.CO2_price = [80, 120, 160]
                self.R_H = [81, 81, 81]
                self.grid_extension = False
            case 2:
                print("High H2 price subscen")
                self.CO2_price = [80, 120, 160]
                self.R_H = [135, 135, 135]
                self.grid_extension = False
            case 3:
                print("High CO2 price subscen")
                self.CO2_price = [130, 250, 480]
                self.R_H = [108, 108, 108]
                self.grid_extension = False
            case 4:
                print("Grid extension")
                self.CO2_price = [80, 120, 160]
                self.R_H = [108, 108, 108]
                self.grid_extension = True
        self.add_future_windcluster = True
        self.EI_bus = pd.DataFrame([
            {"country": "BHEH", "y": 55.13615337829421, "x": 14.898639089359104},
            {"country": "NSEH1", "y": 55.22300, "x": 3.78700},
            {"country": "NSEH2", "y": 55.69354, "x": 3.97940}], index=["BHEH", "NSEH1", "NSEH2"])
        self.EI_capacity = pd.DataFrame([
            {"p_nom_max": 3000, "bus": "BHEH", "carrier": "offwind-dc"},
            {"p_nom_max": 10000, "bus": "NSEH1", "carrier": "offwind-dc"},
            {"p_nom_max": 10000, "bus": "NSEH2", "carrier": "offwind-dc"}])
        self.added_DC_lines = pd.DataFrame(
            {"p_nom": [1400, 2000, 2000, 700], "length": [720, 267, 400, 300], "index_x": [299, 198, 170, 513],
             "index_y": [419, 111, 93, 116], "tags": [
                "North Sea Link 2021: https://tyndp2020-project-platform.azurewebsites.net/projectsheets/transmission/110",
                "hvdc corridor norGer to WesGer 1034: https://tyndp2020-project-platform.azurewebsites.net/projectsheets/transmission/1034",
                "hvdc corridor norGer to WesGer 1034: https://tyndp2020-project-platform.azurewebsites.net/projectsheets/transmission/1034",
                "Hansa Power Bridge 1 https://tyndp2020-project-platform.azurewebsites.net/projectsheets/transmission/176"]})
        self.added_AC_lines = pd.DataFrame(
            {"s_nom": [6000.0,1000.0,500.0,1500.0,300.0,400.0,500.0,1500.0,1000.0,3200.0,900.0], "length": [100.0, 40.0, 237.5, 182.0, 27.0, 46.0, 125.0, 95.0, 175.0, 60.0, 200.0], "x":[24.6, 9.84, 58.425, 44.772, 6.642, 11.316, 30.75, 23.37, 43.05, 14.76, 49.2], "index_x": [0, 26, 28, 85, 119, 142, 170, 180, 225, 303, 490],
             "index_y": [8, 138, 30, 119, 364, 217, 191, 198, 238, 327, 505]})
        self.flexlines_EI = pd.DataFrame(
            {"Pmax": [500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500],
             "from": [523, 523, 523, 523, 523, 523, 523, 522, 522, 522, 522, 522, 522, 521, 521, 521, 521],
             "to": [522, 403, 212, 209, 170, 376, 357, 279, 170, 103, 24, 357, 376, 62, 467, 218, 513],
             "EI": [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0]})

        self.TRM = 0.7
        self.country_selection = ['BE', 'CZ', 'DE', 'DK', 'FI', 'NL', 'NO', 'PL', 'SE', 'UK', "NSEH1", "NSEH2", "BHEH"]
        bidding_zones = ['AL', 'AT', 'BA', 'BE', 'BG', 'CH', 'CZ', 'DE', 'DK1', 'DK2', 'ES', 'FI', 'FR', 'GR', 'HR',
                         'HU', 'IE', 'IT1', 'IT2', 'IT3', 'IT4', 'IT5', 'ME', 'MK', 'NL', 'NO1', 'NO5', 'NO3', 'NO4',
                         'NO2', 'PL', 'PT', 'RO', 'RS', 'SE1', 'SE2', 'SE3', 'SE4', 'SI', 'SK', 'UK', 'CBN', 'TYNDP',
                         'NSEH', 'BHEH']
        self.bidding_zones_overview = pd.DataFrame({"bidding zones": ['AL', 'AT', 'BA', 'BE', 'BG', 'CH', 'CZ', 'DE', 'DK1','DK2', 'ES', 'FI', 'FR', 'GR', 'HR', 'HU', 'IE', 'IT1','IT2', 'IT3', 'IT4', 'IT5', 'ME', 'MK', 'NL', 'NO1','NO5', 'NO3', 'NO4', 'NO2', 'PL', 'PT', 'RO', 'RS','SE1', 'SE2', 'SE3', 'SE4', 'SI', 'SK', 'UK', 'CBN','TYNDP', 'NSEH', 'BHEH'],
                                               "zone_number": [i for i, v in enumerate(bidding_zones)],
                                               "country": ["AL", "AT", "BA", "BE", "BG", "CH", "CZ", "DE", "DK", "DK","ES", "FI", "FR", "GR", "HR", "HU", "IE", "IT", "IT", "IT","IT", "IT", "ME", "MK", "NL", "NO", "NO", "NO", "NO", "NO","PL", "PT", "RO", "RS", "SE", "SE", "SE", "SE", "SI", "SK","UK", "CBN", "TYNDP", "NSEH", "BHEH"]})


class model_data:
    def __init__(self, create_res,reduced_ts, export_files, run_parameter):
        self.CO2_price = run_parameter.CO2_price
        #reading in the files
        busses_raw = pd.read_csv(run_parameter.import_folder+ "PyPSA_elec1024/buses.csv", index_col=0)
        generators_raw = pd.read_csv(run_parameter.import_folder + "PyPSA_elec1024/generators.csv", index_col=0)
        lines_raw = pd.read_csv(run_parameter.import_folder + "PyPSA_elec1024/lines.csv", index_col=0)
        links_raw = pd.read_csv(run_parameter.import_folder + "PyPSA_elec1024/links.csv", index_col=0)
        load_raw = pd.read_csv(run_parameter.import_folder+ "PyPSA_elec1024/load.csv", index_col=0).reset_index(drop=True)
        ror_ts = pd.read_csv(run_parameter.import_folder + "PyPSA_elec1024/hydro_ror_ts.csv", low_memory=False)
        dam_maxsum_ts = pd.read_csv(run_parameter.import_folder + "PyPSA_elec1024/hydro_dam_ts.csv", low_memory=False)
        hydro_database = pd.read_csv(run_parameter.import_folder+ "jrc-hydro-power-plant-database.csv")

        # cleaning the nodes dataframe
        run_parameter.country_selection.append("GB")
        #adding the EI's
        busses_raw = pd.concat([busses_raw, run_parameter.EI_bus])
        busses_filtered = busses_raw[busses_raw["country"].isin(run_parameter.country_selection)].reset_index().reset_index()
        busses_filtered = busses_filtered.replace({"GB": "UK"})
        busses_filtered = busses_filtered[["level_0", "index", "x", "y", "country"]]
        busses_filtered.columns = ["index", "old_index", "LON", "LAT", "country"]
        self.nodes = busses_filtered

        # resolve bidding zones in NO and SE
        self.resolve_bidding_zones()

        #cleaning the conventional plants
        generators = pd.concat([generators_raw,run_parameter.EI_capacity])
        generators_matched = generators.merge(self.nodes[["index", "old_index", "country", "bidding_zone"]], how="left",left_on="bus", right_on="old_index")
        generators_filtered = generators_matched[generators_matched['index'].notnull()] #take only the ones that are in the countries we want to have
        conventionals_filtered = generators_filtered[generators_filtered["carrier"].isin(["CCGT", "OCGT", "nuclear", "biomass", "coal", "lignite", "oil"])]
        conventionals = conventionals_filtered[["p_nom", "carrier", "marginal_cost", "efficiency", "co2_fac","index", "bidding_zone"]].reset_index(drop=True)
        conventionals.columns = ["P_inst", "type", "mc","efficiency", "co2_fac", "bus", "bidding_zone"]
        conventionals["bus"] = conventionals["bus"].astype(int)

        solar_matched = generators_filtered[generators_filtered["carrier"].isin(["solar"])]
        wind_matched = generators_filtered[generators_filtered["carrier"].isin(["onwind", "offwind-ac", "offwind-dc"])]
        solar_filtered = solar_matched[["p_nom_max", "carrier", "marginal_cost", "index", "country", "bidding_zone"]].reset_index(drop=True)
        solar_filtered.columns = ["max", "type", "mc", "bus", "country", "bidding_zone"]
        solar_filtered = solar_filtered.replace({"solar": "Solar"})
        solar_filtered["bus"] = solar_filtered["bus"].astype(int)
        wind_filtered = wind_matched[["p_nom_max", "carrier", "marginal_cost", "index", "country", "bidding_zone"]].reset_index(drop=True)
        wind_filtered.columns = ["max", "type", "mc", "bus", "country", "bidding_zone"]
        wind_filtered = wind_filtered.replace({"onwind": "onwind", "offwind-ac": "offwind", "offwind-dc": "offwind"})
        wind_filtered["bus"] = wind_filtered["bus"].astype(int)

        lines_matched = lines_raw.merge(self.nodes[["index", "old_index"]], how="left", left_on="bus0",right_on="old_index")
        lines_matched = lines_matched.merge(self.nodes[["index", "old_index"]], how="left", left_on="bus1",right_on="old_index")
        lines_filtered = lines_matched[lines_matched['index_x'].notnull()]
        lines_filtered = lines_filtered[lines_filtered['index_y'].notnull()]
        # https://pypsa.readthedocs.io/en/latest/components.html?highlight=parameters#line-types
        lines_filtered["x"] = 0.246 * lines_filtered["length"]
        # add future lines from tyndp data
        lines_added_projects = pd.concat([lines_filtered, run_parameter.added_AC_lines])
        lines = lines_added_projects[["s_nom", "x", "index_x", "index_y"]].reset_index(drop=True)
        lines.columns = ["pmax", "x", "from", "to"]
        lines["from"] = lines["from"].astype(int)
        lines["to"] = lines["to"].astype(int)

        lines_DC_matched = links_raw.merge(self.nodes[["index", "old_index"]], how="left", left_on="bus0",right_on="old_index")
        lines_DC_matched = lines_DC_matched.merge(self.nodes[["index", "old_index"]], how="left", left_on="bus1",right_on="old_index")
        lines_DC_filtered = lines_DC_matched[lines_DC_matched['index_x'].notnull()]
        lines_DC_filtered = lines_DC_filtered[lines_DC_filtered['index_y'].notnull()]
        # See lines_V02.csv
        lines_DC_filtered = pd.concat([lines_DC_filtered, run_parameter.added_DC_lines])
        lines_DC = lines_DC_filtered[["p_nom", "index_x", "index_y"]].reset_index(drop=True)
        lines_DC.columns = ["pmax", "from", "to"]
        lines_DC["from"] = lines_DC["from"].astype(int)
        lines_DC["to"] = lines_DC["to"].astype(int)
        lines_DC.insert(3, "EI", 'N/A')
        lines_DC = pd.concat([lines_DC, run_parameter.flexlines_EI], ignore_index=True)

        lines["max"] = lines["pmax"] * run_parameter.TRM
        lines_DC["max"] = lines_DC["pmax"] * run_parameter.TRM
        self.ac_lines = lines
        self.dc_lines = lines_DC


        # load TYNDP values
        self.tyndp_values(path=run_parameter.import_folder, bidding_zone_encyc=run_parameter.bidding_zones_overview)

        # new demand
        self.demand = demand_columns(self.nodes, load_raw, self.tyndp_load)

        # get new renewables
        self.res_series, self.share_solar, self.share_wind = new_res_mapping(self, old_solar=solar_filtered, old_wind=wind_filtered, create_res_mapping=create_res, location = run_parameter.import_folder, query_ts=False)

        hydro_df = hydro_database.rename(columns={"lat": "LAT", "lon": "LON"})
        hydro_df = hydro_df[hydro_df["country_code"].isin(run_parameter.country_selection)]
        hydro_df = hydro_df.drop(['id', 'dam_height_m', 'volume_Mm3', 'pypsa_id', 'GEO', 'WRI', "country_code"], axis=1)
        hydro_numpy = hydro_df[["LAT", "LON"]].to_numpy()
        dict_bus = self.nodes[["LAT", "LON"]].to_dict()
        bus_vector = give_nearest_bus_relative_position(bus_raw=dict_bus, hydro_numpy=hydro_numpy)
        hydro_df['bus'] = bus_vector
        hydro_df = hydro_df.merge(self.nodes[["country", "bidding_zone"]], left_on = "bus", right_index = True)
        hydro_df = hydro_df.rename(columns={'installed_capacity_MW': 'P_inst'})


        # Hydro reservoir
        default_storage_capacity = 1000  # MWh
        dam = hydro_df[hydro_df["type"] == "HDAM"]
        dam["mc"] = 30  # Euro/MWh
        dam = dam.drop(["pumping_MW", "storage_capacity_MWh"], axis=1)
        # BE, FI have no limits on reservoir
        dam_unlimited = dam[dam["country"].isin(["BE", "FI"])]
        dam_limited = dam[~dam["country"].isin(["BE", "FI"])]
        dam_limited = dam_limited.groupby(["bus"]).sum()[["P_inst"]].reset_index()
        self.reservoir = dam_limited.merge(self.nodes[["country", "bidding_zone"]], left_on = "bus", right_index = True)
        def clear_dam_ts(ts_raw, countries):
            target_year = ts_raw[ts_raw["y"] == 2018.0]
            filtered = target_year.drop(["y", "t", "technology"], axis=1).reset_index(drop=True)
            filtered.columns = filtered.columns.map(
                lambda x: x.replace('00', '').replace("DKW1", "DK").replace('0', ''))
            droped_SE_DE = filtered.drop(columns=["DE", "SE"]).rename(columns={"DELU": "DE"})
            cleared_ts = droped_SE_DE[droped_SE_DE.columns.intersection(countries)]
            NO_SE = filtered[filtered.columns.intersection(["NO1", "NO2", "NO3", "NO4", "NO5", "SE1", "SE2", "SE3", "SE4"])]
            return pd.concat([cleared_ts, NO_SE], axis=1)
        limited_dam_ts = clear_dam_ts(dam_maxsum_ts, run_parameter.country_selection)


        # RoR
        ror = hydro_df[hydro_df["type"] == "HROR"]
        ror = ror.drop(["pumping_MW", "storage_capacity_MWh"], axis=1)
        ror_aggregated = ror.groupby("bus").sum()[["P_inst"]].merge(self.nodes[["country", "bidding_zone"]], left_index = True, right_index = True)
        def clear_hydro_ts(ts_raw, countries):
            target_year = ts_raw[ts_raw["y"] == 2018.0]
            filtered = target_year.drop(["y", "t", "technology"], axis=1).reset_index(drop=True)
            filtered.columns = filtered.columns.map(
                lambda x: x.replace('00', '').replace("DELU", "DE").replace("DKW1", "DK"))
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
        PHS = hydro_df[hydro_df['type'] == 'HPHS']
        PHS["storage_capacity_MWh"] = PHS["storage_capacity_MWh"].fillna(default_storage_capacity)
        PHS["pumping_MW"] = PHS["pumping_MW"].fillna(PHS["P_inst"])
        self.storage = PHS.rename(columns={'P_inst': 'Pmax_out', 'pumping_MW': 'Pmax_in', 'storage_capacity_MWh': 'capacity'}).reset_index(drop=True)

        self.dispatchable_generators = pd.concat([conventionals, dam_unlimited], axis=0).reset_index(drop=True)[["name","type", "mc","efficiency", "co2_fac","P_inst", "bus", "bidding_zone"]]

        #scaling
        #self.future_values()

        if reduced_ts:
            try:
                u = pd.read_csv(run_parameter.import_folder + "poncelet/u_result.csv", index_col=0)
                u_index = u.index[u["value"] == 1.0].to_list()
                self.timesteps_reduced_ts = 24*len(u_index)
            except:
                sys.exit("need to run poncelet algorithm first!")
            self.res_series = {i: self.reduce_timeseries(self.res_series[i], u_index) for i in [0,1,2]}
            self.demand = {i:self.reduce_timeseries(self.demand[i], u_index) for i in [0,1,2]}
            self.share_solar = {i:self.reduce_timeseries(self.share_solar[i], u_index) for i in [0,1,2]}
            self.share_wind = {i:self.reduce_timeseries(self.share_wind[i], u_index)for i in [0,1,2]}
            self.ror_series = self.reduce_timeseries(self.ror_series, u_index)
            self.reservoir_zonal_limit  = self.reduce_timeseries(limited_dam_ts, u_index)

        self.reservoir_zonal_limit = self.reservoir_zonal_limit.sum()
        #self.res_series = self.scaling_country_specific(self.res_series, self.scaling_res,self.nodes)
        #self.demand = self.scaling_country_specific(self.demand, self.scaling_demand, self.nodes)
        self.dispatchable_generators = self.conv_scaling_country_specific()
        if run_parameter.add_future_windcluster:
            self.add_future_windcluster(location=run_parameter.import_folder)

        # Netzausbau
        if run_parameter.grid_extension:
            self.extend_overloaded_lines(type="AC", case_name = run_parameter.case_name)
            self.extend_overloaded_lines(type="DC", case_name = run_parameter.case_name)
        if export_files:
            with open(run_parameter.export_folder + 'powerplants.pkl', 'wb+') as f:
                pickle.dump(self.dispatchable_generators, f)
            with open(run_parameter.export_folder + 'demand.pkl', 'wb+') as f:
                pickle.dump(self.demand, f)
            with open(run_parameter.export_folder + 'P_max.pkl', 'wb+') as f:
                pickle.dump(self.res_series, f)


    def conv_scaling_country_specific(self):
        conventional_h20 = self.dispatchable_generators[self.dispatchable_generators["type"].isin(["HDAM"])]
        conventional_fossil = self.dispatchable_generators[~self.dispatchable_generators["type"].isin(["HDAM"])]
        conventional_fossil_grouped = conventional_fossil.groupby(["bidding_zone", "type"]).sum()["P_inst"]

        tyndp_installed_capacity = self.tyndp_installed_capacity.reset_index()
        #ausgabe
        tyndp_installed_capacity.query('generator == "chp"')
        tyndp_installed_capacity["generator"].replace({"otherres": "biogas", "other":"gas", "ccgt":"CCGT", "ocgt": "OCGT", "gas":"OCGT"}, inplace=True)
        # split chp into ccgt,ocgt, coal
        tyndp_without_chp = tyndp_installed_capacity.query('generator != "chp"')
        chp = tyndp_installed_capacity.query('generator == "chp"')
        chp[[2020, 2030,2035, 2040]] = chp[[2020, 2030,2035, 2040]]/3
        new_chp = pd.DataFrame()
        for type in ["coal", "CCGT", "OCGT"]:
            new_entry = chp.copy()
            new_entry["generator"] = type
            new_chp = pd.concat([new_chp,new_entry])
        tyndp_installed_capacity = pd.concat([tyndp_without_chp, new_chp]).reset_index(drop = True)

        tyndp_installed_capacity_regrouped = tyndp_installed_capacity.groupby(["node", "generator"]).sum()

        def get_conventional_yearly(tyndp_values, df_2020_capacity_bz, df_2020_capacity_bz_grouped, conventional_h20, year, i, CO2_price):
            df_scaled_capacity = df_2020_capacity_bz.copy()
            bidding_zones = list(set(df_2020_capacity_bz["bidding_zone"].unique()) - {"NO2", "NO5"})
            technology = df_scaled_capacity["type"].unique()
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
                        df_scaled_capacity.query('bidding_zone in @bz & type == @tech')["P_inst"] = 0
                    elif (tyndp_zone != 0) & (inst_capacity_bz != 0):
                        factor_total = tyndp_zone / inst_capacity_bz
                        df_scaled_capacity.query('bidding_zone in @bz & type == @tech')["P_inst"]*= factor_total
                        #df_scaled_capacity.loc[df_scaled_capacity['bidding_zone'].isin(bz), "P_inst"] *= factor_total
                    elif (tyndp_zone != 0) & (inst_capacity_bz == 0):
                        number_entries = df_scaled_capacity.query('bidding_zone in @bz').count()[1]
                        df_scaled_capacity.query('bidding_zone in @bz')["P_inst"]= tyndp_zone / number_entries
                    #print("Installed " + str(tech) + " power in BZ " + str(bz) + " in " + str(year) + " = " + str(df_scaled_capacity.groupby(["bidding_zone", "type"]).sum().query('bidding_zone == @bz & type == @tech')["P_inst"].sum()) + " MW")
            # df_scaled_capacity.replace([np.inf, -np.inf], 0, inplace=True)
            df_scaled_capacity["mc"] += df_scaled_capacity["co2_fac"] / df_scaled_capacity["efficiency"] * CO2_price[i]
            df_scaled_capacity = pd.concat([df_scaled_capacity, conventional_h20])
            df_scaled_capacity = df_scaled_capacity.reset_index()
            return df_scaled_capacity
        dispatchable_generators = {i:get_conventional_yearly(tyndp_values=tyndp_installed_capacity_regrouped, df_2020_capacity_bz = conventional_fossil, df_2020_capacity_bz_grouped = conventional_fossil_grouped, conventional_h20= conventional_h20, year = year, i =i, CO2_price = self.CO2_price) for i,year in zip([0,1,2], [2030,2035,2040])}
        return dispatchable_generators
    def scaling_country_specific(self,ts, scaling_factor, bus):
        renewables_supply_new = {}
        renewables_T = ts.T
        renewables_T.index = renewables_T.index.astype(int)
        merge_country = renewables_T.merge(bus, left_index=True, right_index=True)[["country"]]
        merge_factor = merge_country.merge(scaling_factor, how="outer", left_on="country", right_index=True).fillna(
            1).drop("country", axis=1)
        renewables_supply_new.update({i: ts * merge_factor.iloc[:, i] for i in range(4)})
        return renewables_supply_new
    def future_values(self):
        open_entrance_dataset = pd.read_csv("data/PyPSA_elec1024/openentrance-v01-al-2022_05_10.csv").fillna(0).iloc[:-1, :]
        #open_entrance_demand = pd.read_csv("data/north_sea_energy_islands/openentrance_snapshot_1641145432.csv")[:-1]

        open_entrance_dataset["Region"] = open_entrance_dataset["Region"].replace(
            {"Belgium": "BE", "Czech Republic": "CZ", "Denmark": "DK", "Finland": "FI", "Germany": "DE", "Norway": "NO",
             "Poland": "PL", "Sweden": "SE", "The Netherlands": "NL", "United Kingdom": "UK"})

        #open_entrance_dataset[["type_unit", "type_energy", "type_source"]] = open_entrance_dataset['Variable'].str.split('|', 2, expand=True)

        open_entrance_split = open_entrance_dataset.drop(columns=['Type', "PathwayScenario"])
        open_entrance_split_copy = open_entrance_split.copy()
        open_entrance_split_copy["Technology"] = open_entrance_split_copy["Technology"].replace({"P_Biomass":"Biomass","CHP_Biomass_Solid_CCS":"Biomass", "CHP_Biomass_Solid":"Biomass","RES_Hydro_Large":"Hydro", "RES_Hydro_Small": "Hydro", 'RES_PV_Utility_Avg': "Solar", 'RES_PV_Utility_Opt': "Solar",
                                                                                      'Res_PV_Utility_Tracking': "Solar", "RES_Wind_Onshore_Avg": "Wind_onshore", 'RES_Wind_Onshore_Opt': "Wind_onshore",
                                                                                      'RES_Wind_Offshore_Deep':"Wind_offshore",'RES_Wind_Offshore_Transitional':"Wind_offshore", 'RES_Wind_Offshore_Shallow':"Wind_offshore",
                                                                                      'RES_Wind_Onshore_Inf': "Wind_onshore", "RES_Geothermal":"Geothermal", "P_Coal_Hardcoal":"Coal", "CHP_Coal_Hardcoal":"Coal",
                                                                                       'CHP_Coal_Lignite':"Coal", 'P_Gas_CCGT':"Gas",'P_Gas_Engines':"Gas", 'P_Gas_OCGT':"Gas", 'D_Gas_H2':"Gas",'CHP_Gas_CCGT_Natural':"Gas",
                                                                                       'P_Gas_CCS':"Gas", 'CHP_Gas_CCGT_SynGas':"Gas", 'D_CAES':"Storage",'D_Battery_Li-Ion':"Storage", 'D_PHS':"Storage", 'D_PHS_Residual':"Storage",
                                                                                       'CHP_Gas_CCGT_Natural_CCS':"Gas", 'P_Coal_Lignite':"Gas", 'P_Oil':"Oil", 'CHP_Oil':"Oil", 'P_Nuclear':"Nuclear", 'RES_PV_Utility_Inf': "Solar"})

        open_entrance_split["Technology"] = open_entrance_split["Technology"].replace(
            {"P_Biomass": "Biomass", "CHP_Biomass_Solid_CCS": "Biomass", "CHP_Biomass_Solid": "Biomass","RES_Hydro_Large": "Hydro", "RES_Hydro_Small": "Hydro",
             'RES_PV_Utility_Avg': "Solar",'RES_PV_Utility_Opt': "Solar",'Res_PV_Utility_Tracking': "Solar", "RES_Wind_Onshore_Avg": "Wind",
             'RES_Wind_Onshore_Opt': "Wind",'RES_Wind_Offshore_Deep': "Wind", 'RES_Wind_Offshore_Transitional': "Wind",'RES_Wind_Offshore_Shallow': "Wind",
             'RES_Wind_Onshore_Inf': "Wind", "RES_Geothermal": "Geothermal", "P_Coal_Hardcoal": "Coal",
             "CHP_Coal_Hardcoal": "Coal",
             'CHP_Coal_Lignite': "Coal", 'P_Gas_CCGT': "Gas", 'P_Gas_Engines': "Gas", 'P_Gas_OCGT': "Gas",
             'D_Gas_H2': "Gas", 'CHP_Gas_CCGT_Natural': "Gas",
             'P_Gas_CCS': "Gas", 'CHP_Gas_CCGT_SynGas': "Gas", 'D_CAES': "Storage", 'D_Battery_Li-Ion': "Storage",
             'D_PHS': "Storage", 'D_PHS_Residual': "Storage",
             'CHP_Gas_CCGT_Natural_CCS': "Gas", 'P_Coal_Lignite': "Gas", 'P_Oil': "Oil", 'CHP_Oil': "Oil",
             'P_Nuclear': "Nuclear", 'RES_PV_Utility_Inf': "Solar"})

        def check_category(x):
            if x["Technology"] in ["Biomass", "Hydro", "Wind", "Solar"]:
                y = True
            else:
                y = False
            return y

        open_entrance_split["renewable"] = open_entrance_split.apply(lambda x: check_category(x), axis=1)
        self.scaling_res = open_entrance_split[open_entrance_split["renewable"] == True].groupby(["Technology", "Region", "Year"], sort=False).sum().sort_index()
        self.scaling_res_separate = open_entrance_split_copy[open_entrance_split["renewable"] == True].groupby(["Technology", "Region", "Year"], sort=False).sum().sort_index()

        conventionals = open_entrance_split[open_entrance_split["renewable"] == False].groupby(["Technology", "Region", "Year"], sort=False).sum()
        self.scaling_conventional = conventionals

    def tyndp_values(self, path, bidding_zone_encyc):
        tyndp_installed_capacity = pd.read_csv(path+"/TYNDP/capacity_tyndp2020-v04-al-2022_08_08.csv")
        tyndp_installed_capacity["node"] = tyndp_installed_capacity["node"].str.split("00", expand = True)[0]
        tyndp_installed_capacity["generator"] = tyndp_installed_capacity["generator"].replace({"otherres": "biomass"})
        tyndp_installed_capacity = tyndp_installed_capacity.groupby(["node", "generator"]).sum().reset_index()
        tyndp_installed_capacity["node"] = tyndp_installed_capacity["node"].replace({"DKE1": "DK1", "DKW1":"DK2", "NOM1":"NO3", "NON1":"NO4", "NOS0":"NO1", "SE01":"SE1", "SE02":"SE2", "SE03":"SE3", "SE04":"SE4"})
        tyndp_installed_capacity = tyndp_installed_capacity[tyndp_installed_capacity["node"].isin(["BE", "CZ", "DE", "DK1", "DK2", "NL", "NO1", "NO3", "NO4", "PL", "SE1", "SE2", "SE3", "SE4", "UK", "FI"])].dropna(axis = 1).reset_index(drop=True)
        tyndp_installed_capacity.rename(columns = {"ga2030": 2030, "ga2040":2040, "2020":2020}, inplace=True)
        tyndp_installed_capacity["country"] = tyndp_installed_capacity.merge(bidding_zone_encyc, left_on = "node", right_on = "bidding zones")["country"]
        tyndp_installed_capacity.set_index(["node", "generator"], inplace=True)
        tyndp_installed_capacity[2035] = (tyndp_installed_capacity[2030]+tyndp_installed_capacity[2040])/2


        tyndp_demand = pd.read_csv(path+"/TYNDP/load_tyndp2020-v02-al-2022_08_15.csv").dropna(axis=1)#[["y", "t", "BE", "CZ", "DE", "DK1", "DK2", "NL", "NO1", "NO3", "NO4", "PL", "SE1", "SE2", "SE3", "SE4", "UK", "FI"]]
        columns = pd.Series(tyndp_demand.columns.str.split("00", expand=True).get_level_values(0))
        columns = columns.replace({"DKE1": "DK1", "DKW1":"DK2", "NOM1":"NO3", "NON1":"NO4", "NOS0":"NO1", "SE01":"SE1", "SE02":"SE2", "SE03":"SE3", "SE04":"SE4"})
        tyndp_demand.columns = columns
        tyndp_demand = tyndp_demand[["YEAR","BE", "CZ", "DE", "DK1", "DK2", "NL", "NO1", "NO3", "NO4", "PL", "SE1", "SE2", "SE3", "SE4", "UK", "FI"]]
        tyndp_demand_years = {i : tyndp_demand[tyndp_demand["YEAR"] == year].reset_index(drop=True).drop(["YEAR"], axis=1) for year,i in zip([2030, 2040],[0,2])}
        tyndp_demand_years[1] = (tyndp_demand_years[0]+tyndp_demand_years[2])/2

        self.tyndp_installed_capacity = tyndp_installed_capacity
        self.tyndp_load = tyndp_demand_years

    def reduce_timeseries(self, long_ts, u_index):
        short_ts = pd.DataFrame()
        for index in u_index:
            current_day = long_ts.loc[index*24:index*24+23]
            short_ts = pd.concat([short_ts,current_day])
        return short_ts.reset_index(drop=True)

    def resolve_bidding_zones(self):
        busses_NO = self.nodes[self.nodes["country"] == "NO"]
        busses_SE = self.nodes[self.nodes["country"] == "SE"]
        busses_DK = self.nodes[self.nodes["country"] == "DK"]
        busses_others = self.nodes[~self.nodes['country'].isin(["NO", "SE", "DK"])]
        zones_se = {476: 3, 477: 1, 478: 4, 479: 3, 480: 3, 481: 2, 482: 3, 483: 3, 484: 2, 485: 3, 486: 4, 487: 2, 488: 3,
                    489: 4, 490: 1, 491: 3, 492: 3, 493: 3, 494: 2, 495: 3, 496: 2, 497: 3, 498: 3, 499: 2, 500: 4, 501: 2,
                    502: 2, 503: 3, 504: 4, 505: 1, 506: 3, 507: 2, 508: 2, 509: 3, 510: 1, 511: 3, 512: 3, 513: 4, 514: 2,
                    515: 2, 516: 3, 517: 2, 518: 4, 519: 1, 520: 2}
        zones_norge = {389: 1, 390: 4, 391: 5, 392: 3, 393: 4, 394: 1, 395: 3, 396: 3, 397: 1, 398: 2, 399: 3, 400: 5,
                       401: 5, 402: 1, 403: 2, 404: 5, 405: 3, 406: 2, 407: 4, 408: 1, 409: 2, 410: 1, 411: 2, 412: 2,
                       413: 2, 414: 1, 415: 3, 416: 1, 417: 1, 418: 3, 419: 2, 420: 3, 421: 5, 422: 4, 423: 4, 424: 3,
                       425: 1, 426: 2, 427: 5, 428: 3}
        zones_dk = {209: 1, 210: 1, 211: 1, 212:1, 213:1, 214:1, 215:2, 216:2, 217:2, 218:2}
        busses_norge = busses_NO.merge(pd.DataFrame.from_dict(zones_norge, orient="index", columns=["zone"]), how="left",left_index=True, right_index=True)
        busses_norge["bidding_zone"] = busses_norge["country"] + busses_norge["zone"].astype(str)
        busses_SE = busses_SE.merge(pd.DataFrame.from_dict(zones_se, orient="index", columns=["zone"]), how="left",left_index=True, right_index=True)
        busses_SE["bidding_zone"] = busses_SE["country"] + busses_SE["zone"].astype(str)
        busses_DK = busses_DK.merge(pd.DataFrame.from_dict(zones_dk, orient="index", columns=["zone"]), how="left",left_index=True, right_index=True)
        busses_DK["bidding_zone"] = busses_DK["country"] + busses_DK["zone"].astype(str)

        busses_others["bidding_zone"] = busses_others["country"]
        self.nodes = pd.concat([busses_norge.drop(["zone"], axis=1), busses_SE.drop(["zone"], axis=1), busses_DK.drop(["zone"], axis=1), busses_others]).sort_index()

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
    def add_future_windcluster(self, location):
        windfarms = pd.read_csv(location+"additional_windfarm_cluster.csv",  encoding="UTF-8").dropna(axis=1)
        windfarms["Market Zone"] = windfarms["Market Zone"].replace("DELU", "DE").replace("GB", "UK")
        windfarms = windfarms.rename(columns={"Latitude (decimal units)": "LAT", "Longitude (decimal units)":"LON" })
        windfarms = windfarms.reset_index(drop=True)
        windfarms[0] = 1000
        print = False

        if print: plotly_maps_bubbles(df=windfarms, scen=9, maps_folder= location+"kini_locations", name="future_windfarms_locations_all", unit="GW", size_scale=100,title="findfarms", year =0)
        #Belgium
        additional_node = windfarms[windfarms["Market Zone"] == "BE"]
        if print: plotly_maps_bubbles(df=additional_node, scen=9, maps_folder=location + "kini_locations", name="future_windfarms_locations_BE", unit="GW", size_scale=100, title="findfarms", year =0)
        #new_nodes = pd.concat([self.nodes, additional_node])
        additional_dc_lines = pd.DataFrame()
        # attach every of the clusters to a number of onshore points
        # north sea
        for i in range(524, 525):
            #print(i)
            additional_dc_lines = pd.concat([additional_dc_lines, pd.DataFrame({"from": (i, i, i, i, i), "to": (24, 366, 288, 523, 522), "EI":(3,3,3,3,3)})])

        #Deutschland -> hier nehme ich einfach alle
        additional_node = windfarms[windfarms["Market Zone"] == "DE"]
        #additional_node.index = np.arange(len(self.nodes), len(self.nodes) + len(additional_node))
        if print: plotly_maps_bubbles(df=additional_node, scen=9, maps_folder= location+"kini_locations", name="future_windfarms_locations_DE", unit="GW", size_scale=100,title="findfarms", year =0)
        #new_nodes = pd.concat([new_nodes, additional_node])
        # attach every of the clusters to a number of onshore points
        #north sea
        for i in range(525, 529):
            additional_dc_lines = pd.concat([additional_dc_lines, pd.DataFrame({"from":(i, i, i, i, i), "to":(170, 212, 373, 523, 522), "EI":(3,3,3,3,3)})])
        #baltic
        for i in range(529, 531):
            additional_dc_lines = pd.concat([additional_dc_lines, pd.DataFrame({"from":(i, i, i, i), "to":(218, 62, 513, 521), "EI":(3,3,3,3)})])

        # Dänemark
        additional_node = windfarms[windfarms["Market Zone"].isin(["DK1", "DK2"])]
        #additional_node.index = np.arange(len(new_nodes), len(new_nodes) + len(additional_node))
        if print: plotly_maps_bubbles(df=additional_node, scen=9, maps_folder= location+"kini_locations", name="future_windfarms_locations_DK", unit="GW", size_scale=100,title="findfarms",year=0)
        #nodes die ich haben möchte

        for i in range(543, 544):
            additional_dc_lines = pd.concat([additional_dc_lines, pd.DataFrame({"from": (i, i, i, i,i), "to": (212, 426, 380, 522, 523), "EI":(3,3,3,3,3)})])

        # Netherlands
        additional_node = windfarms[windfarms["Market Zone"].isin(["NL"])]
        #additional_node.index = np.arange(len(new_nodes), len(new_nodes) + len(additional_node))
        if print: plotly_maps_bubbles(df=additional_node, scen=9, maps_folder= location+"kini_locations", name="future_windfarms_locations_NL", unit="GW", size_scale=100,title="findfarms")
        #nodes die ich haben möchte
        #additional_node = additional_node[additional_node.index.isin([547,550,551])]
        #additional_node.index = np.arange(len(new_nodes), len(new_nodes) + len(additional_node))
        #plotly_maps_bubbles(df=additional_node, scen=9, maps_folder= location+"kini_locations", name="future_windfarms_locations", unit="GW", size_scale=100,title="findfarms")

        #new_nodes = pd.concat([new_nodes, additional_node])
        for i in range(531, 534):
            additional_dc_lines = pd.concat([additional_dc_lines, pd.DataFrame({"from": (i, i, i, i, i, i), "to": (376, 357, 265, 366, 522, 523) , "EI":(3,3,3,3,3,3)})])

        # UK
        additional_node = windfarms[windfarms["Market Zone"].isin(["UK"])]
        #additional_node.index = np.arange(len(new_nodes), len(new_nodes) + len(additional_node))
        if print: plotly_maps_bubbles(df=additional_node, scen=9, maps_folder= location+"kini_locations", name="future_windfarms_locations_UK", unit="GW", size_scale=100,title="findfarms")
        #nodes die ich haben möchte
        #additional_node = additional_node[additional_node.index.isin([551,560,558, 550, 559, 568, 552, 548, 567, 557, 552, 548, 567, 557, 575, 574, 569, 565, 570, 563, 547, 564, 578, 576])]
        #additional_node.index = np.arange(len(new_nodes), len(new_nodes) + len(additional_node))
        #plotly_maps_bubbles(df=additional_node, scen=9, maps_folder= location+"kini_locations", name="future_windfarms_locations", unit="GW", size_scale=100,title="findfarms")


        for i in range(535, 538):
            additional_dc_lines = pd.concat([additional_dc_lines, pd.DataFrame({"from": (i, i, i, i, i), "to": (300, 292, 307, 522, 523), "EI":(3,3,3,3,3)})])
        for i in range(538, 543):
            additional_dc_lines = pd.concat([additional_dc_lines, pd.DataFrame({"from": (i, i, i, i, i, i), "to": (350, 265, 357 ,24, 522, 523), "EI":(3,3,3,3,3,3)})])


        # Poland
        additional_node = windfarms[windfarms["Market Zone"].isin(["PL"])]
        #additional_node.index = np.arange(len(new_nodes), len(new_nodes) + len(additional_node))
        if print: plotly_maps_bubbles(df=additional_node, scen=9, maps_folder= location+"kini_locations", name="future_windfarms_locations_PL", unit="GW", size_scale=100,title="findfarms")

        #new_nodes = pd.concat([new_nodes, additional_node])
        for i in range(534, 535):
            additional_dc_lines = pd.concat([additional_dc_lines, pd.DataFrame({"from": (i, i, i, i), "to": (470, 518, 62, 521), "EI":(3,3,3,3)})])

        new_dc_lines = pd.concat([self.dc_lines, additional_dc_lines])

        new_dc_lines = new_dc_lines.reset_index(drop=True)
        #new_nodes = new_nodes.reset_index()
        self.dc_lines = new_dc_lines

        #plotly_maps_lines_colorless(P_flow=test, P_flow_DC=test2, bus=self.nodes, scen=9, maps_folder=location+"grid_test")


class kpi_data:
    def __init__(self, run_parameter, scen):
        self.run_parameter = run_parameter
        years = self.run_parameter.years
        read_folder = run_parameter.read_folder = run_parameter.directory + "results/" + run_parameter.case_name + "/" + str(scen) + "/subscen" +str(run_parameter.subscen) + "/"
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
        if (run_parameter.subscen == 0) & (scen == 1):
            try:
                overloaded_AC = self.line_loading.AC["avg"][self.line_loading.AC["avg"]["full_load_h"] >= 0.7 * 504]["full_load_h"]
                overloaded_AC = overloaded_AC * 3
                overloaded_AC.to_csv(run_parameter.export_folder + str(1) +"/subscen" + str(run_parameter.subscen) + "/overloaded_lines_AC.csv")
                overloaded_DC = self.line_loading.DC["avg"][self.line_loading.DC["avg"]["full_load_h"] >= 0.7 * 504]["full_load_h"]
                overloaded_DC = overloaded_DC * 3
                overloaded_DC.to_csv(run_parameter.export_folder + str(1) + "/subscen" + str(run_parameter.subscen) + "/overloaded_lines_DC.csv")
            except:pass

        #electrolyser
        # if scen != 1:
        #     self.electrolyser_location_last_year = self.CAP_E[[years[-1]]].merge(self.run_parameter.electrolyser[scen]["bus"], left_index=True, right_index=True).merge(bus_raw,left_on="bus",right_index=True)[[2, "LAT", "LON"]]
        #     self.electrolyser_location_last_year.rename(columns={2: 0}, inplace=True)
        #     self.CAP_E.index = run_parameter.electrolyser[scen]["name"]


        # Line loading hours
        # self.line_hours.AC = {y: self.line_fullload_hours(flow_file=self.F_AC[y], bus_raw=self.bus_raw, index_file=lines_overview,
        #                                             flexlines_capacity="", tolerance=0.01) for y in years}
        # self.line_hours.DC = {y: line_fullload_hours(flow_file=F_DC[y], bus_raw=bus_raw, index_file=lines_DC_overview,
        #                                                flexlines_capacity=CAP_BH, tolerance=0.01) for y in years}
        # P_flow_total[0] = sum(P_flow_yearly[y][0] for y in years) / len(years)
        # P_flow_total[["from", "to", "LAT_x", "LON_x", "LAT_y", "LON_y"]] = P_flow_yearly[0][
        #     ["from", "to", "LAT_x", "LON_x", "LAT_y", "LON_y"]]
        # P_line_hours_total = sum(P_line_hours[y] for y in years)
        # P_line_hours_total[["from", "to", "LAT_x", "LON_x", "LAT_y", "LON_y"]] = P_line_hours[0][
        #     ["from", "to", "LAT_x", "LON_x", "LAT_y", "LON_y"]]
        # P_line_hours_DC_total = sum(P_DC_line_hours[y] for y in years)
        # P_line_hours_DC_total[["from", "to", "LAT_x", "LON_x", "LAT_y", "LON_y"]] = P_DC_line_hours[0][
        #     ["from", "to", "LAT_x", "LON_x", "LAT_y", "LON_y"]]
        # overloaded_AC_lines = P_line_hours_total[P_line_hours_total[0] >= 0.7 * 3 * 504][0]
        # overloaded_DC_lines = P_line_hours_DC_total[P_line_hours_DC_total[0] >= 0.7 * 3 * 504][0]
        #
        # # sketchy, non volume weighted
        # P_flow_total_bz = P_flow_total.merge(bus_raw[["country"]], how="left", left_on=["from"], right_index=True)
        # P_flow_total_bz = P_flow_total_bz.merge(bus_raw[["country"]], how="left", left_on=["to"], right_index=True)
        #
        # # P_flow_total_bz = P_flow_total_bz.groupby("bidding zone", sort = False).mean().reset_index()
        #
        # P_flow_DC_total[0] = sum(P_flow_DC_yearly[y][0] for y in years) / len(years)
        # P_flow_DC_total[["Pmax", "from", "to", "LAT_x", "LON_x", "LAT_y", "LON_y"]] = P_flow_DC_yearly[years[-1]][
        #     ["Pmax", "from", "to", "LAT_x", "LON_x", "LAT_y", "LON_y"]]
        #
        # P_flow_DC_total_bz = P_flow_DC_total.merge(bus_raw[["country"]], how="left", left_on=["from"], right_index=True)
        # P_flow_DC_total_bz = P_flow_DC_total_bz.merge(bus_raw[["country"]], how="left", left_on=["to"],
        #                                               right_index=True)

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
        for EI in self.run_parameter.EIs:
            data_ei_individual = data_ei_matched[data_ei_matched["EI"] == EI]
            data_ei_from_bus = data_ei_individual.merge(lines_DC_overview[["from"]], how="left", left_index=True,right_index=True).set_index("from")
            data_ei_from_country = data_ei_from_bus.merge(bus_overview["country"], how="left", left_index=True,right_index=True).set_index("country")
            aggregated_trade = data_ei_from_country.groupby("country", axis=0).sum()
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
        zonal_trade_balance = trade_balance_bz_total.groupby("country_from").sum().sub(trade_balance_bz_total.groupby("country_to").sum(), fill_value=0)
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
                index_file["Pmax"].iat[int(i)] = CAP_BH[i][y]
        elif isinstance(CAP_BH, pd.DataFrame):
            CAP_BH.columns = CAP_BH.columns.astype(int)
            for i in CAP_BH.index:
                index_file["Pmax"].iat[int(i)] = CAP_BH.loc[i][0]
        file_without_0 = line_data[line_data != 0].dropna(axis= 1, how="all").abs()
        max_power = index_file[index_file.index.isin(line_data.columns)]["Pmax"]
        max_power_filtered = max_power[(max_power!=0.0)]
        relative = file_without_0.divide(max_power_filtered, axis=1)
        avg = relative.mean(axis=0).to_frame().fillna(0)
        avg["full_load_h"] = relative[relative > 0.70 - full_load_tolerance].count()
        # avg.index = avg.index.astype(int)
        file_bus = avg.merge(index_file[["Pmax", "from", "to"]], how="left", left_index=True, right_index=True)
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