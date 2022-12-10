from dataclasses import dataclass, astuple

import numpy as np
import pandas as pd

from printing_funct import plotly_maps_bubbles

pd.options.mode.chained_assignment = None
from helper_functions import demand_columns, give_nearest_bus_relative_position, Myobject
from mapping import new_res_mapping
import pickle
import requests
import json
from functools import reduce
import os
from sys import platform
import sys


class run_parameter:
    def __init__(self, scenario_name):
        # hier werte je nach betrachtetem Szenario einfügen
        # capture batch system specific parameters for running on a cluster computer
        if platform == "linux" or platform == "linux2":
            self.directory = "/work/seifert/powerinvest/"  # patch on the cluster computer
            self.case_name = sys.argv[1]  # reading from the batch script
            self.years = int(sys.argv[2])
            self.timesteps = int(sys.argv[3])
            self.scen = int(sys.argv[4])
            self.sensitivity_scen = int(sys.argv[5])
        # local execution parameters
        elif (platform == "darwin") or (platform == "win32"):
            self.directory = ""
            self.case_name = scenario_name
            self.years = 1
            self.timesteps = 10
            self.scen = "BZ5"
            self.sensitivity_scen = 0
        self.solving = False
        self.reduced_TS = True
        self.export_model_formulation = self.directory + "results/" + self.case_name + "/model_formulation_scen" + str(
            self.scen) + "_subscen" + str(self.sensitivity_scen) + ".mps"
        self.export_folder = self.directory + "results/" + self.case_name + "/" + str(
            self.scen) + "/" + "subscen" + str(self.sensitivity_scen) + "/"
        self.import_folder = self.directory + "data/"
        os.makedirs(self.export_folder, exist_ok=True)
        #
        self.hours = 504  # 21 representative days
        self.scaling_factor = 8760 / self.hours

    # je nachdem was oben bei scen eingetragen ist wird hier der case betrachtet
    def create_scenarios(self):
        match self.scen:
            case "BZ2":
                self.lookup_dict = {"DEF": "DEII1", "DE6": "DEII1", "DE9": "DEII1", "DE3": "DEII1", "DE4": "DEII1",
                          "DE8": "DEII1", "DED": "DEII1", "DEE": "DEII1", "DEG": "DEII1", "DEA": "DEII2",
                          "DEB": "DEII2", "DEC": "DEII2", "DE1": "DEII2", "DE2": "DEII2", "DE7": "DEII2",
                          "OffBZN": "OffBZN", "OffBZB": "OffBZB"}
            case "BZ3":
                self.lookup_dict = {"DEF": "DEIII1", "DE6": "DEIII1", "DE9": "DEIII1", "DE3": "DEIII2", "DE4": "DEIII2",
                                  "DE8": "DEIII2", "DED": "DEIII2", "DEE": "DEIII2", "DEG": "DEIII2", "DEA": "DEIII3",
                                  "DEB": "DEIII3", "DEC": "DEIII3", "DE1": "DEIII3", "DE2": "DEIII3", "DE7": "DEIII3",
                                  "OffBZN": "OffBZN", "OffBZB": "OffBZB"}
            case "BZ5":
                self.lookup_dict = {"DEF": "DEV1", "DE6": "DEV2", "DE9": "DEV2", "DE3": "DEV3", "DE4": "DEV3",
                                  "DE8": "DEV3", "DED": "DEV3", "DEE": "DEV3", "DEG": "DEV3", "DEA": "DEV4",
                                  "DEB": "DEV4", "DEC": "DEV4", "DE1": "DEV5", "DE2": "DEV5", "DE7": "DEV5",
                                  "OffBZN": "OffBZN", "OffBZB": "OffBZB"}
        match self.sensitivity_scen:
            case 0:
                print("Base scenario sensitivity")
                self.CO2_price = [80, 120, 160]
            case 1:
                print("")

        self.TRM = 0.7
        #self.country_selection = ['BE', 'CZ', 'DE', 'DK', 'FI', 'NL', 'NO', 'PL', 'SE', 'UK', "NSEH1", "NSEH2", "BHEH"]
        self.country_selection_OffBZB = ['BE', 'UK', 'DK1', 'DK', 'NL', 'NSEH1', 'NSEH2', 'OffBZN', 'DE2']
        self.country_selection_OffBZN = ['PL', 'DK2', 'FI', 'NO1', 'NO2', 'NO3', 'NO4', 'NO5', 'SE1', 'SE2', 'SE3',
                                         'SE4', 'BHEH', 'DE1']

class model_data:
    def __init__(self, create_res, reduced_ts, export_files, run_parameter):
        self.CO2_price = run_parameter.CO2_price
        # reading in nodes and merging them into zonal configuration BZ2, BZ3; BZ4
        #df_nodes = pd.read_csv(run_parameter.import_folder + "import_data/df_nodes_to_zones_filtered_final.csv",sep=";", index_col=0)
        #df_nodes = pd.read_csv("data/import_data/df_nodes_to_zones_filtered_final.csv", sep=";", index_col=0)

        #def lookup(row,scen_dict):
            #try:
                #value = scen_dict[row["NUTS_ID"]]
            #except:
                #value = row["country_y"]
            #return value

        #df_nodes[run_parameter.scen] = df_nodes.apply(lambda row: lookup(row, run_parameter.lookup_dict), axis=1)
        self.nodes = pd.read_excel("data\\final_readin_data\\nodes.xlsx", index_col=0)


        #reading in NTCs
        #TODO: flexilines brauchen eine capacity zuweisung - Recherche? Offshore windfarms = Kinis Daten, BHEH =3000, NSEH1+2 = jeweils 10000
        self.ntc_BAU = pd.read_excel("data\\final_readin_data\\NTC_BAU.xlsx")
        self.ntc_BZ2 = pd.read_excel("data\\final_readin_data\\NTC_BZ_2.xlsx")
        self.ntc_BZ3 = pd.read_excel("data\\final_readin_data\\NTC_BZ_3.xlsx")
        self.ntc_BZ5 = pd.read_excel("data\\final_readin_data\\NTC_BZ_5.xlsx")

        #DEMAND
        self.demand = pd.read_excel("data\\final_readin_data\\demand.xlsx")

        #df_demand_final = df_demand_T.rename_axis('index').reset_index()
        #df_demand_final['index'] = df_demand_final['index'].astype(int)
        #df_demand_merged = df_demand_final.merge(df_nodes, on="index",how='left')

        #reading in generation CONV
#       self.dispatchable = pd.read_excel("data\\final_readin_data\\dispatachable.xlsx")
        #generators = pd.read_csv("data\import_data\generators_filtered.csv", sep=";", index_col=0)
        #df_generators = pd.read_csv("data\\import_data\\generators_filtered_v2_gerettet.csv", sep=",", index_col=0)
        #df_generators_merged = df_nodes.merge(df_generators[['index', 'p_nom', 'carrier', 'marginal_cost', 'efficiency']], on="index",how='left')
        #df_offshore = df_generators_merged.loc[(df_generators_merged['carrier'].isin(['offwind-ac','offwind-dc']))]
        #df_only_offshore_wind = df_generators_merged.drop(columns= carrier[('CCGT', 'onwind', 'solar', 'ror', 'nucelar', 'biomass', 'coal'))

        # TODO: aufsummieren der gesamt capacity je carrier and zone !! BZ2 ist eingesetzt! warum geht hier nicht: self.scen?
        #df_generators_merged['Total conventional Capacity'] = df_generators_merged.groupby([run_parameter.scen, 'carrier'])['p_nom'].transform('sum')
        #von Paul:       test = df_generators_merged.groupby([run_parameter.scen, 'carrier']).sum(numeric_only=True)[['p_nom']]

        #df_total_cap = df_generators_merged.drop_duplicates(subset=[run_parameter.scen, 'carrier'])

        #df_generators_merged.to_csv('generators_merged.csv', index=False)
        # test = sjoined_nodes_states4.groupby(["NUTS_ID","Fuel"]).sum(numeric_only=True)[["bidding_zone"]]

    #Reading in res series  timeseries einlesen für wind und solar! mit ninja.renewables - ist jetzt von Paul
        self.res_series = pd.read_excel("data\\final_readin_data\\res_series.xlsx")

    #Readin in conventional
        self.dispatchable = pd.read_excel("data\\final_readin_data\\dispatchable.xlsx")
    #Readin run of river
        self.ror_series = pd.read_excel("data\\final_readin_data\\ror_series.xlsx")
        #old
        #2) solar_filtered
        #solar_raw = pd.read_csv("data\\import_data\\solar_filtered.csv", sep=";", index_col=0)
#        solar_generation = df_generators_merged[df_generators_merged["carrier"].isin(["solar"])]
#            self.solar = solar_generation
        # 3) wind_filtered
        #wind_raw = pd.read_csv("data\\import_data\\wind_filtered.csv", sep=";", index_col=0)
#        wind_generation = df_generators_merged[df_generators_merged["carrier"].isin(["onwind", "offwind-ac", "offwind-dc"])]
#           self.wind = wind_generation

    #Reading in storage
        self.storage = pd.read_excel("data\\final_readin_data\\storage.xlsx")
        self.reservoir = pd.read_excel("data\\final_readin_data\\reservoir.xlsx")

########################
##reduced time series##
#######################

        if reduced_ts:
            try:
                u = pd.read_csv(run_parameter.import_folder + "poncelet/u_result.csv", index_col=0)
                u_index = u.index[u["value"] == 1.0].to_list()
                self.timesteps_reduced_ts = 24 * len(u_index)
            except:
                sys.exit("need to run poncelet algorithm first!")
#            self.res_series = {i: self.reduce_timeseries(self.res_series[i], u_index) for i in [0, 1, 2]}
#            self.demand = {i: self.reduce_timeseries(self.demand[i], u_index) for i in [0, 1, 2]}
#            self.share_solar = {i: self.reduce_timeseries(self.share_solar[i], u_index) for i in [0, 1, 2]}
#            self.share_wind = {i: self.reduce_timeseries(self.share_wind[i], u_index) for i in [0, 1, 2]}
#            self.ror_series = self.reduce_timeseries(self.ror_series, u_index)
#            self.reservoir_zonal_limit = self.reduce_timeseries(limited_dam_ts, u_index)

#        self.reservoir_zonal_limit = self.reservoir_zonal_limit.sum()
#        self.dispatchable_generators = self.conv_scaling_country_specific()
#        if run_parameter.add_future_windcluster:
#            self.add_future_windcluster(location=run_parameter.import_folder)

        # Netzausbau
 #       if run_parameter.grid_extension:
 #           self.extend_overloaded_lines(type="AC", case_name=run_parameter.case_name)
 #           self.extend_overloaded_lines(type="DC", case_name=run_parameter.case_name)
 #       if export_files:
 #           with open(run_parameter.export_folder + 'powerplants.pkl', 'wb+') as f:
 #               pickle.dump(self.dispatchable_generators, f)
 #           with open(run_parameter.export_folder + 'demand.pkl', 'wb+') as f:
 #               pickle.dump(self.demand, f)
 #           with open(run_parameter.export_folder + 'P_max.pkl', 'wb+') as f:
 #               pickle.dump(self.res_series, f)

    def reduce_timeseries(self, long_ts, u_index):
        short_ts = pd.DataFrame()
        for index in u_index:
            current_day = long_ts.loc[index * 24:index * 24 + 23]
            short_ts = pd.concat([short_ts, current_day])
        return short_ts.reset_index(drop=True)

#hier muss jetzt noch gekürzt werden im Script KPI_data
class kpi_data:
    def __init__(self, run_parameter):
        self.run_parameter = run_parameter
        self.run_parameter.years = range(0, run_parameter.years)
        years = self.run_parameter.years

        #todo hier namen des folders eingeben, der die variablen enthält:
        read_folder = run_parameter.read_folder = run_parameter.directory + "results/" + run_parameter.case_name + "/" + str(
            scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/"
        self.bus = pd.read_csv(read_folder + "busses.csv", index_col=0)

        # create empty objects
        self.load_factor = Myobject()
        self.P_R = Myobject()
        self.P_R.max = Myobject()
        self.curtailment = Myobject()
        self.line_loading = Myobject()
        self.line_balance = Myobject()
        with open(read_folder + 'powerplants.pkl', 'rb') as f:
            powerplants_raw = pickle.load(f)
            # powerplants_raw = self.change_column_to_int(powerplants_raw)
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

        bus_raw = self.read_in(y="", string="busses.csv", int_convert=False)
        # overwrite the wind cluster bus country and bidding zone
        bus_raw.loc[524:, ["country", "bidding_zone"]] = bus_raw.loc[524:, ["country", "bidding_zone"]].apply(
            lambda x: x + "_wind_cluster")

        storage = self.read_in(y="", string="storage.csv", int_convert=False)
        lines_overview = self.read_in(y="", string="lines.csv", int_convert=False)
        lines_DC_overview = self.read_in(y="", string="lines_DC.csv", int_convert=False)
        ror_supply = self.read_in(y="", string="ror_supply.csv")
        CAP_lines = self.read_in(y="", string="CAP_BH.csv")
        self.CAP_lines = CAP_lines.T.merge(lines_DC_overview[["from", "to", "EI"]], left_index=True,
                                           right_index=True).merge(bus_raw[["LON", "LAT"]], left_on="from",
                                                                   right_index=True).merge(bus_raw[["LON", "LAT"]],
                                                                                           left_on="to",
                                                                                           right_index=True).sort_index()
        # encyc_powerplants_bus = create_encyclopedia(powerplants_raw[0]["bus"])
        # encyc_storage_bus = create_encyclopedia(storage["bus"])
        if scen != 1:
            self.CAP_E = self.read_in(y="", string="CAP_E.csv").transpose().merge(
                run_parameter.electrolyser[scen][["name", "bus"]], left_index=True, right_index=True).merge(
                bus_raw[["LON", "LAT"]], left_on="bus", right_index=True).set_index("name")

        self.F_AC = {y: self.read_in(y=y, string="_F_AC.csv") for y in years}
        self.timesteps = self.F_AC[0].shape[0]
        self.F_DC = {y: self.read_in(y=y, string="_F_DC.csv") for y in years}
        self.EI_trade = {y: self.EI_connections(lines_DC_overview=lines_DC_overview, bus_overview=bus_raw, year=y) for y
                         in years}
        self.P_R.raw = {y: self.read_in(y=y, string="_P_R.csv") for y in years}
        self.P_DAM = {y: self.read_in(y=y, string="_P_DAM.csv") for y in years}
        self.res_curtailment = {
            y: pd.read_csv(read_folder + str(y) + "_res_curtailment.csv", index_col=0, names=self.P_R.raw[y].columns,
                           header=0) for y in years}
        self.P_C = {y: self.read_in(y=y, string="_P_C.csv") for y in years}
        self.P_S = {y: self.read_in(y=y, string="_P_S.csv") for y in years}
        self.L_S = {y: self.read_in(y=y, string="_L_S.csv") for y in years}
        self.C_S = {y: self.read_in(y=y, string="_C_S.csv") for y in years}
        self.P_loss_load = {y: self.read_in(y=y, string="_p_load_lost.csv") for y in years}
        if scen != 1:
            self.P_H = {
                y: self.read_in(y=y, string="_P_H.csv").transpose().merge(run_parameter.electrolyser[scen]["name"],
                                                                          left_index=True, right_index=True).set_index(
                    "name").T for y in years}

        # calculations
        if scen != 1: self.load_factor.elect = pd.DataFrame({y: (self.P_H[y] / self.CAP_E[y]).mean() for y in years})
        self.P_R.bz = {y: self.prepare_results_files_bz(self.P_R.raw[y], bus_raw) for y in years}
        self.P_R.max.bz = {y: self.prepare_results_files_bz(self.P_R.max.raw[y], bus_raw) for y in years}
        self.P_R.solar = {y: share_solar_raw[y].multiply(self.P_R.raw[y]).dropna(axis=1, how='all') for y in years}
        self.P_R.wind = {y: (share_wind_raw[y] * self.P_R.raw[y]).dropna(axis=1, how='all') for y in years}
        self.zonal_trade_balance = {
            y: self.zonal_trade_balance_function(self.F_AC[y], self.F_DC[y], bus_raw, lines_overview, lines_DC_overview,
                                                 self.run_parameter.scaling_factor) for y in years}

        ## curtailment
        self.curtailment.raw = {y: self.res_curtailment[y] for y in years}
        self.curtailment.bz = {y: self.prepare_results_files_bz(self.curtailment.raw[y], bus_raw) for y in years}
        self.curtailment.bz_relative = {y: pd.DataFrame(self.curtailment.bz[y][0] / (self.P_R.max.bz[y][0])) for y in
                                        years}

        ## electricity sources
        self.generation_temporal = {
            y: self.prepare_results_files_index_temporal(y=y, ror_supply=ror_supply, index_file=powerplants_raw[y],
                                                         scen=scen) for y in years}

        ## line loading
        self.line_loading.AC = {
            y: self.prepare_results_files_lines(y=y, file=self.F_AC, bus_raw=bus_raw, index_file=lines_overview,
                                                yearly=False, full_load_tolerance=0.01) for y in years}
        self.line_loading.DC = {
            y: self.prepare_results_files_lines(y=y, file=self.F_DC, bus_raw=bus_raw, index_file=lines_DC_overview,
                                                yearly=True, CAP_BH=self.CAP_lines, full_load_tolerance=0.01) for y in
            years}
        self.line_loading.AC.update({"avg": (
                    pd.concat([self.line_loading.AC[year][[0, "full_load_h"]] for year in run_parameter.years]).groupby(
                        level=0).sum() / len(run_parameter.years)).merge(lines_overview[["from", "to"]],
                                                                         left_index=True, right_index=True).merge(
            bus_raw[["LAT", "LON"]], left_on="from", right_index=True).merge(bus_raw[["LAT", "LON"]], left_on="to",
                                                                             right_index=True).sort_index(
            ascending=True)})
        # self.line_loading.AC.update({"avg": (sum(self.line_loading.AC[year][0] for year in run_parameter.years)/len(run_parameter.years)).to_frame().merge(lines_overview[["from", "to"]], left_index = True, right_index = True).merge(bus_raw[["LAT","LON"]], left_on = "from", right_index =True).merge(bus_raw[["LAT","LON"]], left_on = "to", right_index =True).sort_index(ascending=True)})
        self.line_loading.DC.update({"avg": (
                    pd.concat([self.line_loading.DC[year][[0, "full_load_h"]] for year in run_parameter.years]).groupby(
                        level=0).sum() / len(run_parameter.years)).merge(lines_DC_overview[["from", "to"]],
                                                                         left_index=True, right_index=True).merge(
            bus_raw[["LAT", "LON"]], left_on="from", right_index=True).merge(bus_raw[["LAT", "LON"]], left_on="to",
                                                                             right_index=True).sort_index(
            ascending=True)})
        # self.line_loading.DC.update({"avg": (sum(self.line_loading.DC[year][0] for year in run_parameter.years)/len(run_parameter.years)).to_frame().merge(lines_DC_overview[["from", "to"]], left_index = True, right_index = True).merge(bus_raw[["LAT","LON"]], left_on = "from", right_index =True).merge(bus_raw[["LAT","LON"]], left_on = "to", right_index =True).sort_index(ascending=True)})

        self.line_balance.AC = {
            y: self.get_trade_balance_yearly(file=self.F_AC[y], bus_raw=bus_raw, index_file=lines_overview) for y in
            years}
        self.line_balance.DC = {
            y: self.get_trade_balance_yearly(file=self.F_DC[y], bus_raw=bus_raw, index_file=lines_DC_overview) for y in
            years}
        self.trade_balance_bz = {y: self.trade_balance(self.line_balance.AC[y], self.line_balance.DC[y]) for y in years}

        # P_R
        self.P_R.nodal_sum = reduce(lambda x, y: x.add(y), list(self.P_R.raw[y] for y in years))
        self.P_R.total_nodes = self.prepare_results_files_nodes(self.P_R.nodal_sum, bus_raw, temporal=0)
        self.P_R.total_bz = self.prepare_results_files_bz(self.P_R.nodal_sum, bus_raw)
        # total curtailments

        self.curtailment.bz_sum = reduce(lambda x, y: x.add(y), list(self.curtailment.bz[y] for y in years))
        self.P_R.max.bz_sum = reduce(lambda x, y: x.add(y), list(self.P_R.max.bz[y] for y in years))
        self.curtailment.sum = reduce(lambda x, y: x.add(y), list(self.curtailment.raw[y] for y in years))
        self.curtailment.bz_relative_sum = pd.DataFrame(self.curtailment.bz_sum[0] / self.P_R.max.bz_sum[0]).rename(
            {0: "relative"}, axis=1)
        self.curtailment.location_sum = self.prepare_results_files_nodes(self.curtailment.sum, bus_raw, temporal=0)
        self.curtailment.location = self.dataframe_creator(run_parameter=run_parameter, dict=self.curtailment.raw,
                                                           bus_raw=bus_raw)

        # further calculations
        ##overloaded lines -> > 70% load über die ganze periode, base case
        if (run_parameter.sensitivity_scen == 0) & (scen == 1):
            try:
                overloaded_AC = self.line_loading.AC["avg"][self.line_loading.AC["avg"]["full_load_h"] >= 0.7 * 504][
                    "full_load_h"]
                overloaded_AC = overloaded_AC * 3
                overloaded_AC.to_csv(run_parameter.export_folder + str(1) + "/subscen" + str(
                    run_parameter.sensitivity_scen) + "/overloaded_lines_AC.csv")
                overloaded_DC = self.line_loading.DC["avg"][self.line_loading.DC["avg"]["full_load_h"] >= 0.7 * 504][
                    "full_load_h"]
                overloaded_DC = overloaded_DC * 3
                overloaded_DC.to_csv(run_parameter.export_folder + str(1) + "/subscen" + str(
                    run_parameter.sensitivity_scen) + "/overloaded_lines_DC.csv")
            except:
                pass

    def dataframe_creator(self, run_parameter, dict, bus_raw):
        df = pd.DataFrame({year: dict[year].sum(axis=0) for year in run_parameter.years}).replace(0, np.nan).dropna(
            axis=0).merge(bus_raw[["LON", "LAT", "country", "bidding_zone"]], left_index=True, right_index=True)
        return df

    def read_in(self, y, string, int_convert=True):
        if isinstance(y, str):
            data = pd.read_csv(self.run_parameter.read_folder + string, index_col=0)
        else:
            data = pd.read_csv(self.run_parameter.read_folder + str(y) + string, index_col=0)
        if int_convert:
            data.columns = data.columns.astype(int)
        return data

    def prepare_results_files_nodes(self, file, bus_raw, temporal):
        file_w_na = file.dropna(axis=1)
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
        lignite: float = 0.0
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
            conventionals = self.oil + self.gas + self.coal + self.lignite + self.nuclear + self.other
            return conventionals

        def to_df(self):
            df = pd.DataFrame()
            for object in self:
                df = pd.concat([df, object], axis=1)
            try:
                df.columns = ["hydro", "oil", "gas", "coal", "lignite", "nuclear", "biomass", "other", "wind", "solar",
                              "P_S", "C_S", "curtailment", "electrolyser"]
            except:
                df.columns = ["hydro", "oil", "gas", "coal", "lignite", "nuclear", "biomass", "other", "wind", "solar",
                              "P_S", "C_S", "curtailment"]
            return df

    def prepare_results_files_index_temporal(self, y, ror_supply, index_file, scen):
        types = {"hydro": ['HDAM'], "oil": ['oil'], "gas": ['CCGT', 'OCGT'], "coal": ['coal'], "lignite": ["lignite"],
                 "nuclear": ['nuclear'], "biomass": ['biomass'], "other": ["other"]}

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
            lignite=get_sum_of_type(conventional_without_0, index_file, types["lignite"]),
            nuclear=get_sum_of_type(conventional_without_0, index_file, types["nuclear"]),
            other=get_sum_of_type(conventional_without_0, index_file, types["other"]),
            biomass=get_sum_of_type(conventional_without_0, index_file, types["biomass"]),
            wind=self.P_R.wind[y].sum(axis=1),
            solar=self.P_R.solar[y].sum(axis=1),
            P_S=self.P_S[y].sum(axis=1),
            C_S=self.C_S[y].sum(axis=1),
            curtailment=self.res_curtailment[y].sum(axis=1),
            electrolyser=self.P_H[y].sum(axis=1) if scen != 1 else pd.DataFrame()
        )
        return generation

    def EI_connections(self, lines_DC_overview, bus_overview, year):
        # data_ei = self.F_DC[year].iloc[:, -self.run_parameter.number_flexlines:].T
        transposed = self.F_DC[year].T
        index_EI_lines = lines_DC_overview[lines_DC_overview["EI"].isin([0, 1, 2])].index
        data_ei = transposed[transposed.index.isin(index_EI_lines)]
        data_ei.index = data_ei.index.astype(int)
        data_ei_matched = data_ei.merge(lines_DC_overview[["EI"]], how="left", left_index=True, right_index=True)
        trade_to_bz = {}
        self.run_parameter.create_scenarios()
        for EI in self.run_parameter.EI_bus.index:
            data_ei_individual = data_ei_matched[data_ei_matched["EI"] == EI]
            data_ei_from_bus = data_ei_individual.merge(lines_DC_overview[["from"]], how="left", left_index=True,
                                                        right_index=True).set_index("from")
            data_ei_from_country = data_ei_from_bus.merge(bus_overview["country"], how="left", left_index=True,
                                                          right_index=True).set_index("country")
            aggregated_trade = data_ei_from_country.groupby("country", axis=0).sum(numeric_only=True)
            trade_to_bz.update({EI: aggregated_trade.iloc[:, :self.timesteps]})
        return trade_to_bz

    def change_column_to_int(self, item):
        for y in self.run_parameter.years:
            item[y].columns = item[y].columns.astype(int)
        return item

    def prepare_results_files_bz(self, file, bus_raw):
        file = file.dropna()
        file_without_0 = file.loc[(file != 0).any(axis=1)]
        file_sum = file_without_0.dropna(how='all').sum(axis=0)
        file_sum.index = file_sum.index.astype(int)
        file_frame = file_sum.to_frame()
        file_ready = file_frame.merge(bus_raw[["country"]], how="left", left_index=True, right_index=True)
        file_ready_bz = file_ready.groupby("country", sort=False).sum()  # .reset_index()
        # file_ready_bz_resolved_names = file_ready_bz.merge(bidding_zones_encyclopedia, how="left", left_on="bidding zone",right_on="zone_number")[["bidding zones", 0]].set_index("bidding zones")
        return file_ready_bz

    def zonal_trade_balance_function(self, F_AC, F_DC, bus_raw, lines_overview, lines_DC_overview, scaling_factor):
        line_balance_total = self.get_trade_balance_yearly(file=F_AC, bus_raw=bus_raw, index_file=lines_overview)
        line_balance_DC_total = self.get_trade_balance_yearly(file=F_DC, bus_raw=bus_raw, index_file=lines_DC_overview)
        trade_balance_bz_total = self.trade_balance(line_balance_total, line_balance_DC_total)
        # trade_balance_bz_total  = trade_balance_bz_total.merge(bidding_zones_encyclopedia, how="left", left_on="bidding zone_from",right_on="zone_number")[["bidding zones", "bidding zone_to", 0]].rename(columns={"bidding zones": "From bidding zone"})
        trade_balance_bz_total = trade_balance_bz_total.sort_values("country_to", axis=0)
        trade_balance_bz_total[0] = trade_balance_bz_total[0] * scaling_factor
        # exports - imports -> yes that is correct in the equation -> from defines where it starts == what country exports
        zonal_trade_balance = trade_balance_bz_total.groupby("country_from").sum(numeric_only=True).sub(
            trade_balance_bz_total.groupby("country_to").sum(numeric_only=True), fill_value=0)
        return zonal_trade_balance

    def trade_balance(self, AC_balance, DC_balance):
        def change_direction(x):
            if x[0] <= 0:
                zwischenspeicher = x["country_x"]
                x["country_x"] = x["country_y"]
                x["country_y"] = zwischenspeicher
                x[0] = -x[0]
            return x

        # balance = AC_balance.append(DC_balance).drop(AC_balance.index[AC_balance[0] == 0].tolist()) # append DC and drop zero loadings
        balance = pd.concat([AC_balance, DC_balance]).drop(AC_balance.index[AC_balance[0] == 0].tolist())
        magnitude = balance.apply(lambda x: change_direction(x), axis=1)
        bz_balance = magnitude.groupby(["country_x", "country_y"], sort=True).sum()[0].reset_index()
        bz_balance_rename = bz_balance.rename(columns={"country_x": "country_to", "country_y": "country_from"})
        interconnectors = bz_balance_rename[bz_balance_rename["country_to"] != bz_balance_rename["country_from"]]
        return interconnectors

    def get_trade_balance_yearly(self, file, bus_raw, index_file):
        # line loading comes in and is aggregated to the yearly node balance
        summed_bus = file.sum(axis=0).to_frame()
        summed_bus.index = summed_bus.index.astype(int)
        file_bus = summed_bus.merge(index_file[["from", "to"]], how="left", left_index=True, right_index=True)
        file_ready = file_bus.merge(bus_raw[["LAT", "LON", "country"]], how="left", left_on="from", right_index=True)
        file_ready = file_ready.merge(bus_raw[["LAT", "LON", "country"]], how="left", left_on="to", right_index=True)
        return file_ready

    def prepare_results_files_lines(self, y, file, index_file, bus_raw, yearly, full_load_tolerance, CAP_BH=""):
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
        file_without_0 = line_data[line_data != 0].dropna(axis=1, how="all").abs()
        max_power = index_file[index_file.index.isin(line_data.columns)]["max"]
        max_power_filtered = max_power[(max_power != 0.0)]
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
        response = requests.get(
            "https://energy-charts.info/charts/power/data_unit/de/year_wind_offshore_unit_2019.json")
        text = response.text
        parsed = json.loads(text)
        self.data = {}
        # self.capacity =
        # test = parsed[17]["name"][0]["en"]
        # test2 = pd.Series(parsed[1]["data"])
        for i in parsed[:-3]:
            print(i)
            self.data.update({i["name"][0]["en"]: pd.Series(i["data"])})
        self.summing_of_non_nan()

    def summing_of_non_nan(self):
        status_na = {}
        non_na = {}
        self.yearly_sum = {}
        for key, value in self.data.items():
            status_na.update({key: value.hasnans})
            if value.hasnans == False:
                non_na.update({key: value})
                self.yearly_sum.update({key: value.sum() / 4})

class gurobi_variables:
    def __init__(self, solved_model):
        all_variables = solved_model.getVars()
        last_item = all_variables[-1].VarName.split(",")
        self.years = int(last_item[0].split("[")[1]) + 1
        self.timesteps = int(last_item[0]) + 1
        self.years = 1
        self.timesteps = 10
        counter = len(all_variables) - 1
        self.additional_columns = {}
        self.results = {}
        while counter > 0:
            current_variable = all_variables[counter].VarName
            variable_name = current_variable.split("[")[0]
            array, counter, irregular_columns, bus_column_irregular = self.get_variable_from_position(
                variables=all_variables, counter=counter)
            self.results.update({variable_name: array})
            if irregular_columns:
                bus_column_irregular.reverse()
                self.additional_columns.update({variable_name: bus_column_irregular})

    def get_variable_from_position(self, variables, counter):
        current_variable = variables[counter].VarName
        bus_column_irregular = []
        irregular_columns = False
        first_run = True
        x = len(current_variable)
        if len(current_variable.split(",")) == 3:
            first_dimension = int(current_variable.split(",")[0].split("[")[1]) + 1
            second_dimension = int(current_variable.split(",")[1]) + 1
            last_dimension = int(current_variable.split(",")[-1].split("]")[0]) + 1
            dimension_counter = 1
            while dimension_counter < last_dimension:
                if int(variables[counter - dimension_counter].VarName.split(",")[-1].split("]")[0]) == int(
                        variables[counter].VarName.split(",")[-1].split("]")[0]):
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
                if int(variables[counter - dimension_counter].VarName.split(",")[-1].split("]")[0]) == int(
                        variables[counter].VarName.split(",")[-1].split("]")[0]):
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
            pd.DataFrame(self.results["P_R"][y, :, :], columns=self.additional_columns["P_R"]).to_csv(
                folder + str(y) + "_P_R.csv")
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
            pd.DataFrame(self.results["res_curtailment"][y, :, :],
                         columns=self.additional_columns["res_curtailment"]).to_csv(
                folder + str(y) + "_res_curtailment.csv")
            # storage
            pd.DataFrame(self.results["P_S"][y, :, :]).to_csv(folder + str(y) + "_P_S.csv")
            pd.DataFrame(self.results["C_S"][y, :, :]).to_csv(folder + str(y) + "_C_S.csv")
            pd.DataFrame(self.results["L_S"][y, :, :]).to_csv(folder + str(y) + "_L_S.csv")
            # AC line flow
            pd.DataFrame(self.results["F_AC"][y, :, :]).to_csv(folder + str(y) + "_F_AC.csv")
            # DC line flow
            pd.DataFrame(self.results["F_DC"][y, :, :]).to_csv(folder + str(y) + "_F_DC.csv")


