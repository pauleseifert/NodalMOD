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
            self.directory = "" #directory anpassen
            self.case_name = scenario_name
            self.years = 1
            self.timesteps = 10
            self.scen = "BZ2"
            self.sensitivity_scen = 0
        self.solving = False
        self.reduced_TS = False
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

class model_data:

    def __init__(self, create_res, reduced_ts, export_files, run_parameter):
        self.CO2_price = run_parameter.CO2_price
        # reading in nodes and merging them into zonal configuration BZ2, BZ3; BZ4
        #df_nodes = pd.read_csv(run_parameter.import_folder + "import_data/df_nodes_to_zones_filtered_final.csv",sep=";", index_col=0)
        df_nodes = pd.read_csv("data/import_data/df_nodes_to_zones_filtered_final.csv", sep=";", index_col=0)

        def lookup(row,scen_dict):
            try:
                value = scen_dict[row["NUTS_ID"]]
            except:
                value = row["country_y"]
            return value

        df_nodes[run_parameter.scen] = df_nodes.apply(lambda row: lookup(row, run_parameter.lookup_dict), axis=1)
        self.nodes = df_nodes

        #self.df_nodes = pd.read_csv("data/import_data/df_nodes.csv", sep=",", index_col=0)

        #reading in NTCs
        #TODO: flexilines brauchen eine capacity zuweisung - Recherche? Offshore windfarms = Kinis Daten, BHEH =3000, NSEH1+2 = jeweils 10000
        self.ntc_BZ_2 = pd.read_csv("data\\import_data\\NTC\\NTC_BZ_2.csv", sep=";")
        self.ntc_BZ_3 = pd.read_csv("data\\import_data\\NTC\\NTC_BZ_3.csv", sep=";")
        self.ntc_BZ_5 = pd.read_csv("data\\import_data\\NTC\\NTC_BZ_5.csv", sep=";")

        #DEMAND
        df_demand = pd.read_csv("data\\import_data\\demand.csv", sep=",")
        df_demand_T = df_demand.T

        df_demand_final = df_demand_T.rename_axis('index').reset_index()
        df_demand_final['index'] = df_demand_final['index'].astype(int)
        df_demand_merged = df_demand_final.merge(df_nodes, on="index",how='left')



        #reading in generation (erst mergen mit den BZ Scenarios und den NUTS)
        #TODO: haben wir doppelte generation von offshore wind drin? müssen wir den DF filtern? Mergen mit den BZ Scenarios anhand der Nodes Index?
        #TODO: überall wo df steht self austauschen?
        #generators = pd.read_csv("data\import_data\generators_filtered.csv", sep=";", index_col=0)
        df_generators = pd.read_csv("data\\import_data\\generators_filtered.csv", sep=";", index_col=0)
        #mergen der nodes der OffBZ in den generator df
        df_generators_merged = df_nodes.merge(df_generators[['index', 'p_nom', 'carrier', 'marginal_cost', 'efficiency']], on="index",how='left')
        df_offshore = df_generators_merged.loc[(df_generators_merged['carrier'].isin(['offwind-ac','offwind-dc']))]
        #df_only_offshore_wind = df_generators_merged.drop(columns= carrier[('CCGT', 'onwind', 'solar', 'ror', 'nucelar', 'biomass', 'coal'))

        # aufsummieren der gesamt capacity je carrier and zone !! BZ2 ist eingesetzt! warum geht hier nicht: self.scen?
        df_generators_merged['Total conventional Capacity'] = df_generators_merged.groupby([run_parameter.scen, 'carrier'])['p_nom'].transform('sum')
 #von Paul:       test = df_generators_merged.groupby([run_parameter.scen, 'carrier']).sum(numeric_only=True)[['p_nom']]

        df_total_cap = df_generators_merged.drop_duplicates(subset=[run_parameter.scen, 'carrier'])

        #df_generators_merged.to_csv('generators_merged.csv', index=False)
        #self.generators = df_generators_merged


        #TODO: die OffBZ (also fie carrier offshore) ausgliedern und OffBZ hinzufügen? Sollten wir das tun?
        #TODO: die restlichen OffBZ mit carrier und marginal costs eintragen
            #filter = offwind_ac, offwind_dc und nan
            #options_offwind = ['offwind-ac', 'offwind-dc', '']
            # selecting rows based on condition
            #df_offwind = df_generators_merged.loc[df_generators_merged['carrier'].isin(options_offwind)]
        #merged_df = df1.merge(df2, on="Name",suffixes=('_left', '_right'))

        #auf Basis der Zones:
        # 1) conventionals
        #TODO: Komma Fehler beim einlesen in der Excel
            #conventionals_raw = pd.read_csv("data\\import_data\\conventionals_filtered.csv", sep=";", index_col=0)
#            conventionals_filtered = df_generators_merged[df_generators_merged["carrier"].isin(["CCGT", "OCGT", "nuclear", "biomass", "coal", "lignite", "oil"])]
            #kann nicht weiter gefiltered werden, weil es dann probleme bei den versch. Scen gibt mit "BZ2"
            #conventionals = conventionals_filtered[
                #["p_nom", "carrier", "marginal_cost", "efficiency", "co2_fac", "index", "bidding_zone"]].reset_index(
                #drop=True)
            #conventionals.columns = ["P_inst", "type", "mc", "efficiency", "co2_fac", "bus", "bidding_zone"]
#           self.conventionals = conventionals_filtered

        # TODO: Aufsummieren der Convetionals nach BZ und Fuel Type und Capacities
        #Funktion zum groupen und aufsummeiren der generations and fuels
        #test = sjoined_nodes_states4.groupby(["NUTS_ID","Fuel"]).sum(numeric_only=True)[["bidding_zone"]]
        #auf Basis der Nodes:
        # 2) solar_filtered
        #solar_raw = pd.read_csv("data\\import_data\\solar_filtered.csv", sep=";", index_col=0)
#        solar_generation = df_generators_merged[df_generators_merged["carrier"].isin(["solar"])]
#            self.solar = solar_generation
        # 3) wind_filtered
        #wind_raw = pd.read_csv("data\\import_data\\wind_filtered.csv", sep=";", index_col=0)
#        wind_generation = df_generators_merged[df_generators_merged["carrier"].isin(["onwind", "offwind-ac", "offwind-dc"])]
#           self.wind = wind_generation

        #Funktion zum groupen und aufsummeiren der generations and fuels
        #test = sjoined_nodes_states4.groupby(["NUTS_ID","Fuel"]).sum(numeric_only=True)[["bidding_zone"]]

        #demand einlesen mit tyndp_load! Auf basis der BZ
        #tiemseries einlesen für wind und solar! mir ninja.renewables

