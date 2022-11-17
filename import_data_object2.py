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
            self.directory = "\Users\Dell\Documents\GitHub\MulCarNI" #directory anpassen
            self.case_name = scenario_name
            self.years = 1
            self.timesteps = 10
            self.scen = 1
            self.sensitivity_scen = 0
            self.scen = 1  # hier szenario 5 erstellen für zonen
            self.sensitivit_scen = 0
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
            case 1:
                self.electrolyser = []
                print("BASE case")
            #case 2:

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
        #TODO: durchlaufen lassen ohne das man es einzeln machen muss?
        #df_nodes = pd.read_csv(run_parameter.import_folder + "import_data/df_nodes_to_zones_filtered_final.csv",sep=";", index_col=0)
        #df_nodes = pd.read_csv("data/import_data/df_nodes_to_zones_filtered_final.csv", sep=";", index_col=0)

        lookup_dictBZ2 = {"DEF": "DEII1", "DE6": "DEII1", "DE9": "DEII1", "DE3": "DEII1", "DE4": "DEII1",
                          "DE8": "DEII1", "DED": "DEII1", "DEE": "DEII1", "DEG": "DEII1", "DEA": "DEII2",
                          "DEB": "DEII2", "DEC": "DEII2", "DE1": "DEII2", "DE2": "DEII2", "DE7": "DEII2",
                          "OffBZN": "OffBZN", "OffBZB": "OffBZB"}
        lookup_dictBZ3 = {"DEF": "DEII1", "DE6": "DEII1", "DE9": "DEII1", "DE3": "DEII2", "DE4": "DEII2",
                          "DE8": "DEII2", "DED": "DEII2", "DEE": "DEII2", "DEG": "DEII2", "DEA": "DEII3",
                          "DEB": "DEII3", "DEC": "DEII3", "DE1": "DEII3", "DE2": "DEII3", "DE7": "DEII3",
                          "OffBZN": "OffBZN", "OffBZB": "OffBZB"}
        lookup_dictBZ5 = {"DEF": "DEII1", "DE6": "DEII2", "DE9": "DEII2", "DE3": "DEII3", "DE4": "DEII3",
                          "DE8": "DEII3", "DED": "DEII3", "DEE": "DEII3", "DEG": "DEII3", "DEA": "DEII4",
                          "DEB": "DEII4", "DEC": "DEII4", "DE1": "DEII5", "DE2": "DEII5", "DE7": "DEII5",
                          "OffBZN": "OffBZN", "OffBZB": "OffBZB"}

        def lookup(row):
            try:
                value = lookup_dictBZ5[row["NUTS_ID"]]
            except:
                value = row["country_y"]
            return value

        df_nodes['BZ_2'] = df_nodes.apply(lambda row: lookup(row), axis=1)
        df_nodes['BZ_3'] = df_nodes.apply(lambda row: lookup(row), axis=1)
        df_nodes['BZ_5'] = df_nodes.apply(lambda row: lookup(row), axis=1)
        self.nodes = pd.read_csv("data/import_data/df_nodes.csv", sep=";", index_col=0)

        #reading in NTCs
        #TODO: flexilines brauchen eine capacity zuweisung - Recherche? Offshore windfarms = Kinis Daten, BHEH =3000, NSEH1+2 = jeweils 10000
        self.ntc_BZ_2 = pd.read_csv("data\import_data\NTC\NTC_BZ_2.csv", sep=";", index_col=0)
        self.ntc_BZ_3 = pd.read_csv("data\import_data\NTC\NTC_BZ_3.csv", sep=";", index_col=0)
        self.ntc_BZ_5 = pd.read_csv("data\import_data\NTC\NTC_BZ_5.csv", sep=";", index_col=0)

        #reading in generation:

        #Funktion zum groupen und aufsummeiren der generations and fuels
        #test = sjoined_nodes_states4.groupby(["NUTS_ID","Fuel"]).sum(numeric_only=True)[["bidding_zone"]]

        generators_raw = pd.read_csv(run_parameter.import_folder + "PyPSA_elec1024/generators.csv", index_col=0)
        lines_raw = pd.read_csv(run_parameter.import_folder + "PyPSA_elec1024/lines.csv", index_col=0)
        links_raw = pd.read_csv(run_parameter.import_folder + "PyPSA_elec1024/links.csv", index_col=0)
        load_raw = pd.read_csv(run_parameter.import_folder + "PyPSA_elec1024/load.csv",index_col=0).reset_index(drop=True)
        ror_ts = pd.read_csv(run_parameter.import_folder + "PyPSA_elec1024/hydro_ror_ts.csv", low_memory=False)
        dam_maxsum_ts = pd.read_csv(run_parameter.import_folder + "PyPSA_elec1024/hydro_dam_ts.csv",low_memory=False)
        hydro_database = pd.read_csv(run_parameter.import_folder + "jrc-hydro-power-plant-database.csv")

