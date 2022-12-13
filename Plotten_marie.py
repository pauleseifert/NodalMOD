#todo: wie sollen sources zusammenaddiert werden bei genration?
#todo: storage charge muss negativ werden


import pandas as pd
import time
from selenium import webdriver
import os
from colour import Color
from dataclasses import dataclass
import pickle
import timeit
import geopandas as gpd
import gurobipy as gp
import pandas as pd
from gurobipy import GRB
from gurobipy import Model
from collections import ChainMap
import plotly.graph_objects as go
from sys import platform
import sys
from import_object_data_Zonal_Configuration import run_parameter, kpi_data
import plotly.io as pio
import plotly.express as px

import os
import pandas as pd

from import_object_data_Zonal_Configuration import kpi_data, run_parameter, model_data
from printing_funct import plot_bar2_electrolyser, kpi_development2, radar_chart, plot_generation, plotly_maps_bubbles, \
    plotly_maps_lines, plotly_maps_size_lines, plotly_maps_lines_hours

import pickle
import timeit
import geopandas as gpd
import gurobipy as gp
import pandas as pd
from gurobipy import GRB
from gurobipy import Model
from collections import ChainMap

from helper_functions import ren_helper2, demand_helper2, create_encyclopedia
from import_object_data_Zonal_Configuration import model_data, run_parameter


#scen_folder = run_parameter.export_folder +"maps/sensitivity/"
#scen = 'BAU'

#plot_generation(generation_temporal = kpis[scen].generation_temporal, maps_folder=scen_folder, scen=scen)
#plot_generation(generation_temporal = kpis[scen].generation_temporal, maps_folder=scen_folder, scen=scen)


#def plot_generation(generation_temporal, maps_folder, scen, year):
#    scaling_factor = 1000
#    fig = go.Figure()
#    if scen != 1:
#        fig.add_trace(go.Scatter(x=generation_temporal[year].electrolyser.index,y=-generation_temporal[year].electrolyser / scaling_factor, name='Electrolysis', fill='tozeroy', stackgroup='two', mode='none', fillcolor="rgba(0, 230, 230, 0.8)"))
#        fig.add_trace(go.Scatter(x=generation_temporal[year].C_S.index, y=-generation_temporal[year].C_S / scaling_factor, name='Storage charge', fill='tonexty', stackgroup='two', mode='none',fillcolor="rgba(153, 0, 153, 0.8)"))



#df = px.data.iris() # iris is a pandas DataFrame
#fig = px.scatter(df, x="sepal_width", y="sepal_length")
#pio.write_image(fig=fig, file="results//test_energy_generation.png", width=2000, height=800)

run_parameter = run_parameter(scenario_name = "Offshore_Bidding_Zone_Scenario")
data = model_data(create_res = False,reduced_ts = False, export_files= False, run_parameter = run_parameter)

#read in
if platform == "linux" or platform == "linux2":
    directory = "/work/seifert/"
    case_name = sys.argv[1]
    scen = int(sys.argv[2])
elif platform == "win32":
    directory = ""
    case_name = "Offshore_Bidding_Zone_Scenario"
    scen = run_parameter.scen
    years = [0]

match run_parameter.scen:
    case "BAU":number_zones = 20
    case "BZ2":number_zones = 21
    case "BZ3":number_zones = 22
    case "BZ5":number_zones = 24



##########################
###total generation######
#########################

#prepare conventionals:
#for z in range(number_zones):
#    df_P_C_0 = pd.read_csv("results/"+ str(run_parameter.case_name) +"/"+ str(run_parameter.scen) +"/subscen"+ str(run_parameter.sensitivity_scen) + "/" + str(number_zones) + "_P_C.csv", usecols=range(1,9))
match run_parameter.scen:
    case "BAU":
        df_P_C_0 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/0_P_C.csv", usecols=range(9))
        df_P_C_1 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/1_P_C.csv", usecols=range(9))
        df_P_C_2 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/2_P_C.csv", usecols=range(9))
        df_P_C_3 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/3_P_C.csv", usecols=range(9))
        df_P_C_4 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/4_P_C.csv", usecols=range(9))
        df_P_C_5 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/5_P_C.csv", usecols=range(9))
        df_P_C_6 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/6_P_C.csv", usecols=range(9))
        df_P_C_7 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/7_P_C.csv", usecols=range(9))
        df_P_C_8 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/8_P_C.csv", usecols=range(9))
        df_P_C_9 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/9_P_C.csv", usecols=range(9))
        df_P_C_10 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/10_P_C.csv", usecols=range(9))
        df_P_C_11 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/11_P_C.csv", usecols=range(9))
        df_P_C_12 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/12_P_C.csv", usecols=range(9))
        df_P_C_13 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/13_P_C.csv", usecols=range(9))
        df_P_C_14 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/14_P_C.csv", usecols=range(9))
        df_P_C_15 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/15_P_C.csv", usecols=range(9))
        df_P_C_16 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/16_P_C.csv", usecols=range(9))
        df_P_C_17 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/17_P_C.csv", usecols=range(9))
        df_P_C_18 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/18_P_C.csv", usecols=range(9))
        df_P_C_19 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/19_P_C.csv", usecols=range(9))

        df_sum_p_c = df_P_C_1 + df_P_C_2 + df_P_C_3 + df_P_C_4 + df_P_C_5 + df_P_C_6 + df_P_C_7 + df_P_C_8 + df_P_C_9 + df_P_C_10 + df_P_C_11 + df_P_C_12+ df_P_C_13 + df_P_C_14 + df_P_C_15 + df_P_C_16 + df_P_C_17 + df_P_C_18 + df_P_C_19

    case "BZ2":

        df_P_C_0 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/0_P_C.csv", usecols=range(9))
        df_P_C_1 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/1_P_C.csv", usecols=range(9))
        df_P_C_2 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/2_P_C.csv", usecols=range(9))
        df_P_C_3 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/3_P_C.csv", usecols=range(9))
        df_P_C_4 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/4_P_C.csv", usecols=range(9))
        df_P_C_5 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/5_P_C.csv", usecols=range(9))
        df_P_C_6 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/6_P_C.csv", usecols=range(9))
        df_P_C_7 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/7_P_C.csv", usecols=range(9))
        df_P_C_8 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/8_P_C.csv", usecols=range(9))
        df_P_C_9 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/9_P_C.csv", usecols=range(9))
        df_P_C_10 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/10_P_C.csv", usecols=range(9))
        df_P_C_11 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/11_P_C.csv", usecols=range(9))
        df_P_C_12 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/12_P_C.csv", usecols=range(9))
        df_P_C_13 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/13_P_C.csv", usecols=range(9))
        df_P_C_14 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/14_P_C.csv", usecols=range(9))
        df_P_C_15 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/15_P_C.csv", usecols=range(9))
        df_P_C_16 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/16_P_C.csv", usecols=range(9))
        df_P_C_17 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/17_P_C.csv", usecols=range(9))
        df_P_C_18 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/18_P_C.csv", usecols=range(9))
        df_P_C_19 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/19_P_C.csv", usecols=range(9))
        df_P_C_20 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/20_P_C.csv", usecols=range(9))

        df_sum_p_c = df_P_C_1 + df_P_C_2 + df_P_C_3 + df_P_C_4 + df_P_C_5 + df_P_C_6 + df_P_C_7 + df_P_C_8 + df_P_C_9 + df_P_C_10 + df_P_C_11 + df_P_C_12+ df_P_C_13 + df_P_C_14 + df_P_C_15 + df_P_C_16 + df_P_C_17 + df_P_C_18 + df_P_C_19+ df_P_C_20

    case "BZ3":
        df_P_C_0 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/0_P_C.csv", usecols=range(9))
        df_P_C_1 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/1_P_C.csv", usecols=range(9))
        df_P_C_2 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/2_P_C.csv", usecols=range(9))
        df_P_C_3 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/3_P_C.csv", usecols=range(9))
        df_P_C_4 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/4_P_C.csv", usecols=range(9))
        df_P_C_5 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/5_P_C.csv", usecols=range(9))
        df_P_C_6 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/6_P_C.csv", usecols=range(9))
        df_P_C_7 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/7_P_C.csv", usecols=range(9))
        df_P_C_8 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/8_P_C.csv", usecols=range(9))
        df_P_C_9 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/9_P_C.csv", usecols=range(9))
        df_P_C_10 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/10_P_C.csv", usecols=range(9))
        df_P_C_11 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/11_P_C.csv", usecols=range(9))
        df_P_C_12 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/12_P_C.csv", usecols=range(9))
        df_P_C_13 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/13_P_C.csv", usecols=range(9))
        df_P_C_14 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/14_P_C.csv", usecols=range(9))
        df_P_C_15 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/15_P_C.csv", usecols=range(9))
        df_P_C_16 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/16_P_C.csv", usecols=range(9))
        df_P_C_17 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/17_P_C.csv", usecols=range(9))
        df_P_C_18 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/18_P_C.csv", usecols=range(9))
        df_P_C_19 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/19_P_C.csv", usecols=range(9))
        df_P_C_20 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/20_P_C.csv", usecols=range(9))
        df_P_C_21 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/21_P_C.csv", usecols=range(9))

        df_sum_p_c = df_P_C_1 + df_P_C_2 + df_P_C_3 + df_P_C_4 + df_P_C_5 + df_P_C_6 + df_P_C_7 + df_P_C_8 + df_P_C_9 + df_P_C_10 + df_P_C_11 + df_P_C_12+ df_P_C_13 + df_P_C_14 + df_P_C_15 + df_P_C_16 + df_P_C_17 + df_P_C_18 + df_P_C_19+ df_P_C_20 + df_P_C_21

    case "BZ5":
        df_P_C_0 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/0_P_C.csv", usecols=range(9))
        df_P_C_1 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/1_P_C.csv", usecols=range(9))
        df_P_C_2 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/2_P_C.csv", usecols=range(9))
        df_P_C_3 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/3_P_C.csv", usecols=range(9))
        df_P_C_4 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/4_P_C.csv", usecols=range(9))
        df_P_C_5 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/5_P_C.csv", usecols=range(9))
        df_P_C_6 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/6_P_C.csv", usecols=range(9))
        df_P_C_7 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/7_P_C.csv", usecols=range(9))
        df_P_C_8 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/8_P_C.csv", usecols=range(9))
        df_P_C_9 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/9_P_C.csv", usecols=range(9))
        df_P_C_10 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/10_P_C.csv", usecols=range(9))
        df_P_C_11 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/11_P_C.csv", usecols=range(9))
        df_P_C_12 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/12_P_C.csv", usecols=range(9))
        df_P_C_13 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/13_P_C.csv", usecols=range(9))
        df_P_C_14 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/14_P_C.csv", usecols=range(9))
        df_P_C_15 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/15_P_C.csv", usecols=range(9))
        df_P_C_16 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/16_P_C.csv", usecols=range(9))
        df_P_C_17 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/17_P_C.csv", usecols=range(9))
        df_P_C_18 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/18_P_C.csv", usecols=range(9))
        df_P_C_19 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/19_P_C.csv", usecols=range(9))
        df_P_C_20 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/20_P_C.csv", usecols=range(9))
        df_P_C_21 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/21_P_C.csv", usecols=range(9))
        df_P_C_22 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/22_P_C.csv", usecols=range(9))
        df_P_C_23 = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/23_P_C.csv", usecols=range(9))

        df_sum_p_c = df_P_C_1 + df_P_C_2 + df_P_C_3 + df_P_C_4 + df_P_C_5 + df_P_C_6 + df_P_C_7 + df_P_C_8 + df_P_C_9 + df_P_C_10 + df_P_C_11 + df_P_C_12+ df_P_C_13 + df_P_C_14 + df_P_C_15 + df_P_C_16 + df_P_C_17 + df_P_C_18 + df_P_C_19+ df_P_C_20 + df_P_C_21 +df_P_C_22 + df_P_C_23


del df_sum_p_c['Unnamed: 0']
df_sum_p_c = df_sum_p_c.div(1000)


#prepare curtailment
df_curtail = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/_res_curtailment.csv")
del df_curtail['Unnamed: 0']
df_curtail = df_curtail.sum(axis=1)
df_curtail = df_curtail.div(1000)

#prepare storage
df_storage_charge= pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/_S_ext.csv")
df_storage_charge = df_storage_charge.sum(axis=1)
df_storage_charge = df_storage_charge.div(1000)
#todo: discharge wird nicht als negativ angezeigt ?
df_storage_discharge= pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/_S_inj.csv")
df_storage_discharge = df_storage_discharge.sum(axis=1)
df_storage_discharge = df_storage_discharge.div(1000)
df_storage_discharge = - df_storage_discharge
#prepare res
df_ror = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/ror_supply.csv")
del df_ror['Unnamed: 0']
df_ror = df_ror.sum(axis=1)
df_ror = df_ror.div(1000)

#renewables todo: davon habe ich curtailment abgezogen, ist das korrekt?
df_res = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/renewables.csv")
del df_res['Unnamed: 0']
df_res = df_res.sum(axis=1)
df_res = df_res.div(1000)
df_res = df_res - df_curtail

#prepare DAM
df_ror = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/ror_supply.csv")
del df_ror['Unnamed: 0']
df_ror = df_ror.sum(axis=1)
df_ror = df_ror.div(1000)


fig = go.Figure()
fig.add_trace(go.Scatter(x=df_sum_p_c.index, y= df_sum_p_c['2'],name='Biomass', fill='tonexty', stackgroup='one', mode='none', fillcolor="rgba(1, 135, 4,  0.8)"))
fig.add_trace(go.Scatter(x=df_sum_p_c.index, y= df_sum_p_c['5'], name= 'Nuclear', fill = 'tonexty', stackgroup= 'one', mode= 'none', fillcolor= "rgba(255, 0, 0, 0.8)"))
fig.add_trace(go.Scatter(x=df_sum_p_c.index, y= df_sum_p_c['1'], name= 'Coal', fill = 'tonexty', stackgroup= 'one', mode= 'none', fillcolor= "rgba(77, 38, 0, 0.8)"))
fig.add_trace(go.Scatter(x=df_sum_p_c.index, y= df_sum_p_c['6'], name= 'Lignite', fill = 'tonexty', stackgroup= 'one', mode= 'none', fillcolor= "rgba(179, 89, 0, 0.8)"))
fig.add_trace(go.Scatter(x=df_sum_p_c.index, y= df_sum_p_c['7'], name= 'Oil', fill = 'tonexty', stackgroup= 'one', mode= 'none', fillcolor= "rgba(0, 0, 0, 0.8)"))
fig.add_trace(go.Scatter(x=df_sum_p_c.index, y= df_sum_p_c['0'], name= 'CCGT', fill = 'tonexty', stackgroup= 'one', mode= 'none', fillcolor= "rgb(255, 218, 30)"))
fig.add_trace(go.Scatter(x=df_sum_p_c.index, y= df_sum_p_c['4'], name='OCGT', fill='tonexty', stackgroup='one',mode='none', fillcolor="rgba(255, 165, 0, 0.8)"))
fig.add_trace(go.Scatter(x=df_sum_p_c.index, y= df_sum_p_c['3'], name= 'HDAM', fill = 'tonexty', stackgroup= 'one', mode= 'none', fillcolor= "rgba(208, 194, 172, 0.8)"))
fig.add_trace(go.Scatter(x=df_storage_discharge.index, y= df_storage_discharge.tolist(),name='Storage discharge', fill='tonexty', stackgroup='one', mode='none', fillcolor="rgba(145, 0, 0, 0.8)"))
fig.add_trace(go.Scatter(x=df_ror.index, y= df_ror.tolist(),name='Run of River', fill='tonexty', stackgroup='one', mode='none', fillcolor="rgba(198, 210, 190, 0.8)"))
fig.add_trace(go.Scatter(x=df_storage_charge.index, y= df_storage_charge.tolist(),name='Storage charge', fill='tonexty', stackgroup='one', mode='none', fillcolor="rgba(145, 0, 0, 0.8)"))
fig.add_trace(go.Scatter(x=df_res.index, y= df_res.tolist(),name='Renewables', fill='tonexty', stackgroup='one', mode='none', fillcolor="rgba(0, 0, 255, 0.8)"))
fig.add_trace(go.Scatter(x=df_curtail.index, y= df_curtail.tolist(),name='Curtailment', fill='tonexty', stackgroup='one', mode='none', fillcolor="rgba(68, 218, 247)"))



fig.update_layout(
    xaxis= dict(title='Timesteps',dtick=24),
    yaxis=dict(title='Generation [GW]'),
    font = dict(size = 30,
                    #family = "Serif"
                    ),
    legend=dict(x=0, y=-0.2, bgcolor='rgba(255, 255, 255, 0)', bordercolor='rgba(255, 255, 255, 0)',font_size = 22),
    legend_orientation="h",
    plot_bgcolor='rgba(0,0,0,0)',
    yaxis_gridcolor = "rgba(166, 166, 166, 0.5)"
)
#fig.write_html(maps_folder + "scen_"+str(scen) +"_year_"+str(year) + "_electricity_gen.html")
pio.write_image(fig=fig, file="results//" + str(run_parameter.scen) + "energy_generation.png" , width=2000, height=800)

##########################
#####Bubbles map#########
#########################

Z_dict = {}
keys = range(len(data.nodes[run_parameter.scen].unique()))
values = data.nodes[run_parameter.scen].unique()
for i in keys:
        Z_dict[i] = values[i]

test = pd.DataFrame(Z_dict.items(), columns = ["index","zones"]).set_index("index")

#prepare curtailment
df_curtail = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/_res_curtailment.csv")
del df_curtail['Unnamed: 0']
df_curtail = df_curtail.sum(axis=0)
df_curtail.rename(index=Z_dict)
df_curtail = df_curtail.div(1000) # dann ist es in GWh!

df_curtail_merged = df_curtail.merge(Z_dict, on="index",how='left')

#for scen in scenarios: plotly_maps_bubbles(df=kpis[scen].curtailment.location, scen=scen, maps_folder=scen_folder, name="curtailment", size_scale=2, unit="TWh", title="Curtailment", year=2)



unit_dict = {"TWh": 1000000, "GWh":1000, "GW":1000}
columns = ["LAT", "LON"]
fig = go.Figure()
df = df[columns]
#for idx, row in df.iterrows():
fig.add_trace(
    go.Scattermapbox(
        lon=df['LON'],
        lat=df['LAT'],
        text=df.index,
        marker=dict(
            size=round(df[year]/(unit_dict[unit]/5000*size_scale),5) if flexible_bubble_size == True else 30,
            color=round(df[year]/unit_dict[unit],5),
            sizemode='area',
            showscale = True,
            colorbar=dict(title=title+" in "+unit, orientation = "h", y = -0.1, xpad = 100),
            cmin = min_legend,
            cmax = max_legend
        ),
        name="Node"+str(df.index),
    ),
)
fig = plotly_map(fig, 4.7*zoom, hoffset = hoffset, voffset=voffset)
fig.update_layout(
    font = dict(size=30)
)
#fig.show()


#fig.write_html(maps_folder + str(scen) + "_"+name+".html")
fig.write_image(maps_folder + "scen_"+str(scen) +"_year_"+str(year) + "_"+name+".pdf", width=2000, height=1600)

