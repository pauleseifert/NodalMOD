#todo negativ scatter plot

import matplotlib.pyplot as plt
from sys import platform
import pickle
import sys
import plotly.io as pio
import seaborn as sb
import numpy as np
import plotly.graph_objects as go
from colour import Color
import pandas as pd
from import_object_data_Zonal_Configuration import model_data, run_parameter
from matplotlib import font_manager


#plt.rcParams["font.family"] = "Times New Roman"

run_parameter = run_parameter(scenario_name = "Offshore_Bidding_Zone_Scenario")
data = model_data(create_res = False,reduced_ts = False, export_files= False, run_parameter = run_parameter)


# read in
if platform == "linux" or platform == "linux2":
    directory = "/work/seifert/"
    case_name = sys.argv[1]
    scen = int(sys.argv[2])
elif platform == "win32":
    directory = ""
    case_name = "Offshore_Bidding_Zone_Scenario"
    scen = run_parameter.scen
    years = [0]


def change_column_to_int(item):
    for y in run_parameter.years:
        item[y].columns = item[y].columns.astype(int)
    return item

match run_parameter.scen:
    case "BAU":
        number_zones = 20
    case "BZ2":
        number_zones = 21
    case "BZ3":
        number_zones = 22
    case "BZ5":
        number_zones = 24

#Todo:set PLOT 1=generation, 2=bubbles 3=price 4 = ntc
plot = 1
match plot:
    case 1:
        #renewables
        #with open("data/final_readin_data/share_wind.pkl", 'rb') as f df_wind_share = pickle.load(f)
        #df_wind_share = pd.read_pickle("data/final_readin_data/share_wind.pkl")
       # df_wind_share = change_column_to_int(df_wind_share)

        df_solar_share = pd.read_excel("data/final_readin_data/share_solar.xlsx")
        df_wind_share = pd.read_excel("data/final_readin_data/share_wind.xlsx")


        solar_share = df_solar_share.transpose()
        solar_share = solar_share.mean()
        #solar_share['index'] = solar_share.index
        #solar_share['index']  = solar_share['index'] .astype(str).astype(int)
        #solar_share = solar_share.merge(data.nodes[['index', run_parameter.scen]], on="index", how='left')
        #del solar_share['index']
        #solar_share= solar_share.groupby([run_parameter.scen]).mean()
        #solar_share = solar_share.transpose()

        wind_share = df_wind_share.transpose()
        wind_share = wind_share.mean()
        #wind_share['index']  = wind_share['index'] .astype(str).astype(int)
        #wind_share = wind_share.merge(data.nodes[['index', run_parameter.scen]], on="index", how='left')
        #del wind_share['index']
        #wind_share = wind_share.groupby([run_parameter.scen]).mean()
        #wind_share = wind_share.transpose()



        #with open('data/final_readin_data/share_wind.pkl', 'rb') as f:share_solar_raw = pickle.load(f)
         #   share_solar_raw = self.change_column_to_int(share_solar_raw)
        #with open('data/final_readin_data/share_solar.pkl', 'rb') as f: share_wind_raw = pickle.load(f)
        #    share_wind_raw = self.change_column_to_int(share_wind_raw)

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
        df_sum_p_c['gas'] = df_sum_p_c['0'] + df_sum_p_c['4']



        #prepare curtailment
        df_curtail = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/_res_curtailment.csv")
        del df_curtail['Unnamed: 0']
        df_curtail = df_curtail.sum(axis=1)
        df_curtail = df_curtail.div(1000)

        #prepare storage
        df_storage_charge= pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/_S_ext.csv")
        df_storage_charge = df_storage_charge.sum(axis=1)
        df_storage_charge = - df_storage_charge.div(1000)
        #todo: discharge wird nicht als negativ angezeigt ?
        df_storage_discharge= pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/_S_inj.csv")
        df_storage_discharge = df_storage_discharge.sum(axis=1)
        df_storage_discharge = df_storage_discharge.div(1000)
        df_storage_discharge = df_storage_discharge
        #prepare ror
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
        df_res_solar = df_res * solar_share
        df_res_wind = df_res * wind_share

        #prepare DAM
        df_ror = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/ror_supply.csv")
        del df_ror['Unnamed: 0']
        df_ror = df_ror.sum(axis=1)
        df_ror = df_ror.div(1000)
        df_sum_p_c['hydro'] = df_sum_p_c['3'] + df_ror.tolist()

        #prepare demand
        df_demand = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/demand.csv")
        del df_demand['Unnamed: 0']
        df_demand = df_demand.sum(axis=1)
        df_demand = df_demand.div(1000)

        #prepare demand
        df_load_lost = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/_p_load_lost.csv")
        del df_load_lost['Unnamed: 0']
        df_load_lost = df_load_lost.sum(axis=1)
        df_load_lost = df_load_lost.div(1000)

#test
        df_test = df_demand - df_storage_charge

        fig = go.Figure()
        #fig.add_trace(go.Scatter(x=df_test.index, y= df_test.tolist(),name='Demand', mode = 'lines'))
        #fig.add_trace(go.Scatter(x=df_demand.index, y= df_demand.tolist(),name='Demand', mode = 'lines'))
        fig.add_trace(go.Scatter(x=df_sum_p_c.index, y= df_sum_p_c['2'],name='Biomass', fill='tonexty', stackgroup='one', mode='none', fillcolor="rgb(255, 160, 251)"))
        fig.add_trace(go.Scatter(x=df_sum_p_c.index, y= df_sum_p_c['5'], name= 'Nuclear', fill = 'tonexty', stackgroup= 'one', mode= 'none', fillcolor= "rgb(199, 241, 21)"))
        fig.add_trace(go.Scatter(x=df_sum_p_c.index, y= df_sum_p_c['1'], name= 'Coal', fill = 'tonexty', stackgroup= 'one', mode= 'none', fillcolor= "rgb(105, 60, 37)"))
        fig.add_trace(go.Scatter(x=df_sum_p_c.index, y= df_sum_p_c['6'], name= 'Lignite', fill = 'tonexty', stackgroup= 'one', mode= 'none', fillcolor= "rgb(128, 112, 72)"))
        fig.add_trace(go.Scatter(x=df_sum_p_c.index, y= df_sum_p_c['7'], name= 'Oil', fill = 'tonexty', stackgroup= 'one', mode= 'none', fillcolor= "rgba(0, 0, 0, 0.8)"))
        fig.add_trace(go.Scatter(x=df_sum_p_c.index, y= df_sum_p_c['gas'], name= 'Gas', fill = 'tonexty', stackgroup= 'one', mode= 'none', fillcolor= "rgb(227, 212, 133)"))
        #fig.add_trace(go.Scatter(x=df_sum_p_c.index, y= df_sum_p_c['4'], name='OCGT', fill='tonexty', stackgroup='one',mode='none', fillcolor="rgba(255, 165, 0, 0.8)"))
        fig.add_trace(go.Scatter(x=df_sum_p_c.index, y= df_sum_p_c['hydro'], name= 'Hydro', fill = 'tonexty', stackgroup= 'one', mode= 'none', fillcolor= "rgb(117, 212, 253)"))
        fig.add_trace(go.Scatter(x=df_storage_discharge.index, y= df_storage_discharge.tolist(),name='Storage discharge', fill='tonexty', stackgroup='one', mode='none', fillcolor="rgba(145, 0, 0, 0.8)"))
        #fig.add_trace(go.Scatter(x=df_ror.index, y= df_ror.tolist(),name='Run of River', fill='tonexty', stackgroup='one', mode='none', fillcolor="rgba(198, 210, 190, 0.8)"))
        fig.add_trace(go.Scatter(x=df_storage_charge.index, y= df_storage_charge.tolist(),name='Storage charge', fill='tozeroy', stackgroup='two', mode='none', fillcolor="rgb(174, 160, 251)"))
        #fig.add_trace(go.Scatter(x=df_res.index, y= df_res.tolist(),name='Renewables', fill='tonexty', stackgroup='one', mode='none', fillcolor="rgba(0, 0, 255, 0.8)"))
        fig.add_trace(go.Scatter(x=df_curtail.index, y= df_curtail.tolist(),name='Curtailment', fill='tonexty', stackgroup='one', mode='none', fillcolor="rgb(254, 3, 2)"))
        fig.add_trace(go.Scatter(x=df_load_lost.index, y= df_load_lost.tolist(),name='Loss Load', fill='tonexty', stackgroup='one', mode='none', fillcolor="rgb(254, 134, 98)"))
        fig.add_trace(go.Scatter(x=df_res_wind.index, y= df_res_wind.tolist(),name='Wind', fill='tonexty', stackgroup='one', mode='none', fillcolor="rgba(0, 0, 255, 0.8)"))
        fig.add_trace(go.Scatter(x=df_res_solar.index, y= df_res_solar.tolist(),name='Solar', fill='tonexty', stackgroup='one', mode='none', fillcolor="rgb(255, 252, 38)"))

        fig.update_layout(
            xaxis= dict(title='Timesteps',dtick=24),#.format(font='Times New Roman'),
            yaxis=dict(title='Generation [GW]'),#.format(font='Times New Roman'),
            font = dict(size = 30,
                            family = "Times New Roman"
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
    case 2:
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


###################
###price##########
###################
    case 3:
        test = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/model_data.csv")
        test = pd.DataFrame(test)
        test.drop(['Unnamed: 0'], axis=1)
        test.drop(['index'], axis=1)

        BAU = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str('BAU') + "/subscen" + str(run_parameter.sensitivity_scen) + "/model_data.csv")
        BAU = pd.DataFrame(test)
        BAU.drop(['Unnamed: 0'], axis=1)
        BAU.drop(['index'], axis=1)


        if run_parameter.scen == 'BZ5':
            price_duration_BAU = BAU[['2']].copy()
            #price_duration_BAU = price_duration_BAU.pivot_table(columns=['2'], aggfunc='size')
            #price_duration_BAU = pd.DataFrame(price_duration_BAU.items(), columns=["index", "duration"])#.set_index("index")
            #price_duration_BAU.sort_values(by=['index'])
            #price_duration_BAU['aggregated'] = [503, 501,500,457,446, 335, 325,130, 80, 49,48, 29,28,1]
            #price_duration_BAU.loc[len(price_duration_BAU.index)] = [152.28091, 0, 0]

            price_duration_DEV1 = test[['6']].copy()
            #price_duration_DEV1 = price_duration_DEV1.pivot_table(columns=['6'], aggfunc='size')
            #price_duration_DEV1 = pd.DataFrame(price_duration_DEV1.items(), columns=["index", "duration"])#.set_index("index")
            #price_duration_DEV1.sort_values(by=['index'])
            #price_duration_DEV1['aggregated'] = [504,426,425, 421,167,115,24,23,10]
            #price_duration_DEV1.loc[len(price_duration_DEV1.index)] = [3000, 0, 0]


            price_duration_DEV2 = test[['4']].copy()
            #price_duration_DEV2= price_duration_DEV2.pivot_table(columns=['4'], aggfunc='size')
            #price_duration_DEV2 = pd.DataFrame(price_duration_DEV2.items(), columns=["index", "duration"])#.set_index("index")
            #price_duration_DEV2.sort_values(by=['index'])
            #price_duration_DEV2['aggregated'] = [504, 496,491, 488,425,424,423,9,1]
            #price_duration_DEV2.loc[len(price_duration_DEV2.index)] = [104.98408, 0, 0]

            price_duration_DEV3 = test[['5']].copy()
            #price_duration_DEV3= price_duration_DEV3.pivot_table(columns=['5'], aggfunc='size')
            #price_duration_DEV3 = pd.DataFrame(price_duration_DEV3.items(), columns=["index", "duration"])#.set_index("index")
            #price_duration_DEV3.sort_values(by=['index'])
            #price_duration_DEV3['aggregated'] = [504,495, 302, 196,186,93,91, 40,39,17,16,15, 1]
            #price_duration_DEV3.loc[len(price_duration_DEV3.index)] = [151.25231, 0, 0]

            price_duration_DEV4 = test[['3']].copy()
            #price_duration_DEV4= price_duration_DEV4.pivot_table(columns=['3'], aggfunc='size')
            #price_duration_DEV4 = pd.DataFrame(price_duration_DEV4.items(), columns=["index", "duration"])#.set_index("index")
            #price_duration_DEV4.sort_values(by=['index'])
            #price_duration_DEV4['aggregated'] = [504, 489, 262,261]
            #price_duration_DEV4.loc[len(price_duration_DEV4.index)] = [82.44828, 0, 0]

            price_duration_DEV5 = test[['2']].copy()
            #price_duration_DEV5= price_duration_DEV5.pivot_table(columns=['2'], aggfunc='size')
            #price_duration_DEV5 = pd.DataFrame(price_duration_DEV5.items(), columns=["index", "duration"])#.set_index("index")
            #price_duration_DEV5.sort_values(by=['index'])
            #price_duration_DEV5['aggregated'] = [504, 502,501,458,447,336,326,131,81,49,48,29,28,1]
            #price_duration_DEV5.loc[len(price_duration_DEV5.index)] = [152.28091, 0, 0]

            #    palette = sb.color_palette("mako_r", 5)
        #    sb.set(rc={"figure.figsize": (10, 7)})
            plt.rcParams["font.family"] = "Times New Roman"
            sb.set(rc={'axes.facecolor': 'white', 'figure.facecolor': 'white'})

            p = sb.lineplot(x=price_duration_DEV1.index, y=price_duration_DEV1['6'],label = 'DEV1', color='red')
            p = sb.lineplot(x=price_duration_DEV2.index, y=price_duration_DEV2['4'],label = 'DEV2', color='blue')
            p = sb.lineplot(x=price_duration_DEV3.index, y=price_duration_DEV3['5'], label = 'DEV3', color='green')
            p = sb.lineplot(x=price_duration_DEV4.index, y=price_duration_DEV4['3'],label = 'DEV4', color='yellow')
            p = sb.lineplot(x=price_duration_DEV5.index, y=price_duration_DEV5['2'],label = 'DEV5', color = 'pink')
            p = sb.lineplot(x=price_duration_BAU.index, y=price_duration_BAU['2'], label = 'BAU', color = 'black')

           #sns.lineplo.lineplot(x=price_duration_DEV1['duration'], y=price_duration_DEV1.index)
            plt.ylim(-120, 200)
            plt.xlim(0,505)
          #  p.set_title("Price duration", fontsize=30)
            p.set_xlabel("Time [h]", fontsize=15)
            p.set_ylabel("Price [€/MWh]", fontsize=15)
            p.font = dict(family="Times New Roman"),
 #           p.legend()
  #          p.set_style("whitegrid")
            plt.show()


########################
#########NTC###########
#######################


    case 4:

        zone_lon_lat = pd.read_excel('data/final_readin_data/zonen_lon_lat.xlsx')


        max_cap = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/lines_NTC.csv")
        max_cap = pd.DataFrame(max_cap)
        max_cap = max_cap.reset_index()
        max_cap['index'] = max_cap['index'].astype(int)


        line_flow = pd.read_csv("results/" + str(run_parameter.case_name) + "/" + str(run_parameter.scen) + "/subscen" + str(run_parameter.sensitivity_scen) + "/_F_NTC.csv")
        line_flow = pd.DataFrame(line_flow)
        line_flow  = line_flow.abs()#todo macht alle werte positiv
        line_flow = line_flow.mean(axis=0)
        line_flow = line_flow.reset_index()
        line_flow.drop(index = 0)#todo warum geht das nicht ?????????????????????
        line_flow['index'] = line_flow['index'].astype(int)

        flow = pd.merge(max_cap, line_flow, on=['index'])




        colors = list(Color("green").range_to(Color("red"), 8))
        limits = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 75)]
        text = ["0%-10%", "10%-20%", "20%-30%", "30%-40%", "40%-50%", "50%-60%", "60%-70%", "70%-75%"]
        # P_flow_DC = P_flow_DC[P_flow_DC["Pmax"]> 0.0]
        # P_flow['text'] = "AC line number "+P_flow['index'].astype(str) + '<br>Mean loading ' + (round(P_flow[0] *100,1)).astype(str) + ' %'
        # P_flow_DC['text'] = "DC line number " + P_flow_DC['index'].astype(str) + '<br>Mean loading ' + (round(P_flow_DC[0] * 100, 1)).astype(str) + ' %'
        #df = pd.concat([P_flow, P_flow_DC], axis=0, ignore_index=True).reset_index(drop=True)

        fig = go.Figure()
        limits_groups = {}
        for i in range(len(limits)):
            limits_groups.update({i: df[(df[0] < limits[i][1] / 100) & (df[0] >= limits[i][0] / 100)]})
            lons = np.empty(3 * len(limits_groups[i]))
            lons[::3] = limits_groups[i]["LON_x"]
            lons[1::3] = limits_groups[i]["LON_y"]
            lons[2::3] = None
            lats = np.empty(3 * len(limits_groups[i]))
            lats[::3] = limits_groups[i]["LAT_x"]
            lats[1::3] = limits_groups[i]["LAT_y"]
            lats[2::3] = None
            fig.add_trace(
                go.Scattermapbox(
                    lon=lons,
                    lat=lats,
                    mode='lines',
                    hovertemplate=
                    '<i>Price</i>: $%{lons:.2f}' +
                    '<br><b>X</b>: %{lats}<br>' +
                    '<b>%{text}</b>',
                    text=["line nr. " + str(row.index) for i, row in limits_groups[i].iterrows()],
                    line=dict(
                        width=3,
                        color=str(colors[i])
                    ),
                    opacity=1,
                    name=text[i],
                )
            )
        fig.add_trace(
            go.Scattermapbox(
                lon=bus['LON'],
                lat=bus['LAT'],
                hoverinfo='text',
                text=["bus nr." + str(row["index"]) for i, row in bus.iterrows()],
                mode='markers',
                marker=dict(
                    size=2,
                    color='rgb(0, 0, 0)'
                ),
                name="Bus"
            ))

        fig = plotly_map(fig, 4.7)
        fig.update_layout(
            font=dict(size=30,
                      # family = "Serif"
                      ),
            legend_title_text='Line loading',
        )
        # fig.show()
        # fig.write_html(maps_folder + str(scen) + "_line_loading.html")
        fig.write_image(maps_folder + "scen_" + str(scen) + "_line_loading.pdf", width=2000, height=1600)
        # fig.show()

        def find_duplicate_lines(self, lines):
            # get rid of multiple lines in the same columns
            grouped_lines = lines.groupby(["from", "to"]).size()
            grouped_lines = grouped_lines[grouped_lines > 1]
            for index, count in grouped_lines.items():
                duplicate_index = lines[(lines["from"] == index[0]) & (lines["to"] == index[1])].index
                lines = fix_multiple_parallel_lines(duplicate_index, lines)
            single_lines_same_order = lines.sort_index().reset_index(drop=True)

            # get rid of multiple lines in the other columns

            grouped_lines_oo = pd.concat([single_lines_same_order, single_lines_same_order.rename(
                columns={"to": "from", "from": "to"})]).groupby(["from", "to"]).size()
            grouped_lines_oo = grouped_lines_oo[grouped_lines_oo > 1]

            for index, count in grouped_lines_oo.items():
                duplicate_index = single_lines_same_order[
                    ((single_lines_same_order["from"] == index[0]) & (single_lines_same_order["to"] == index[1])) | (
                                (single_lines_same_order["to"] == index[0]) & (
                                    single_lines_same_order["from"] == index[1]))].index
                single_lines_same_order = fix_multiple_parallel_lines(duplicate_index, single_lines_same_order)
                # grouped_lines_oo.drop([index[::-1]], inplace=True)
            single_lines_oo = single_lines_same_order.sort_index().reset_index(drop=True)

            return single_lines_oo

