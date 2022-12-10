import os

import pandas as pd

from import_object_data_Zonal_Configuration import kpi_data, run_parameter
from printing_funct import plot_bar2_electrolyser, kpi_development2, radar_chart, plot_generation, plotly_maps_bubbles, \
    plotly_maps_lines, plotly_maps_size_lines, plotly_maps_lines_hours

#list of scenarios to calculate
#scenarios = [1]
run_parameter = run_parameter(scenario_name="Offshore_Bidding_Zone_Scenario")
kpis = kpi_data(run_parameter = run_parameter )#, scen= scen) for scen in scenarios}
#kpis = {scen : kpi_data(run_parameter = run_parameter, scen= scen) for scen in scenarios}

#scenario spanning analysis
scen_folder = run_parameter.export_folder +"maps/sensitivity" +str(run_parameter.sensitivity_scen)+"/"
if not os.path.exists(scen_folder):
    os.makedirs(scen_folder)

if run_parameter.sensitivity_scen == 0:
    #base scenario
    scen1 = pd.DataFrame({"BHEI": [0, 0, 0], "NSEI 1": [0, 0, 0], "NSEI 2": [0, 0, 0], "Landing points": [0, 0, 0]}, index=[2030, 2035, 2040])
    scen2 = pd.DataFrame({"BHEI": kpis[2].CAP_E.loc["electrolyser_Bornholm"][run_parameter.years].to_list(), "NSEI 1": kpis[2].CAP_E.loc["electrolyser_NS1"][run_parameter.years].to_list(), "NSEI 2": kpis[2].CAP_E.loc["electrolyser_NS2"][run_parameter.years].to_list(), "Landing points": [0, 0, 0]},index=[2030, 2035, 2040])
    scen3 = pd.DataFrame({"BHEI": kpis[3].CAP_E.loc["electrolyser_Bornholm"][run_parameter.years].to_list(), "NSEI 1": kpis[3].CAP_E.loc["electrolyser_NS1"][run_parameter.years].to_list(), "NSEI 2": kpis[3].CAP_E.loc["electrolyser_NS2"][run_parameter.years].to_list(), "Landing points": kpis[3].CAP_E.iloc[3:,:3].sum(axis=0).to_list()},index=[2030, 2035, 2040])
    scen4 = pd.DataFrame({"BHEI": kpis[4].CAP_E.loc["electrolyser_Bornholm"][run_parameter.years].to_list(), "NSEI 1": kpis[4].CAP_E.loc["electrolyser_NS1"][run_parameter.years].to_list(), "NSEI 2": kpis[4].CAP_E.loc["electrolyser_NS2"][run_parameter.years].to_list(), "Landing points": kpis[4].CAP_E.iloc[3:,:3].sum(axis=0).to_list()},index=[2030, 2035, 2040])
    df = pd.concat([scen1, scen2, scen3, scen4], axis=1, keys=["BAU", "OFFSH", "COMBI", "STAKE"])
    df2 = pd.concat([scen1.T, scen2.T, scen3.T, scen4.T], axis=1,keys=["BAU", "OFFSH", "COMBI", "STAKE"]).T

    df2_reordert = pd.melt(df2, value_vars=["BHEI", "NSEI 1", "NSEI 2", "Landing points"], ignore_index=False).reset_index().set_index("level_0").pivot(columns=["level_1", "variable"], values="value")

    plot_bar2_electrolyser(df=df/1000, title = "Electrolyser installed capacity [GW]", caption = "elec_capacity_scen", position_label= 3, maps_folder=scen_folder)
    plot_bar2_electrolyser(df=df2_reordert/1000, title="Electrolyser installed capacity [GW]", caption="elec_capacity_years",position_label=3, maps_folder=scen_folder)


    scen1 = pd.DataFrame({"BHEI": kpis[1].CAP_lines[kpis[1].CAP_lines["EI"] == 0.0].sum(axis=0)[run_parameter.years].to_list(), "NSEI 1": kpis[1].CAP_lines[kpis[1].CAP_lines["EI"] == 1.0].sum(axis=0)[run_parameter.years].to_list(), "NSEI 2": kpis[1].CAP_lines[kpis[1].CAP_lines["EI"] == 2.0].sum(axis=0)[run_parameter.years].to_list()}, index=[2030, 2035, 2040])
    scen2 = pd.DataFrame({"BHEI": kpis[2].CAP_lines[kpis[2].CAP_lines["EI"] == 0.0].sum(axis=0)[run_parameter.years].to_list(), "NSEI 1": kpis[2].CAP_lines[kpis[2].CAP_lines["EI"] == 1.0].sum(axis=0)[run_parameter.years].to_list(), "NSEI 2": kpis[2].CAP_lines[kpis[2].CAP_lines["EI"] == 2.0].sum(axis=0)[run_parameter.years].to_list()},index=[2030, 2035, 2040])
    scen3 = pd.DataFrame({"BHEI": kpis[3].CAP_lines[kpis[3].CAP_lines["EI"] == 0.0].sum(axis=0)[run_parameter.years].to_list(), "NSEI 1": kpis[3].CAP_lines[kpis[3].CAP_lines["EI"] == 1.0].sum(axis=0)[run_parameter.years].to_list(), "NSEI 2": kpis[3].CAP_lines[kpis[3].CAP_lines["EI"] == 2.0].sum(axis=0)[run_parameter.years].to_list()},index=[2030, 2035, 2040])
    scen4 = pd.DataFrame({"BHEI": kpis[4].CAP_lines[kpis[4].CAP_lines["EI"] == 0.0].sum(axis=0)[run_parameter.years].to_list(), "NSEI 1": kpis[4].CAP_lines[kpis[4].CAP_lines["EI"] == 1.0].sum(axis=0)[run_parameter.years].to_list(), "NSEI 2": kpis[4].CAP_lines[kpis[4].CAP_lines["EI"] == 2.0].sum(axis=0)[run_parameter.years].to_list()}, index=[2030, 2035, 2040])
    df = pd.concat([scen1, scen2, scen3, scen4], axis=1, keys=["BAU", "OFFSH", "COMBI", "STAKE"])
    plot_bar2_electrolyser(df=df/1000, title = "EI to shore installed line capacity [GW]", caption = "line_capacity", position_label=2, maps_folder=scen_folder)

    #alexandras Kategorie 3
    scen1 = pd.DataFrame({"BHEI": kpis[1].CAP_lines.query('EI == 3 & to == 521').sum(axis=0)[run_parameter.years].to_list(), "NSEI 1": kpis[1].CAP_lines.query('EI == 3 & to == 522').sum(axis=0)[run_parameter.years].to_list(), "NSEI 2": kpis[1].CAP_lines.query('EI == 3 & to == 522').sum(axis=0)[run_parameter.years].to_list()}, index=[2030, 2035, 2040])
    scen2 = pd.DataFrame({"BHEI": kpis[2].CAP_lines.query('EI == 3 & to == 521').sum(axis=0)[run_parameter.years].to_list(), "NSEI 1": kpis[2].CAP_lines.query('EI == 3 & to == 522').sum(axis=0)[run_parameter.years].to_list(), "NSEI 2": kpis[2].CAP_lines.query('EI == 3 & to == 522').sum(axis=0)[run_parameter.years].to_list()}, index=[2030, 2035, 2040])
    scen3 = pd.DataFrame({"BHEI": kpis[3].CAP_lines.query('EI == 3 & to == 521').sum(axis=0)[run_parameter.years].to_list(), "NSEI 1": kpis[3].CAP_lines.query('EI == 3 & to == 522').sum(axis=0)[run_parameter.years].to_list(), "NSEI 2": kpis[3].CAP_lines.query('EI == 3 & to == 522').sum(axis=0)[run_parameter.years].to_list()}, index=[2030, 2035, 2040])
    scen4 = pd.DataFrame({"BHEI": kpis[4].CAP_lines.query('EI == 3 & to == 521').sum(axis=0)[run_parameter.years].to_list(), "NSEI 1": kpis[4].CAP_lines.query('EI == 3 & to == 522').sum(axis=0)[run_parameter.years].to_list(), "NSEI 2": kpis[4].CAP_lines.query('EI == 3 & to == 522').sum(axis=0)[run_parameter.years].to_list()}, index=[2030, 2035, 2040])
    df = pd.concat([scen1, scen2, scen3, scen4], axis=1, keys=["BAU", "OFFSH", "COMBI", "STAKE"])
    plot_bar2_electrolyser(df=df/1000, title = "Cluster to EI installed line capacity [GW]", caption = "cluster_ei_capacity", position_label=2, maps_folder=scen_folder)
    #alexandras Kategorie 2
    scen1 = pd.DataFrame({"Wind cluster": kpis[1].CAP_lines.query('EI == 3 & to not in [521, 522, 523]').sum(axis=0)[run_parameter.years].to_list()}, index=[2030, 2035, 2040])
    scen2 = pd.DataFrame({"Wind cluster": kpis[2].CAP_lines.query('EI == 3 & to not in [521, 522, 523]').sum(axis=0)[run_parameter.years].to_list()}, index=[2030, 2035, 2040])
    scen3 = pd.DataFrame({"Wind cluster": kpis[3].CAP_lines.query('EI == 3 & to not in [521, 522, 523]').sum(axis=0)[run_parameter.years].to_list()}, index=[2030, 2035, 2040])
    scen4 = pd.DataFrame({"Wind cluster": kpis[4].CAP_lines.query('EI == 3 & to not in [521, 522, 523]').sum(axis=0)[run_parameter.years].to_list()}, index=[2030, 2035, 2040])
    df = pd.concat([scen1, scen2, scen3, scen4], axis=1, keys=["BAU", "OFFSH", "COMBI", "STAKE"])
    plot_bar2_electrolyser(df=df/1000, title = "Cluster to shore installed line capacity [GW]", caption = "cluster_shore_capacity", position_label=2, maps_folder=scen_folder)


    def year_wise_normalisation(df):
        for year in df.index:
            # df.loc[year] = (df.loc[year]-min(df.loc[year]))/(max(df.loc[year])-min(df.loc[year]))
            df.loc[year] = df.loc[year] / max(df.loc[year])
        return df

    # base scenario
    curtailment = pd.DataFrame({"BAU": [kpis[1].curtailment.raw[year].sum().sum() for year in run_parameter.years],"OFFSH": [kpis[2].curtailment.raw[year].sum().sum() for year in run_parameter.years],"COMBI": [kpis[3].curtailment.raw[year].sum().sum() for year in run_parameter.years],"STAKE": [kpis[4].curtailment.raw[year].sum().sum() for year in run_parameter.years]},index=[2030, 2035, 2040])
    electrolyser_capacity = pd.DataFrame({"BAU": [0, 0, 0], "OFFSH": kpis[2].CAP_E[run_parameter.years].sum(axis=0).to_list(),"COMBI": kpis[3].CAP_E[run_parameter.years].sum(axis=0).to_list(),"STAKE": kpis[4].CAP_E[run_parameter.years].sum(axis=0).to_list()}, index=[2030, 2035, 2040])
    hydrogen = pd.DataFrame({"BAU": [0, 0, 0], "OFFSH": [kpis[2].P_H[year].sum().sum() for year in run_parameter.years],"COMBI": [kpis[3].P_H[year].sum().sum() for year in run_parameter.years],"STAKE": [kpis[4].P_H[year].sum().sum() for year in run_parameter.years]},index=[2030, 2035, 2040])
    conventional = pd.DataFrame({"BAU": [kpis[1].generation_temporal[year].ts_conventional().sum() for year in run_parameter.years],"OFFSH": [kpis[2].generation_temporal[year].ts_conventional().sum() for year in run_parameter.years],"COMBI": [kpis[3].generation_temporal[year].ts_conventional().sum() for year in run_parameter.years],"STAKE": [kpis[4].generation_temporal[year].ts_conventional().sum() for year in run_parameter.years]},index=[2030, 2035, 2040])
    df = pd.concat([curtailment, electrolyser_capacity, hydrogen, conventional], axis=1,keys=["curtailment", "electrolyser capacity", "hydrogen production", "conventional usage"])
    curtailment = year_wise_normalisation(curtailment)
    electrolyser_capacity = year_wise_normalisation(electrolyser_capacity)
    hydrogen = year_wise_normalisation(hydrogen)
    conventional = year_wise_normalisation(conventional)

    df2 = pd.concat([curtailment, electrolyser_capacity, hydrogen, conventional], axis=1,keys=["curtailment", "electrolyser capacity", "hydrogen production", "conventional usage"])
    kpi_development2(data=df, title="Table 3 - Base scenario", folder=scen_folder)
    radar_chart(maps_folder=scen_folder, data=df, data2=df2)

#Excel export
kpi_folder = run_parameter.export_folder +"kpis/"+"sensitivity_scen"+str(run_parameter.sensitivity_scen)
if not os.path.exists(kpi_folder):
    os.makedirs(kpi_folder)

#CAP_BH
def excel_export(run_parameter, kpis, type, col_sum = False, ts_reduction_backscaling = False):
    with pd.ExcelWriter(run_parameter.export_folder + "/kpis/" +"sensitivity_scen"+str(run_parameter.sensitivity_scen)+"/"+ type+'.xlsx') as writer:
        for scen in scenarios:
            try:
                entry = eval("kpis[" + str(scen) + "]." + type)
                if isinstance(entry, dict):
                    new_df = pd.DataFrame()
                    for year in run_parameter.years:
                        if col_sum:
                            if isinstance(entry[year], kpi_data.temporal_generation):
                                generation = entry[year].to_df().sum(axis=0)
                                new_df = pd.concat([new_df, generation], axis=1)
                            else:
                                island_entry = entry[year].sum(axis=0)
                                island_entry.index = run_parameter.electrolyser[scen]["name"].to_list()
                                new_df = pd.concat([new_df, island_entry], axis=1)
                        else:
                            new_df = pd.concat([new_df, entry[year][0]], axis=1)
                    entry = new_df.set_axis(run_parameter.years, axis=1)
                if ts_reduction_backscaling:
                    entry = entry * run_parameter.scaling_factor
                entry.to_excel(writer, sheet_name='Scen_'+str(scen))
            except:
                pass
excel_export(run_parameter = run_parameter, kpis = kpis, type = "CAP_E")
excel_export(run_parameter = run_parameter, kpis = kpis, type = "CAP_lines")
excel_export(run_parameter = run_parameter, kpis = kpis, type = "P_H", col_sum= True, ts_reduction_backscaling = True)
excel_export(run_parameter = run_parameter, kpis = kpis, type = "curtailment.bz", ts_reduction_backscaling = True)
excel_export(run_parameter = run_parameter, kpis = kpis, type = "curtailment.bz_relative")
excel_export(run_parameter = run_parameter, kpis = kpis, type = "zonal_trade_balance", ts_reduction_backscaling = True)
excel_export(run_parameter = run_parameter, kpis = kpis, type = "generation_temporal", col_sum=True, ts_reduction_backscaling=True)
excel_export(run_parameter = run_parameter, kpis = kpis, type = "load_factor.elect")


#Maps

for scen in scenarios: plot_generation(generation_temporal = kpis[scen].generation_temporal, maps_folder=scen_folder, scen=scen, year=run_parameter.years[0])
#for scen in scenarios: plot_generation(generation_temporal = kpis[scen].generation_temporal, maps_folder=scen_folder, scen=scen, year=run_parameter.years[1])
for scen in scenarios: plot_generation(generation_temporal = kpis[scen].generation_temporal, maps_folder=scen_folder, scen=scen, year=run_parameter.years[2])

for scen in scenarios: plotly_maps_bubbles(df=kpis[scen].curtailment.location, scen=scen, maps_folder=scen_folder, name="curtailment", size_scale=2, unit="TWh", title="Curtailment", year=2)
if run_parameter.sensitivity_scen == 0:
    for scen in [2,3,4]: plotly_maps_bubbles(df=kpis[scen].CAP_E, year=2, scen=scen, maps_folder=scen_folder, name="electrolyser_location", size_scale=8, unit="GW", title="Electrolyser capacity", flexible_bubble_size = True,  zoom = 1.22 , hoffset = -4, voffset = -2, min_legend=0, max_legend=40)
else:
    for scen in [3,4]: plotly_maps_bubbles(df=kpis[scen].CAP_E, year=2, scen=scen, maps_folder=scen_folder, name="electrolyser_location", size_scale=8, unit="GW", title="Electrolyser capacity", flexible_bubble_size = True,  zoom = 1.22 , hoffset = -4, voffset = -2)
for scen in scenarios: plotly_maps_lines(P_flow=kpis[scen].line_loading.AC["avg"], P_flow_DC= kpis[scen].line_loading.DC["avg"], bus=kpis[scen].bus,  scen=scen, maps_folder=scen_folder)
for scen in scenarios: plotly_maps_lines_hours(P_flow=kpis[scen].line_loading.AC["avg"], P_flow_DC= kpis[scen].line_loading.DC["avg"], bus=kpis[scen].bus, scen=scen, maps_folder=scen_folder, timesteps=run_parameter.timesteps)
for scen in scenarios: plotly_maps_size_lines(P_flow=kpis[scen].line_loading.AC, P_flow_DC = kpis[scen].line_loading.DC, CAP_lines=kpis[scen].CAP_lines , bus=kpis[scen].bus, scen=scen, year=2,  maps_folder=scen_folder, zoom = 1.25, offset = -4)

