import pandas as pd
from helper_functions import merge_timeseries_demand, merge_timeseries_demand_entsoe, map_distance, distance_calc, append_BHEH, renewables_scaling_country_specific, conv_scaling_country_specific
from mapping import hydro_mapping, new_res_mapping, mapping_ror, zones_busses_dam
from aggregate_openentrance import open_entrance_scaling_factors


def read_data(create_hydro, create_renewables, zones_CM, bidding_zones, bidding_zones_encyclopedia, TRM, location_export_csv, reduced_TS):


    busses_raw = pd.read_csv(location_export_csv + "buses.csv", index_col=0)
    #dam_raw = pd.read_csv(location_export_csv + "Dam.csv", index_col=0)
    conventionals_raw = pd.read_csv(location_export_csv + "generators.csv", index_col=0)
    #hydro_all_raw = pd.read_csv(location_export_csv + "hydro_all.csv", index_col=0)
    lines_raw = pd.read_csv(location_export_csv + "lines.csv", index_col=0)

    links_raw = pd.read_csv(location_export_csv + "links.csv", index_col=0)
    load_raw = pd.read_csv(location_export_csv + "load.csv", index_col=0).reset_index(drop=True)
    #ren_pot_raw = pd.read_csv(location_export_csv + "renpot.csv", index_col=0)
    #storage_raw = pd.read_csv(location_export_csv + "storage.csv", index_col=0)
    ror_ts = pd.read_csv(location_export_csv + "hydro_ror_ts.csv")
    dam_maxsum_ts = pd.read_csv(location_export_csv + "hydro_dam_ts.csv")

    #reindexing and selecting countries
    zones_CM.append("GB")
    busses_raw = pd.concat([busses_raw, pd.DataFrame([{"country": "BHEH", "y": 55.13615337829421,"x": 14.898639089359104},
                                    {"country": "NSEH1", "y": 55.22300,"x": 3.78700},
                                    {"country": "NSEH2", "y": 55.69354,"x": 3.97940}], index = ["BHEH", "NSEH1", "NSEH2"])])
    busses_filtered = busses_raw[busses_raw["country"].isin(zones_CM)].reset_index().reset_index()
    busses_filtered = busses_filtered.replace({"GB":"UK"})
    busses_filtered = busses_filtered[["level_0", "index", "x", "y", "country"]]
    busses_filtered.columns = ["index","old_index", "LON", "LAT", "country"]
    bus_CM = busses_filtered.copy()

    conventionals_matched = conventionals_raw.merge(busses_filtered[["index", "old_index", "country"]], how="left", left_on="bus",right_on="old_index")
    conventionals_filtered = conventionals_matched[conventionals_matched['index'].notnull()]
    solar_matched = conventionals_filtered[conventionals_filtered["carrier"].isin(["solar"])]
    wind_matched = conventionals_filtered[conventionals_filtered["carrier"].isin(["onwind", "offwind-ac", "offwind-dc"])]
    conventionals_filtered = conventionals_filtered[conventionals_filtered["carrier"].isin(["CCGT", "OCGT", "nuclear", "biomass", "coal", "lignite", "oil"])]
    generators_CM = conventionals_filtered[["p_nom", "carrier", "marginal_cost", "index"]].reset_index(drop=True)
    generators_CM.columns = ["pmax", "type", "mc", "bus"]
    generators_CM["bus"] = generators_CM["bus"].astype(int)

    solar_filtered = solar_matched[["p_nom_max", "carrier", "marginal_cost", "index", "country"]].reset_index(drop=True)
    solar_filtered.columns = ["max", "type", "mc", "bus", "country"]
    solar_filtered = solar_filtered.replace({"solar": "Solar"})
    solar_filtered["bus"] = solar_filtered["bus"].astype(int)
    wind_matched = pd.concat([wind_matched, pd.DataFrame([{"p_nom_max": 3000, "country": "BHEH", "carrier": "offwind-dc", "marginal_cost": 0.015, "index":521},
                      {"p_nom_max": 10000, "country": "NSEH1", "carrier": "offwind-dc", "marginal_cost": 0.015, "index":522},
                      {"p_nom_max": 10000, "country": "NSEH2", "carrier": "offwind-dc", "marginal_cost": 0.015, "index":523}])])
    wind_filtered = wind_matched[["p_nom_max", "carrier", "marginal_cost", "index", "country"]].reset_index(drop=True)
    wind_filtered.columns = ["max", "type", "mc", "bus", "country"]
    wind_filtered = wind_filtered.replace({"onwind": "Wind", "offwind-ac": "Wind", "offwind-dc": "Wind"})
    wind_filtered["bus"] = wind_filtered["bus"].astype(int)

    lines_matched = lines_raw.merge(busses_filtered[["index", "old_index"]], how="left", left_on="bus0",right_on="old_index")
    lines_matched = lines_matched.merge(busses_filtered[["index", "old_index"]], how="left", left_on="bus1",right_on="old_index")
    lines_filtered = lines_matched[lines_matched['index_x'].notnull()]
    lines_filtered = lines_filtered[lines_filtered['index_y'].notnull()]
    #https://pypsa.readthedocs.io/en/latest/components.html?highlight=parameters#line-types
    lines_filtered["x"] = 0.246*lines_filtered["length"]
    #lines_filtered.to_csv("data/PyPSA_elec1024/lines_after_matching.csv")
    lines_tyndp = pd.read_csv(location_export_csv + "lines_after_matching-v03_tyndp.csv", index_col=0)

    lines = lines_tyndp[["s_nom", "x", "index_x", "index_y"]].reset_index(drop=True)
    lines.columns = ["Pmax", "x", "from", "to"]
    lines["from"] = lines["from"].astype(int)
    lines["to"] = lines["to"].astype(int)


    lines_DC_matched = links_raw.merge(busses_filtered[["index", "old_index"]], how="left", left_on="bus0",right_on="old_index")
    lines_DC_matched = lines_DC_matched.merge(busses_filtered[["index", "old_index"]], how="left", left_on="bus1",right_on="old_index")
    lines_DC_filtered = lines_DC_matched[lines_DC_matched['index_x'].notnull()]
    lines_DC_filtered = lines_DC_filtered[lines_DC_filtered['index_y'].notnull()]
    # See lines_V02.csv
    added_DC_lines = pd.DataFrame({"p_nom": [1400, 2000,2000, 700], "length": [720,267,400,300], "index_x":[299,198, 170, 513],"index_y":[419,111, 93, 116], "tags": ["North Sea Link 2021: https://tyndp2020-project-platform.azurewebsites.net/projectsheets/transmission/110", "hvdc corridor norGer to WesGer 1034: https://tyndp2020-project-platform.azurewebsites.net/projectsheets/transmission/1034", "hvdc corridor norGer to WesGer 1034: https://tyndp2020-project-platform.azurewebsites.net/projectsheets/transmission/1034", "Hansa Power Bridge 1 https://tyndp2020-project-platform.azurewebsites.net/projectsheets/transmission/176"]})
    lines_DC_filtered = pd.concat([lines_DC_filtered, added_DC_lines])
    lines_DC = lines_DC_filtered[["p_nom", "index_x", "index_y"]].reset_index(drop=True)
    lines_DC.columns = ["Pmax", "from", "to"]
    lines_DC["from"] = lines_DC["from"].astype(int)
    lines_DC["to"] = lines_DC["to"].astype(int)
    lines_DC.insert(3, "EI", 'N/A')
    flexlines_EI = pd.DataFrame({"Pmax": [500,500,500, 500, 500, 500, 500, 500,500,500, 500, 500, 500, 500, 500, 500, 500], "from":[523, 523, 523, 523 , 523, 523, 523, 522, 522,522, 522, 522, 522, 521, 521, 521, 521], "to":[522, 403, 212,209, 170, 376, 357, 279, 170, 103, 24, 357, 376, 62, 467, 218, 513], "EI": [1,1,1,1,1,1,1,2,2,2,2,2,2,0,0,0,0]})
    lines_DC = pd.concat([lines_DC, flexlines_EI], ignore_index=True)

    def demand_columns(busses_filtered, load_raw):
        def get_value(x):
            try:
                return busses_filtered[busses_filtered["old_index"] == x.strip()]["index"].values[0]
            except:
                pass
        load_raw.columns = load_raw.columns.to_series().apply(get_value)
        filtered_load = load_raw.loc[:,load_raw.columns.notnull()]
        filtered_load.columns = filtered_load.columns.astype(int)
        return filtered_load
    #new demand
    demand = demand_columns(busses_filtered, load_raw)
    #get new renewables
    renewables_supply, share_solar, share_wind = new_res_mapping(busses_filtered, solar_filtered, wind_filtered, bidding_zones_encyclopedia, create_renewables, location_export_csv)

    #TRM schon weg? Unklar im Datenset!
    lines["max"] = lines["Pmax"]*TRM
    lines_DC["max"] = lines_DC["Pmax"]*TRM

    generators = generators_CM.rename(columns={"cost":"mc"})
    #generators_CM = match_bus_index(column="bus", data = generators, nodes_locations= bus_CM, zones_CM= zones_CM)
    #generators_CM = generators_CM[~generators_CM.type.str.contains("Hydro")]
    #generators_CM = generators_CM.drop(['Qmin', 'Qmax'], axis=1)

    if create_hydro == True:
        hydro_raw = hydro_mapping(bus_CM)

    #hydro_CM = match_bus_hydro("bus", hydro_raw, bus_CM)
    hydro_CM = hydro_raw.rename(columns={'installed_capacity_MW': 'pmax', 'country_code': 'zone'})

    # Hydro reservoir
    default_storage_capacity = 1000  # MWh
    dam = hydro_CM[hydro_CM["type"] == "HDAM"]
    dam["pmin"] = 0
    dam["mc"] = 30 #Euro/MWh
    dam = dam.drop(["pumping_MW", "storage_capacity_MWh"], axis=1)
    #BE, FI have no limits on reservoir
    dam_unlimited = dam[dam["zone"].isin(["BE", "FI"])]
    dam_limited = dam[~dam["zone"].isin(["BE", "FI"])]
    dam_limited_sum = dam_limited.groupby(["bus"]).sum()[["pmax"]].reset_index()
    def clear_dam_ts(ts_raw, countries):
        target_year = ts_raw[ts_raw["y"] == 2019.0]
        filtered = target_year.drop(["y", "t", "technology"],axis=1).reset_index(drop=True)
        filtered.columns = filtered.columns.map(lambda x: x.replace('00', '').replace("DKW1","DK").replace('0', ''))
        droped_SE_DE = filtered.drop(columns=["DE", "SE"]).rename(columns={"DELU":"DE"})
        cleared_ts = droped_SE_DE[droped_SE_DE.columns.intersection(countries)]
        NO_SE = filtered[filtered.columns.intersection(["NO1", "NO2", "NO3", "NO4", "NO5", "SE1", "SE2", "SE3", "SE4"])]
        return pd.concat([cleared_ts, NO_SE], axis=1)
    limited_dam_ts = clear_dam_ts(dam_maxsum_ts, zones_CM)
    encyc_dam_zones = zones_busses_dam(bus_overview_limited_dam=bus_CM, limited_dam = dam_limited_sum)
    #dam["storage_capacity_MWh"] = dam["storage_capacity_MWh"].fillna(default_storage_capacity)

    # RoR
    ror = hydro_CM[hydro_CM["type"] == "HROR"]
    ror["mc"] = 0
    ror["pmin"] = 0
    ror = ror.drop(["pumping_MW", "storage_capacity_MWh"], axis=1)

    def clear_hydro_ts(ts_raw, countries):
        target_year = ts_raw[ts_raw["y"] == 2019.0]
        filtered = target_year.drop(["y", "t", "technology"],axis=1).reset_index(drop=True)
        filtered.columns = filtered.columns.map(lambda x: x.replace('00', '').replace("DELU","DE").replace("DKW1","DK"))
        cleared_ts = filtered[filtered.columns.intersection(countries)]
        norway = filtered[filtered.columns.intersection(["NO1", "NO2", "NO3", "NO4", "NO5"])]
        return pd.concat([cleared_ts, norway], axis=1)
    ror_ts = clear_hydro_ts(ror_ts, zones_CM)
    ror_supply = mapping_ror(ror_overview=ror, ror_ts=ror_ts, bus_overview=bus_CM)

    # PHS
    PHS = hydro_CM[hydro_CM['type'] == 'HPHS']
    PHS["storage_capacity_MWh"] = PHS["storage_capacity_MWh"].fillna(default_storage_capacity)
    PHS["pumping_MW"] = PHS["pumping_MW"].fillna(PHS["pmax"])
    storage_CM = PHS.rename(columns={'pmax': 'Pmax_out','pumping_MW':'Pmax_in','storage_capacity_MWh': 'capacity'}).reset_index(drop=True)


    merged_generators = pd.concat([generators_CM, dam_unlimited], axis=0).reset_index(drop =True)
    merged_generators = merged_generators.drop(["name", "zone", "LAT", "LON", 'avg_annual_generation_GWh', "pmin"], axis =1)


    def reduce_timeseries(long_ts, u_index):
        short_ts = pd.DataFrame()
        for index in u_index:
            current_day = long_ts.loc[index*24:index*24+23]
            short_ts = short_ts.append(current_day)
        return short_ts.reset_index(drop=True)

    # scaling factors
    res_scaling_factors, conv_scaling_factors, demand_scaling_factors = open_entrance_scaling_factors()
    #renewables_supply_for_poncelet = renewables_scaling_country_specific(renewables_supply, res_scaling_factors, bus_CM,bidding_zones_encyclopedia, True, False)
    #renewables_supply_for_poncelet.to_csv(location_export_csv + "renewables_for_poncelet.csv")

    if reduced_TS:
        renewables_reduced = pd.read_csv(location_export_csv + "/poncelet/poncelet_ts.csv", index_col=0)
        u = pd.read_csv(location_export_csv + "/poncelet/u_result.csv", index_col=0)
        u_index = u.index[u["value"] == 1.0].to_list()
        renewables_reduced.columns = renewables_supply.columns
        demand = reduce_timeseries(demand, u_index)
        share_solar = reduce_timeseries(share_solar, u_index)
        share_wind = reduce_timeseries(share_wind, u_index)
        ror_supply = reduce_timeseries(ror_supply, u_index)
        dam_supply_sum = reduce_timeseries(limited_dam_ts, u_index).sum()
        renewables_supply_4_years = renewables_scaling_country_specific(renewables_reduced, res_scaling_factors, bus_CM, bidding_zones_encyclopedia, False)
    else:
        dam_supply_sum = limited_dam_ts.sum()
        renewables_supply_4_years = renewables_scaling_country_specific(renewables_supply, res_scaling_factors, bus_CM, bidding_zones_encyclopedia,True)

    demand_4_years = renewables_scaling_country_specific(demand, demand_scaling_factors, bus_CM, bidding_zones_encyclopedia, True)
    conventionals_supply_4_years = conv_scaling_country_specific(merged_generators, conv_scaling_factors, bus_CM)

    return demand_4_years, lines, lines_DC, conventionals_supply_4_years, renewables_supply_4_years, ror_supply, dam_limited_sum, dam_supply_sum, encyc_dam_zones, busses_filtered, storage_CM, share_solar, share_wind
