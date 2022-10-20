import requests
import pandas as pd
import json
import time
import os.path


type = "pv"
#type = "wind"
location = "data/PyPSA_elec1024/"
def get_data(type, lat, lon, token):
    api_base = 'https://www.renewables.ninja/api/'
    s = requests.session()
    # Send token header with each request
    s.headers = {'Authorization': 'Token ' + token}
    url = api_base + 'data/' + type

    if type == "pv":
        args = {
            'lat': lat,
            'lon': lon,
            'date_from': '2019-01-01',
            'date_to': '2019-12-31',
            'dataset': 'merra2',
            'capacity': 1.0,
            'system_loss': 0.0,
            'tracking': 0,
            'tilt': 45,
            'azim': 180,
            'format': 'json'
        }
    if type=="wind":
        args = {
            'lat': lat,
            'lon': lon,
            'date_from': '2019-01-01',
            'date_to': '2019-12-31',
            'capacity': 1.0,
            'height': 100,
            'turbine': 'Vestas V80 2000',
            'format': 'json',
            'raw': True
        }

    r = s.get(url, params=args)
    print(r.status_code)

    # Parse JSON to get a pandas.DataFrame of data and dict of metadata
    parsed_response = json.loads(r.text)

    data = pd.read_json(json.dumps(parsed_response['data']), orient='index')["electricity"]
    return data.reset_index(drop=True), r.status_code


#time.sleep(3600)
bus_CM = pd.read_csv(location + "bus_" + type + ".csv", index_col=0)
if os.path.isfile(location + "res_ninja_"+type+"_ts.csv"):
    timeseries = pd.read_csv(location + "res_ninja_"+type+"_ts.csv", index_col=0)
    timeseries.columns = timeseries.columns.astype(int)
    counter_hour = 1
else:
    test,status_code = get_data(type, bus_CM.iloc[0]["LAT"], bus_CM.iloc[0]["LON"],'975cd56278181790d6bff8802f9ca0d16279f7e6')
    test.name = int(bus_CM.iloc[0]["bus"])
    test.to_csv(location + "res_ninja_"+type+"_ts.csv")
    timeseries = pd.read_csv(location + "res_ninja_" + type + "_ts.csv", index_col=0)
    timeseries.columns = timeseries.columns.astype(int)
    counter_hour = 2

token_ntnu = '975cd56278181790d6bff8802f9ca0d16279f7e6'
token_tu = '221eb41954bc7041e8a6e5e329074fb49f374d36'
token_paul = '6efb01abe98c420d3405e698244d2b8425237cb5'
token_martin = 'fe658e1ce4f91ca8371ed1b1bbbf1bb6dadeae40'
token = token_ntnu

for bus_index,bus in bus_CM.iterrows():
    if int(bus["bus"]) not in timeseries.columns:
        try_counter = 0
        status_code = int
        while try_counter<3:
            try:
                print("getting a new timeseries of bus: " + str(int(bus["bus"])) + " in query " + str(counter_hour))
                new_ts_column, status_code = get_data(type, bus_CM.at[bus_index, "LAT"], bus_CM.at[bus_index, "LON"], token)
                new_ts_column.name = int(bus["bus"])
                timeseries = pd.concat([timeseries, new_ts_column], axis = 1, sort = True)
                try_counter = 3
            except:
                print("error")
                try_counter += 1
            counter_hour += 1
            time.sleep(11)
        match counter_hour:
            case 51:
                timeseries.to_csv(location + "res_ninja_" + type + "_ts.csv")
                token = token_tu
                print("change of token!")
            case 101:
                print("change of token")
                timeseries.to_csv(location + "res_ninja_" + type + "_ts.csv")
                token = token_paul
            case 151:
                print("change of token")
                timeseries.to_csv(location + "res_ninja_" + type + "_ts.csv")
                token = token_martin
            case 201:
                print("limit reached. time to sleep")
                timeseries.to_csv(location + "res_ninja_" + type + "_ts.csv")
                token = token_ntnu
                counter_hour = 1
                time.sleep(1550)
timeseries = timeseries.reindex(sorted(timeseries.columns), axis=1)
timeseries.to_csv(location + "res_ninja_"+type+"_ts.csv")
print("done")
