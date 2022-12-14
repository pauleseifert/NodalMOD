import gurobipy as gp
import pandas as pd
from collections import ChainMap
from import_object_data_Zonal_Configuration import gurobi_variables, run_parameter, model_data

run_parameter = run_parameter(scenario_name = "Offshore_Bidding_Zone_Scenario")
data = model_data(create_res = False,reduced_ts = True, export_files= True, run_parameter = run_parameter)
match run_parameter.scen:
    case "BAU":number_zones = 19
    case "BZ2":number_zones = 20
    case "BZ3":number_zones = 21
    case "BZ5":number_zones = 23

model = gp.read(run_parameter.export_model_formulation)

print("model read. setting parameters")

model.optimize()



variables = gurobi_variables(solved_model=model)
variables.export_csv(folder = run_parameter.export_folder, scen=run_parameter.scen, number_zones=number_zones)
#list(range(len(data.nodes[run_parameter.scen].unique())))

Z = list(range(len(data.nodes[run_parameter.scen].unique())))
Z_dict = {}
keys = range(len(data.nodes[run_parameter.scen].unique()))
values = data.nodes[run_parameter.scen].unique()
T = range(run_parameter.timesteps)
for i in keys:
        Z_dict[i] = values[i]
dict = {t: {z: model.getConstrByName("Injection_equality" + "[" + str(z) + "," + str(t) + "]").Pi for z in Z}for t in T}
test = pd.DataFrame.from_dict(dict, orient = "index").reset_index()
#sort value und dann load duration curve

print("scenario "+str(run_parameter.scen)+ " with sensitivity scenario "+ str(run_parameter.sensitivity_scen)+ " has an objective value of "+ str(model.getObjective().getValue()))

print("done")
