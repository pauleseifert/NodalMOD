import gurobipy as gp
from collections import ChainMap
from import_object_data_Zonal_Configuration import gurobi_variables, run_parameter

run_parameter= run_parameter(scenario_name = "Offshore_Bidding_Zone_Scenario")

model = gp.read(run_parameter.export_model_formulation)

print("model read. setting parameters")

model.optimize()

#Z_dict = [ { 'BE':1 , 'CZ':2 , 'DEV5':3 , 'DEV4':4 , 'DEV3':5 , 'DEV2':6 , 'DEV1':7 , 'OffBZB':8 , 'DK1':9 , 'DK2':10 , 'FI':11 , 'UK':12, 'NL':13 , 'NO2':14 , 'NO4':15 , 'NO3':16 , 'NO1':17 , 'NO5':18 , 'PL':19, 'SE3':20 , 'SE1':21 , 'SE4':22 , 'SE2':23, 'OffBZN':24} ]
#G_dict =[{ 'CCGT': 1, 'coal': 2, 'biomass': 3, 'HDAM': 4, 'OCGT': 5, 'nuclear': 6, 'lignite': 7, 'oil': 8}]
#G_Z_dict = ChainMap(Z_dict, G_dict)
#all_variables = model.getVars()
#map(lambda x: 'ACHTUNG' if type(x) == str else x, all_variables.VarName) #transform zones into integer
#map(lambda x: 0 if x == "X" else x, all_variables)#tranforms conventionals into integer


variables = gurobi_variables(solved_model=model)
variables.export_csv(folder = run_parameter.export_folder, scen=run_parameter.scen)
print("scenario "+str(run_parameter.scen)+ " with sensitivity scenario "+ str(run_parameter.sensitivity_scen)+ " has an objective value of "+ str(model.getObjective().getValue()))

print("done")