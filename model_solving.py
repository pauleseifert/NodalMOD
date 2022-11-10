import gurobipy as gp
from sys import platform
import sys
from import_data_object import gurobi_variables, run_parameter

run_parameter= run_parameter(scenario_name = "Offshore_Bidding_Zone_Scenario")


if (platform == "darwin") or (platform == "win32"):
    directory = ""
    case_name = "Offshore_Bidding_Zone_Scenario"
    scen = 1
    sensitivit_scen = 0

model = gp.read(run_parameter.export_model_formulation )
print("model read. setting parameters")

model.optimize()
variables = gurobi_variables(solved_model=model)
variables.export_csv(folder = run_parameter.export_folder, scen=scen)
print("scenario "+str(scen)+ " with sensitivity scenario "+ str(sensitivit_scen)+ " has an objective value of "+ str(model.getObjective().getValue()))

print("done")