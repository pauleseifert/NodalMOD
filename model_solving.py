import gurobipy as gp
import csv

from import_object_data_Zonal_Configuration import gurobi_variables, run_parameter

run_parameter= run_parameter(scenario_name = "Offshore_Bidding_Zone_Scenario")

model = gp.read(run_parameter.export_model_formulation)

print("model read. setting parameters")

model.optimize()

#if model.SolCount == 0:
#    print("Model has no solution")
#    exit(1)
#var_names = []
#for var in model.getVars():
#    # Or use list comprehensions instead
#    if 'x' == str(var.VarName[0]) and var.X > 0.1:
#        var_names.append(var.VarName)
# Write to csv
#with open('out.csv', 'wb') as myfile:
#     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
#     wr.writerows(var_names)

variables = gurobi_variables(solved_model=model)
variables.export_csv(folder = run_parameter.export_folder, scen=run_parameter.scen)
print("scenario "+str(run_parameter.scen)+ " with sensitivity scenario "+ str(run_parameter.sensitivity_scen)+ " has an objective value of "+ str(model.getObjective().getValue()))

print("done")