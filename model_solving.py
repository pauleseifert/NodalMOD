import gurobipy as gp

from import_data_object import gurobi_variables, run_parameter

run_parameter= run_parameter(scenario_name = "Energy_island_scenario")

model = gp.read(run_parameter.export_model_formulation)

print("model read. setting parameters")

model.optimize()
variables = gurobi_variables(solved_model=model)
variables.export_csv(folder = run_parameter.export_folder, scen=run_parameter.scen)
print("scenario "+str(run_parameter.scen)+ " with sensitivity scenario "+ str(run_parameter.sensitivity_scen)+ " has an objective value of "+ str(model.getObjective().getValue()))

print("done")