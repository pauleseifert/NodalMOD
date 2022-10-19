import gurobipy as gp
from sys import platform
import sys
from import_data_object import gurobi_variables

if platform == "linux" or platform == "linux2":
    directory = "/work/seifert/powerinvest/"
    case_name = sys.argv[1]
    scen = int(sys.argv[2])
    sensitivit_scen = int(sys.argv[3])
elif platform == "darwin":
    directory = ""
    case_name = "13_09"
    scen = 3
    sensitivit_scen = 0
folder = directory + "results/"+ case_name+"/"+str(scen)+"/"+"subscen" +str(sensitivit_scen)+"/"

model = gp.read(folder+'my_problem.mps')
print("model read. setting parameters")

model.optimize()
variables = gurobi_variables(solved_model=model)
variables.export_csv(folder = folder, scen=scen)
print("scenario "+str(scen)+ " with sensitivity scenario "+ str(sensitivit_scen)+ " has an objective value of "+ str(model.getObjective().getValue()))

print("done")