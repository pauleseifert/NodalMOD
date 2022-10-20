import gurobipy as gp
import numpy as np
import pandas as pd
from helper_functions import export
from sys import platform
import sys

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

print(folder)
model = gp.read(folder+'my_problem.mps')
print("model read. setting parameters")
model.printStats()

model.optimize()
#model.write(folder+"model_solution.sol")
variables = model.getVars()
constraints = model.getConstrs()
#test = constraints[0].Pi
def get_variables(variables, folder):
    last_item = variables[-1].VarName.split(",")
    Y = range(int(last_item[0].split("[")[1])+1)
    T = range(int(last_item[1])+1)
    S = range(int(last_item[2].split("]")[0])+1)
    match scen:
        case 1: E =[0]
        case 2: E= range(3)
        case 3: E= range(17)
        case 4: E= range(17)
    G = range(572)
    R = range(530)
    DAM = range(95)
    L = range(817)
    N = range(544)
    C = range(274)
    I = range(122)
    LDC = range(149)
    Z = range(13)

    counter = 0
    def get_var_value(Y, T, special, counter, variables, size):
        first_run = True
        match size:
            case 1:
                new = np.zeros(len(special))
                columns_year = []
                for s in special:
                    new[s] = variables[counter].X
                    entry = int(variables[counter].VarName.split("[")[-1].split("]")[0])
                    columns_year.append(entry)
                    counter += 1
            case 2:
                new = np.zeros((len(Y), len(special)))
                columns_year = []
                for y in Y:
                    for s in special:
                        new[y, s] = variables[counter].X
                        if first_run:
                            entry = int(variables[counter].VarName.split(",")[-1].split("]")[0])
                            columns_year.append(entry)
                        counter += 1
                    first_run=False
            case 3:
                new = np.zeros((len(Y), len(T), len(special)))
                columns_year = []
                for y in Y:
                    for t in T:
                        for s in special:
                            new[y, t, s] = variables[counter].X
                            if first_run:
                                entry = int(variables[counter].VarName.split(",")[-1].split("]")[0])
                                columns_year.append(entry)
                            counter += 1
                        first_run = False
            #test_xr = xr.DataArray(new, dims=["year", "time", "plant/bus"], indexes=["","",test])
        #next_element_loop = variables[counter]
        return new, counter, columns_year
    #reading the variables. Achtung! Die Reihenfolge muss passen!
    P_C, counter, P_C_columns = get_var_value(Y, T, G, 0, variables, 3)
    P_R, counter, P_R_columns = get_var_value(Y, T, R, counter,variables, 3)
    P_DAM, counter, P_DAM_columns = get_var_value(Y, T, DAM, counter,variables, 3)
    res_curtailment, counter, curtailment_columns = get_var_value(Y, T, R, counter, variables, 3)
    CAP_BH, counter,CAP_BH_columns  = get_var_value(Y, T, I, counter,variables, 2)
    F_AC, counter, F_AC_columns = get_var_value(Y, T, L, counter, variables, 3)
    F_DC, counter, F_DC_columns = get_var_value(Y, T, LDC, counter, variables, 3)
    p_load_lost,counter, p_load_lost_columns = get_var_value(Y, T, N, counter,variables, 3)
    if scen in [2,3,4]:
        CAP_E, counter, CAP_E_columns  = get_var_value(Y, T, E, counter, variables, 2)
        P_H, counter, P_H_columns = get_var_value(Y, T, E, counter, variables, 3)
    P_S, counter, P_S_columns = get_var_value(Y, T, S, counter, variables, 3)
    C_S, counter, D_S_columns = get_var_value(Y, T, S, counter, variables, 3)
    L_S, counter, L_S_columns = get_var_value(Y, T, S, counter, variables, 3)
    additionals = dict({"P_R": P_R_columns, "CAP_BH":CAP_BH_columns})
    if scen in [1]:
        export(folder,scen, Y, P_C, P_R, P_DAM, res_curtailment, "", "" ,CAP_BH,P_S, C_S, L_S, F_AC, F_DC, p_load_lost, additionals)
    if scen in [2,3,4]:
        export(folder,scen, Y, P_C, P_R, P_DAM, res_curtailment, P_H, CAP_E ,CAP_BH,P_S, C_S, L_S, F_AC, F_DC, p_load_lost, additionals)
    return Y, T, E, G, S, R, DAM, L, N, C, I, LDC, Z
Y, T, E, G, S, R,DAM, L, N, C, I, LDC, Z = get_variables(variables, folder)

def get_constraints(constr, folder, Y, T, E, G, S, R,DAM, L, N, C, I, LDC, Z):
    counter = 0
    def get_const_value(Y, T, special, counter, variables, size):
        if size == 2:
            new = np.zeros((len(Y), len(special)))
            for y in Y:
                for s in special:
                    new[y, s] = variables[counter].Pi
                    counter += 1
        if size == 3:
            new = np.zeros((len(Y), len(T), len(special)))
            for y in Y:
                for t in T:
                    for s in special:
                        new[y, t, s] = variables[counter].Pi
                        counter += 1
        return new, counter
    GenerationLimitUp, counter = get_const_value(Y, T, G, 0, constr, 3)
    DAMLimit_Up, counter = get_const_value(Y, T, DAM, counter, constr, 3)
    DAMSumUp, counter = get_const_value(Y, T, Z, counter, constr, 2)
    ResGenerationLimitUp, counter = get_const_value(Y, T, R, counter, constr, 3)
    curtailment, counter = get_const_value(Y, T, R, counter, constr, 3)
    limitLoadLoss, counter = get_const_value(Y, T, N, counter, constr, 3)
    StoragePowerOutput, counter =get_const_value(Y, T, S, counter, constr, 3)
    StoragePowerInput, counter =get_const_value(Y, T, S, counter, constr, 3)
    StorageLevelGen,counter=get_const_value(Y, T, S, counter, constr, 3)
    StorageLevelCap,counter=get_const_value(Y, T, S, counter, constr, 3)
    Storage_balance_init, counter = get_const_value(Y, T[:1], S, counter, constr, 3)
    Storage_balance, counter = get_const_value(Y, T[:-1], S, counter, constr, 3) # index falsch
    Storage_end_level, counter = get_const_value(Y, T[:1], S, counter, constr, 3) #hier stimmt an sich der index nicht
    Injection_equality, counter = get_const_value(Y, T, N, counter, constr, 3)
    for y in Y:
        pd.DataFrame(Injection_equality[y, :, :]).to_csv(folder + str(y)+"_price_node.csv")
get_constraints(constraints, folder, Y, T, E, G, S, R,DAM, L, N, C, I, LDC, Z)
#model.write(folder+'my_solution.sol')
