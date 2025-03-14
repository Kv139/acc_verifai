
import pandas as pd
import os
import re
import numpy as np


dir_path = "./outputs/persistent_ce"

files = os.listdir(dir_path)

print(len(files))

ce_reg = r'.*\d_cex.*'
nc_reg = r'.*no_cex.*'
att_reg = r".*attacker_crash.*"

reg = [ce_reg,nc_reg,att_reg]

def classify_simulations(files):
    classified_cases = [[],[],[]]
    for simulation in files:
        for i in range(len(reg)):
            found = re.match(reg[i],simulation)
            if found:
                classified_cases[i].append(simulation)
                break
    return classified_cases

final = classify_simulations(files)

sum = 0
for i in final:
    sum += len(i)

print(sum == len(files))


file_path = "outputs/persistent_attack_ce.csv"

persistent_ce_params_df = pd.read_csv(file_path)
persistent_mab_params_df = pd.read_csv("outputs/persistent_attack_mab.csv")


print(persistent_ce_params_df.columns)

distance_init_ce = persistent_ce_params_df["point.params.inter_vehicle_distance"]
distance_init_mab = persistent_mab_params_df["point.params.inter_vehicle_distance"]

print(distance_init_ce.describe(),distance_init_mab.describe())

labels = ["mean", "std", "min", "25%", "50%", "75%", "max"]

print("          Mab                 CE")
for i in range(1,len(distance_init_mab.describe())):
    print(f"{labels[i-1]}: {distance_init_mab.describe()[i]}, {distance_init_ce.describe()[i]}")


