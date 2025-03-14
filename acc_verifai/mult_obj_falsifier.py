"""
Framework for experimentation of platooning attacks.
Author: Iván Salgado & Juan José Torres
Email: ifsalgad@ucsc.edu & jtorr127@ucsc.edu 
"""

import time
import os
import numpy as np
from dotmap import DotMap
import traceback
import argparse

from verifai.samplers.scenic_sampler import ScenicSampler
from verifai.scenic_server import ScenicServer
from verifai.falsifier import generic_falsifier, generic_parallel_falsifier
from verifai.monitor import multi_objective_monitor, specification_monitor
from verifai.falsifier import generic_falsifier
import pandas as pd
import random
import networkx as nx


def announce(message):
    lines = message.split('\n')
    size = max([len(p) for p in lines]) + 4

    def pad(line):
        ret = '* ' + line
        ret += ' ' * (size - len(ret) - 1) + '*'
        return ret
    lines = list(map(pad, lines))
    m = '\n'.join(lines)
    border = '*' * size
    print(border)
    print(m)
    print(border)

iteration = 1
counterex = 1
route = ""

def create_distances_csv(name, d1, d2, d3, p1, p2, p3, p4):
    distances = {"c0_c1": [], "c1_c2": [], "c2_c3": [], "p1": [],  "p2": [],  "p3": [],  "p4": []}
    for i in d1:
        distances['c0_c1'].append(i[0])
    for j in d2:
        distances['c1_c2'].append(j[0])
    for k in d3:
        distances['c2_c3'].append(k[0])
    for m in p1:
        distances['p1'].append(m[0])
    for n in p2:
        distances['p2'].append(n[0])
    for x in p3:
        distances['p3'].append(x[0])
    for y in p4:
        distances['p4'].append(y[0])        
    df = pd.DataFrame.from_dict(distances)
    df.to_csv(name + f".csv")
    
"""
Single-objective specification. This monitor is similar to the one above, but takes a
minimum over the distances from each vehicle. If the ego vehicle is less than 5 meters
away from any vehicle at any point, a counterexample is returned.
"""


class distance_multi(multi_objective_monitor):
    def __init__(self,num_objectives=3):

        self.num_ojectives = num_objectives

        priority_graph = nx.DiGraph()
        priority_graph.add_node(0)
        priority_graph.add_node(1)
        priority_graph.add_node(2)

        print(f"Initialized priority graph with {self.num_ojectives} objectives")

        self.simulation_count = 0

        self.iteration = 0
        self.counterex = 0
        self.ego_crash = 0
        self.route = ""

        self.velocites = []
        self.dict_keys = ["attacker_speed", "f1_speed","f2_speed","f3_speed"]


        self.no_crash = 0
        self.c2_c3_counter = 0
        self.c3_c4_counter = 0

        self.min_dist_counter = [0,0,0]

        def specification(simulation):
             
            ego_velocity = simulation.result.records
            simulation_velocites = []
            for i in self.dict_keys:
                simulation_velocites.append(ego_velocity[i])  
           
            self.velocites.append(simulation_velocites) 

            positions = np.array(simulation.result.trajectory)
            global route
            self.route = route
            p1 = positions[:, [0], :]
            p2 = positions[:, [1], :]
            p3 = positions[:, [2], :]
            p4 = positions[:, [3], :]
            distances0 = positions[:, [0], :] - positions[:, [1], :]
            distances1 = positions[:, [1], :] - positions[:, [2], :]
            distances2 = positions[:, [2], :] - positions[:, [3], :]

            distances0 = np.linalg.norm(distances0, axis=2)
            distances1 = np.linalg.norm(distances1, axis=2)
            distances2 = np.linalg.norm(distances2, axis=2)

            rho0 = np.min(distances0) - 5
            rho1 = np.min(distances1) - 5
            rho2 = np.min(distances2) - 5

            min_distances = [rho0, rho1, rho2]
            
            classify_outcome(rho0,rho1,rho2)

            print(f"Distances between cars respectively {rho0}, {rho1}, {rho2}")
            

            attacker = 0
            dist_victims = []
            for i in range(3):
                if i != attacker:
                    dist_victims.append(min_distances[i])
            
            rho_victims = min(dist_victims)
           
            idx = np.argmin(dist_victims)
            self.min_dist_counter[idx+1] += 1
           
            min_distances[0] = min_distances[0] * -1
            rho_attacker = min_distances[attacker] 
            rho = max(rho_victims, rho_attacker) # if rho_attacker > 0 then it was < 0 during simulation indicating a crash. 

            if rho>0:
                if rho_attacker <0:
                    name = self.route+"/distances_" + str(self.iteration) + "_no_cex" # Neet to be able to associate distances file with rho files for classification later 
                    create_distances_csv(name, distances0, distances1, distances2, p1, p2, p3, p4) 
                    self.iteration += 1
              
                elif rho_attacker >0:
                    name = self.route+"/distances_" + str(self.iteration) + "_attacker_crash"
                    create_distances_csv(name, distances0, distances1, distances2, p1, p2, p3, p4)
                    self.iteration += 1
                
            if rho<0:
                name = self.route+"/distances_" + str(self.counterex) + "_cex"
                create_distances_csv(name, distances0, distances1, distances2, p1, p2, p3, p4)
             
            self.simulation_count += 1 
             
            print(f"Robustness values are : {min_distances}")

            return min_distances

        def classify_outcome(attacker_c1,c2_c3,c3_c4):
            if attacker_c1 > 0: #attacker did not crash
                if c2_c3 > 0 and c3_c4 > 0: #neither did any cars
                    self.no_crash += 1

                elif c2_c3 > 0 and c3_c4 < 0: #cars 3-4 crashed
                    self.counterex += 1
                    self.c3_c4_counter +=1

                elif c2_c3 < 0 and c3_c4 > 0: #cars 2-3 crashed
                    self.counterex += 1
                    self.c2_c3_counter += 1
                
                elif c2_c3 < 0 and c3_c4 < 0: #both cars 2-3, 3-4 crashed
                    self.counterex += 1
                    self.c2_c3_counter += 1
                    self.c3_c4_counter += 1
            
            elif attacker_c1 <= 0:
                self.ego_crash += 1
                if c3_c4 < 0:
                    self.c3_c4_counter += 1
                elif c2_c3 < 0:
                    self.c2_c3_counter += 1
                else:
                    print("Error classifying outcome for attacker < 0")
        

        super().__init__(specification,priority_graph)


"""
Runs all experiments in a directory.
"""


def run_experiments(path, parallel=False, model=None,
                    sampler_type=None, headless=False, num_workers=5, output_dir='outputs',
                    experiment_name=None, map_path=None, lgsvl=False, routepath=""):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(routepath):
        os.mkdir(routepath)
    global route
    route = routepath
    paths = []
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for name in files:
                fname = os.path.join(root, name)
                if os.path.splitext(fname)[1] == '.scenic':
                    paths.append(fname)
    else:
        paths = [path]
    for p in paths:
        try:
            falsifier = run_experiment(p, parallel=parallel,
                                       model=model, sampler_type=sampler_type, headless=headless,
                                       num_workers=num_workers)
        except:
            announce(f'ERROR FOR SCRIPT {p}:\n\n{traceback.format_exc()}')
            continue
        df = pd.concat([falsifier.error_table.table,
                        falsifier.safe_table.table])
        if experiment_name is not None:
            outfile = experiment_name
        else:
            root, _ = os.path.splitext(p)
            outfile = root.split('/')[-1]
            if parallel:
                outfile += '_parallel'
            if model:
                outfile += f'_{model}'
            if sampler_type:
                outfile += f'_{sampler_type}'
        outfile += '.csv'
        outpath = os.path.join(output_dir, outfile)
        announce(f'SAVING OUTPUT TO {outpath}')


        df.to_csv(outpath)


"""
Runs a single falsification experiment.
Arguments:
    path: Path to Scenic script to be run.
    parallel: Whether or not to enable parallelism.
    model: Which simulator model to use (e.g. carla, lgsvl, newtonian, etc.)
    sampler_type: Which VerifAI sampelr to use (e.g. halton, scenic, ce, mab, etc.)
    headless: Whether or not to display each simulation.
    num_workers: Number of parallel workers. Only used if parallel is true.
"""


def run_experiment(path, parallel=False, model=None,
                   sampler_type=None, headless=False, num_workers=5):
    announce(f'RUNNING SCENIC SCRIPT {path}')
    model = f'scenic.simulators.{model}.model' if model else None


    if model:
        params['model'] = model


    num_objectives = 3
    iterations = 500
    falsifier_params = DotMap(
        n_iters=iterations,
        save_error_table=True,
        save_safe_table=True,
    )
    server_options = DotMap(maxSteps=1000, verbosity=2)
    monitor = distance_multi(num_objectives)

    opts = DotMap(priority_graph=monitor.graph)
    params = {'verifaiSamplerType': sampler_type, 'verifaiSamplerParams':opts} if sampler_type else {}
    sampler = ScenicSampler.fromScenario(path,params=params)


    falsifier_cls = generic_parallel_falsifier if parallel else generic_falsifier

    falsifier = falsifier_cls(sampler=sampler, falsifier_params=falsifier_params,
                              server_class=ScenicServer,
                              server_options=server_options,
                              monitor=monitor)
                              #,scenic_path=path,
                              #scenario_params=params, num_workers=num_workers)
    t0 = time.time()
    print('Running falsifier')
    falsifier.run_falsifier()
    t = time.time() - t0
    print()

    with open("experiment_results.txt", 'a') as file:
        file.write(f"\n \n \n Results \n \n ============= \n")
        file.write(f"Results for {path} running verifai sampler{sampler_type} and {iterations} iterations \n")
        file.write(f'Generated {len(falsifier.samples)} samples in {t} seconds with {falsifier.num_workers} workers \n')
        file.write(f'Number of counterexamples: {len(falsifier.error_table.table)} \n')
        file.write(f'Sampling time: {falsifier.total_sample_time} \n Simulation time: {falsifier.total_simulate_time} \n')
        file.write(f"Confidence interval: {falsifier.get_confidence_interval()} \n")

        file.write(f"\n Displaing counts over all iterations \n ========================= \n Crash incidents and counts are the following \n ")
        file.write(f"CounterExamples: {monitor.counterex} \n")
        file.write(f"No crashes: {monitor.no_crash} \n")
        file.write(f"Attacker crashed: {monitor.ego_crash} \n")

        file.write(f"Car 2 and 3 crashes total {monitor.c2_c3_counter}, \n Car 3 and 4 crashes total {monitor.c3_c4_counter} \n")

        file.write(f"Distance minimums between cars was as follows {monitor.min_dist_counter}")



    print(
        f'Generated {len(falsifier.samples)} samples in {t} seconds with {falsifier.num_workers} workers')
    print(f'Number of counterexamples: {len(falsifier.error_table.table)}')
    if not parallel:
        print(f'Sampling time: {falsifier.total_sample_time}')
        print(f'Simulation time: {falsifier.total_simulate_time}')
    print(f'Confidence interval: {falsifier.get_confidence_interval()}')

    
    print(("Displaing counts over all iterations \n ========================="))
    print("Crash incidents and counts are the following")
    print(f"CounterExamples: {monitor.counterex}")
    print(f"No crashes: {monitor.no_crash}")
    print(f"Attacker crashed: {monitor.ego_crash}")

    print(f"Car 2 and 3 crashes total {monitor.c2_c3_counter}, \n Car 3 and 4 crashes total {monitor.c3_c4_counter}")

    print(f"Distance minimums between cars was as follows {monitor.min_dist_counter}")

    velocities_df = pd.DataFrame(monitor.velocites)
    velocities_df.to_csv(path+"_"+str(iterations)+str(sampler_type+"_velocities.csv",index=False)

    return falsifier


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', type=str, default='./scenarios-dddas/persistent_attack.scenic',
                        help='Path to Scenic script')
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--num-workers', type=int, default=5,
                        help='Number of parallel workers')
    parser.add_argument('--sampler-type', '-s', type=str, default=None,
                        help='verifaiSamplerType to use')
    parser.add_argument('--experiment-name', '-e', type=str, default=None,
                        help='verifaiSamplerType to use')
    parser.add_argument('--model', '-m', type=str, default=None)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--lgsvl', '-l', action='store_true')
    parser.add_argument('--route', '-r', type=str, default="outputs")
    args = parser.parse_args()

   
    run_experiments(args.path, args.parallel,
                    model=args.model, sampler_type=args.sampler_type, headless=args.headless,
                    num_workers=args.num_workers, experiment_name=args.experiment_name, lgsvl=args.lgsvl,routepath=args.route)