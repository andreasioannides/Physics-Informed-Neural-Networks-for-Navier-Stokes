from PINN_V8 import PINN

import sys
import numpy as np
import yaml
from datetime import datetime
import os
import shutil

def get_boundary_conditions(config_file: dict, dim: int):
    """Returns boundary conditions dynamically based on the dimension."""

    if dim == 2:
        return config_file['boundary_conditions']['2d']['bc']
    elif dim == 3:
        return config_file['boundary_conditions']['3d']['bc']
    
def main():
    general_path = os.getcwd()
    general_config_path = os.path.join(general_path, 'config_1.yaml')
    with open(general_config_path, "r", encoding="utf-8") as file:
        config_file = yaml.safe_load(file)

    # Get arguments
    if len(sys.argv) > 1:
        nodes_filename = sys.argv[1]
    else:
        nodes_filename = config_file['paths']['nodes']

    if len(sys.argv) > 2:
        experiment_foldername = sys.argv[2]
    else:
        experiment_foldername = None
        
    # Create savings folder
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    general_results_foldername = config_file['paths']['general_results_foldername']

    path_parts = [general_path, general_results_foldername]
    if experiment_foldername:
        path_parts.append(experiment_foldername)
    path_parts.append(timestamp + "_" + os.path.splitext(nodes_filename)[0])

    savings_path = os.path.join(*path_parts)
    os.makedirs(savings_path, exist_ok=True)
    os.makedirs(os.path.join(savings_path, config_file['paths']['model_foldername']), exist_ok=True)

    # Copy configuration file to results directory
    config_output_path = os.path.join(savings_path, 'config.yaml')
    shutil.copy(general_config_path, config_output_path)

    # Load nodes
    general_nodes_foldername = config_file['paths']['nodes_general_foldername']

    path_parts = [general_path, general_nodes_foldername]
    if experiment_foldername:
        path_parts.append(experiment_foldername)
    path_parts.append(nodes_filename)

    precision = config_file['training']['precision']
    if precision == 'single':
        dtype = np.float32
    elif precision == 'double':
        dtype = np.float64

    nodes_path = os.path.join(*path_parts)
    with open(nodes_path, 'r') as f:
        first_line = f.readline()
        n_inlet, n_outlet, n_walls, n_internal = map(int, first_line.strip().split())
    nodes = np.loadtxt(nodes_path, skiprows=1, dtype=dtype)
    
    dim = nodes.shape[1]

    # Update configuration file 
    config_file['EXPERIMENT'] = experiment_foldername
    config_file['flow']['dimensions'] = dim
    config_file['nodes']['inlet'] = n_inlet
    config_file['nodes']['outlet'] = n_outlet
    config_file['nodes']['walls'] = n_walls
    config_file['nodes']['collocation'] = n_internal
    config_file['paths']['nodes_txt'] = nodes_filename

    with open(config_output_path, "w") as file:
        yaml.dump(config_file, file)
    
    # Training
    boundary_conditions = get_boundary_conditions(config_file, dim)
    optimization_method = config_file['optimizer']['use']
    
    model = PINN(config_file, dim=dim)
    model.train(nodes, boundary_conditions, savings_path, optimization_method, display=0)

if __name__ == "__main__": # run only from terminal (python main.py)
    main()
