# Physics-Informed-Neural-Networks-for-Navier-Stokes
This projects aims the development and training of Physics-Informed Neural Networks (PINNs) to solve the steady incompressible Navier-Stokes PDEs within a converging-diverging nozzle. The network is trained exclusively on point cloud data. The code supports both 2D and 3D equations, single and double arithmetic precision and first-, second-, and hybrid-order optimizers. The supported optimizers are Adam (via Keras), L-BFGS-B, BFGS, Broyden, SLSQP (via SciPy), Adam + L-BFGS-B, and Adam + BFGS. All hypermarameters are defined in the config.yaml file.\n

main.py takes two arguments:
  - nodes_txt_file (e.g. 3D_nodes.txt)
  - experiment_foldername (e.g. Adam_training)
  
The first argument is mandatory. The "experiment_foldername" argument can be ommited and the results will be saved in the general results foldername, the "Results". 

For a single training (e.g. testing), execute one of these two lines:
  1) python3.11 main.py nodes_txt_file experiment_foldername
  2) nohup python3.11 main.py nodes_txt_file > output.log 2>output.err &
    
For multiple training, execute the bash file with one of these two line:
  1) ./run_main.sh 
  2) nohup ./run_main.sh > output.log 2>output.err &
