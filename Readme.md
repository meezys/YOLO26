This file contains the source code for training YOLO26x as a teacher model and distilling knowledge down to a YOLO26n model to be deployed on the ADS-DV vehicle
Training data is uploaded via cloud.

# sbatch_setup branch
The results/data obtained in this branch are made possible thanks to the Warwick DCS Batch Compute System. The use of the cluster is documented in the 'jupyter.sbatch' file, the time results are based on the falcon partition. 
As this was a long job, a python script was generated from the jupyter notebook to ensure stable overnight-style training. 
