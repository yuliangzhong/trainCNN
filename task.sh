env2lmod
module load gcc/6.3.0 python_gpu/3.8.5 cuda/11.0.3
bsub -n 4 -W 18:00 python CNNtry1.py
bsub -n 4 -W 18:00 python CNNtry2.py
bsub -n 4 -W 18:00 python CNNtry3.py
bsub -n 4 -W 18:00 python CNNtry4.py
bsub -n 4 -W 18:00 python CNNtry5.py
