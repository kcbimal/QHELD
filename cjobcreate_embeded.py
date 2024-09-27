stage = 'stage_1'
root = 'Ta_3.319_300K'
par_dir = 'C:/Users/bkc/Downloads/Tantalum_recover/Ta_QMD'
filepath = par_dir + '/normal_modes/'+stage+'/'

#%%
# Define the content of the shell script
script_content = '''#!/bin/bash
#SBATCH --qos=debug
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=16
#SBATCH -C cpu
#SBATCH -J stage_1
#SBATCH -t 00:30:00
#SBATCH -A m3845
export SLURM_CPU_BIND="cores"
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMP_NUM_THREADS=8
export HDF5_USE_FILE_LOCKING=FALSE

# QE parallelization parameters (problem-dependent)
nimage=1
npool=1
nband=16
ntg=1
ndiag=1

flags="-nimage $nimage -npool $npool -nband $nband -ntg $ntg -ndiag $ndiag"

module load espresso/7.0-libxc-5.2.2-cpu

'''

#%%
# Generate the lines with $i going from 0 to 19
for i in range(20):
    script_content += f'srun pw.x $flags -input Ta-{i}.in > {root}_QE_in_{stage}_{i}.out\n'
filename ='pjob_all.sh'
# Write the content to a shell script file
with open(filepath + filename, 'w') as f:
    f.write(script_content)