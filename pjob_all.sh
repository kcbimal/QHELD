#!/bin/bash
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

srun pw.x $flags -input Ta-0.in > Ta_3.319_300K_QE_in_stage_1_0.out
srun pw.x $flags -input Ta-1.in > Ta_3.319_300K_QE_in_stage_1_1.out
srun pw.x $flags -input Ta-2.in > Ta_3.319_300K_QE_in_stage_1_2.out
srun pw.x $flags -input Ta-3.in > Ta_3.319_300K_QE_in_stage_1_3.out
srun pw.x $flags -input Ta-4.in > Ta_3.319_300K_QE_in_stage_1_4.out
srun pw.x $flags -input Ta-5.in > Ta_3.319_300K_QE_in_stage_1_5.out
srun pw.x $flags -input Ta-6.in > Ta_3.319_300K_QE_in_stage_1_6.out
srun pw.x $flags -input Ta-7.in > Ta_3.319_300K_QE_in_stage_1_7.out
srun pw.x $flags -input Ta-8.in > Ta_3.319_300K_QE_in_stage_1_8.out
srun pw.x $flags -input Ta-9.in > Ta_3.319_300K_QE_in_stage_1_9.out
srun pw.x $flags -input Ta-10.in > Ta_3.319_300K_QE_in_stage_1_10.out
srun pw.x $flags -input Ta-11.in > Ta_3.319_300K_QE_in_stage_1_11.out
srun pw.x $flags -input Ta-12.in > Ta_3.319_300K_QE_in_stage_1_12.out
srun pw.x $flags -input Ta-13.in > Ta_3.319_300K_QE_in_stage_1_13.out
srun pw.x $flags -input Ta-14.in > Ta_3.319_300K_QE_in_stage_1_14.out
srun pw.x $flags -input Ta-15.in > Ta_3.319_300K_QE_in_stage_1_15.out
srun pw.x $flags -input Ta-16.in > Ta_3.319_300K_QE_in_stage_1_16.out
srun pw.x $flags -input Ta-17.in > Ta_3.319_300K_QE_in_stage_1_17.out
srun pw.x $flags -input Ta-18.in > Ta_3.319_300K_QE_in_stage_1_18.out
srun pw.x $flags -input Ta-19.in > Ta_3.319_300K_QE_in_stage_1_19.out
