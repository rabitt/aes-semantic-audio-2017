#!/bin/bash
# we need 1 node, with 1 process per node:
#PBS -l nodes=1:ppn=1
#PBS -l walltime=04:00:00
#PBS -l mem=120gb
#PBS -N tsne_500_50_5000
#PBS -M justin.salamon@gmail.com
#PBS -j oe
#PBS -m abe
#PBS -p 1000

module purge
module load numpy/intel/1.12.0 scipy/intel/0.18.1 scikit-learn/intel/0.18.1
SRCDIR=$HOME/dev/aes-semantic-audio-2016
RUNDIR=$SCRATCH/aes-semantic-audio-2016/run-${PBS_JOBID/.*}
mkdir $RUNDIR
cd $RUNDIR
python $SRCDIR/experiment-scripts/contour_tsne.py $SRCDIR/notebooks/analysis/vizdata/features-salience.npz $SRCDIR/notebooks/analysis/vizdata/feature-names.npz $SRCDIR/notebooks/analysis/vizdata/tsne_output/tsne_500_50_5000.npz --min_duration 0.5 --perplexity 50.0 --n_iter 5000


