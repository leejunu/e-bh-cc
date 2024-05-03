#!/bin/bash
#$ -N mxkn-derand_10-aknf_1.0-logistic-mvr-05022024
#$ -t 1-60
#$ -o job_output/$JOB_NAME/$JOB_ID-$TASK_ID.log
#$ -j y

module load python/3.11.5
source venv_kn_mvr/bin/activate

# user set this before-hand
n_exp=100
n_machines=15
n_seeds_per=$(( ($n_exp + $n_machines - 1) / $n_machines ))    # rounds up n_exp/n_machines

n_amps=4    # 8, 10, 12, 14

# there should be n_amps * n_machines tasks (as qsub -t input), with n_exp seeds per amp
whichseed=$(( ($SGE_TASK_ID-1) %  $n_machines))   
whichamp=$(( ($SGE_TASK_ID-1) / $n_machines))    

# amp=$(bc <<<"scale=3;$whichamp+2") # so that amp starts at 1
amp=$(( $whichamp * 2 + 8 ))

startseed=$(( $whichseed * $n_seeds_per ))
temp=$(( $startseed + $n_seeds_per )) 
endseed=$(( ($temp < $n_exp ? $temp : $n_exp) - 1))   # make sure we do exactly n_exp experiments

python -u mxknockoffs_CC.py --name '05022024-mxkn-derand_10-aknf_1.0-logistic-mvr' --start_seed $startseed --end_seed $endseed --m 200 --n 500 --tp_freq 20 --tp_alt_sign --amp $amp  --kmethod 'mvr' --derand 10 --exact_cs_switchpoint 0.6 --mc_max_samples 1000 --mc_batch_size 100 --power_guarantee --confidence_required --rel_mc_error 0.1 --alpha_fdr 0.1 --akn_factor 1.0 --mask 'feature_selection' --model 'logistic'