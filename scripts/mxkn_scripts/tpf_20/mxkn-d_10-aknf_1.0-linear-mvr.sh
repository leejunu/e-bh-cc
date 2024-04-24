#!/bin/bash
#$ -N mxkn-derand_10-aknf_1.0-linear-mvr
#$ -t 1-200
#$ -o job_output/$JOB_NAME/$JOB_ID-$TASK_ID.log
#$ -j y

module load python/3.11.5
source venv_kn_mvr/bin/activate

# user set this before-hand
n_exp=100
n_machines=50
n_seeds_per=$(( ($n_exp + $n_machines - 1) / $n_machines ))    # rounds up n_exp/n_machines

n_amps=4    # 4, 5, 6, 7

# there should be n_amps * n_machines tasks (as qsub -t input), with n_exp seeds per amp
whichseed=$(( ($SGE_TASK_ID-1) /  $n_amps))   
whichamp=$(( ($SGE_TASK_ID-1) % $n_amps))    

# amp=$(bc <<<"scale=3;$whichamp+2") # so that amp starts at 1
amp=$(( $whichamp + 4 ))

startseed=$(( $whichseed * $n_seeds_per ))
temp=$(( $startseed + $n_seeds_per )) 
endseed=$(( ($temp < $n_exp ? $temp : $n_exp) - 1))   # make sure we do exactly n_exp experiments

python -u mxknockoffs_CC.py --name 'mxkn-derand_10-aknf_1.0-linear-mvr' --start_seed $startseed --end_seed $endseed --m 200 --n 500 --tp_freq 20 --tp_alt_sign --amp $amp  --kmethod 'mvr' --derand 10 --exact_cs_switchpoint 0.6 --mc_max_samples 2000 --mc_batch_size 100 --power_guarantee --confidence_required --rel_mc_error 0.1 --alpha_fdr 0.1 --akn_factor 1.0 --mask 'none'