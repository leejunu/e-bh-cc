#!/bin/bash
#$ -N outlier-m_200-n_200-train_500-theta_3-prop_0.4-03252024.sh
#$ -t 1-90
#$ -o job_output/$JOB_NAME/$JOB_ID-$TASK_ID.log
#$ -j y

module load python/3.11.5
source venv_numba/bin/activate

# user set this before-hand
n_exp=1000
n_machines=10
n_seeds_per=$(( ($n_exp + $n_machines - 1) / $n_machines ))    # rounds up n_exp/n_machines

n_amps=9    # [2.0, 2.25, ...., 4.0]

# there should be n_amps * n_machines tasks (as qsub -t input), with n_exp seeds per amp
whichseed=$(( ($SGE_TASK_ID-1) %  $n_machines))   
whichamp=$(( ($SGE_TASK_ID-1) / $n_machines))   

amp=$(bc <<<"scale=1;$whichamp*.25+2.0") 

startseed=$(( $whichseed * $n_seeds_per ))
temp=$(( $startseed + $n_seeds_per )) 
endseed=$(( ($temp < $n_exp ? $temp : $n_exp) - 1))   # make sure we do exactly n_exp experiments

python -u outlierdetection_CC.py --name '03252024-outlier-m_200-n_200-train_500-theta_3-prop_0.4' --start_seed $startseed --end_seed $endseed --prop_outliers 0.4 --m 200 --n 200 --n_train 500 --amp $amp --exact_cs_switchpoint 0.6 --mc_max_samples 2500 --mc_batch_size 100 --power_guarantee --confidence_required --rel_mc_error 0.1 --alpha_fdr 0.1 --theta_type 3 --weight_scores