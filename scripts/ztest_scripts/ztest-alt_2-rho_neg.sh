#!/bin/bash
#$ -N ztest-alt_2-rho_neg
#$ -t 2-12
#$ -o job_output/$JOB_NAME/$JOB_ID-$TASK_ID.log
#$ -j y

module load python/3.11.5
source venv_ebhcc/bin/activate

amp=$(bc <<<"scale=3;$SGE_TASK_ID/2") # 1, 1.5, ..., 5.5, 6

alt=2

python -u ztesting_CC.py --name 'ztest-alt_2-rho_neg' --n_exp 1000 --m 100 --tp 10 --tp_first --amp $amp --alpha_fdr 0.05 --rho -0.9 --e_method 'lrt' --e_alt $alt --power_guarantee --rel_mc_error 0.1  --mc_batch_size 100 --mc_max_samples 5000 --exact_cs_switchpoint 0.6 --confidence_required --mask 'pvalues' 