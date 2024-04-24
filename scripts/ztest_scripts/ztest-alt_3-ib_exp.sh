#!/bin/bash
#$ -N ztest-alt_3-ib_exp
#$ -t 2-12
#$ -o job_output/$JOB_NAME/$JOB_ID-$TASK_ID.log
#$ -j y

module load python/3.11.5
source venv_ebhcc/bin/activate

amp=$(bc <<<"scale=3;$SGE_TASK_ID/2") # 1, 1.5, ..., 5.5, 6
alt=3

python -u ztesting_CC.py --name 'ztest-alt_3-no_ib' --n_exp 1000 --m 100 --tp 10 --tp_first --amp $amp --alpha_fdr 0.05 --rho 0.5 --e_method 'lrt' --e_alt $alt --power_guarantee --rel_mc_error 0.1  --mc_batch_size 100 --mc_max_samples 5000 --exact_cs_switchpoint 0.6 --confidence_required --mask 'pvalues' 

python -u ztesting_CC.py --name 'ztest-alt_3-ib' --n_exp 1000 --m 100 --tp 10 --tp_first --amp $amp --alpha_fdr 0.05 --rho 0.5 --e_method 'lrt' --e_alt $alt --rel_mc_error 0.1  --mc_batch_size 100 --mc_max_samples 5000 --exact_cs_switchpoint 0.6 --confidence_required --mask 'pvalues' --initial_boost
