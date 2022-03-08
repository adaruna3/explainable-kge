#!/bin/sh
# get embeddings
python ./explainable_kge/experiments/standard_setting.py --config_file ./explainable_kge/experiments/configs/std_tucker_dt_vh+_corr_ran.yaml --options "{'logging': {'log_num':0}}"
# get explanations
python ./explainable_kge/experiments/explainable_setting.py --config_file ./explainable_kge/experiments/configs/std_tucker_dt_vh+_corr_ran.yaml --options "{'logging': {'log_num':0}, 'explain': {'experiment': 'feedback', 'ground_explanations': True, 'explanations': 'all'}}"
# output the complete set of explanations
python ./explainable_kge/experiments/get_feedback_explanations.py --config_file ./explainable_kge/experiments/configs/std_tucker_dt_vh+_corr_ran.yaml