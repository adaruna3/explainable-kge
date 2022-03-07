#!/bin/sh
# get embeddings
python ./explainable_kge/experiments/standard_setting.py --config_file ./explainable_kge/experiments/configs/std_tucker_dt_vh+_clean_ran.yaml --options "{'logging': {'log_num':0}}"
python ./explainable_kge/experiments/standard_setting.py --config_file ./explainable_kge/experiments/configs/std_tucker_dt_vh+_clean_ran.yaml --options "{'logging': {'log_num':1}}"
python ./explainable_kge/experiments/standard_setting.py --config_file ./explainable_kge/experiments/configs/std_tucker_dt_vh+_clean_ran.yaml --options "{'logging': {'log_num':2}}"
python ./explainable_kge/experiments/standard_setting.py --config_file ./explainable_kge/experiments/configs/std_tucker_dt_vh+_clean_ran.yaml --options "{'logging': {'log_num':3}}"
python ./explainable_kge/experiments/standard_setting.py --config_file ./explainable_kge/experiments/configs/std_tucker_dt_vh+_clean_ran.yaml --options "{'logging': {'log_num':4}}"
# Run 0
python ./explainable_kge/experiments/explainable_setting.py --config_file ./explainable_kge/experiments/configs/std_tucker_dt_vh+_clean_ran.yaml --options "{'logging': {'log_num':0}, 'explain': {'experiment': 'preferences', 'ground_explanations': True}}"
# Run 1
python ./explainable_kge/experiments/explainable_setting.py --config_file ./explainable_kge/experiments/configs/std_tucker_dt_vh+_clean_ran.yaml --options "{'logging': {'log_num':1}, 'explain': {'experiment': 'preferences', 'ground_explanations': True}}"
# Run 2
python ./explainable_kge/experiments/explainable_setting.py --config_file ./explainable_kge/experiments/configs/std_tucker_dt_vh+_clean_ran.yaml --options "{'logging': {'log_num':2}, 'explain': {'experiment': 'preferences', 'ground_explanations': True}}"
# Run 3
python ./explainable_kge/experiments/explainable_setting.py --config_file ./explainable_kge/experiments/configs/std_tucker_dt_vh+_clean_ran.yaml --options "{'logging': {'log_num':3}, 'explain': {'experiment': 'preferences', 'ground_explanations': True}}"
# Run 4
python ./explainable_kge/experiments/explainable_setting.py --config_file ./explainable_kge/experiments/configs/std_tucker_dt_vh+_clean_ran.yaml --options "{'logging': {'log_num':4}, 'explain': {'experiment': 'preferences', 'ground_explanations': True}}"
# output the complete set of explanations
python ./explainable_kge/experiments/get_preferences_explanations.py --config_file ./explainable_kge/experiments/configs/std_tucker_dt_vh+_clean_ran.yaml