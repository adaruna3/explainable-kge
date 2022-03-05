#!/bin/sh
# get embeddings
python ./explainable_kge/experiments/standard_setting.py --config_file ./explainable_kge/experiments/configs/std_tucker_dt_vh+_clean_ran.yaml --options "{'logging': {'log_num':0}}"
python ./explainable_kge/experiments/standard_setting.py --config_file ./explainable_kge/experiments/configs/std_tucker_dt_vh+_clean_ran.yaml --options "{'logging': {'log_num':1}}"
python ./explainable_kge/experiments/standard_setting.py --config_file ./explainable_kge/experiments/configs/std_tucker_dt_vh+_clean_ran.yaml --options "{'logging': {'log_num':2}}"
python ./explainable_kge/experiments/standard_setting.py --config_file ./explainable_kge/experiments/configs/std_tucker_dt_vh+_clean_ran.yaml --options "{'logging': {'log_num':3}}"
python ./explainable_kge/experiments/standard_setting.py --config_file ./explainable_kge/experiments/configs/std_tucker_dt_vh+_clean_ran.yaml --options "{'logging': {'log_num':4}}"
# Run 0
python ./explainable_kge/experiments/explainable_setting.py --config_file ./explainable_kge/experiments/configs/std_tucker_dt_vh+_clean_ran.yaml --options "{'logging': {'log_num':0}, 'explain': {'xmodel': 'decision_tree', 'locality': 'global'}}"
python ./explainable_kge/experiments/explainable_setting.py --config_file ./explainable_kge/experiments/configs/std_tucker_dt_vh+_clean_ran.yaml --options "{'logging': {'log_num':0}, 'explain': {'xmodel': 'logit', 'locality': 'global'}}"
python ./explainable_kge/experiments/explainable_setting.py --config_file ./explainable_kge/experiments/configs/std_tucker_dt_vh+_clean_ran.yaml --options "{'logging': {'log_num':0}, 'explain': {'xmodel': 'decision_tree', 'locality': 'local3', 'locality_k': 14}}"
# Run 1
python ./explainable_kge/experiments/explainable_setting.py --config_file ./explainable_kge/experiments/configs/std_tucker_dt_vh+_clean_ran.yaml --options "{'logging': {'log_num':1}, 'explain': {'xmodel': 'decision_tree', 'locality': 'global'}}"
python ./explainable_kge/experiments/explainable_setting.py --config_file ./explainable_kge/experiments/configs/std_tucker_dt_vh+_clean_ran.yaml --options "{'logging': {'log_num':1}, 'explain': {'xmodel': 'logit', 'locality': 'global'}}"
python ./explainable_kge/experiments/explainable_setting.py --config_file ./explainable_kge/experiments/configs/std_tucker_dt_vh+_clean_ran.yaml --options "{'logging': {'log_num':1}, 'explain': {'xmodel': 'decision_tree', 'locality': 'local3', 'locality_k': 14}}"
# Run 2
python ./explainable_kge/experiments/explainable_setting.py --config_file ./explainable_kge/experiments/configs/std_tucker_dt_vh+_clean_ran.yaml --options "{'logging': {'log_num':2}, 'explain': {'xmodel': 'decision_tree', 'locality': 'global'}}"
python ./explainable_kge/experiments/explainable_setting.py --config_file ./explainable_kge/experiments/configs/std_tucker_dt_vh+_clean_ran.yaml --options "{'logging': {'log_num':2}, 'explain': {'xmodel': 'logit', 'locality': 'global'}}"
python ./explainable_kge/experiments/explainable_setting.py --config_file ./explainable_kge/experiments/configs/std_tucker_dt_vh+_clean_ran.yaml --options "{'logging': {'log_num':2}, 'explain': {'xmodel': 'decision_tree', 'locality': 'local3', 'locality_k': 14}}"
# Run 3
python ./explainable_kge/experiments/explainable_setting.py --config_file ./explainable_kge/experiments/configs/std_tucker_dt_vh+_clean_ran.yaml --options "{'logging': {'log_num':3}, 'explain': {'xmodel': 'decision_tree', 'locality': 'global'}}"
python ./explainable_kge/experiments/explainable_setting.py --config_file ./explainable_kge/experiments/configs/std_tucker_dt_vh+_clean_ran.yaml --options "{'logging': {'log_num':3}, 'explain': {'xmodel': 'logit', 'locality': 'global'}}"
python ./explainable_kge/experiments/explainable_setting.py --config_file ./explainable_kge/experiments/configs/std_tucker_dt_vh+_clean_ran.yaml --options "{'logging': {'log_num':3}, 'explain': {'xmodel': 'decision_tree', 'locality': 'local3', 'locality_k': 14}}"
# Run 4
python ./explainable_kge/experiments/explainable_setting.py --config_file ./explainable_kge/experiments/configs/std_tucker_dt_vh+_clean_ran.yaml --options "{'logging': {'log_num':4}, 'explain': {'xmodel': 'decision_tree', 'locality': 'global'}}"
python ./explainable_kge/experiments/explainable_setting.py --config_file ./explainable_kge/experiments/configs/std_tucker_dt_vh+_clean_ran.yaml --options "{'logging': {'log_num':4}, 'explain': {'xmodel': 'logit', 'locality': 'global'}}"
python ./explainable_kge/experiments/explainable_setting.py --config_file ./explainable_kge/experiments/configs/std_tucker_dt_vh+_clean_ran.yaml --options "{'logging': {'log_num':4}, 'explain': {'xmodel': 'decision_tree', 'locality': 'local3', 'locality_k': 14}}"
# plot the results
python ./explainable_kge/logger/viz_utils.py --config_file ./explainable_kge/experiments/configs/std_tucker_dt_vh+_clean_ran.yaml --options "{'explain': {'xmodel': 'decision_tree', 'locality': 'global'}, 'plotting': {'mode': 'cross_val_plot', 'output_pdf': 'global_decision_tree'}}"
python ./explainable_kge/logger/viz_utils.py --config_file ./explainable_kge/experiments/configs/std_tucker_dt_vh+_clean_ran.yaml --options "{'explain': {'xmodel': 'logit', 'locality': 'global'}, 'plotting': {'mode': 'cross_val_plot', 'output_pdf': 'global_logit'}}"
python ./explainable_kge/logger/viz_utils.py --config_file ./explainable_kge/experiments/configs/std_tucker_dt_vh+_clean_ran.yaml --options "{'explain':{'xmodel': 'decision_tree', 'locality': 'local3', 'locality_k': 14}, 'plotting': {'mode': 'cross_val_plot', 'output_pdf': 'local_decision_tree'}}"