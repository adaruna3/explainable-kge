#!/bin/sh
# generate d bar dataset based on AMT performance
python explainable_kge/experiments/generate_d_bar.py --config_file ./explainable_kge/experiments/configs/std_tucker_dt_vh+_corr_ran.yaml --options "{'feedback': {'noise_reduction_rate': 0.84, 'correction_rate': 0.84}}"
python explainable_kge/experiments/generate_d_bar.py --config_file explainable_kge/experiments/configs/std_tucker_dt_vh+_corr_ran.yaml --options "{'feedback': {'noise_reduction_rate': 0.00, 'correction_rate': 0.00}}"
# python explainable_kge/experiments/generate_d_bar.py --config_file explainable_kge/experiments/configs/std_tucker_dt_vh+_corr_ran.yaml --options "{'feedback': {'noise_reduction_rate': 0.05, 'correction_rate': 0.05}}"
# python explainable_kge/experiments/generate_d_bar.py --config_file explainable_kge/experiments/configs/std_tucker_dt_vh+_corr_ran.yaml --options "{'feedback': {'noise_reduction_rate': 0.10, 'correction_rate': 0.10}}"
# python explainable_kge/experiments/generate_d_bar.py --config_file explainable_kge/experiments/configs/std_tucker_dt_vh+_corr_ran.yaml --options "{'feedback': {'noise_reduction_rate': 0.15, 'correction_rate': 0.15}}"
# python explainable_kge/experiments/generate_d_bar.py --config_file explainable_kge/experiments/configs/std_tucker_dt_vh+_corr_ran.yaml --options "{'feedback': {'noise_reduction_rate': 0.20, 'correction_rate': 0.20}}"
# python explainable_kge/experiments/generate_d_bar.py --config_file explainable_kge/experiments/configs/std_tucker_dt_vh+_corr_ran.yaml --options "{'feedback': {'noise_reduction_rate': 0.25, 'correction_rate': 0.25}}"
# python explainable_kge/experiments/generate_d_bar.py --config_file explainable_kge/experiments/configs/std_tucker_dt_vh+_corr_ran.yaml --options "{'feedback': {'noise_reduction_rate': 0.30, 'correction_rate': 0.30}}"
# python explainable_kge/experiments/generate_d_bar.py --config_file explainable_kge/experiments/configs/std_tucker_dt_vh+_corr_ran.yaml --options "{'feedback': {'noise_reduction_rate': 0.35, 'correction_rate': 0.35}}"
# python explainable_kge/experiments/generate_d_bar.py --config_file explainable_kge/experiments/configs/std_tucker_dt_vh+_corr_ran.yaml --options "{'feedback': {'noise_reduction_rate': 0.40, 'correction_rate': 0.40}}"
# python explainable_kge/experiments/generate_d_bar.py --config_file explainable_kge/experiments/configs/std_tucker_dt_vh+_corr_ran.yaml --options "{'feedback': {'noise_reduction_rate': 0.45, 'correction_rate': 0.45}}"
# python explainable_kge/experiments/generate_d_bar.py --config_file explainable_kge/experiments/configs/std_tucker_dt_vh+_corr_ran.yaml --options "{'feedback': {'noise_reduction_rate': 0.50, 'correction_rate': 0.50}}"
# python explainable_kge/experiments/generate_d_bar.py --config_file explainable_kge/experiments/configs/std_tucker_dt_vh+_corr_ran.yaml --options "{'feedback': {'noise_reduction_rate': 0.55, 'correction_rate': 0.55}}"
# python explainable_kge/experiments/generate_d_bar.py --config_file explainable_kge/experiments/configs/std_tucker_dt_vh+_corr_ran.yaml --options "{'feedback': {'noise_reduction_rate': 0.60, 'correction_rate': 0.60}}"
# python explainable_kge/experiments/generate_d_bar.py --config_file explainable_kge/experiments/configs/std_tucker_dt_vh+_corr_ran.yaml --options "{'feedback': {'noise_reduction_rate': 0.65, 'correction_rate': 0.65}}"
# python explainable_kge/experiments/generate_d_bar.py --config_file explainable_kge/experiments/configs/std_tucker_dt_vh+_corr_ran.yaml --options "{'feedback': {'noise_reduction_rate': 0.70, 'correction_rate': 0.70}}"
# python explainable_kge/experiments/generate_d_bar.py --config_file explainable_kge/experiments/configs/std_tucker_dt_vh+_corr_ran.yaml --options "{'feedback': {'noise_reduction_rate': 0.75, 'correction_rate': 0.75}}"
# python explainable_kge/experiments/generate_d_bar.py --config_file explainable_kge/experiments/configs/std_tucker_dt_vh+_corr_ran.yaml --options "{'feedback': {'noise_reduction_rate': 0.80, 'correction_rate': 0.80}}"
# python explainable_kge/experiments/generate_d_bar.py --config_file explainable_kge/experiments/configs/std_tucker_dt_vh+_corr_ran.yaml --options "{'feedback': {'noise_reduction_rate': 0.85, 'correction_rate': 0.85}}"
# python explainable_kge/experiments/generate_d_bar.py --config_file explainable_kge/experiments/configs/std_tucker_dt_vh+_corr_ran.yaml --options "{'feedback': {'noise_reduction_rate': 0.90, 'correction_rate': 0.90}}"
# python explainable_kge/experiments/generate_d_bar.py --config_file explainable_kge/experiments/configs/std_tucker_dt_vh+_corr_ran.yaml --options "{'feedback': {'noise_reduction_rate': 0.95, 'correction_rate': 0.95}}"
# python explainable_kge/experiments/generate_d_bar.py --config_file explainable_kge/experiments/configs/std_tucker_dt_vh+_corr_ran.yaml --options "{'feedback': {'noise_reduction_rate': 1.00, 'correction_rate': 1.00}}"
# get embeddings
python explainable_kge/experiments/standard_setting.py --config_file ./explainable_kge/experiments/configs/std_tucker_dt_vh+_corr_ran.yaml --options "{'dataset': {'name':'VH+_CORR_RAN_DENOISED_84_CORRECTED_84'}}"
python explainable_kge/experiments/standard_setting.py --config_file explainable_kge/experiments/configs/std_tucker_dt_vh+_corr_ran.yaml --options "{'dataset': {'name':'VH+_CORR_RAN_DENOISED_0_CORRECTED_0'}}"
# python explainable_kge/experiments/standard_setting.py --config_file explainable_kge/experiments/configs/std_tucker_dt_vh+_corr_ran.yaml --options "{'dataset': {'name':'VH+_CORR_RAN_DENOISED_5_CORRECTED_5'}}"
# python explainable_kge/experiments/standard_setting.py --config_file explainable_kge/experiments/configs/std_tucker_dt_vh+_corr_ran.yaml --options "{'dataset': {'name':'VH+_CORR_RAN_DENOISED_10_CORRECTED_10'}}"
# python explainable_kge/experiments/standard_setting.py --config_file explainable_kge/experiments/configs/std_tucker_dt_vh+_corr_ran.yaml --options "{'dataset': {'name':'VH+_CORR_RAN_DENOISED_15_CORRECTED_15'}}"
# python explainable_kge/experiments/standard_setting.py --config_file explainable_kge/experiments/configs/std_tucker_dt_vh+_corr_ran.yaml --options "{'dataset': {'name':'VH+_CORR_RAN_DENOISED_20_CORRECTED_20'}}"
# python explainable_kge/experiments/standard_setting.py --config_file explainable_kge/experiments/configs/std_tucker_dt_vh+_corr_ran.yaml --options "{'dataset': {'name':'VH+_CORR_RAN_DENOISED_25_CORRECTED_25'}}"
# python explainable_kge/experiments/standard_setting.py --config_file explainable_kge/experiments/configs/std_tucker_dt_vh+_corr_ran.yaml --options "{'dataset': {'name':'VH+_CORR_RAN_DENOISED_30_CORRECTED_30'}}"
# python explainable_kge/experiments/standard_setting.py --config_file explainable_kge/experiments/configs/std_tucker_dt_vh+_corr_ran.yaml --options "{'dataset': {'name':'VH+_CORR_RAN_DENOISED_35_CORRECTED_35'}}"
# python explainable_kge/experiments/standard_setting.py --config_file explainable_kge/experiments/configs/std_tucker_dt_vh+_corr_ran.yaml --options "{'dataset': {'name':'VH+_CORR_RAN_DENOISED_40_CORRECTED_40'}}"
# python explainable_kge/experiments/standard_setting.py --config_file explainable_kge/experiments/configs/std_tucker_dt_vh+_corr_ran.yaml --options "{'dataset': {'name':'VH+_CORR_RAN_DENOISED_45_CORRECTED_45'}}"
# python explainable_kge/experiments/standard_setting.py --config_file explainable_kge/experiments/configs/std_tucker_dt_vh+_corr_ran.yaml --options "{'dataset': {'name':'VH+_CORR_RAN_DENOISED_50_CORRECTED_50'}}"
# python explainable_kge/experiments/standard_setting.py --config_file explainable_kge/experiments/configs/std_tucker_dt_vh+_corr_ran.yaml --options "{'dataset': {'name':'VH+_CORR_RAN_DENOISED_55_CORRECTED_55'}}"
# python explainable_kge/experiments/standard_setting.py --config_file explainable_kge/experiments/configs/std_tucker_dt_vh+_corr_ran.yaml --options "{'dataset': {'name':'VH+_CORR_RAN_DENOISED_60_CORRECTED_60'}}"
# python explainable_kge/experiments/standard_setting.py --config_file explainable_kge/experiments/configs/std_tucker_dt_vh+_corr_ran.yaml --options "{'dataset': {'name':'VH+_CORR_RAN_DENOISED_65_CORRECTED_65'}}"
# python explainable_kge/experiments/standard_setting.py --config_file explainable_kge/experiments/configs/std_tucker_dt_vh+_corr_ran.yaml --options "{'dataset': {'name':'VH+_CORR_RAN_DENOISED_70_CORRECTED_70'}}"
# python explainable_kge/experiments/standard_setting.py --config_file explainable_kge/experiments/configs/std_tucker_dt_vh+_corr_ran.yaml --options "{'dataset': {'name':'VH+_CORR_RAN_DENOISED_75_CORRECTED_75'}}"
# python explainable_kge/experiments/standard_setting.py --config_file explainable_kge/experiments/configs/std_tucker_dt_vh+_corr_ran.yaml --options "{'dataset': {'name':'VH+_CORR_RAN_DENOISED_80_CORRECTED_80'}}"
# python explainable_kge/experiments/standard_setting.py --config_file explainable_kge/experiments/configs/std_tucker_dt_vh+_corr_ran.yaml --options "{'dataset': {'name':'VH+_CORR_RAN_DENOISED_85_CORRECTED_85'}}"
# python explainable_kge/experiments/standard_setting.py --config_file explainable_kge/experiments/configs/std_tucker_dt_vh+_corr_ran.yaml --options "{'dataset': {'name':'VH+_CORR_RAN_DENOISED_90_CORRECTED_90'}}"
# python explainable_kge/experiments/standard_setting.py --config_file explainable_kge/experiments/configs/std_tucker_dt_vh+_corr_ran.yaml --options "{'dataset': {'name':'VH+_CORR_RAN_DENOISED_95_CORRECTED_95'}}"
# python explainable_kge/experiments/standard_setting.py --config_file explainable_kge/experiments/configs/std_tucker_dt_vh+_corr_ran.yaml --options "{'dataset': {'name':'VH+_CORR_RAN_DENOISED_100_CORRECTED_100'}}"
# TEST
python explainable_kge/experiments/standard_setting.py --config_file ./explainable_kge/experiments/configs/std_tucker_dt_vh+_corr_ran.yaml --options "{'sess_mode': 'TEST', 'dataset': {'name':'VH+_CORR_RAN_DENOISED_84_CORRECTED_84'}}"
python explainable_kge/experiments/standard_setting.py --config_file explainable_kge/experiments/configs/std_tucker_dt_vh+_corr_ran.yaml --options "{'sess_mode': 'TEST', 'dataset': {'name':'VH+_CORR_RAN_DENOISED_0_CORRECTED_0'}}"
# python explainable_kge/experiments/standard_setting.py --config_file explainable_kge/experiments/configs/std_tucker_dt_vh+_corr_ran.yaml --options "{'sess_mode': 'TEST', 'dataset': {'name':'VH+_CORR_RAN_DENOISED_5_CORRECTED_5'}}"
# python explainable_kge/experiments/standard_setting.py --config_file explainable_kge/experiments/configs/std_tucker_dt_vh+_corr_ran.yaml --options "{'sess_mode': 'TEST', 'dataset': {'name':'VH+_CORR_RAN_DENOISED_10_CORRECTED_10'}}"
# python explainable_kge/experiments/standard_setting.py --config_file explainable_kge/experiments/configs/std_tucker_dt_vh+_corr_ran.yaml --options "{'sess_mode': 'TEST', 'dataset': {'name':'VH+_CORR_RAN_DENOISED_15_CORRECTED_15'}}"
# python explainable_kge/experiments/standard_setting.py --config_file explainable_kge/experiments/configs/std_tucker_dt_vh+_corr_ran.yaml --options "{'sess_mode': 'TEST', 'dataset': {'name':'VH+_CORR_RAN_DENOISED_20_CORRECTED_20'}}"
# python explainable_kge/experiments/standard_setting.py --config_file explainable_kge/experiments/configs/std_tucker_dt_vh+_corr_ran.yaml --options "{'sess_mode': 'TEST', 'dataset': {'name':'VH+_CORR_RAN_DENOISED_25_CORRECTED_25'}}"
# python explainable_kge/experiments/standard_setting.py --config_file explainable_kge/experiments/configs/std_tucker_dt_vh+_corr_ran.yaml --options "{'sess_mode': 'TEST', 'dataset': {'name':'VH+_CORR_RAN_DENOISED_30_CORRECTED_30'}}"
# python explainable_kge/experiments/standard_setting.py --config_file explainable_kge/experiments/configs/std_tucker_dt_vh+_corr_ran.yaml --options "{'sess_mode': 'TEST', 'dataset': {'name':'VH+_CORR_RAN_DENOISED_35_CORRECTED_35'}}"
# python explainable_kge/experiments/standard_setting.py --config_file explainable_kge/experiments/configs/std_tucker_dt_vh+_corr_ran.yaml --options "{'sess_mode': 'TEST', 'dataset': {'name':'VH+_CORR_RAN_DENOISED_40_CORRECTED_40'}}"
# python explainable_kge/experiments/standard_setting.py --config_file explainable_kge/experiments/configs/std_tucker_dt_vh+_corr_ran.yaml --options "{'sess_mode': 'TEST', 'dataset': {'name':'VH+_CORR_RAN_DENOISED_45_CORRECTED_45'}}"
# python explainable_kge/experiments/standard_setting.py --config_file explainable_kge/experiments/configs/std_tucker_dt_vh+_corr_ran.yaml --options "{'sess_mode': 'TEST', 'dataset': {'name':'VH+_CORR_RAN_DENOISED_50_CORRECTED_50'}}"
# python explainable_kge/experiments/standard_setting.py --config_file explainable_kge/experiments/configs/std_tucker_dt_vh+_corr_ran.yaml --options "{'sess_mode': 'TEST', 'dataset': {'name':'VH+_CORR_RAN_DENOISED_55_CORRECTED_55'}}"
# python explainable_kge/experiments/standard_setting.py --config_file explainable_kge/experiments/configs/std_tucker_dt_vh+_corr_ran.yaml --options "{'sess_mode': 'TEST', 'dataset': {'name':'VH+_CORR_RAN_DENOISED_60_CORRECTED_60'}}"
# python explainable_kge/experiments/standard_setting.py --config_file explainable_kge/experiments/configs/std_tucker_dt_vh+_corr_ran.yaml --options "{'sess_mode': 'TEST', 'dataset': {'name':'VH+_CORR_RAN_DENOISED_65_CORRECTED_65'}}"
# python explainable_kge/experiments/standard_setting.py --config_file explainable_kge/experiments/configs/std_tucker_dt_vh+_corr_ran.yaml --options "{'sess_mode': 'TEST', 'dataset': {'name':'VH+_CORR_RAN_DENOISED_70_CORRECTED_70'}}"
# python explainable_kge/experiments/standard_setting.py --config_file explainable_kge/experiments/configs/std_tucker_dt_vh+_corr_ran.yaml --options "{'sess_mode': 'TEST', 'dataset': {'name':'VH+_CORR_RAN_DENOISED_75_CORRECTED_75'}}"
# python explainable_kge/experiments/standard_setting.py --config_file explainable_kge/experiments/configs/std_tucker_dt_vh+_corr_ran.yaml --options "{'sess_mode': 'TEST', 'dataset': {'name':'VH+_CORR_RAN_DENOISED_80_CORRECTED_80'}}"
# python explainable_kge/experiments/standard_setting.py --config_file explainable_kge/experiments/configs/std_tucker_dt_vh+_corr_ran.yaml --options "{'sess_mode': 'TEST', 'dataset': {'name':'VH+_CORR_RAN_DENOISED_85_CORRECTED_85'}}"
# python explainable_kge/experiments/standard_setting.py --config_file explainable_kge/experiments/configs/std_tucker_dt_vh+_corr_ran.yaml --options "{'sess_mode': 'TEST', 'dataset': {'name':'VH+_CORR_RAN_DENOISED_90_CORRECTED_90'}}"
# python explainable_kge/experiments/standard_setting.py --config_file explainable_kge/experiments/configs/std_tucker_dt_vh+_corr_ran.yaml --options "{'sess_mode': 'TEST', 'dataset': {'name':'VH+_CORR_RAN_DENOISED_95_CORRECTED_95'}}"
# python explainable_kge/experiments/standard_setting.py --config_file explainable_kge/experiments/configs/std_tucker_dt_vh+_corr_ran.yaml --options "{'sess_mode': 'TEST', 'dataset': {'name':'VH+_CORR_RAN_DENOISED_100_CORRECTED_100'}}"