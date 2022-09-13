# python run_answer_quality_avg.py --obj_name car --base_name UA-DETRAC_share --test_date 2020-09-19 --ground_truth 39311_res608_yolo --ground_truth_freq 1 --predictions 39311_res608_yolo --sample_frac 0.5 --constraints 39311_res608_yolo --use_val True --val_frac 0.5 --our_alg True
# python run_answer_quality_avg.py --obj_name car --base_name UA-DETRAC_share --test_date 2020-09-19 --ground_truth 39311_res608_yolo --ground_truth_freq 1 --predictions 39311_res608_yolo --sample_frac 0.332 --constraints 39311_res608_yolo --use_val True --val_frac 0.332 --our_alg True
# Make the correction set size to be 500, also use it as the sample size when varying resolution.

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import sys
sys.path.append('../')

from Estimation.run_answer_quality_avg import estimate_avg_error

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
fig = plt.figure(figsize=(9,2.8))

num_40771 = 1720
num_40775 = 975

data_path = "../Data/"
obj_name = "car"
base_name = "UA-DETRAC_share"
test_date = "2020-09-19"
ground_truth_freq = 1
our_alg = "True"
constraints = "40771_res608_yolo"
cons_obj_name = None
use_val = "True"

#AVG; sample fraction
sample_sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500]
limit_sample_sizes = [10, 20, 30, 40, 50]

corrected_error_40771 = []
ground_truth = "40771_res608_yolo"
predictions = "40771_res608_yolo"
val_frac = 500/num_40771
for sample_size in sample_sizes:
    sample_frac = sample_size/num_40771
    _, estimate_err, _ = estimate_avg_error(obj_name, base_name, test_date, ground_truth, predictions, sample_frac, ground_truth_freq, constraints, use_val, val_frac, our_alg, data_path, cons_obj_name)
    corrected_error_40771.append(estimate_err*100)

corrected_error_40775 = []
ground_truth = "40775_res608_yolo"
predictions = "40775_res608_yolo"
val_frac = 500/num_40775
for sample_size in sample_sizes:
    sample_frac = sample_size/num_40775
    _, estimate_err, _ = estimate_avg_error(obj_name, base_name, test_date, ground_truth, predictions, sample_frac, ground_truth_freq, constraints, use_val, val_frac, our_alg, data_path, cons_obj_name)
    corrected_error_40775.append(estimate_err*100)

corrected_error_40771_limit50 = []
ground_truth = "40771_res608_yolo"
predictions = "40771_res608_yolo"
val_frac = 50/num_40771
for sample_size in limit_sample_sizes:
    sample_frac = sample_size/num_40771
    _, estimate_err, _ = estimate_avg_error(obj_name, base_name, test_date, ground_truth, predictions, sample_frac, ground_truth_freq, constraints, use_val, val_frac, our_alg, data_path, cons_obj_name)
    corrected_error_40771_limit50.append(estimate_err*100)


plt.subplot(1,2,1)
corrected_error_diff_40771_40771_limit50 = np.abs(np.array(corrected_error_40771)[:5]-np.array(corrected_error_40771_limit50))
corrected_error_diff_40771_40775 = np.abs(np.array(corrected_error_40771)-np.array(corrected_error_40775))
plt.plot(limit_sample_sizes, corrected_error_diff_40771_40771_limit50,linewidth=3, color = 'orange', marker='x',markersize=7)
plt.plot(sample_sizes, corrected_error_diff_40771_40775,linewidth=3, color = 'red', marker='x',markersize=7)
plt.xlabel("Sample size",fontsize=16)
plt.ylabel("Error difference (%)",fontsize=16)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
ax=plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
plt.grid(axis='y',linestyle='--',linewidth=1)
plt.xlim(0,100)
plt.ylim(0,30)
fig.legend(["Error bound difference between limited and unlimited frame access to original video", "Error bound difference between original video and similar video"],bbox_to_anchor=(0.5,1.26), loc='upper center',ncol=1,fontsize=15)
plt.tight_layout()

#AVG; frame resolution
resolutions = [608, 416, 384, 320, 288, 256, 224, 192, 160, 128]

corrected_error_40771 = []
ground_truth = "40771_res608_yolo"
val_frac = 500/num_40771
sample_frac = 500/num_40771
for resolution in resolutions:
    predictions = "40771_res"+str(resolution)+"_yolo"
    _, estimate_err, _ = estimate_avg_error(obj_name, base_name, test_date, ground_truth, predictions, sample_frac, ground_truth_freq, constraints, use_val, val_frac, our_alg, data_path, cons_obj_name)
    corrected_error_40771.append(estimate_err*100)

corrected_error_40775 = []
ground_truth = "40775_res608_yolo"
val_frac = 500/num_40775
sample_frac = 500/num_40775
for resolution in resolutions:
    predictions = "40775_res"+str(resolution)+"_yolo"
    _, estimate_err, _ = estimate_avg_error(obj_name, base_name, test_date, ground_truth, predictions, sample_frac, ground_truth_freq, constraints, use_val, val_frac, our_alg, data_path, cons_obj_name)
    corrected_error_40775.append(estimate_err*100)


plt.subplot(1,2,2)
corrected_error_diff_40771_40775 = np.abs(np.array(corrected_error_40771)-np.array(corrected_error_40775))
plt.plot(resolutions, corrected_error_diff_40771_40775,linewidth=3, color = 'red', marker='x',markersize=7)
plt.xlabel("Frame resolution",fontsize=16)
plt.ylabel("Error difference (%)",fontsize=16)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
ax=plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
plt.grid(axis='y',linestyle='--',linewidth=1)
plt.xlim(100,608)
plt.ylim(0,30)
plt.tight_layout()

fig.savefig("Figure10.pdf", bbox_inches='tight')