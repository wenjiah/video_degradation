# python run_answer_quality_avg.py --obj_name car --base_name UA-DETRAC --test_date 2020-09-19 --ground_truth freq1_res608_yolo --ground_truth_freq 1 --predictions freq1_res256_yolo --sample_frac 0.1 --constraints freq1_res608_yolo --cons_obj_name person --use_val True --val_frac 1 --our_alg True
# python run_answer_quality_avg.py --obj_name car --base_name UA-DETRAC --test_date 2020-09-19 --ground_truth freq1_res608_yolo --ground_truth_freq 1 --predictions freq1_res320_yolo --sample_frac 0.05 --constraints freq1_face_mtcnn --cons_obj_name face --use_val True --val_frac 1 --our_alg True
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import sys
sys.path.append('../')

from Estimation.run_answer_quality_avg import estimate_avg_error
from Estimation.run_answer_quality_max import estimate_max_error

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
fig = plt.figure(figsize=(9,2.6))

data_path = "../Data/"
obj_name = "car"
base_name = "UA-DETRAC"
test_date = "2020-09-19"
ground_truth = "freq1_res608_yolo"
ground_truth_freq = 1
our_alg = "True"
predictions_list = ["freq1_res256_yolo", "freq1_res320_yolo"]
constraints_list = ["freq1_res608_yolo", "freq1_face_mtcnn"]
cons_obj_name_list = ["person", "face"]
sample_fracs = [0.1, 0.05]
use_val = "True"

#AVG
val_fracs = [0.009, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
corrected_err_list = []
for i in range(2):
    predictions = predictions_list[i]
    constraints = constraints_list[i]
    cons_obj_name = cons_obj_name_list[i]
    sample_frac = sample_fracs[i]
    
    corrected_err = []
    for val_frac in val_fracs:
        _, estimate_err, _ = estimate_avg_error(obj_name, base_name, test_date, ground_truth, predictions, sample_frac, ground_truth_freq, constraints, use_val, val_frac, our_alg, data_path, cons_obj_name)
        corrected_err.append(estimate_err*100)
    corrected_err_list.append(corrected_err)

val_error_pre = None
for val_sample_frac in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]:
    _, val_error, _ = estimate_avg_error(obj_name, base_name, test_date, ground_truth, predictions="freq1_res608_yolo", sample_frac=val_sample_frac, ground_truth_freq=1, constraints="freq1_res608_yolo", use_val="False", val_frac=0.1, our_alg="True", data_path="../Data/", cons_obj_name=None)
    if val_error_pre != None and val_error_pre-val_error<0.02:
        val_frac_select = val_sample_frac
        break
    val_error_pre = val_error

plt.subplot(1,2,1)
plt.plot(val_fracs, corrected_err_list[0], linewidth=3, marker='x',markersize=7)
plt.plot(val_fracs, corrected_err_list[1], linewidth=3, color='green', marker='x',markersize=7)
plt.vlines(val_frac_select,-0.4,125,color='black', linestyle=':',linewidth=2)
plt.xlabel("Correction set fraction",fontsize=16)
plt.ylabel("Analytical error (%)",fontsize=16)
xticks = [0.02, 0.04, 0.06, 0.08, 0.1]
plt.xticks(xticks,fontsize=16)
plt.yticks(fontsize=16)
ax=plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
plt.grid(axis='y',linestyle='--',linewidth=1)
plt.ylim(-0.4,125)
fig.legend(["Error bound w/ correction set (f=0.1, p=256x256, c=\"person\")", "Error bound w/ correction set (f=0.05, p=320x320, c=\"face\")"],bbox_to_anchor=(0.499,1.33), loc='upper center',ncol=1,fontsize=16)
plt.tight_layout()
fig.text(0.25,-0.02,"(ⅰ) AVG",fontsize=16)

#MAX
val_fracs = [0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
corrected_err_list = []
for i in range(2):
    predictions = predictions_list[i]
    constraints = constraints_list[i]
    cons_obj_name = cons_obj_name_list[i]
    sample_frac = sample_fracs[i]
    
    corrected_err = []
    for val_frac in val_fracs:
        _, estimate_err, _ = estimate_max_error(obj_name, base_name, test_date, ground_truth, predictions, sample_frac, ground_truth_freq, constraints, use_val, val_frac, our_alg, data_path, cons_obj_name)
        corrected_err.append(estimate_err*100)
    corrected_err_list.append(corrected_err)

val_error_pre = None
for val_sample_frac in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]:
    _, val_error, _ = estimate_max_error(obj_name, base_name, test_date, ground_truth, predictions="freq1_res608_yolo", sample_frac=val_sample_frac, ground_truth_freq=1, constraints="freq1_res608_yolo", use_val="False", val_frac=0.1, our_alg="True", data_path="../Data/", cons_obj_name=None)
    if val_error_pre != None and val_error_pre-val_error<0.02:
        val_frac_select = val_sample_frac
        break
    val_error_pre = val_error

plt.subplot(1,2,2)
plt.plot(val_fracs, corrected_err_list[0], linewidth=3, marker='x',markersize=7)
plt.plot(val_fracs, corrected_err_list[1], linewidth=3, color='green', marker='x',markersize=7)
plt.vlines(val_frac_select,-0.4,125,color='black', linestyle=':',linewidth=2)
plt.xlabel("Correction set fraction",fontsize=16)
plt.ylabel("Analytical error (%)",fontsize=16)
xticks = [0.02, 0.04, 0.06, 0.08, 0.1]
plt.xticks(xticks, fontsize=16)
plt.yticks(fontsize=16)
ax=plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
plt.grid(axis='y',linestyle='--',linewidth=1)
plt.ylim(-0.4,125)
plt.tight_layout()
fig.text(0.75,-0.02,"(ⅱ) MAX",fontsize=16)
fig.savefig("Figure9.pdf", bbox_inches='tight')