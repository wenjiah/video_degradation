# python car_distribution.py --obj_name car --base_name night-street --test_date 2017-12-17 --ground_truth_freq 0.02 --predictions freq002_res608_yolo
import argparse

import numpy as np
import pandas as pd
import scipy.stats

import sys
sys.path.append('../')

from Data.generate_fnames import get_csv_fname

def get_data(base_name, date, obj_name, data_path, predictions):
    # get predicted data (after running NN model)
    csv_fname = get_csv_fname(data_path, base_name, date, predictions)
    df = pd.read_csv(csv_fname)
    df = df[df['object_name'] == obj_name]
    pred_idx = df.groupby('frame').size()
    preds = np.zeros(pred_idx.index.max() + 1)
    for idx in pred_idx.index:
        preds[idx] = pred_idx.at[idx]

    return preds

def car_num(data_path, obj_name, base_name, test_date, predictions, ground_truth_freq):
    Y_pred = get_data(base_name, test_date, obj_name, data_path, predictions)

    interval = int(1/ground_truth_freq)
    predictions = Y_pred[:len(Y_pred):interval]

    car_num_list = [len(predictions[predictions==0]), len(predictions[predictions==1]), len(predictions[predictions==2]), len(predictions[predictions==3]), len(predictions[predictions>3])]
    return car_num_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../Data/')
    parser.add_argument('--obj_name', required=True)
    parser.add_argument('--base_name', required=True)
    parser.add_argument('--test_date', required=True)
    parser.add_argument('--ground_truth_freq', type=float, required=True)
    parser.add_argument('--predictions', required=True)
    args = parser.parse_args()

    data_path = args.data_path
    obj_name = args.obj_name
    base_name = args.base_name
    test_date = args.test_date
    predictions = args.predictions
    ground_truth_freq = args.ground_truth_freq

    car_num_list = car_num(data_path, obj_name, base_name, test_date, predictions, ground_truth_freq)
    print("# of frames where there is 0 car, 1 car, 2 cars, 3 cars, or more than 3 cars:")
    print(car_num_list)
