import joblib
import json
import csv
import pickle
import argparse
import numpy as np
import pandas as pd
import os

from scipy.stats import pearsonr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader,TensorDataset
import random
from AAmodel_class import *
from AAmetric import *

def get_train_data(fea1,fea2,k,path):
    select_file =path + f"{k}.pkl"
    with open(select_file, "rb") as f:
        select=pickle.load(f)

    fea1 = pd.read_csv(fea1, header=0, index_col=None)
    fea2 = pd.read_csv(fea2, header=None, index_col=None)
    fea = pd.concat([fea2, fea1], axis=1)

    features = fea.iloc[:, :2616]
    features = features.iloc[:, select]
    return features

def test(Model,test_data,path):
    for i in range(1, 6):
        Model.load_state_dict(torch.load(path + 'model_{}'.format(str(i)) + '.pkl'))
        Model.eval()
        pp = []
        with torch.no_grad():
            test_dataa = torch.tensor(test_data.to_numpy(), dtype=torch.float32)
            predict = Model(test_dataa)
            pred = predict.detach().cpu().numpy().ravel().tolist()
            if isinstance(pred, list):
                pp.extend(pred)
            else:
                pp.extend([pred])

        result = {'pred': pp}
        resultt = pd.DataFrame(result)
        resultt.to_csv(path + 'predict{}.csv'.format(i), index_label=False, index=False)

def analyse(path):
    combined_results = []
    for i in range(1, 6):
        model_predictions = pd.read_csv(path + 'predict{}.csv'.format(i))
        predictions = model_predictions.iloc[:,0].values
        binary_predictions = np.where(predictions > 0.5, 1, 0)
        combined_results.append(binary_predictions)
    combined_df = pd.DataFrame(np.column_stack(combined_results), columns=[f'model_{i}_pred' for i in range(1, 6)])
    combined_df['final_result'] = combined_df.apply(lambda row: 1 if row.sum() > 2 else 0, axis=1)
    combined_df.to_csv(path + 'prediction.csv', index=False)



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('fea1', type=str,help='fea1:56 dim')
    parser.add_argument('fea2', type=str, help='fea2:2560 dim')
    parser.add_argument('path', type=str,help='save path')
    parser.add_argument('k', type=int, default=1012,help='feature selection index')

    args = parser.parse_args()

    test_data = get_train_data(args.fea1,args.fea2,args.k,args.path)
    model_full = FullModel(in_dim=args.k)
    # device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model_full.to(device)
    test(model_full,test_data,args.path)
    analyse(args.path)


if __name__ == "__main__":
    main()


