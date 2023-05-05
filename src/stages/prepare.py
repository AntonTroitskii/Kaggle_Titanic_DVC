import sys
import zipfile
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import os
import shutil
from typing import Text

import yaml
from src.utils import pars, files



def prepare_data(config_path: Text) -> None:

    with open(config_path) as config_file:
        params = yaml.safe_load(config_file)
    
    input = Path(params['data']['input'])

    # clear previous data
    data_dir = Path(params['data']['data_dir'])
    init_dir = Path(params['data']['init_dir'])
    files.clear_all(data_dir, [init_dir])

    # unzip data
    unzip_dir = Path(params['data']['unzip_dir'])
    with zipfile.ZipFile(input, 'r') as zipf:
        zipf.extractall(unzip_dir)

    # prepare data for prediction
    prepared_folder = Path(params['data']['prepared_dir'])

    files.make_folder(prepared_folder)
    test_crs = unzip_dir / params['data']['test_name']
    test_dst = prepared_folder / params['data']['prep_pred_name']
    shutil.copy(test_crs, test_dst)

    # prepare train test data
    df = pd.read_csv(unzip_dir / params['data']['train_name'])
    y = df[[params['base']['target']]]

    X_train, X_test = train_test_split(df,
                                       test_size=params['prepare']['split'],
                                       random_state=params['base']['seed'],
                                       shuffle=True,
                                       stratify=y)

    train_path = prepared_folder / params['data']['prep_train_name']
    test_path = prepared_folder / params['data']['prep_test_name']

    pd.DataFrame(X_train, columns=df.columns).to_csv(train_path, index=False)
    pd.DataFrame(X_test, columns=df.columns).to_csv(test_path, index=False)


if __name__ == '__main__':

    prepare_data(config_path=pars.parse_config())
    

