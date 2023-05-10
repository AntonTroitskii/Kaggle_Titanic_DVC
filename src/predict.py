import pandas as pd
from typing import Text
from utils import pars
import yaml
from pathlib import Path
import pickle
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit

def predict(config_path: Text):
    
    with open(config_path) as config_file:
        params = yaml.safe_load(config_file)
    
    # get pathes    
    featurize_dir = Path(params['data']['featurize_dir'])
    predict_dir = Path(params['data']['predict_dir'])
    
    sub_path = Path(params['data']['unzip_dir']) / params['data']['sub_name']    
    pred_path = featurize_dir / params['data']['predpkl']
    train_path = featurize_dir / params['data']['trainpkl']
    test_path = featurize_dir / params['data']['testpkl']
    model_path = Path(params['train']['model_dir']) / params['train']['model']
    
        
    # load train data
    
    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)        
    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)

    train_all = pd.concat([train_data, test_data], ignore_index=True)
    
    X_train_all = train_all.iloc[:, :-1]
    y_train_all = train_all.iloc[:, -1]
    
    # Train and Predict
    
    ## load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)    
    
    ## cross-val-score
    sgk = StratifiedShuffleSplit(
        n_splits=params['evaluate']['n_split'],
        random_state=params['evaluate']['seed'])
    cvs = cross_val_score(model, X_train_all, y_train_all, cv=sgk, n_jobs=-1)
    print(f'cvs = {cvs}')
    print(f'cvs = {cvs.mean()}')
        

    model.fit(X_train_all, y_train_all)
    
    # read pred data    
    with open(pred_path, 'rb') as f:
        X_pred = pickle.load(f)
    y_pred = model.predict(X_pred)
    
    ## read submission file to replce the res
    df_sub = pd.read_csv(sub_path)
    target = params['base']['target']
    df_sub[target] = y_pred
    
    ## safe result in result dir
    pred_name = 'precition.csv'
    pred_path = predict_dir / pred_name
    
    df_sub.to_csv(pred_path, index=False)

if __name__ == '__main__':
    
    predict(config_path=pars.parse_config())
    