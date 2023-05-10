from pathlib import Path
import pandas as pd
import yaml
import sys
import numpy as np
import scipy

# if __name__ == '__main__':
#     import files
# else:
#     from ..utils import files

from utils import pars, files
from typing import Text
import pickle

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


def read_df(path: Path):
    return pd.read_csv(path)


def get_X_y(path: Path, params):
    df = pd.read_csv(path)
    if 'PassengerId' in df.columns:
        df = df.drop(columns=['PassengerId'], axis=1)
    if params['base']['target'] in df.columns:
        X = df.drop(columns=params['base']['target'], axis=1)
        y = df[params['base']['target']]
    else:
        X = df
        y = None
        
    return X, y


pipe_age = make_pipeline(SimpleImputer(
    missing_values=pd.NA, strategy='median'), StandardScaler())
pipe_cat = make_pipeline(OneHotEncoder(handle_unknown='ignore'))
pipe_cat_nan = make_pipeline(SimpleImputer(missing_values=pd.NA, strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore'))


pipe_transfrom = ColumnTransformer(
    [('num', pipe_age, ['Age', 'Fare']),
     ('cat', pipe_cat, ['Sex']),
     ('can_nan', pipe_cat_nan, ['Embarked'])
     ]
)

# embarked deleted because it's not sugnificant as Age, Fare and Sex


def transfrom_X(X: pd.DataFrame) -> pd.DataFrame:
    X_tr = pipe_transfrom.fit_transform(X)
    if isinstance(X_tr, np.ndarray):
        df_X_tr = pd.DataFrame(
            data=X_tr, columns=pipe_transfrom.get_feature_names_out())
        return df_X_tr
    elif isinstance(X_tr, scipy.sparse.csr_matrix):
        X_tr = X_tr.toarray()
        df_X_tr = pd.DataFrame(
            data=X_tr, columns=pipe_transfrom.get_feature_names_out())
        return df_X_tr
    else:
        return None


def append_label_save_pkl(X, y, feature_dir, filename):
    if y is not None:
        output = pd.concat([X, y], axis=1)
    else:
        output = X
        
    fpath = feature_dir / filename
    with open(fpath, 'wb') as f:
        pickle.dump(output, f)


def featurize(config_path: Text) -> None:

    with open(config_path) as config_file:
        params = yaml.safe_load(config_file)

    prepared_dir = Path(params['data']['prepared_dir'])
    featurize_dir = Path(params['data']['featurize_dir'])

    files.make_folder(featurize_dir)

    train_path = prepared_dir / params['data']['prep_train_name']
    test_path = prepared_dir / params['data']['prep_test_name']
    pred_path = prepared_dir / params['data']['prep_pred_name']

    X_train, y_train = get_X_y(train_path, params)
    X_test, y_test = get_X_y(test_path, params)
    X_pred, y_pred = get_X_y(pred_path, params) # read data fro prediction y

    X_train_tr = transfrom_X(X_train)
    X_test_tr = transfrom_X(X_test)
    X_pred_tr = transfrom_X(X_pred)

    append_label_save_pkl(X_train_tr, y_train, featurize_dir,  params['data']['trainpkl'])
    append_label_save_pkl(X_test_tr, y_test, featurize_dir, params['data']['testpkl'])
    
    # write data for prediction y_pred = None
    append_label_save_pkl(X_pred_tr, y_pred, featurize_dir, params['data']['predpkl'])
    


if __name__ == '__main__':

    featurize(config_path=pars.parse_config())
