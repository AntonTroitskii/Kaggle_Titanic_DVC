import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
import pickle
import json
import sys
import yaml
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc, accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
from typing import Text


from src.utils import pars, files


def evaluate(config_path: Text):

    with open(config_path) as config_file:
        params = yaml.safe_load(config_file)

    model_path = Path(params['train']['model_dir']) / params['train']['model']
    features_dir = Path(params['data']['featurize_dir'])

    # pathes of files
    reports_dir = Path(params['evaluate']['reports_dir'])
    scores_path = reports_dir / params['evaluate']['scores']
    importance_path = reports_dir / params['evaluate']['importance']
    classes_path = reports_dir / params['evaluate']['classes']
    cvs_path = reports_dir / params['evaluate']['cvs']
    cvs_mean_path = reports_dir / params['evaluate']['cvs_mean']
    plots_path = reports_dir / params['evaluate']['plots']

# read test data
    testpkl = features_dir / params['data']['testpkl']
    X_test, y_test = files.load_X_y_from_pkl(testpkl)

# open model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

# save png of importance of features
    featrues = [(i, j) for i, j in zip(
        model.feature_names_in_, model.feature_importances_)]
    featrues.sort(key=lambda x: x[1], reverse=True)
    df_featrures = pd.DataFrame(featrues, columns=['fname', 'importance'])
    sns.barplot(data=df_featrures, y='fname', x='importance')
    plt.savefig(importance_path, bbox_inches='tight')

# make predictions
    predictions_by_class = model.predict_proba(X_test)
    predictions = predictions_by_class[:, -1]

# generate scores
    precision, recall, thresholds = precision_recall_curve(y_test, predictions)
    auc_test = auc(recall, precision)
    y_pred = model.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_pred)

# cross validation
# read train data
    trainpkl = Path('data/featurized') / params['data']['trainpkl']
    X_train, y_train = files.load_X_y_from_pkl(trainpkl)

    sgk = StratifiedShuffleSplit(
        n_splits=params['evaluate']['n_split'],
        random_state=params['evaluate']['seed'])
    cvs_train = cross_val_score(model, X_train, y_train, cv=sgk, n_jobs=-1)

# save test actual and predicted
    df = pd.DataFrame({'actual': y_test, 'predict': y_pred})
    df.to_csv(classes_path, index=False)

# save scores
    with open(scores_path, 'w') as f:
        scroe_dict = {
            'cvs mean': cvs_train.mean(),
            'cvs std': cvs_train.std(),
            'cvs min': cvs_train.min(),
            'test auc': auc_test,
            'test accuracy': accuracy_test}
        json.dump(scroe_dict, f)

# save plots
# cvs
    cvs_mean_df = pd.DataFrame({'cvs_mean': [cvs_train.mean()]})
    cvs_mean_df.to_csv(cvs_mean_path, index=False)

    cvs_y = list(cvs_train)
# append mean value to cvs
    cvs_x = [i+1 for i in range(0, len(cvs_y))]
    cvs_df = pd.DataFrame({'fold': cvs_x, 'cvs_y': cvs_y})
    cvs_df.to_csv(cvs_path, index=False)


if __name__ == '__main__':

    evaluate(config_path=pars.parse_config())
