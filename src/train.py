from sklearn.ensemble import RandomForestClassifier
import pickle

import sys
from pathlib import Path
from typing import Text

for i in sys.path:
    print(i)

from utils import pars, files
import yaml


def train(config_path: Text):

    with open(config_path) as config_file:
        params = yaml.safe_load(config_file)

    featurized_dir = Path(params['data']['featurize_dir'])
    model_dir = Path(params['train']['model_dir'])
    model_path = model_dir / params['train']['model']

    train_path = featurized_dir / params['data']['trainpkl']
    # test_path = featurized_dir / params['data']['testpkl']

    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)

    X_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]

    clf = RandomForestClassifier(
        n_estimators=params['train']['n_estim'],
        random_state=params['train']['seed'])

    clf.fit(X_train, y_train)

    files.make_folder(model_dir)
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)


if __name__ == '__main__':

    train(config_path=pars.parse_config())
