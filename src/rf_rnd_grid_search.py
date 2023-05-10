import itertools
import subprocess
import random

random.seed(0)
num_exp = 100

for _ in range(num_exp):
    
    params = {
        'n_estim_val': random.randint(50, 2000),
        'crit_val': random.choice(['gini', 'entropy']),
        'max_depth_val': random.randint(2, 30)
    }
    

    subprocess.run([
        'dvc', 'exp', 'run', '--queue',
        '--set-param', f'train.n_estim={params["n_estim_val"]}',
        '--set-param', f'train.crit={params["crit_val"]}',
        '--set-param', f'train.max_d={params["max_depth_val"]}'
        ])
