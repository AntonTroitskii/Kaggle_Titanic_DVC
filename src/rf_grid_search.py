import itertools
import subprocess

# n_estim_vals = [100, 200, 300, 500, 70]
# crit_vals = ['gini', 'entropy']
# max_depth_vals = [5, 10, 15, 20]

n_estim_vals = [50, 60, 70, 80, 100]
# n_estim_vals = [100, 200, 300, 500, 70]
crit_vals = ['gini', 'entropy']
max_depth_vals = [5, 10, 15]




# for n_estim in itertools.product(n_estim_vals):
#     subprocess.run([
#         'dvc', 'exp', 'run', '--queue',
#          '-S', f'train.n_estim={n_estim}'
#         ])

# n_estim_vals = [100, 200]
# crit_vals = ['gini']
# max_depth_vals = [5]


for n_estim, crit, max_d in itertools.product(n_estim_vals, crit_vals, max_depth_vals):
    subprocess.run([
        'dvc', 'exp', 'run', '--queue',
         '--set-param', f'train.n_estim={n_estim}',
         '--set-param', f'train.crit={crit}',
         '--set-param', f'train.max_d={max_d}'
        ])
