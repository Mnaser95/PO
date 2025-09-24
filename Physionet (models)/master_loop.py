import os
import numpy as np
import random
import tensorflow as tf
from experiment_prep_and_run import experiment_prep_and_run


all_acc = []
seeds=[i for i in range(0,10)]

# #subbb = [2,6,8,9,18,29,42,56,65,69,78,90,93,106,108] # A
# #subbb = [4,15,28,47,48,60,61,84,91,98] # B
# subbb =  [2,6,8,4,15,28,11,77,87,54] # random
# #subbb =  [28,60,61,91,98,15,47] # selected


#subbb = [2,6,8,9,18,29,42,69,78,93,106,108] # A
#subbb = [15,28,48,60,61,84,98] # B
subbb =  [2,6,8,4,15,28,11,77,87,54] # random
#subbb =  [28,60,61,91,98,15,47] # selected

for seed in seeds:

    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    current_acc=experiment_prep_and_run(subbb,seed)
    all_acc.append(current_acc)

stop=1