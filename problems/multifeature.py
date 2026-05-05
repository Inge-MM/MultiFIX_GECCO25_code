import os

# OR(AND(circle,A), AND(!triangle, B))
def get_y(labels, INPUT):
    y = []
    circle = labels['circle']
    a = labels['Feature1']
    b = labels['Feature2']
    triangle = labels['triangle']
    c = labels['Feature3']
    d = labels['Feature4']

    if INPUT == 'img_c':
        y = circle
    elif INPUT == 'img_t':
        y = triangle
    elif INPUT == 'tab_a':
        for i in range(labels.shape[0]):
            if a[i] > b[i]: y.append(1)
            else: y.append(0)
    elif INPUT == 'tab_b':
        for i in range(labels.shape[0]):
            if c[i] > d[i]: y.append(1)
            else: y.append(0)
    else:
        for i in range(labels.shape[0]):
            if circle[i] == 1:
                if a[i] > b[i]: y1 = 1
                else: y1 = 0
            else: y1 = 0
            if triangle[i] == 0: #!triangle
                if c[i] > d[i]: y2 = 1
                else: y2 = 0
            else: y2 = 0
            if (y1 + y2) > 0: y.append(1)
            else: y.append(0)
    return y 

# binary problem
global OUT_SIZE
OUT_SIZE = 1

# Define input
global INPUT
INPUT = 'fusion' # Choose from 'img', 'tab', 'fusion' OR 'img_c'/'img_t' OR 'tab_a'/'tab_b'

# use this for training ↓ with hpo.py
#TRAINING = os.environ.get('TRAINING')

# use this for making GP files ↓
global TRAINING
TRAINING = 'ft_comp_comp'   # Choose from:  'ft_comp_comp', 'ft_comp_part', 'ft_comp_none'
                            #               'ft_part_part', 'ft_part_none'
                            #               'ft_none_none'
# Define the training mode: 'end' for end-to-end, 'seq' for sequential, or 'hyb' for hybrid

# Define which weights to load for hybrid and sequential training
global WTS
WTS = 'single' # Choose from 'ae' or 'single'

# Define whethe to temporarily freeze img block (only for TRAINING = 'seq' and WTS = 'ae')
global TEMP_FREEZE
TEMP_FREEZE = False

# Initialise directory to save models to
global MODEL_DIR
MODEL_DIR = './models/Multifeature/'

global AE_DIR
AE_DIR = './models/img_encoders/'

def print_problem():
    print("Multifeature Problem: OR(AND(circle,A), AND(!triangle, B)); TRAINING = ", TRAINING, "; INPUT = ", INPUT, "; OUT_SIZE = ", OUT_SIZE, "; WTS = ", WTS, "; TEMP_FREEZE = ", TEMP_FREEZE)

### FOR XAI ###
global LR
global WD
global IMG_FTS
global TAB_FTS
global BEST_FOLD
global GP_DIR

GP_DIR = './gp_files/Multifeature/' + TRAINING + '_' + WTS + '_' + str(TEMP_FREEZE) + '/'

# setting ACTUAL best seeds:

if INPUT == 'fusion': 
    IMG_FTS = 2
    TAB_FTS = 2

    if TRAINING == 'ft_comp_comp':
        LR = 0.001
        WD = 0.0
        BEST_FOLD = 2
    
    elif TRAINING == 'ft_comp_part': # done but not using for GP
        LR = 0.0001
        WD = 0.0001
        BEST_FOLD = 3
    
    elif TRAINING == 'ft_comp_none': # done but not using for GP
        LR = 0.0001
        WD = 0.0001
        BEST_FOLD = 3

    elif TRAINING == 'ft_part_part': # done
        LR = 0.001
        WD = 0.001
        BEST_FOLD = 3

    
    elif TRAINING == 'ft_part_none':
        LR = 0.001
        WD = 0.0001
        BEST_FOLD = 4

    
    elif TRAINING == 'ft_none_none':
        LR = 0.001
        WD = 0.0
        BEST_FOLD = 4

    else:
        print('NO KNOWN BEST FOLD')



# I accidentally ran the things below for each model, so I need to rerun
'''
else: 
    IMG_FTS = 2
    TAB_FTS = 2
    BEST_FOLD = 3
    # when I ran everything with these settings, i also used the original LR and WD but those don't get used so I will not be rerunning those experimentts
'''