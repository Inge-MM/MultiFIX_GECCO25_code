# OR(AND(circle,A), AND(!triangle, B))
def get_y(labels):
    y = []
    circle = labels['circle']
    a = labels['Feature1']
    b = labels['Feature2']
    triangle = labels['triangle']
    c = labels['Feature3']
    d = labels['Feature4']
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
INPUT = 'fusion' # Choose from 'img', 'tab', 'fusion'

# Define the training mode: 'end' for end-to-end, 'seq' for sequential, or 'hyb' for hybrid
global TRAINING
TRAINING = 'hyb'  # Choose from 'end', 'seq', or 'hyb'

# Define which weights to load for hybrid and sequential training
global WTS
WTS = 'ae' # Choose from 'ae' or 'single'

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

GP_DIR = './gp_files/ANDOR/' + TRAINING + '_' + WTS + '_' + str(TEMP_FREEZE) + '/'

if TRAINING == 'end': 
    IMG_FTS = 3
    TAB_FTS = 2
    LR = 1e-3
    WD = 1e-3
    BEST_FOLD = 2
elif TRAINING == 'hyb': 
    if WTS == 'ae': 
        IMG_FTS = 2
        TAB_FTS = 3
        LR = 1e-3
        WD = 1e-4
        BEST_FOLD = 2
    else: # single
        IMG_FTS = 3
        TAB_FTS = 3
        LR = 1e-3
        WD = 1e-4
        BEST_FOLD = 2
else: # seq
    if WTS == 'ae':
        if TEMP_FREEZE == False: 
            IMG_FTS = 2
            TAB_FTS = 3
            LR = 1e-3
            WD = 1e-3
            BEST_FOLD = 0
        else: 
            IMG_FTS = 2
            TAB_FTS = 3
            LR = 1e-4
            WD = 0
            BEST_FOLD = 4
    else: 
        IMG_FTS = 2
        TAB_FTS = 2
        LR = 1e-3
        WD = 0
        BEST_FOLD = 3