# AND(circle,F1)
def get_y(labels):
    y = []
    circle = labels['circle']
    a = labels['Feature2']
    b = labels['Feature6']
    for i in range(labels.shape[0]):
        if circle[i] == 1:
            if a[i] < b[i]: y.append(1)
            else: y.append(0)
        else: y.append(0)
    return y

# binary problem
global OUT_SIZE
OUT_SIZE = 1 # do not change

### ARGUMENTS ###

# Define input
global INPUT
INPUT = 'img' # Choose from 'img', 'tab', 'fusion' #TODO use as argument


# Define the training mode: 'end' for end-to-end, 'seq' for sequential, or 'hyb' for hybrid
global TRAINING
TRAINING = 'mid-train'  # Choose from 'end', uhijuiijuij
# Define which weights to load for hybrid and sequential training
global WTS
WTS = 'single' # Choose from 'ae' for autoencoder wts or 'single' for single modality wts

# Define whethe to temporarily freeze img block (only for TRAINING = 'seq' and WTS = 'ae')
global TEMP_FREEZE
TEMP_FREEZE = False

### END OF ARGUMENTS ###
        
# Initialise directory to save models to
global MODEL_DIR
MODEL_DIR = f'./models/{INPUT}/'

global AE_DIR
AE_DIR = './models/img_encoders/'

def print_problem():
    print("!!!!!!!!!!!!!!!! AND Problem: AND(circle,A); TRAINING = ", TRAINING, "; INPUT = ", INPUT, "; OUT_SIZE = ", OUT_SIZE, "; WTS = ", WTS, "; TEMP_FREEZE = ", TEMP_FREEZE)

### FOR XAI ###
global LR
global WD
global IMG_FTS
global TAB_FTS
global BEST_FOLD
global GP_DIR

GP_DIR = './gp_files/AND/' + TRAINING + '_' + WTS + '_' + str(TEMP_FREEZE) + '/'

if TRAINING == 'end': 
    IMG_FTS = 2
    TAB_FTS = 1
    LR = 1e-3
    WD = 0
    BEST_FOLD = 1
elif TRAINING == 'hyb': 
    if WTS == 'ae': 
        IMG_FTS = 3
        TAB_FTS = 2
        LR = 1e-3
        WD = 1e-3
        BEST_FOLD = 4
    else: # single
        IMG_FTS = 3
        TAB_FTS = 2
        LR = 1e-3
        WD = 1e-4
        BEST_FOLD = 0
else: # seq
    if WTS == 'ae':
        if TEMP_FREEZE == False: 
            IMG_FTS = 2
            TAB_FTS = 1
            LR = 1e-3
            WD = 1e-3
            BEST_FOLD = 4
        else: 
            IMG_FTS = 3
            TAB_FTS = 2
            LR = 1e-4
            WD = 1e-3
            BEST_FOLD = 0
    else: 
        IMG_FTS = 2
        TAB_FTS = 2
        LR = 1e-3
        WD = 1e-4
        BEST_FOLD = 1
