import sys
sys.path.append('./dependencies')
# Import custom methods and architectures
from utils import *
from architectures import *
from dataset import *

# Add the problem-specific config path to the Python system path
sys.path.append('./problems')

# GLOBAL CONSTANTS
# Define the maximum number of epochs for training

global REPO_DIR
REPO_DIR = '.'

global DATA_DIR
DATA_DIR = 'ADD DATA DIR HERE'

global DEVICE
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

global SEED
SEED = 0

global MAX_EPOCHS
MAX_EPOCHS = 75

# Define the number of epochs to wait before early stopping (patience)
global PATIENCE
PATIENCE = 5

## Change problem here!
# and_
# multiclass
# multifeature
# xor
from and_ import *