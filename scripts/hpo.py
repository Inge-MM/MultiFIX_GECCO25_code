# Import necessary packages for data manipulation and system path adjustments
import pandas as pd
import torch
import itertools
import sys

from config_file import *
sys.path.append('./dependencies')
from train import get_model

print_problem()

# Load and preprocess the data
# Load the label data
labels = pd.read_csv(DATA_DIR + 'fusion_labels.csv')
# Load tabular data and remove unnecessary columns for model input
tab = pd.read_csv(DATA_DIR + 'tab_data.csv')
tab.drop(columns=['A', 'B', 'C', 'id', 'Feature14', 'Feature13', 'Feature12', 'Feature11', 'Feature10'], inplace=True)
# Combine the tabular data and labels into a single DataFrame for training
data = pd.concat([tab, labels], axis=1)
# Apply a transformation to create a new target column 'y'

data['y'] = get_y(data, INPUT)


# Define hyperparameters for grid search during hyperparameter optimization (HPO)
n_runs = 5  # Number of runs for each set of hyperparameters
lrs = [1e-3, 1e-4, 1e-5]  # List of learning rates to test
wds = [1e-3, 1e-4, 0]     # List of weight decay values to test
if INPUT == 'fusion':
    z_img = [2] #[0, 1, 2, 3] # List of img_fts to test
    z_tab = [2] #[0, 1, 2, 3] # List of tab_fts to test
elif INPUT == 'img':
    if OUT_SIZE == 0: # in case of regression task
        z_img = [1] # Output node for single modality
    else:
        z_img = [OUT_SIZE] # Output node for single modality
    z_tab = [0] # Not using tabular data
elif INPUT == 'tab':
    z_img = [0] # Not using image data
    if OUT_SIZE == 0: # in case of regression task
        z_tab = [1] # Otput node for single modality
    else: 
        z_tab = [OUT_SIZE] # Otput node for single modality
elif INPUT in ['img_t', 'img_c']:
    z_img = [OUT_SIZE]
    z_tab = [0]
elif INPUT in ['tab_a', 'tab_b']:
    z_tab = [OUT_SIZE]
    z_img = [0]

# Initialize a DataFrame to store results from each test run
test_results = pd.DataFrame(columns=['LR', 'WD', 'Z_img', 'Z_tab', 'Fold', 'Loss', 'AUROC', 'BAcc', 'model'])

# Perform grid search over the combinations of learning rates and weight decays
for TAB_FTS, IMG_FTS, WD, LR  in itertools.product(z_tab, z_img, wds, lrs):
    if IMG_FTS == 0 and TAB_FTS == 0: continue
    set_seed(SEED) # Set the random seed for reproducibility
    
    # Log the current hyperparameter combination
    print("*** LR = ", LR, "; WD = ", WD, "; IMG_FTS = ", IMG_FTS, "; TAB_FTS = ", TAB_FTS, " ***")
    nas = {"img_fts": IMG_FTS, "tab_fts": TAB_FTS}
    
    # Train the model with the current hyperparameters and update the results DataFrame
    test_results = get_model(data, LR, WD, nas, SEED, test_results)

## Log all relevant Results
# Group results by learning rate and weight decay to calculate the mean values across folds
grouped_results = test_results.groupby(['LR', 'WD', 'Z_img', 'Z_tab'])[['Loss','AUROC','BAcc']].mean().reset_index()

# Identify the hyperparameter combination that resulted in the lowest loss
best_params = grouped_results.loc[grouped_results['Loss'].idxmin()]
best_lr = best_params['LR']
best_wd = best_params['WD']
best_img = best_params['Z_img']
best_tab = best_params['Z_tab']

# Filter the original results for the best LR and WD
best_models = test_results[(test_results['LR'] == best_lr) & (test_results['WD'] == best_wd) & (test_results['Z_img'] == best_img) & (test_results['Z_tab'] == best_tab)]
# Select the row with the lowest BCELoss
best_fold = best_models.loc[best_models['Loss'].idxmin()]
models_to_save = best_models['model']
best_model = best_fold['model']


# save best model for single modality experiments
i = 0
if INPUT != 'fusion':
    for model in models_to_save:
        save_dir = MODEL_DIR + INPUT + str(i) + '.pth'
        torch.save(model.state_dict(), save_dir)
        i += 1
        
# save best model after hpo
if INPUT == 'fusion':
    save_dir = MODEL_DIR + INPUT + '_' + TRAINING + '_' + WTS + '_Freeze_' + str(TEMP_FREEZE) + '.pth'
    torch.save(best_model.state_dict(), save_dir)
    print("Saved best model!")
        
# Display the best hyperparameters and associated metrics
print("************************** RESULTS **************************\n")
print("BEST PARAMETERS: ")
print(f"  -> LR: {best_lr}")
print(f"  -> WD: {best_wd}")
print(f"  -> Z_img: {best_img}")
print(f"  -> Z_tab: {best_tab}")
print(f"AVERAGE METRICS ACROSS FOLDS:")
print(f"  -> Loss: {best_params['Loss']:.5f}")
print(f"  -> AUROC: {best_params['AUROC']:.5f}")
print(f"  -> BAcc: {best_params['BAcc']:.5f}")
print(f"BEST FOLD: {best_fold['Fold']}")

print("\n\n\n************************** 5 folds from best Hyper Parameters **************************\n")
# Save and log each model trained with the best hyperparameters
for idx, row in best_models.iterrows():
    print("Model no. ", row['Fold'])
    print("* Loss: ", row['Loss'])
    print("* AUROC: ", row['AUROC'])
    print("* BAcc: ", row['BAcc'])
    print()