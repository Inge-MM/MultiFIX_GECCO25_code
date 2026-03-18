import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

# Add the project-specific dependencies path to the Python system path
sys.path.append('./scripts')
from config_file import *

def get_dataloaders(data, train_idx, test_idx, seed):
    train_data = data.iloc[train_idx]
    unique_labels = np.unique(train_data['y'])
    if len(unique_labels) > 5: # for regression
        kbins = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
        y_binned = kbins.fit_transform(train_data['y'].values.reshape(-1,1)).reshape(-1)
        train_data, val_data = train_test_split(train_data, test_size=0.20, shuffle=True, stratify=y_binned, random_state=seed)
    else: train_data, val_data = train_test_split(train_data, test_size=0.20, shuffle=True, stratify=train_data['y'], random_state=seed)
    test_data = data.iloc[test_idx]
    
    train_dataset = BinaryDataset(train_data, DATA_DIR, OUT_SIZE)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataset = BinaryDataset(val_data, DATA_DIR, OUT_SIZE)
    val_loader = DataLoader(val_dataset, shuffle=True)
    test_dataset = BinaryDataset(test_data, DATA_DIR, OUT_SIZE)
    test_loader = DataLoader(test_dataset, shuffle=True)
    
    return train_loader, val_loader, test_loader

def train_epoch(dataloader, model, loss_fn, optim):
    """
    Train the model for one epoch.

    Parameters:
        dataloader: DataLoader object providing training data
        model: The neural network model to be trained
        loss_fn: Loss function used to compute the loss
        optim: Optimizer used for updating the model weights

    Returns:
        total_loss: Average loss over the epoch
    """
    model.to(DEVICE)  # Move model to the training DEVICE
    model.train()     # Set model to training mode
    total_loss = 0.   # Initialize total loss

    # Iterate over the training data
    for i, (img, tab, _, _, y) in enumerate(dataloader):
        img, tab, y = img.to(DEVICE, dtype = torch.float), tab.to(DEVICE, dtype = torch.float), y.to(DEVICE, dtype = torch.float)
        optim.zero_grad()  # Clear previous gradients
        prob = model(img, tab)  # Forward pass
        loss = loss_fn(prob, y)  # Compute loss
        loss.backward()  # Backward pass
        optim.step()  # Update weights
        total_loss += loss.item()  # Accumulate loss

    total_loss /= (i + 1)  # Compute average loss over the epoch
    
    return total_loss

def val_epoch(dataloader, model, loss_fn):
    """
    Validate the model for one epoch.

    Parameters:
        dataloader: DataLoader object providing validation data
        model: The neural network model to be validated
        loss_fn: Loss function used to compute the loss

    Returns:
        total_loss: Average loss over the epoch
    """
    model.to(DEVICE)  # Move model to the validation DEVICE
    model.eval()      # Set model to evaluation mode
    total_loss = 0.   # Initialize total loss

    # Iterate over the validation data
    for i, (img, tab, _, _, y) in enumerate(dataloader):
        img, tab, y = img.to(DEVICE, dtype = torch.float), tab.to(DEVICE, dtype = torch.float), y.to(DEVICE, dtype = torch.float)
        prob = model(img, tab)  # Forward pass
        loss = loss_fn(prob, y)  # Compute loss
        total_loss += loss.item()  # Accumulate loss

    total_loss /= (i + 1)  # Compute average loss over the epoch
    
    return total_loss

def eval_model(model, dataloader, loss_fn):
    """
    Evaluate the model performance on the provided data.

    Parameters:
        model: The trained neural network model
        dataloader: DataLoader object providing test data

    Returns:
        total_loss: Average loss over the test data
        auroc: Area Under the Receiver Operating Characteristic Curve (AUROC)
        bacc: Balanced Accuracy Score (BAcc)
    """
    model.to(DEVICE)  # Move model to the evaluation DEVICE
    model.eval()      # Set model to inference mode
    
    # Initialize metric variables
    total_loss = 0.
    all_y, all_prob = [], []

    with torch.no_grad():  # No gradient computation needed for evaluation
        # Iterate over the test data
        for i, (img, tab, _, _, y) in enumerate(dataloader):
            img, tab, y = img.to(DEVICE, dtype = torch.float), tab.to(DEVICE, dtype = torch.float), y.to(DEVICE, dtype = torch.float)
            prob = model(img, tab)  # Forward pass
            loss = loss_fn(prob, y)  # Compute loss
            total_loss += loss.item()  # Accumulate loss
            all_prob.append(prob.detach().cpu().numpy())  # Collect probabilities
            all_y.append(y.detach().cpu().numpy())  # Collect true labels
            
    total_loss /= (i + 1)  # Compute average loss over the test data
    
    if len(np.unique(all_y)) > 5: # if regression don't calculate auroc and bacc
        auroc = 0.5
        bacc = 0.5
    else: auroc, bacc = get_metrics(np.asarray(all_y).reshape(-1,1), np.asarray(all_prob).reshape(-1,1), OUT_SIZE)  # Compute additional metrics

    return total_loss, auroc, bacc

def train_model(train_dataloader, val_dataloader, loss_fn, lr, wd, model):
    """
    Train and validate a model with specified hyperparameters.

    Parameters:
        train_dataloader: DataLoader object providing training data
        val_dataloader: DataLoader object providing validation data
        lr: Learning rate for the optimizer
        wd: Weight decay for the optimizer

    Returns:
        model: The model with the best weights after training
    """
    # initialise optimiser
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd) 
    
    no_improvement = 0  # Counter for early stopping
    best_loss = np.inf  # Initialize best loss
    best_model_wts = copy.deepcopy(model.state_dict())  # Save the best model weights
    train_hist = []  # History of training losses
    val_hist = []    # History of validation losses
    
    # Training loop
    for epoch in range(MAX_EPOCHS):
        if INPUT == 'fusion':# and model.img_block is not None:
            if TEMP_FREEZE and epoch == 15:
                print("Defreezing the img block after 20 epochs!")
                for param in model.img_block.resnet.parameters():
                    param.requires_grad = True
        train_loss = train_epoch(train_dataloader, model, loss_fn, optim)
        train_hist.append(train_loss)
        val_loss = val_epoch(val_dataloader, model, loss_fn)
        val_hist.append(val_loss)
        print("#{}: train loss = {:.5f}; val loss = {:.5f}".format(epoch, train_loss, val_loss))
        
        # Check for improvement and apply early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())  # Update best model weights
            no_improvement = 0
        else:
            no_improvement += 1
        if no_improvement >= PATIENCE:  # Early stopping criteria
            break
    
    # Load the best model weights
    model.load_state_dict(best_model_wts)
    
    return model

def initialise_model(fold, nas, XAI=False):
    """
    Initialise the model according to input data, output size and training procedure.

    Parameters:
        fold: cross validation fold
        nas: number of extracted features per data modality

    Returns:
        results: initialised model with right configuration
    """
    if INPUT == 'img': model = single_img_net(OUT_SIZE)

    elif INPUT in ['img_t', 'img_c']: model = single_img_net(OUT_SIZE)
    
    elif INPUT in ['tab_a', 'tab_b']: model = single_tab_net(OUT_SIZE)

    elif INPUT == 'tab': single_tab_net(OUT_SIZE)

    #TODO if input == A etc
    # if model.circle_weights != none load weights
    
    elif INPUT == 'fusion' and TRAINING in ['ft_comp', 'ft_part', 'ft_none']: # ft = fusion test
        model = multifix_net_test(nas, OUT_SIZE)
        # img
        if TRAINING in ['ft_part', 'ft_comp']:
            if model.img_c_block is not None:
                img_c_wts = torch.load(MODEL_DIR + 'img_c' + str(fold) + '.pth', map_location=DEVICE)
                #resnet_c_wts = {k: v for k, v in img_c_wts.items() if not k.startswith('classifier')}
                model.img_c_block.load_state_dict(img_c_wts, strict=False)
            # tab
            if model.tab_a_block is not None:
                tab_a_wts = torch.load(MODEL_DIR + 'tab_a' + str(fold) + '.pth', map_location=DEVICE)
                #fts_a_wts = {k.replace('fts.', '', 1): v for k, v in tab_a_wts.items() if not k.startswith('classifier')}
                model.tab_a_block.load_state_dict(tab_a_wts)#, strict=False)
            
        if TRAINING == 'ft_comp':
            if model.img_t_block is not None:
                img_t_wts = torch.load(MODEL_DIR + 'img_t' + str(fold) + '.pth', map_location=DEVICE)
                #resnet_t_wts = {k: v for k, v in img_t_wts.items() if not k.startswith('classifier')}
                model.img_t_block.load_state_dict(img_t_wts, strict=False)
            # tab
            if model.tab_b_block is not None:
                tab_b_wts = torch.load(MODEL_DIR + 'tab_b' + str(fold) + '.pth', map_location=DEVICE)
                #fts_b_wts = {k.replace('fts.', '', 1): v for k, v in tab_b_wts.items() if not k.startswith('classifier')}
                model.tab_b_block.load_state_dict(tab_b_wts)#, strict=False)

        # freezing params
        if TRAINING == 'ft_part':
            for block in [model.img_c_block, model.tab_a_block]:
                for param in block.parameters():
                    param.requires_grad = False

        if TRAINING == 'ft_comp':
            # freezing params
            for block in [model.img_t_block, model.tab_b_block]:
                for param in block.parameters():
                    param.requires_grad = False


    #elif TRAINING == 'ft_none':
    #    model = multifix_net_test(nas, OUT_SIZE)

    else: # fusion
        model = multifix_net(nas, OUT_SIZE)
        if not XAI:
            # Training procedure
            if TRAINING != 'end': # if hybrid or sequential training, load wts from img and tab blocks
                if WTS == 'ae':
                    if model.img_block is not None:
                        print("Loading AE wts in image feature engineering block!")
                        # load image feature engineering weights (up to latent representation)
                        img_wts = torch.load(AE_DIR + 'img' + str(fold) + '.pth', map_location=DEVICE)
                        resnet_wts = {k: v for k, v in img_wts.items() if not k.startswith('classifier')}
                        model.img_block.resnet.load_state_dict(resnet_wts)#, strict=False)
                else: # single modality wts
                    print("Loading single modality wt(s) in feature engineering blocks!")
                    if model.img_block is not None:
                        # load image feature engineering weights (up to latent representation)
                        img_wts = torch.load(MODEL_DIR + 'img' + str(fold) + '.pth', map_location=DEVICE)
                        resnet_wts = {k: v for k, v in img_wts.items() if not k.startswith('classifier')}
                        model.img_block.resnet.load_state_dict(resnet_wts, strict=False)
                    if model.tab_block is not None:
                        # load tabular feature engineering weights (up to latent representation)
                        tab_wts = torch.load(MODEL_DIR + 'tab' + str(fold) + '.pth', map_location=DEVICE)
                        fts_wts = {k.replace('fts.', '', 1): v for k, v in tab_wts.items() if not k.startswith('classifier')}
                        model.tab_block.fts.load_state_dict(fts_wts)#, strict=False)
            if TRAINING == 'seq': # if sequential training freeze img and tab blocks
                if model.img_block is not None:
                    print("Freezing IMG feature engineering block!")
                    # freeze img block up to latent representation
                    for param in model.img_block.resnet.parameters():
                        param.requires_grad = False
                if WTS == 'single':
                    if model.tab_block is not None:
                        print("Freezing feature engineering blocks!")
                        # freeze tab block up to latent representation
                        for param in model.tab_block.fts.parameters():
                            param.requires_grad = False
                
    # Initialize loss function according to output size
    if OUT_SIZE == 1: loss_fn = nn.BCELoss() # binary classification
    elif OUT_SIZE > 1: loss_fn = nn.CrossEntropyLoss() # multiclass classification
    else: loss_fn = nn.MSELoss() # regression
    
    return model, loss_fn

def get_model(data, lr, wd, nas, seed, results):
    """
    Train and evaluate models with different hyperparameters.

    Parameters:
        data: Dataset to be split into training and test sets
        lr: Learning rate for the optimizer
        wd: Weight decay for the optimizer
        seed: Random seed for reproducibility
        results: DataFrame to store results

    Returns:
        results: Updated DataFrame with new results
    """
    # Split data into training and testing folds
    for i, (train_idx, test_idx) in enumerate(split_data(data, seed)):
        train_loader, val_loader, test_loader = get_dataloaders(data, train_idx, test_idx, seed)
        model, loss_fn = initialise_model(i, nas) # initialise the model according to prediction type and training procedure
        model = train_model(train_loader, val_loader, loss_fn, lr, wd, model)  # Train the model
        loss, auroc, bacc = eval_model(model, test_loader, loss_fn)  # Evaluate the model
        new_row = {'LR': lr, 'WD': wd, 'Z_img': nas['img_fts'], 'Z_tab': nas['tab_fts'], 'Fold': i, 'Loss': loss, 'AUROC': auroc, 'BAcc': bacc, 'model': model}
        new_row = pd.DataFrame([new_row]) # Convert the new_row into a DataFrame
        # Concatenate the new_row_df to the existing results DataFrame
        results = pd.concat([results, new_row], ignore_index=True)
        print("*** RUN {} ***\n   *Test Loss={:.5f}\n   *AUC={:.5f}\n   *BAcc={:.5f}".format(i, loss, auroc, bacc))
        # to avoid cuda out of memory error
        model.cpu()
        del model
        torch.cuda.empty_cache()
        
    return results