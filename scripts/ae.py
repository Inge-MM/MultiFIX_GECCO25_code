from config_file import *
import copy

def train_ae(loader):
    DEVICE = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
    ae = Autoencoder().to(DEVICE)
    num_epochs = 100
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(ae.parameters(), lr=0.0001)

    for epoch in range(num_epochs):
        total_loss = 0.0

        for i, (img, _, _, _, _) in enumerate(loader):
            x = img.to(DEVICE, dtype=torch.float)
            out = ae(x)
            loss = criterion(out, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        average_loss = total_loss / len(loader)
        print("___epoch ", epoch, ": MSE = ", average_loss)
    model_wts = copy.deepcopy(ae.encoder.state_dict())

    return model_wts

def get_encoder(data, seed):
    """
    Train and save encoder weights for image feature enineering blocks

    Parameters:
        data: Dataset to be split into training and test sets
        seed: Random seed for reproducibility

    Returns:
        results: Updated DataFrame with new results
    """
    save_path = './models/img_encoders/'
    
    # Split data into training and testing folds
    for i, (train_idx, test_idx) in enumerate(split_data(data, seed)):
        print("Training Fold #", str(i))
        train_loader, val_loader, test_loader = get_dataloaders(data, train_idx, test_idx, seed)
        encoder_wts = train_ae(train_loader)  # Train the model
        filename = 'img_encoder_' + str(i) + '.pth'
        torch.save(encoder_wts, save_path + filename)
        print("-> saved model #", str(i))
        
################# METHODS ARE OVER #################

# Here it doesn't matter the label since it's unsupervised learning
labels = pd.read_csv(DATA_DIR + 'fusion_labels.csv')
# Load tabular data and remove unnecessary columns for model input
tab = pd.read_csv(DATA_DIR + 'tab_data.csv')
tab.drop(columns=['A', 'B', 'C', 'id', 'Feature14', 'Feature13', 'Feature12', 'Feature11', 'Feature10'], inplace=True)
# Combine the tabular data and labels into a single DataFrame for training
data = pd.concat([tab, labels], axis=1)

set_seed(SEED) # Set the random seed for reproducibility
print("Starting autoencoder training")
get_encoder(data, SEED)
