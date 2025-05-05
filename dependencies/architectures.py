import torch
import torch.nn as nn
from torchvision import models

# Define the image processing network using ResNet18 as the backbone
class img_net(nn.Module):
    def __init__(self, img_fts=1):
        super(img_net, self).__init__()
        
        # Load pre-trained ResNet18 model and remove its fully connected layer
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Replace the final fully connected layer with a custom head (512 -> 128) followed by ReLU
        self.resnet.fc = nn.Sequential(
                nn.Linear(512, 128), # 128 latent representation
                nn.ReLU())
        
        # Classifier to output the desired number of features (img_fts)
        self.classifier = nn.Sequential(
            nn.Linear(128, img_fts), # Reduce the dimension from 128 to the number of image features
            nn.Sigmoid())  # Apply sigmoid to normalize the output to [0,1] range
        
    # Forward pass through the image processing network
    def forward(self, x, tab):
        x = self.resnet(x)  # Pass image through ResNet18 backbone
        x = self.classifier(x)  # Pass the features through the classifier
        return x

# Define the tabular data processing network
class tab_net(nn.Module):
    def __init__(self, tab_fts=1):
        super(tab_net, self).__init__()
        
        # Define the architecture for processing tabular data
        w = 128  # Intermediate width for layers
        d = 0.125  # Dropout rate

        # Feature extractor for tabular data
        self.fts = nn.Sequential(
            nn.Linear(10, w),  # Initial input size is 15 (features in tabular data)
            nn.BatchNorm1d(w), # Batch normalization for stable training
            nn.ReLU(),         # ReLU activation
            nn.Dropout(d),     # Dropout to prevent overfitting
            nn.Linear(w, w),   # Second layer
            nn.BatchNorm1d(w), 
            nn.ReLU(),
            nn.Linear(w, 128), # 128 latent representation
            nn.ReLU())
        
        # Classifier for tabular features, output size matches tab_fts
        self.classifier = nn.Sequential(
            nn.Linear(128, tab_fts),  # Reduce to desired number of tabular features
            nn.Sigmoid())  # Sigmoid activation for binary classification
        
    # Forward pass through the tabular data processing network
    def forward(self, img, x):
        x = self.fts(x)  # Pass tabular data through feature extractor
        x = self.classifier(x)  # Classify the extracted features
        return x

# Define the fusion network to combine both image and tabular data features
class fusion_net(nn.Module):
    def __init__(self, img_fts, tab_fts, out_size=1):
        super().__init__()
        
        self.z_img = img_fts
        self.z_tab = tab_fts
        fts_nr = self.z_img + self.z_tab
        w = 128  # Intermediate layer size
        d = 0.125  # Dropout rate

        # Fusion layers to combine image and tabular features
        self.fusion = nn.Sequential(
            nn.Linear(fts_nr, w),  # Input is the concatenation of image and tabular features
            nn.BatchNorm1d(w),     # Batch normalization
            nn.ReLU(),
            nn.Dropout(d),         # Dropout for regularization
            nn.Linear(w, w),       # Second fusion layer
            nn.BatchNorm1d(w),
            nn.ReLU(),
            nn.Linear(w, 128),     # 128 latent representation
            nn.ReLU())
        
        # Classifier to produce the final output (binary classification)
        if out_size == 1: # binary classification
            self.classifier = nn.Sequential(
                nn.Linear(128, 1),  # Output a single value for binary classification
                nn.Sigmoid())  # Sigmoid to output a probability (0 to 1)
        elif out_size > 1: # multiclass
            self.classifier = nn.Linear(128,out_size)
        else: # regression
            self.classifier = nn.Linear(128,1)
        
    # Forward pass through the fusion network
    def forward(self, a, b):
        if self.z_img != 0:
            if self.z_tab != 0:
                x = torch.cat((a, b), dim=1)  # Concatenate image and tabular features along feature axis
            else: x = a
        else: x = b
        x = self.fusion(x)  # Pass through fusion layers
        x = self.classifier(x)  # Final classification layer
        return x

# Define the main multifix network that integrates image, tabular, and fusion networks
class multifix_net(nn.Module):
    def __init__(self, nas_config, output_size):
        super().__init__()

        """ Image Processing Network """
        self.img_fts = nas_config['img_fts']  # Get the number of image features from the config
        if self.img_fts != 0: self.img_block = img_net(self.img_fts)  # Initialize the image network
        else: self.img_block = None

        """ Tabular Data Processing Network """
        self.tab_fts = nas_config['tab_fts']  # Get the number of tabular features from the config
        if self.tab_fts != 0: self.tab_block = tab_net(self.tab_fts)  # Initialize the tabular network
        else: self.tab_block = None
        
        """ Fusion Network """
        # Combine image and tabular features in the fusion network   
        self.fusion_block = fusion_net(self.img_fts, self.tab_fts, output_size)

    # Forward pass through the entire multifix network
    def forward(self, img, tab):
        if self.img_fts != 0: img = self.img_block(img, tab)  # Pass image data through image network
        if self.tab_fts != 0: tab = self.tab_block(img, tab)  # Pass tabular data through tabular network

        # Combine image and tabular features and pass through fusion network
        x = self.fusion_block(img, tab)

        return x
    
########### AUTOENCODER ###########

class ResBlock(nn.Module):
    """
    A two-convolutional layer residual block.
    """
    
    def __init__(self, c_in, c_out, k, s=1, p=1, mode='encode'):
        assert mode in ['encode', 'decode'], "Mode must be either 'encode' or 'decode'."
        super(ResBlock, self).__init__()
        if mode == 'encode':
            self.conv1 = nn.Conv2d(c_in, c_out, k, s, p)
            self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1)
        elif mode == 'decode':
            self.conv1 = nn.ConvTranspose2d(c_in, c_out, k, s, p)
            self.conv2 = nn.ConvTranspose2d(c_out, c_out, 3, 1, 1)
        self.relu = nn.ReLU()
        self.BN = nn.BatchNorm2d(c_out)
        self.resize = s > 1 or (s == 1 and p == 0) or c_out != c_in
    
    def forward(self, x):
        conv1 = self.BN(self.conv1(x))
        relu = self.relu(conv1)
        conv2 = self.BN(self.conv2(relu))
        if self.resize:
            x = self.BN(self.conv1(x))
        return self.relu(x + conv2)
    
class Decoder(nn.Module):
    """
    Decoder class, mainly consisting of two residual blocks.
    """
    
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(128,512),
            nn.ReLU(),
            nn.Linear(512,512*7*7),
            nn.ReLU(),
            nn.Unflatten(1, torch.Size([512,7,7]))
        )
        # 512x7x7 -> 3x200x200
        self.rb1 = ResBlock(512, 256, 3, 2, 0, 'decode')
        self.rb2 = ResBlock(256, 128, 3, 2, 0, 'decode')
        self.rb3 = ResBlock(128, 64, 5, 2, 0, 'decode')
        self.rb4 = ResBlock(64, 32, 5, 3, 0, 'decode')
        self.out_conv = nn.ConvTranspose2d(32, 3, 4, 1, 0)
        self.sig = nn.Sigmoid()
        self.deconv = nn.Sequential(
            self.rb1,
            self.rb2,
            self.rb3,
            self.rb4,
            self.out_conv,
            self.sig
        )
        
    def forward(self, x):
        x = self.fc(x)
        x = self.deconv(x)
        return x
    
class Autoencoder(nn.Module):
    """
    Autoencoder class, combines encoder and decoder model.
    """
    
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = img_net().resnet
        self.decoder = Decoder()
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

##### SINGLE MODALITY #####

# Define the image processing network using ResNet18 as the backbone
class single_img_net(nn.Module):
    def __init__(self, img_fts=1):
        super(single_img_net, self).__init__()
        
        # Load pre-trained ResNet18 model and remove its fully connected layer
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Replace the final fully connected layer with a custom head (512 -> 128) followed by ReLU
        self.resnet.fc = nn.Sequential(
                nn.Linear(512, 128), # 128 latent representation
                nn.ReLU())
        
        # Classifier to produce the final output (binary classification)
        if img_fts == 1: # binary classification
            self.classifier = nn.Sequential(
                nn.Linear(128, 1),  # Output a single value for binary classification
                nn.Sigmoid())  # Sigmoid to output a probability (0 to 1)
        elif img_fts > 1: # multiclass
            self.classifier = nn.Linear(128,img_fts)
        else: # regression
            self.classifier = nn.Linear(128,1)
        
    # Forward pass through the image processing network
    def forward(self, x, tab):
        x = self.resnet(x)  # Pass image through ResNet18 backbone
        x = self.classifier(x)  # Pass the features through the classifier
        return x

# Define the tabular data processing network
class single_tab_net(nn.Module):
    def __init__(self, tab_fts=1):
        super(single_tab_net, self).__init__()
        
        # Define the architecture for processing tabular data
        w = 128  # Intermediate width for layers
        d = 0.125  # Dropout rate

        # Feature extractor for tabular data
        self.fts = nn.Sequential(
            nn.Linear(7, w),  # Initial input size is 15 (features in tabular data)
            nn.BatchNorm1d(w), # Batch normalization for stable training
            nn.ReLU(),         # ReLU activation
            nn.Dropout(d),     # Dropout to prevent overfitting
            nn.Linear(w, w),   # Second layer
            nn.BatchNorm1d(w), 
            nn.ReLU(),
            nn.Linear(w, 128), # 128 latent representation
            nn.ReLU())
        
        # Classifier to produce the final output (binary classification)
        if tab_fts == 1: # binary classification
            self.classifier = nn.Sequential(
                nn.Linear(128, 1),  # Output a single value for binary classification
                nn.Sigmoid())  # Sigmoid to output a probability (0 to 1)
        elif tab_fts > 1: # multiclass
            self.classifier = nn.Linear(128,tab_fts)
        else: # regression
            self.classifier = nn.Linear(128,1)
        
    # Forward pass through the tabular data processing network
    def forward(self, img, x):
        x = self.fts(x)  # Pass tabular data through feature extractor
        x = self.classifier(x)  # Classify the extracted features
        return x