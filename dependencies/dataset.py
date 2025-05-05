import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
from torchvision import transforms
import sys

class BinaryDataset(Dataset):
    def __init__(self, info, data_dir, out_size):
        """
        Initializes the BinaryDataset class.

        Parameters:
        info (pd.DataFrame): A dataframe containing all the relevant data. Columns include:
                             - 'id' (unique identifier for each sample)
                             - 'A', 'B', 'C' (tabular features)
                             - 'circle', 'rectangle', 'triangle' (image-related labels)
                             - 'y' (target label)
        """
        # Store the img_dir according to server
        self.img_dir = data_dir + 'imgs'
        
        # Store the 'id' column as an array
        self.ids = np.asarray(info['id'])
        
        # Store the tabular data, excluding specific columns ('A', 'B', 'C', shape labels, and 'y')
        self.tab = np.asarray(info.drop(columns=['A','B','C','circle','rectangle','triangle','id','y']))
        
        # Store the target labels for tabular data (columns 'A', 'B', 'C')
        self.y_tab = np.asarray(info[['A','B','C']])
        
        # Store the target labels related to images (columns 'circle', 'rectangle', 'triangle')
        self.y_img = np.asarray(info[['circle','rectangle','triangle']])
        
        # Store the primary target label (column 'y')
        self.y = np.asarray(info['y'])
        
        # Determine if the task is multiclass classification or not
        self.out_size = out_size
        self.multiclass = self._is_multiclass()

    def _is_multiclass(self):
        """
        Determines if the target labels ('y') represent a multiclass classification problem.
        Multiclass is assumed if there are more than 2 and fewer than 5 unique labels.

        Returns:
        bool: True if it's a multiclass task, False otherwise.
        """
        multiclass = False
        # Find the unique values in the target labels
        unique_labels = np.unique(self.y)
        
        # If there are between 2 and 5 unique labels, it's a multiclass task
        #if len(unique_labels) > 2 and len(unique_labels) < 5: 
        #    multiclass = True
        if self.out_size > 1: multiclass = True
        
        return multiclass

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
        int: The length of the dataset (number of samples).
        """
        return len(self.y)

    def __getitem__(self, index):
        """
        Retrieves a single data sample, including the image, tabular data, and labels, by index.

        Parameters:
        index (int): Index of the sample to retrieve.

        Returns:
        tuple: Contains the following elements:
               - img (torch.Tensor): Normalized image data loaded from file.
               - tab (np.ndarray): Tabular features.
               - y_img (np.ndarray): Image-related labels.
               - y_tab (np.ndarray): Tabular-related labels.
               - y (np.ndarray): Target label, one-hot encoded if multiclass.
        """
        # Define the directory where image data is stored
        #img_dir = '../dummy_data/and_or/imgs'
        
        # Construct the path to the image file based on the sample's id
        img_path = os.path.join(self.img_dir, str(self.ids[index]) + '.npy')
        
        # Load the image from a NumPy file
        img = np.load(img_path)
        
        # Normalize the image between 0 and 1
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        
        # Apply a series of transformations to the image (e.g., convert to Tensor)
        trans = transforms.Compose([transforms.ToTensor()])
        img = trans(img)
        
        # Get the tabular data for the sample
        tab = self.tab[index]
        
        # Get the image-related labels (circle, rectangle, triangle)
        y_img = self.y_img[index]
        
        # Get the tabular-related labels (A, B, C)
        y_tab = self.y_tab[index]
        
        # Get the target label
        y = self.y[index].reshape(-1)
        
        # If it's a multiclass classification task, one-hot encode the target label
        if self.multiclass:
            one_hot_y = np.zeros(4)  # Create an array of zeros for 4 classes
            one_hot_y[y] = 1         # Set the index corresponding to the label to 1
            y = one_hot_y.squeeze()

        # Return the image, tabular features, image-related labels, tabular labels, and the target label
        return img, tab, y_img, y_tab, y
