import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None, target_transform=None):
        self.img_labels = labels
        self.img_dataset = images
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = self.img_dataset[idx]
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
        
class CustomDataloader():
    # def __init__(self, images, labels, normalization_stats, batch_size, split = True):
    def __init__(self, images, labels, batch_size, split = True):
        self._data = images
        self._label = labels
        # self._normalization_stats = normalization_stats
        # # Feature engineering, such as logarithmic transformations of characteristic parameters.
        # self.preprocessor()
        # # Normalization of model inputs
        # self._data = (self._data - self._normalization_stats.mean[:self._data.shape[1]])/self._normalization_stats.std[:self._data.shape[1]]
        if(split):
            #In training phase, the dataset needs to be randomly split into training set and validation set.
            self.splitting_datasets()
            # train_dataset = DataList(self._training_set)
            train_dataset = CustomDataset(self._training_set,self._training_label)
            self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
            # test_dataset = DataList(self._valid_set)
            test_dataset = CustomDataset(self._valid_set,self._valid_label)
            self.test_loader = DataLoader(test_dataset, batch_size=batch_size*100, shuffle=False, pin_memory=True)
        else:
            # In testing phase, the test set does not need to be split.
            # test_dataset = DataList(self._data)
            test_dataset = CustomDataset(self._data,self._label)
            self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    def splitting_datasets(self):
        """Splitting the dataset."""
        index = np.random.rand(self._data.shape[0]) < 0.63
        self._training_set = self._data[index]
        self._training_label = self._label[index]
        self._valid_set = self._data[~index]
        self._valid_label = self._label[~index]