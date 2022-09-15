
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):

    def __init__(self, feature_path, label_path):
        super(MyDataset, self).__init__()
        self.feature_paths = glob.glob(os.path.join(feature_path, '*.npy'))
        self.label_paths = glob.glob(os.path.join(label_path, '*.npy'))


    def __len__(self):
        return len(self.feature_paths)


    def __getitem__(self, index):
        feature_data = np.load(self.feature_paths[index])
        label_data = np.load(self.label_paths[index])
        feature_data = torch.from_numpy(feature_data) 
        label_data = torch.from_numpy(label_data)
        feature_data.unsqueeze_(0)  
        label_data.unsqueeze_(0)
        return feature_data, label_data


if __name__ == "__main__":

    feature_path = "D:\\pycharm\\py\\myseis\\unet_inter\\data\\feature\\"
    label_path = "D:\\pycharm\\py\\myseis\\unet_inter\\data\\label\\"
    # feature_path = 'D:\\Seismic\\f_l\\feature\\'
    # label_path = "D:\\Seismic\\f_l\\label\\"
    seismic_dataset = MyDataset(feature_path, label_path)
    train_loader = torch.utils.data.DataLoader(dataset=seismic_dataset,
                                               batch_size=32,
                                               shuffle=False)
    print('Dataset size:', len(seismic_dataset))
    print('train_loader:', len(train_loader))
