import numpy as np
import torch
from torch.utils.data import Dataset
import os.path as osp
import h5py

class PreDataset(Dataset):
    def __init__(self, data_path, data_split='train', dataname=None, flag = 'sup'):
        self.root = data_path
        if dataname == 'flickr25k':
            # self.dset = dataname + '_vgg_bow_split2'  # 数据集实际名称
            # self.dset = dataname + '_vgg_bow_glove_split'  # 数据集实际名称
            if flag == 'sup':
                self.dset = dataname + '_clip_split_sup'
            else:
                self.dset = dataname + '_clip_split'  # 数据集实际名称
        elif dataname == 'nuswide':
            # self.dset = dataname + '_vgg_bow_clean_split'  # 数据集实际名称
            # self.dset = dataname + '_vgg_bow_glove_clean_split'  # 数据集实际名称
            # self.dset = dataname + '_clip_split'  # 数据集实际名称
            if flag == 'sup':
                self.dset = dataname + '_clip_split_sup'
            else:
                self.dset = dataname + '_clip_split'  # 数据集实际名称
        elif dataname == 'coco':
            if flag == 'sup':
                self.dset = dataname + '_clip_split_sup'
            else:
                self.dset = dataname + '_clip_split'  # 数据集实际名称

        # print(self.dset)
        self.data = h5py.File(osp.join(self.root, dataname, '{}.mat'.format(self.dset)))
        if data_split == 'train':
            self.images = self.data['I_tr'][:].T
            self.texts = self.data['T_tr'][:].T
            # self.texts = self.data['glove_mean_tr'][:].T
            self.labels = self.data['L_tr'][:].T
        if data_split == 'test':
            self.images = self.data['I_te'][:].T
            self.texts = self.data['T_te'][:].T
            # self.texts = self.data['glove_mean_te'][:].T
            self.labels = self.data['L_te'][:].T
        if data_split == 'retrieval':
            self.images = self.data['I_db'][:].T
            self.texts = self.data['T_db'][:].T
            # self.texts = self.data['glove_mean_db'][:].T
            self.labels = self.data['L_db'][:].T
        if data_split == 'all':
            # print(self.data['I_db'][:].T.shape)
            # print(self.data['I_te'][:].T.shape)
            self.images = np.concatenate((self.data['I_db'][:].T, self.data['I_te'][:].T), axis=0)
            self.texts = np.concatenate((self.data['T_db'][:].T, self.data['T_te'][:].T), axis=0)
            self.labels = np.concatenate((self.data['L_db'][:].T, self.data['L_te'][:].T), axis=0)


        self.length = len(self.labels)

    def __getitem__(self, index):

        img = torch.as_tensor(self.images[index], dtype=torch.float32)
        text = torch.as_tensor(self.texts[index], dtype=torch.float32)
        label = torch.as_tensor(self.labels[index], dtype=torch.float32)
        return img, text, label, index

    def __len__(self):
        return self.length





if __name__ == '__main__':
    pass
