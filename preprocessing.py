import pickle
import h5py
import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
import torchvision.models.resnet as models
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class CustomedDataset(Dataset):
    def __init__(self, dataset, img_dir, file_paths, transform=None):
        self.dataset = dataset
        self.matcontent = sio.loadmat(file_paths)
        self.image_files = np.squeeze(self.matcontent['image_files'])
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx][0]
        if self.dataset == 'CUB':
            split_idx = 6 
        elif self.dataset == 'SUN':
            split_idx = 7
        elif self.dataset == 'AWA2':
            split_idx = 5
        image_file = os.path.join(self.img_dir,
                                    '/'.join(image_file.split('/')[split_idx:]))
        image = Image.open(image_file)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


def extract_features(config):

    img_dir = f'data/{config.dataset}'
    file_paths = f'data/xlsa17/data/{config.dataset}/res101.mat'
    save_path = f'data/{config.dataset}/feature_map_ResNet_101_{config.dataset}.hdf5'
    attribute_path = f'w2v/{config.dataset}_attribute.pkl'

    # region feature extractor
    resnet101 = models.resnet101(pretrained=True).to(config.device)
    resnet101 = nn.Sequential(*list(resnet101.children())[:-2]).eval()

    data_transforms = transforms.Compose([
        transforms.Resize(448),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    Dataset = CustomedDataset(config.dataset, img_dir, file_paths, data_transforms)
    dataset_loader = torch.utils.data.DataLoader(Dataset,
                                                 batch_size=config.batch_size,
                                                 shuffle=False,
                                                 num_workers=config.nun_workers)

    with torch.no_grad():
        all_features = []
        for _, imgs in enumerate(dataset_loader):
            imgs = imgs.to(config.device)
            features = resnet101(imgs)
            all_features.append(features.cpu().numpy())
        all_features = np.concatenate(all_features, axis=0)

    # get remaining metadata
    matcontent = Dataset.matcontent
    labels = matcontent['labels'].astype(int).squeeze() - 1

    split_path = os.path.join(f'data/xlsa17/data/{config.dataset}/att_splits.mat')
    matcontent = sio.loadmat(split_path)
    trainval_loc = matcontent['trainval_loc'].squeeze() - 1
    # train_loc = matcontent['train_loc'].squeeze() - 1
    # val_unseen_loc = matcontent['val_loc'].squeeze() - 1
    test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
    test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1
    att = matcontent['att'].T
    original_att = matcontent['original_att'].T

    # construct attribute w2v
    with open(attribute_path,'rb') as f:
        w2v_att = pickle.load(f)
    if config.dataset == 'CUB':
        assert w2v_att.shape == (312,300)
    elif config.dataset == 'SUN':
        assert w2v_att.shape == (102,300)
    elif config.dataset == 'AWA2':
        assert w2v_att.shape == (85,300)

    compression = 'gzip' if config.compression else None 
    f = h5py.File(save_path, 'w')
    f.create_dataset('feature_map', data=all_features,compression=compression)
    f.create_dataset('labels', data=labels,compression=compression)
    f.create_dataset('trainval_loc', data=trainval_loc,compression=compression)
    # f.create_dataset('train_loc', data=train_loc,compression=compression)
    # f.create_dataset('val_unseen_loc', data=val_unseen_loc,compression=compression)
    f.create_dataset('test_seen_loc', data=test_seen_loc,compression=compression)
    f.create_dataset('test_unseen_loc', data=test_unseen_loc,compression=compression)
    f.create_dataset('att', data=att,compression=compression)
    f.create_dataset('original_att', data=original_att,compression=compression)
    f.create_dataset('w2v_att', data=w2v_att,compression=compression)
    f.close()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='AWA2')
    parser.add_argument('--compression', '-c', action='store_true', default=False)
    parser.add_argument('--batch_size', '-b', type=int, default=200)
    parser.add_argument('--device', '-g', type=str, default='cuda:0')
    parser.add_argument('--nun_workers', '-n', type=int, default='16')
    config = parser.parse_args()
    extract_features(config)
