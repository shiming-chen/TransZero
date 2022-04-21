import os
import pickle
from torchvision import transforms
from torch.utils.data import Dataset, Subset, DataLoader
from PIL import Image


class BaseDataset(Dataset):
    def __init__(self, dataset_path, image_files, labels, transform=None):
        super(BaseDataset, self).__init__()
        self.dataset_path = dataset_path
        self.image_files = image_files
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        label = self.labels[idx]
        image_file = self.image_files[idx]
        image_file = os.path.join(self.dataset_path, image_file)
        image = Image.open(image_file)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


class UNIDataloader():
    def __init__(self, config):
        self.config = config
        with open(config.pkl_path, 'rb') as f:
            self.info = pickle.load(f)

        self.seenclasses = self.info['seenclasses'].to(config.device)
        self.unseenclasses = self.info['unseenclasses'].to(config.device)

        (self.train_set,
         self.test_seen_set,
         self.test_unseen_set) = self.torch_dataset()

        self.train_loader = DataLoader(self.train_set,
                                       batch_size=config.batch_size,
                                       shuffle=True,
                                       num_workers=config.num_workers)
        self.test_seen_loader = DataLoader(self.test_seen_set,
                                           batch_size=config.batch_size,
                                           shuffle=False,
                                           num_workers=config.num_workers)
        self.test_unseen_loader = DataLoader(self.test_unseen_set,
                                             batch_size=config.batch_size,
                                             shuffle=False,
                                             num_workers=config.num_workers)

    def torch_dataset(self):
        data_transforms = transforms.Compose([
            transforms.Resize(self.config.img_size),
            transforms.CenterCrop(self.config.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        baseset = BaseDataset(self.config.dataset_path,
                              self.info['image_files'],
                              self.info['labels'],
                              data_transforms)

        train_set = Subset(baseset, self.info['trainval_loc'])
        test_seen_set = Subset(baseset, self.info['test_seen_loc'])
        test_unseen_set = Subset(baseset, self.info['test_unseen_loc'])

        return train_set, test_seen_set, test_unseen_set
