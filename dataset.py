import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode


class Mvtec(Dataset):
    def __init__(self, root_dir, object_type=None, split=None, defect_type=None, im_size=None, transform=None):
        
        if split == 'train':
            # defect_type = 'good'
            csv_name = '{}_train.csv'.format(object_type)
        else:
            csv_name = '{}_{}.csv'.format(object_type, defect_type)

        csv_file = os.path.join(root_dir, object_type, csv_name)
        # self.image_folder = os.path.join(root_dir, object_type, split, defect_type)
        self.data_frame = pd.read_csv(csv_file)
        self.image_dir = os.path.join(root_dir, object_type)
        if transform:
            self.transform = transform
        else:
            imagenet_mean = [0.485, 0.456, 0.406]
            imagenet_std = [0.229, 0.224, 0.225]
            self.im_size = (224, 224) if im_size is None else (im_size, im_size)
            normalize_tf = transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
            self.transform = transforms.Compose([transforms.Resize(tuple(self.im_size), interpolation=InterpolationMode.LANCZOS), transforms.ToTensor(), normalize_tf])
        self.num_classes = 1


    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.data_frame.iloc[idx, 0])
        image = Image.open(img_name)
        if image.mode == 'L':
            image = image.convert('RGB')
        image = self.transform(image)
        labels = self.data_frame.iloc[idx, 1]
        sample = {'data': image, 'label': labels}

        return sample

    def getclasses(self):
        classes = [str(i) for i in range(self.num_classes)]
        c = dict()
        for i in range(len(classes)):
            c[i] = classes[i]
        return c