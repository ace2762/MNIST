from torchvision import datasets
from torchvision import transforms
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import numpy as np
import gzip
import torch
import matplotlib.pyplot as plt


class DealDataset(Dataset):
    def __init__(self, data_folder, data_name, label_name, transform=None):
        with gzip.open(os.path.join(data_folder, label_name), 'rb') as lb_path:                   # rb表示的是读取二进制数据
            y_train = np.frombuffer(lb_path.read(), dtype=np.uint8, offset=8)                     # offset8，读取标签label
        with gzip.open(os.path.join(data_folder, data_name), 'rb') as img_path:
            x_train = np.frombuffer(
                img_path.read(), dtype=np.uint8, offset=16)\
                .reshape(len(y_train), 28, 28)                # offset16，读取数据image, 图片像素28*28，unsigned byte即uint8
        self.train_data = x_train
        self.train_target = y_train
        self.transform = transform

    def __getitem__(self, idx):
        data, target = self.train_data[idx], self.train_target[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data, target

    def __len__(self):
        return len(self.train_target)


if __name__ == '__main__':
    trainDataset = DealDataset('./mnist_dataset', "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
                               transform=transforms.ToTensor())

    train_loader = DataLoader(
        dataset=trainDataset,
        batch_size=10,
        shuffle=False,
    )

    images, labels = next(iter(train_loader))                      # dataloader是可迭代对象，iter(dxxr)返回迭代器,可用next访问
    img = torchvision.utils.make_grid(images, nrow=5)                                             # 多个图拼在一起，生成网格
    img = img.numpy().transpose(1, 2, 0)                                              # image参数顺序(size,size,channels)
    '''
    std = [0.5, 0.5, 0.5]
    mean = [0.5, 0.5, 0.5]
    img = img * std + mean
    '''
    print(labels)
    plt.imshow(img)                                                  # 该方法参数为(channels,size,size)，所以之前要transpose
    plt.show()
