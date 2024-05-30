from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os


def load_data(root_path, dir, batch_size):
    """加载数据集"""
    transform = transforms.Compose(
        [transforms.Resize([224, 224]),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
         ])  # 图像预处理
    data = datasets.ImageFolder(root=os.path.join(root_path, dir), transform=transform)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4)
    return data, data_loader


def load_train(root_path, dir, batch_size):
    """加载训练集和验证集"""
    transform_dict = {
        'train': transforms.Compose(
            [transforms.RandomResizedCrop(224),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
             ]),
        'val': transforms.Compose(
            [transforms.Resize([224, 224]),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
             ])}  # 图像预处理
    data = datasets.ImageFolder(root=os.path.join(root_path, dir), transform=transform_dict['train'])
    # data = datasets.ImageFolder(root=os.path.join(root_path, dir))
    # train_size = int(0.8 * len(data))  # 80%训练集
    # val_size = len(data) - train_size  # 20%验证集
    # data_train, data_val = random_split(data, [train_size, val_size])  # 随机划分训练集和验证集
    # data_train.dataset.transform = transform_dict['train']
    # data_val.dataset.transform = transform_dict['val']
    # train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4)
    # val_loader = DataLoader(data_val, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4)
    train_loader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4)
    # return train_loader, val_loader
    return train_loader
