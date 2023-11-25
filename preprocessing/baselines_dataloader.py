import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name())
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.data import Subset, DataLoader
from torch.utils.data import Dataset
import os
from PIL import Image
import pandas as pd
import numpy as np
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import random


# help function for downscaling of images
def resize_image(image, scale_percent):
    target_width = int(image.size[0] * scale_percent / 100) 
    target_height = int(image.size[1] * scale_percent / 100)
    target_dim = (target_height, target_width)
    resized_image = transforms.Resize(size=target_dim)(image)
    return resized_image

class ChestXrayDataset(Dataset):
    def __init__(self, csv_file, root_dir, scale_percent=100, transform=None, is_rgb=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            scale_percent (int): Percent of original size for scaling the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.data_frame = self.data_frame.head(3000)  # only keep the first 3000 pictures
        #self.data_frame = self.data_frame.sample(n=1000, random_state=1).reset_index(drop=True)# random get 2000 pictures
        #sample_size = min(2000, len(self.data_frame))  # Ensure sample size is not larger than the dataset
        #self.data_frame = self.data_frame.sample(n=sample_size, random_state=1).reset_index(drop=True)  # random sample 2000 pictures
        self.root_dir = root_dir
        self.scale_percent = scale_percent
        self.transform = transform
        #self.original_images = []  # 实例属性，用于存储原始图像
        self.is_rgb = is_rgb

        diseases = ["Cardiomegaly", "Pleural_Effusion", "Edema", "Atelectasis", "Consolidation"]
        labels = np.array([self.data_frame[disease].values if disease in self.data_frame.columns else np.zeros(len(self.data_frame)) for disease in diseases])
        labels[np.isnan(labels)] = 0
        labels[labels == -1] = 1
        self.targets = torch.from_numpy(labels).float().T
        print(labels)  # 在__init__方法中打印labels数组
        print(self.targets)  # 在__init__方法中打印self.targets张量

    def __len__(self):
        return len(self.data_frame)

    
    def __getitem__(self, idx):
        if isinstance(idx, list) or isinstance(idx, torch.Tensor):  # Check if idx is a list or tensor
            images = []
            targets = []
            for i in idx:
                img_name = os.path.join(self.root_dir, str(self.data_frame.at[i, 'Path'])) 
                original_image = Image.open(img_name).convert('RGB')
                #self.original_images.append(original_image)  # 更新实例属性
                if self.is_rgb:
                    image = original_image  # Keep as RGB
                else:
                    image = original_image.convert('L')  # Convert to grayscale
                image = self.transform(image)
                target = self.targets[i]
                #if torch.all(target == 0):
                #     # 这个样本的所有标签都是 0，您可以选择重新选择一个样本或处理这种情况
                #    pass
                images.append(image)
                targets.append(target)
            return torch.stack(images), torch.stack(targets)
        else:
            img_name = os.path.join(self.root_dir, str(self.data_frame.at[idx, 'Path'])) 
            img_path = self.data_frame.iloc[idx, 0] # for debug
            original_image = Image.open(img_name).convert('RGB')
            #self.original_images.append(original_image)  # 更新实例属性
            if self.is_rgb:
                    image = original_image  # Keep as RGB
            else:
                    image = original_image.convert('L')  # Convert to grayscale
            image = resize_image(image, self.scale_percent)

            if self.transform:
                image = self.transform(image)
        print(self.targets[idx])  # 在__getitem__方法中打印指定索引的标签数据
        return image, self.targets[idx]
        



def load_data(name, root='./data', download=True, save_pre_data=True):

    data_dict = ['MNIST', 'EMNIST', 'FashionMNIST', 'CelebA', 'CIFAR10', 'QMNIST', 'SVHN', "IMAGENET", 'CIFAR100', 'ChestXrays']
    assert name in data_dict, "The dataset is not present"

    if not os.path.exists(root):
        os.makedirs(root, exist_ok=True)

    if name == 'MNIST':
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
        trainset = torchvision.datasets.MNIST(root=root, train=True, download=download, transform=transform)
        testset = torchvision.datasets.MNIST(root=root, train=False, download=download, transform=transform)
    
    elif name == 'ChestXrays':
        # Set the paths to CSV files and image directories

        print("Loading ChestXrays dataset")
        # Define the transformations
        transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to fit the input size of the model
        transforms.ToTensor(),  # Convert images to PyTorch Tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize images
        ])

        # Get the current script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Build the CSV file relative paths
        csv_file_path_train = os.path.join(script_dir, '../../data/CheXpert/archive/train.csv')
        csv_file_path_test = os.path.join(script_dir, '../../data/CheXpert/archive/valid.csv')

        # Build the root directory path
        root_dir = os.path.join(script_dir, '../../data/')

        # Create the datasets
        trainset = ChestXrayDataset(csv_file=csv_file_path_train, root_dir=root_dir, transform=transform)
        testset = ChestXrayDataset(csv_file=csv_file_path_test, root_dir=root_dir, transform=transform)
        # 创建原始图像数据集
        original_set = ChestXrayDataset(csv_file=csv_file_path_train, root_dir=root_dir, transform=transform, is_rgb=True)

        print("here")
        # Now you can access the labels of all images in the trainset and testset
        print("trainset.image is",trainset[0])
        print("testset.image is",testset[0])
        #  loading first pic to check
        first_image, _ = trainset[0]  # first element is Tensor，second element is label
        print("first pic Tensor is ", first_image)
        print("orginal_set.image is",original_set[0])



    elif name == 'EMNIST':
        # byclass, bymerge, balanced, letters, digits, mnist
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
        trainset = torchvision.datasets.EMNIST(root=root, train=True, split= 'letters', download=download, transform=transform)
        testset = torchvision.datasets.EMNIST(root=root, train=False, split= 'letters', download=download, transform=transform)



    elif name == 'FashionMNIST':
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
        trainset = torchvision.datasets.FashionMNIST(root=root, train=True, download=download, transform=transform)
        testset = torchvision.datasets.FashionMNIST(root=root, train=False, download=download, transform=transform)

    elif name == 'CelebA':
        # Could not loaded possibly for google drive break downs, try again at week days
        target_transform = transforms.Compose([transforms.ToTensor()])
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CelebA(root=root, split='train', target_type=list, download=download, transform=transform, target_transform=target_transform)
        testset = torchvision.datasets.CelebA(root=root, split='test', target_type=list, download=download, transform=transform, target_transform=target_transform)

    elif name == 'CIFAR10':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])
        trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=download, transform=transform)
        testset = torchvision.datasets.CIFAR10(root=root, train=False, download=download, transform=transform)
        trainset.targets = torch.Tensor(trainset.targets)
        testset.targets = torch.Tensor(testset.targets)

    elif name == 'CIFAR100':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        trainset = torchvision.datasets.CIFAR100(root=root, train=True, transform=transform, download=True)
        testset = torchvision.datasets.CIFAR100(root=root, train=False, transform=transform, download=True)
        trainset.targets = torch.Tensor(trainset.targets)
        testset.targets = torch.Tensor(testset.targets)

    elif name == 'QMNIST':
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
        trainset = torchvision.datasets.QMNIST(root=root, what='train', compat=True, download=download, transform=transform)
        testset = torchvision.datasets.QMNIST(root=root, what='test', compat=True, download=download, transform=transform)

    elif name == 'SVHN':
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
        trainset = torchvision.datasets.SVHN(root=root, split='train', download=download, transform=transform)
        testset = torchvision.datasets.SVHN(root=root, split='test', download=download, transform=transform)
        trainset.targets = torch.Tensor(trainset.labels)
        testset.targets = torch.Tensor(testset.labels)


        
    '''
    elif name == 'IMAGENET':
        train_val_transform = transforms.Compose([
            transforms.ColorJitter(hue=.05, saturation=.05),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.ColorJitter(hue=.05, saturation=.05),
            transforms.ToTensor(),
        ])
        # transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                          std=[0.229, 0.224, 0.225])])
        trainset = torchvision.datasets.ImageFolder(root='./data/tiny-imagenet-200/train', transform=train_val_transform)
        testset = torchvision.datasets.ImageFolder(root='./data/tiny-imagenet-200/val', transform=test_transform)
        trainset.targets = torch.Tensor(trainset.targets)
        testset.targets = torch.Tensor(testset.targets)
    '''

    len_classes_dict = {
        'MNIST': 10,
        'EMNIST': 26, # ByClass: 62. ByMerge: 814,255 47.Digits: 280,000 10.Letters: 145,600 26.MNIST: 70,000 10.
        'FashionMNIST': 10,
        'CelebA': 0,
        'CIFAR10': 10,
        'QMNIST': 10,
        'SVHN': 10,
        'IMAGENET': 200,
        'CIFAR100': 100,
        'ChestXrays':5
    }

    len_classes = len_classes_dict[name]
    print("here it works")
    return trainset, testset, original_set, len_classes

def random_sample_from_global_data(trainset, num_samples=100):
    """
    Randomly sample data from the entire dataset.

    Args:
    - trainset (Dataset): The entire dataset.
    - num_samples (int): Number of samples to draw.

    Returns:
    - indices (list): List of randomly chosen indices.
    """
    all_indices = list(range(len(trainset)))
    sampled_indices = random.sample(all_indices, num_samples)
    return torch.tensor(sampled_indices)


def divide_data(num_client=1, num_local_class=10, dataset_name='ChestXrays', i_seed=0):

    torch.manual_seed(i_seed)

    trainset, testset, original_set, len_classes = load_data(dataset_name, download=True, save_pre_data=False)
    num_samples = int(0.01 * len(trainset))
    num_classes = len_classes

    
    if num_local_class == -1:
        num_local_class = num_classes
    assert 0 < num_local_class <= num_classes, "number of local class should smaller than global number of class"

    trainset_config = {'users': [],
                       'user_data': {},
                       'num_samples': []}
    config_division = {}  # Count of the classes for division
    config_class = {}  # Configuration of class distribution in clients
    config_data = {}  # Configuration of data indexes for each class : Config_data[cls] = [0, []] | pointer and indexes

    for i in range(num_client):
        config_class['f_{0:05d}'.format(i)] = []
        for j in range(num_local_class):
            cls = (i+j) % num_classes
            if cls not in config_division:
                config_division[cls] = 1
                config_data[cls] = [0, []]

            else:
                config_division[cls] += 1
            config_class['f_{0:05d}'.format(i)].append(cls)


    # Diagnostic print statements:
    print("Class distribution among clients:", config_class)
    print("Number of clients needing each class:", config_division)

    # print(config_class)
    # print(config_division)


    previous_partition_data = []


    for cls in config_division.keys():
        indexes = torch.nonzero(trainset.targets == cls)
        num_datapoint = indexes.shape[0]
        indexes = indexes[torch.randperm(num_datapoint)]
        num_partition = num_datapoint // config_division[cls]
        for i_partition in range(config_division[cls]):
            if i_partition == config_division[cls] - 1:
                partition = indexes[i_partition * num_partition:]
                #config_data[cls][1].append(indexes[i_partition * num_partition:])
            else:
                partition = indexes[i_partition * num_partition: (i_partition + 1) * num_partition]
                #config_data[cls][1].append(indexes[i_partition * num_partition: (i_partition + 1) * num_partition])
            
            config_data[cls][1].append(partition)
            previous_partition_data.append(partition)

    for user in tqdm(config_class.keys()):
        user_data_indexes = torch.tensor([])
        for cls in config_class[user]:

            if len(config_data[cls][1][config_data[cls][0]]) == 0:  # If the partition is empty
               # Take some data from the previous_partition_data list
                if previous_partition_data:  # Check if there's data left in previous_partition_data
                    extra_data = previous_partition_data.pop()  # Take and remove the last element
                    config_data[cls][1][config_data[cls][0]] = extra_data
                else:
                    print(f"Warning: No data left in previous_partition_data for user {user} and class {cls}.")


            user_data_index = config_data[cls][1][config_data[cls][0]]
            user_data_indexes = torch.cat((user_data_indexes, user_data_index))
            config_data[cls][0] += 1

        # If a client has no data, randomly sample from the global dataset
        if len(user_data_indexes) == 0:
            user_data_indexes = random_sample_from_global_data(trainset, num_samples)

        user_data_indexes = user_data_indexes.squeeze().int().tolist()
        user_data = Subset(trainset, user_data_indexes)
        #user_targets = trainset.target[user_data_indexes.tolist()]
        trainset_config['users'].append(user)
        trainset_config['user_data'][user] = user_data
        trainset_config['num_samples'] = len(user_data)

    # Diagnostic print statements:
    for user, user_data in trainset_config['user_data'].items():
        print(f"Client {user} has {len(user_data)} data points.")

    #
    # test_loader = DataLoader(trainset_config['user_data']['f_00001'])
    # for i, (x,y) in enumerate(test_loader):
    #     print(i)
    #     print(y)

    total_data_points = 0
    for client_id, data in trainset_config['user_data'].items():
        num_data_points = len(data)
        total_data_points += num_data_points
        print(f"Client {client_id} has {num_data_points} data points.")
    print(f"Total data points distributed: {total_data_points}")



    return trainset_config, testset


if __name__ == "__main__":
    # 'MNIST', 'EMNIST', 'FashionMNIST', 'CelebA', 'CIFAR10', 'QMNIST', 'SVHN'
    # data_dict = ['MNIST', 'EMNIST', 'FashionMNIST', 'CIFAR10', 'QMNIST', 'SVHN']
    data_dict = ['ChestXrays']
    
    for name in data_dict:
        print(name)
        divide_data(num_client=20, num_local_class=2, dataset_name=name, i_seed=0)

        
if torch.cuda.is_available():
    # 打印 CUDA 设备数量
    num_cuda_devices = torch.cuda.device_count()
    print(f"可用的 CUDA 设备数量：{num_cuda_devices}")

    # 打印每个 CUDA 设备的信息
    for i in range(num_cuda_devices):
        device = torch.device(f"cuda:{i}")
        print(f"CUDA 设备 {i} 的信息：")
        print(torch.cuda.get_device_properties(device))
else:
    print("没有可用的 CUDA 设备")