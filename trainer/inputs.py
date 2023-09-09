# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the \"License\");
# you may not use this file except in compliance with the License.\n",
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an \"AS IS\" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import pickle
from google.cloud import storage
import numpy as np
from torchvision import transforms
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data.sampler import SubsetRandomSampler

from trainer import metadata


class waferMap(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.X[idx]
        y = self.Y[idx]

        if self.transform:
            x = self.transform(x)
        return x, y


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class BinaryNoiseGeneration:
    """Randomly generate noise"""

    def __init__(self, p=0.1):
        assert isinstance(p, float)
        self.p = p

    def __call__(self, img):
        assert isinstance(img, np.ndarray)
        # img = img.astype(bool)
        # print(a)
        img_c = img.copy()
        mask = np.random.rand(*img_c.shape) < self.p
        img_c[np.where((img_c == 2) & (mask))] = 1
        img_c[np.where((img_c == 1) & (mask))] = 2
        # out = np.multiply(img, where = mask).astype(np.uint8)
        return img_c


def simsiam_transform():
    return TwoCropsTransform(
        transforms.Compose(
            [
                BinaryNoiseGeneration(p=0.05),
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.RandomRotation(degrees=(0, 360)),
                transforms.ToTensor(),
            ]
        )
    )


def base_transform():
    return [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]


class waferMapFromPickle(waferMap):
    def __init__(self, args, pickle_file, transform=simsiam_transform()):
        file = open(pickle_file, "rb")
        data = pickle.load(file)
        # print(data.head(10))
        # data = data.tail(10)
        data = data[data.failureType != "none"]
        print(data.head(10))
        if args.job_type == "train":
            X = data[data.trainTestLabel == "Training"].waferMap.tolist()
            y = data[data.trainTestLabel == "Training"].failureType.tolist()
        super(waferMapFromPickle, self).__init__(X=X, Y=y, transform=transform)


class CSVDataset(Dataset):
    def __init__(self, args, csv_files, device, transform=None):
        """
        Args:
            args: arguments passed to the python script
            csv_files (list): Path to the list of csv files with annotations.
            device (string): PyTorch device on which to load the dataset.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataframe = None
        for csv_file in csv_files:
            if self.dataframe is None:
                self.dataframe = pd.read_csv(csv_file, header=0)
            else:
                self.dataframe = pd.concat(
                    [self.dataframe, pd.read_csv(csv_file, header=0)]
                )
        self.device = device
        self.transform = transform

        # Convert the categorical columns in place to a numerical category
        # Example: Payment_Type =
        #       ['Credit Card' 'Cash' 'No Charge' 'Dispute' 'Unknown']
        # Converted: Payment_Type = [0, 1, 2, 3, 4]
        if args.embed_categorical_columns:
            for category in metadata.CATEGORICAL_COLUMNS:
                self.dataframe[category].replace(
                    {val: i for i, val in enumerate(self.dataframe[category].unique())},
                    inplace=True,
                )

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # When retrieving an item from the dataset, get the features and the
        # target. In this template, the target is 0th column and the features
        # are all the other columns.
        features = self.dataframe.iloc[idx, 1:].values
        target = self.dataframe.iloc[idx, :1].values

        # Load the data as a tensor
        item = {
            "features": torch.from_numpy(features).to(self.device),
            "target": torch.from_numpy(target).to(self.device),
        }

        if self.transform:
            item = self.transform(item)

        return item


def load_train_data(args, device):
    train_dataset = waferMapFromPickle(args=args, pickle_file=args.train_files[0])
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=args.shuffle_train
    )
    return train_loader


def load_data(args, device):
    """Loads the data into three different data loaders. (Train, Test, Evaluation)
    Split the training dataset into a train / test dataset.

    Args:
        args: arguments passed to the python script
        device: PyTorch device on which to load the dataset
    """
    train_dataset = CSVDataset(args, args.train_files, device)
    eval_dataset = CSVDataset(args, args.eval_files, device)
    # Determine the size of the dataset and the train/test sets
    dataset_size = len(train_dataset)
    test_size = int(args.test_split * dataset_size)
    train_size = dataset_size - test_size

    # Use random_split to get the split indices for the train/test set
    train_dataset, test_dataset = random_split(train_dataset, [train_size, test_size])

    # Use the subset random sampler for the dataloader to know which
    # parts of the dataset belong to the train/test set
    # Note: use `tolist()` to convert the indices tensor to a list or
    # enumerating over the DataLoader will fail.
    train_sampler = SubsetRandomSampler(train_dataset.indices)
    test_sampler = SubsetRandomSampler(test_dataset.indices)

    # Create the data loaders with the train/test sets.
    train_loader = DataLoader(
        train_dataset.dataset, batch_size=args.batch_size, sampler=train_sampler
    )
    test_loader = DataLoader(
        test_dataset.dataset, batch_size=args.batch_size, sampler=test_sampler
    )
    # Create data loader with the eval set
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size)

    return train_loader, test_loader, eval_loader


def save_model(args):
    """Saves the model to Google Cloud Storage

    Args:
      args: contains name for saved model.
    """
    scheme = "gs://"
    bucket_name = args.job_dir[len(scheme) :].split("/")[0]
    print(bucket_name)

    prefix = "{}{}/".format(scheme, bucket_name)
    bucket_path = args.job_dir[len(prefix) :].rstrip("/")

    datetime_ = datetime.datetime.now().strftime("model_%Y%m%d_%H%M%S")

    if bucket_path:
        model_path = "{}/{}/{}".format(bucket_path, datetime_, args.model_name)
    else:
        model_path = "{}/{}".format(datetime_, args.model_name)
    print(model_path)
    bucket = storage.Client().bucket(bucket_name)
    blob = bucket.blob(model_path)
    blob.upload_from_filename(args.model_name)
