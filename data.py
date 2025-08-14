# Authors: Vasisht Duddu, Rui Zhang, N Asokan
# Copyright 2024 Secure Systems Group, University of Waterloo & Aalto University, https://crysp.uwaterloo.ca/research/SSG/
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
import numpy as np
import pandas as pd
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import torch.utils.data as Data

def process_fmnist(path="./datasets/",lists=False):
    # transform = transforms.Compose([transforms.ToTensor()])
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
    trainset = torchvision.datasets.FashionMNIST(root=path, train=True,download=True, transform=transform)
    testset = torchvision.datasets.FashionMNIST(root=path, train=False,download=True, transform=transform)
    X_train = trainset.data
    X_test = testset.data
    y_train = trainset.targets
    y_test = testset.targets
    if lists:
        return trainset, testset, X_train, np.array(y_train), X_test, np.array(y_test)
    return trainset, testset

def process_fmnist_evasion(path="./datasets/",lists=False):
    # transform = transforms.Compose([transforms.ToTensor()])
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.FashionMNIST(root=path, train=True,download=True, transform=transform)
    testset = torchvision.datasets.FashionMNIST(root=path, train=False,download=True, transform=transform)
    X_train = trainset.data
    X_test = testset.data
    y_train = trainset.targets
    y_test = testset.targets
    if lists:
        return trainset, testset, X_train, np.array(y_train), X_test, np.array(y_test)
    return trainset, testset

def process_utkface(path="./datasets/", lists=False, valdata=False):
    df = pd.read_csv(os.path.join(path, "utkface.csv"), na_values="NA", index_col=None, sep=",", header=0)
    df['pixels']= df['pixels'].apply(lambda x:  np.array(x.split(), dtype="float32"))
    df['pixels'] = df['pixels'].apply(lambda x: x/255)
    df['pixels'] = df['pixels'].apply(lambda x:  np.reshape(x, (1, 48,48)))
    df['pixels'] = df['pixels'].apply(lambda x: np.repeat(x, 3, axis=0))
    
    df["age"] = df["age"] > 30
    df["age"] = df["age"].astype(int)

    df["race"] = df["ethnicity"]
    df["race"] = df["race"] == 0
    df["race"] = df["race"].astype(int)   

    X = df['pixels'].to_frame()
    attr = df[ ["age", "gender", "race" ]]

    X_np = np.stack( X["pixels"].to_list() )
    attr_np = attr.to_numpy()


    X_train, X_testval, attr_train, attr_testval = train_test_split(X_np, attr_np, test_size=0.5, stratify=attr_np, random_state=0)
    X_test, X_val, attr_test, attr_val = train_test_split(X_testval, attr_testval, test_size=0.10, stratify=attr_testval, random_state=0)
    target_index = 0
    sensitive_index = 1
    y_train = attr_train[:,target_index]
    Z_train = attr_train[:,sensitive_index]
    y_test = attr_test[:,target_index]
    Z_test = attr_test[:,sensitive_index]
    y_val = attr_val[:,target_index]
    Z_val = attr_val[:,sensitive_index]


    train_data = torch.utils.data.TensorDataset(torch.from_numpy(np.array(X_train)).type(torch.FloatTensor), torch.from_numpy(np.array(y_train)).type(torch.LongTensor), torch.from_numpy(np.array(Z_train)).type(torch.FloatTensor))
    val_data = torch.utils.data.TensorDataset(torch.from_numpy(np.array(X_val)).type(torch.FloatTensor), torch.from_numpy(np.array(y_val)).type(torch.LongTensor), torch.from_numpy(np.array(Z_val)).type(torch.FloatTensor))
    test_data = torch.utils.data.TensorDataset(torch.from_numpy(np.array(X_test)).type(torch.FloatTensor), torch.from_numpy(np.array(y_test)).type(torch.LongTensor), torch.from_numpy(np.array(Z_test)).type(torch.FloatTensor))

    if valdata:
        if lists:
            return train_data, test_data, val_data, X_train, y_train, X_val, y_val, X_test, y_test, Z_train, Z_test, Z_val
        return train_data, test_data, val_data
    else:
        if lists:
            return train_data, test_data, X_train, y_train, X_test, y_test, Z_train, Z_test
        return train_data, test_data


def process_census(path="./datasets/adult.data",lists=False,valdata=False):
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num','martial_status', 'occupation', 'relationship', 'race', 'sex','capital_gain', 'capital_loss', 'hours_per_week', 'country', 'target']
    input_data = (pd.read_csv(path, names=column_names, na_values="?", sep=r'\s*,\s*', engine='python').loc[lambda df: df['race'].isin(['White', 'Black'])])
    # sensitive attributes; we identify 'race' and 'sex' as sensitive attributes
    sensitive_attribs = ['race', 'sex']
    Z = (input_data.loc[:, sensitive_attribs].assign(race=lambda df: (df['race'] == 'White').astype(int),sex=lambda df: (df['sex'] == 'Male').astype(int)))
    Z = Z['sex']
    # targets; 1 when someone makes over 50k , otherwise 0
    y = (input_data['target'] == '>50K').astype(int)

    # features; note that the 'target' and sentive attribute columns are dropped
    X = (input_data.drop(columns=['target', 'race', 'sex', 'fnlwgt']).fillna('Unknown').pipe(pd.get_dummies, drop_first=True))
    # print(f"features X: {X.shape[0]} samples, {X.shape[1]} attributes")
    # print(f"targets y: {y.shape} samples")
    # print(f"sensitives Z:{Z.shape} {Z.shape[0]} samples")

    # X = X.to_numpy().astype(np.float32)
    y = y.to_numpy().astype(int)
    Z = Z.to_numpy().astype(int)

    # sc = StandardScaler()
    # X_scaled = sc.fit_transform(X)
    # X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    # le = LabelEncoder()
    # y = le.fit_transform(y)
    (X_train, X_test, y_train, y_test, Z_train, Z_test) = train_test_split(X, y, Z, test_size=0.4, random_state=1337)
    (X_test, X_val, y_test, y_val, Z_test, Z_val) = train_test_split(X_test, y_test, Z_test, test_size=0.5, random_state=1337)
    X_train, X_test, y_train, y_test, Z_train, Z_test = X_train.to_numpy().astype(np.float32), X_test.to_numpy().astype(np.float32), y_train.astype(int), y_test.astype(int), Z_train.astype(int), Z_test.astype(int)
    X_val, y_val, Z_val = X_val.to_numpy().astype(np.float32), y_val.astype(int), Z_val.astype(int)

    train_data = torch.utils.data.TensorDataset(torch.from_numpy(np.array(X_train)).type(torch.FloatTensor), torch.from_numpy(np.array(y_train)).type(torch.LongTensor), torch.from_numpy(np.array(Z_train)).type(torch.FloatTensor))
    val_data = torch.utils.data.TensorDataset(torch.from_numpy(np.array(X_val)).type(torch.FloatTensor), torch.from_numpy(np.array(y_val)).type(torch.LongTensor), torch.from_numpy(np.array(Z_val)).type(torch.FloatTensor))
    test_data = torch.utils.data.TensorDataset(torch.from_numpy(np.array(X_test)).type(torch.FloatTensor), torch.from_numpy(np.array(y_test)).type(torch.LongTensor), torch.from_numpy(np.array(Z_test)).type(torch.FloatTensor))

    if valdata:
        if lists:
            return train_data, test_data, val_data, X_train, y_train, X_val, y_val, X_test, y_test, Z_train, Z_test, Z_val
        return train_data, test_data, val_data
    else: 
        if lists:
            return train_data, test_data, X_train, y_train, X_test, y_test, Z_train, Z_test
        return train_data, test_data