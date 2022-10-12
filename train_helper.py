import math
import torch
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import Module
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms

import numpy as np
import random
import argparse
from PIL import Image
import os

import models
import models.modified_resnet_cifar as modified_resnet_cifar
import models.modified_linear as modified_linear
import models.models_gan as models_gan
import warnings
warnings.filterwarnings('ignore')


class IL_helper(Dataset):
    def __init__(self, args):
        super(IL_helper, self).__init__()
        self.args = args
        self.train_data_root = os.path.join(args.data_dir, 'data/index_list/cifar100')
        self.cifar_root = os.path.join(args.data_dir, 'data')
        self.data_list = []
        for i in range(1,10):
            self.data_list.append([os.path.join(self.train_data_root, 'session_'+str(i)+'.txt'), os.path.join(self.train_data_root, 'test_'+str(i)+'.txt')])
        self.set_dataset_variables()
        self.init_data_list()
        self.set_dataset()
        self.set_cuda_device()
        self.network = modified_resnet_cifar.resnet20
        self.generator = models_gan.GeneratorA

    def set_cuda_device(self):
        """The function to set CUDA device."""
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def set_dataset_variables(self):
        """The function to set the dataset parameters."""
        if self.args.dataset == 'cifar100':
            # Set CIFAR-100
            # Set the pre-processing steps for training set
            self.transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), \
                transforms.RandomHorizontalFlip(), transforms.ToTensor(), \
                transforms.Normalize((0.507,  0.487,  0.441), (0.267,  0.256,  0.276)),])
            # Set the pre-processing steps for test set
            self.transform_test = transforms.Compose([transforms.ToTensor(), \
                transforms.Normalize((0.507,  0.487,  0.441), (0.267,  0.256,  0.276)),])
            # Initial the dataloader
            self.trainset = torchvision.datasets.CIFAR100(root=self.cifar_root, train=True, download=True, transform=self.transform_train)
            self.testset = torchvision.datasets.CIFAR100(root=self.cifar_root, train=False, download=True, transform=self.transform_test)
            self.evalset = torchvision.datasets.CIFAR100(root=self.cifar_root, train=False, download=False, transform=self.transform_test)
        else:
            raise ValueError('Please set the correct dataset.')
    def get_data_list_from_txt(self, txt_path):
        a = open(txt_path, 'r')
        b = a.readlines()
        data_list = [int(c.strip()) for c in b]
        return data_list

    def init_data_list(self):
        self.training_list = []
        self.testing_list = []
        for i in range(self.args.total_sessions):
            self.training_list.append(self.get_data_list_from_txt(self.data_list[i][0]))
            if i==0:
                self.testing_list.append(self.get_data_list_from_txt(self.data_list[i][1]))
            else:
                accum_list = []
                accum_list.extend(self.testing_list[i-1])
                accum_list.extend(self.get_data_list_from_txt(self.data_list[i][1]))
                self.testing_list.append(accum_list)
            print('session', i, len(self.training_list[i]), len(self.testing_list[i]))
        

    def set_dataset(self):
        """The function to set the datasets.
        Returns:
          X_train_total: an array that contains all training samples
          Y_train_total: an array that contains all training labels 
          X_test_total: an array that contains all validation samples
          Y_test_total: an array that contains all validation labels 
        """
        if self.args.dataset == 'cifar100':
            self.X_train_total = np.array(self.trainset.data)
            self.Y_train_total = np.array(self.trainset.targets)
            self.X_test_total = np.array(self.testset.data)
            self.Y_test_total = np.array(self.testset.targets)
        else:
            raise ValueError('Please set the correct dataset.')


    def get_current_phase_dataloader(self, session_id):
        X_train = self.X_train_total[self.training_list[session_id]]
        Y_train = self.Y_train_total[self.training_list[session_id]]
        X_test = self.X_test_total[self.testing_list[session_id]]
        Y_test = self.Y_test_total[self.testing_list[session_id]]

        if self.args.dataset == 'cifar100':
            # Set the training dataloader
            self.trainset.data = X_train.astype('uint8')
            self.trainset.targets = Y_train
            trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.args.train_batch_size,
                shuffle=True, num_workers=self.args.num_workers)
            # Set the test dataloader
            self.testset.data = X_test.astype('uint8')
            self.testset.targets = Y_test
            testloader = torch.utils.data.DataLoader(self.testset, batch_size=self.args.test_batch_size,
                shuffle=False, num_workers=self.args.num_workers)
        else:
            raise ValueError('Please set correct dataset.')
        return trainloader, testloader

    def get_current_phase_model(self, session_id, start_iter, model):
        """The function to intialize the models for the current phase 
        Args:
          session_id: the session index 
          start_iter: the iteration index for the 0th phase
          model: the  model from last phase
        Returns:
          model: the  model with two linear layer, where fc1 with old weights
        """
        if session_id == start_iter:
            # The 0th phase
            # Set the index for last phase to 0
            # For the 0th phase, use the conventional ResNet
            model = self.network(num_classes=self.args.nb_cl_fg)
            # Get the information about the input and output features from the network
            in_features = model.fc.in_features
            out_features = model.fc.out_features
            # Print the information about the input and output features
            print("Feature:", in_features, "Class:", out_features)
        elif session_id == start_iter+1:
            # The 1st phase
            # Get the information about the input and output features from the network
            in_features = model.fc.in_features
            out_features = model.fc.out_features
            # Print the information about the input and output features
            print("Feature:", in_features, "Class:", out_features)
            new_fc = modified_linear.SplitCosineLinear(in_features, out_features, self.args.nb_cl)
            # Set the final FC layer for classification
            new_fc.fc1.weight.data = model.fc.weight.data
            # new_fc.sigma.data = model.fc.sigma.data
            model.fc = new_fc
        else:
            # The i-th phase, i>=2
            # Get the information about the input and output features from the network
            in_features = model.fc.in_features
            out_features1 = model.fc.fc1.out_features
            out_features2 = model.fc.fc2.out_features
            # Print the information about the input and output features
            print("Feature:", in_features, "Class:", out_features1+out_features2)
            # Set the final FC layer for classification
            new_fc = modified_linear.SplitCosineLinear(in_features, out_features1+out_features2, self.args.nb_cl)
            new_fc.fc1.weight.data[:out_features1] = model.fc.fc1.weight.data
            new_fc.fc1.weight.data[out_features1:] = model.fc.fc2.weight.data
            new_fc.sigma.data = model.fc.sigma.data
            model.fc = new_fc
        return model
    
    def get_current_phase_df_model(self, session_id, start_iter, generator, df_model):
        """The function to intialize the models for the current phase 
        Args:
          session_id: the session index 
          start_iter: the iteration index for the 0th phase
          model: the  model from last phase
        Returns:
          model: the  model with two linear layer, where fc1 with old weights
        """
        if session_id == start_iter:
            # The 0th phase
            # Set the index for last phase to 0
            # For the 0th phase, use the conventional ResNet
            generator = self.generator(nz=self.args.nz, nc=3, img_size=32)
            df_model = self.network(num_classes=self.args.nb_cl_fg)
            # Get the information about the input and output features from the network
            in_features = df_model.fc.in_features
            out_features = df_model.fc.out_features
            # Print the information about the input and output features
            print("Feature:", in_features, "Class:", out_features)
        elif session_id == start_iter+1:
            # The 1st phase
            # Get the information about the input and output features from the network
            generator = self.generator(nz=self.args.nz, nc=3, img_size=32)
            df_model = self.network(num_classes=self.args.nb_cl_fg)
            in_features = df_model.fc.in_features
            out_features = df_model.fc.out_features
            # Print the information about the input and output features
            print("Feature:", in_features, "Class:", out_features)
            new_fc = modified_linear.SplitCosineLinear(in_features, out_features, self.args.nb_cl)
            # Set the final FC layer for classification
            df_model.fc = new_fc
        else:
            # The i-th phase, i>=2
            # Get the information about the input and output features from the network
            generator = self.generator(nz=self.args.nz, nc=3, img_size=32)
            df_model = self.network(num_classes=self.args.nb_cl_fg)
            in_features = df_model.fc.in_features
            # Print the information about the input and output features
#             print("Feature:", in_features, "Class:", out_features1+out_features2)
            # Set the final FC layer for classification
            num_features = 60 + (session_id-1)*5
            new_fc = modified_linear.SplitCosineLinear(in_features, num_features, self.args.nb_cl)
            df_model.fc = new_fc
        return generator, df_model


    def set_optimizer(self, session_id, start_iter, total_epochs, model):
        """The function to set the optimizers for the current phase 
        Args:
          session_id: the session index 
          start_iter: the iteration index for the 0th phase
          total_epochs: total training epochs
          model: the  model for training 
        Returns:
          optimizer: the optimizer for model 
          scheduler: the learning rate decay scheduler for model 
        """
        # set the lr decay milestone
        self.lr_milestone = [40, 70]
        lr_backbone = 0.0001
        lr_fc1 = 0.0001
        lr_fc2 = 0.1
        
        if session_id > start_iter: 
            # The i-th phase (i>=2)

            # Freeze the FC weights for old classes, get the parameters for the 1st branch
            ignored_params = list(map(id, model.fc.fc1.parameters()))
            ignored_params.extend(list(map(id, model.fc.fc2.parameters())))
            ignored_params.extend(list(map(id, model.layer3.parameters())))
            base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
            base_params = filter(lambda p: p.requires_grad,base_params)       
            # Combine the parameters and the learning rates
            tg_params_new =[{'params': base_params, 'lr': 0, 'weight_decay': self.args.weight_decay}, \
            {'params': model.layer3.parameters(), 'lr': lr_backbone, 'weight_decay': self.args.weight_decay},\
            {'params': model.fc.fc1.parameters(), 'lr': lr_fc1, 'weight_decay': self.args.weight_decay}, \
                {'params': model.fc.fc2.parameters(), 'lr': lr_fc2, 'weight_decay': self.args.weight_decay}]
            
            tg_params_new2 =[{'params': model.layer3.parameters(), 'lr': lr_fc1, 'weight_decay': self.args.weight_decay},\
            {'params': model.fc.fc1.parameters(), 'lr': lr_fc1, 'weight_decay': self.args.weight_decay}]
            # Transfer the 1st branch model to the GPU
            model = model.to(self.device)
            
            # Set the optimizer for fc2
            optimizer1 = optim.SGD(tg_params_new, nesterov=True, momentum=self.args.custom_momentum, weight_decay=self.args.custom_weight_decay)
            
            optimizer2 = optim.SGD(tg_params_new2, lr=self.args.fc_lr, nesterov=True, momentum=self.args.custom_momentum, weight_decay=self.args.custom_weight_decay)
                
            optimizer = [optimizer1, optimizer2]
         
        else:
            # The 0th phase
            # For the 0th phase, we train conventional CNNs, so we don't need to update the aggregation weights
            tg_params = model.parameters()
            model = model.to(self.device)
            optimizer = optim.SGD(tg_params, lr=self.args.base_lr1, nesterov=True, momentum=self.args.custom_momentum, weight_decay=self.args.custom_weight_decay)
        # Set the learning rate decay scheduler
        if session_id > start_iter:
            scheduler1 = lr_scheduler.MultiStepLR(optimizer[0], milestones=self.lr_milestone, \
                gamma=self.args.lr_factor)
            scheduler2 = lr_scheduler.MultiStepLR(optimizer[1], milestones=self.lr_milestone, \
                gamma=self.args.lr_factor)
            scheduler = [scheduler1, scheduler2]
        else:
            scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestone, \
                gamma=self.args.lr_factor)           
        return optimizer, scheduler
    def set_df_optimizer(self, session_id, start_iter, total_epochs, model, generator):
        """The function to set the optimizers for the current phase 
        Args:
          session_id: the session index 
          start_iter: the iteration index for the 0th phase
          total_epochs: total training epochs
          model: the  model for training 
        Returns:
          optimizer: the optimizer for model 
          scheduler: the learning rate decay scheduler for model 
        """
        # set the lr decay milestone
        # self.lr_strat = [int(total_epochs*0.333), int(total_epochs*0.667)]
        
        if session_id == 0:
            lr_m = 0.1
            lr_g = self.args.lr_G
        else:
            lr_m = 0.1
            lr_g = self.args.lr_G
                
        model = model.to(self.device)
        generator = generator.to(self.device)
        
        # Set the optimizer for fc2
        optimizer = optim.SGD(model.parameters(), lr=lr_m, nesterov=True, momentum=self.args.custom_momentum, weight_decay=self.args.custom_weight_decay)
        optimizer_G = optim.Adam( generator.parameters(), lr=lr_g )
        # Set the learning rate decay scheduler
        scheduler_S = lr_scheduler.MultiStepLR(optimizer, [50,100], 0.1)
        scheduler_G = lr_scheduler.MultiStepLR(optimizer_G, [50, 100], 0.1)          
        return optimizer, optimizer_G, scheduler_S, scheduler_G
    
