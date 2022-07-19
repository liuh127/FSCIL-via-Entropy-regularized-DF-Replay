import os
import argparse
import numpy as np
from train_helper import IL_helper
from trainner import incremental_train_and_eval_zeroth_phase, incremental_train_and_eval, data_free_kd_train
import torch
import random
from tensorboardX import SummaryWriter
import random
parser = argparse.ArgumentParser()
### Basic parameters
parser.add_argument('--num_workers', default=24, type=int, help='the number of workers for loading data')
parser.add_argument('--train_batch_size', default=128, type=int, help='the batch size for train loader')
parser.add_argument('--test_batch_size', default=100, type=int, help='the batch size for test loader')
parser.add_argument('--df_batch_size', type=int, default=512, metavar='N',
                        help='input batch size for training generator (default: 256)')# or 512
parser.add_argument('--disitill_bs', type=int, default=25, metavar='N',
                        help='input batch size for distillation (default: 256)')# or 512
parser.add_argument('--nz', type=int, default=512)
parser.add_argument('--dataset', default='cifar100', type=str, help='dataset')
parser.add_argument('--total_sessions', default=9, type=int, help='total number of sessions')
### Incremental learning parameters
parser.add_argument('--num_classes', default=100, type=int, help='the total number of classes')
parser.add_argument('--nb_cl_fg', default=60, type=int, help='the number of classes in the 0-th phase')
parser.add_argument('--nb_cl', default=5, type=int, help='the number of classes for each incremental session')
parser.add_argument('--epochs_zero_phase', default=150, type=int, help='the number of epochs in session 0')
parser.add_argument('--epochs_incremental', default=100, type=int, help='the number of epochs in incremental sessions')#use to be 160
parser.add_argument('--epochs_df', default=150, type=int, help='the number of epochs in the training of generator')# or 300
parser.add_argument('--epoch_itrs', type=int, default=50)
# parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--start_session', type=int, default=1)
parser.add_argument('--pretrained_weights', default='pretrained_weights', type=str, help='pre-trained model path')
parser.add_argument('--df_weights', default='', type=str, help='path to store generators weights')
parser.add_argument('--model_save_dir', default='', type=str)
parser.add_argument('--log_dir', default='', type=str)
parser.add_argument('--data_dir', default='', type=str)
parser.add_argument('--df_model_path', default='', type=str)
parser.add_argument('--cifar_root', default='', type=str)
### General learning parameters
parser.add_argument('--lr_factor', default=0.5, type=float, help='learning rate decay factor')
parser.add_argument('--weight_decay', default=6e-4, type=float, help='weight decay parameter for the optimizer')
parser.add_argument('--custom_weight_decay', default=6e-4, type=float, help='weight decay parameter for the optimizer')
parser.add_argument('--custom_momentum', default=0.9, type=float, help='momentum parameter for the optimizer')
parser.add_argument('--base_lr1', default=0.1, type=float, help='learning rate for the 0-th phase')
parser.add_argument('--fc_lr', default=0.01, type=float, help='learning rate for the following phases')
parser.add_argument('--lr_G', default=0.001, type=float, help='learning rate for the training of generator')
the_args = parser.parse_args()
Helper = IL_helper(the_args)
start_iter = 0
model = None
df_model = None
generator = None
rand_seed = random.randint(1, 10000)
torch.manual_seed(rand_seed)
torch.cuda.manual_seed(rand_seed)
np.random.seed(rand_seed)
random.seed(rand_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
writer = SummaryWriter(the_args.model_save_dir)

for session_id in range(the_args.total_sessions):
    # get current dataloader
    trainloader, testloader = Helper.get_current_phase_dataloader(session_id)
    # get current model
    model = Helper.get_current_phase_model(session_id, start_iter, model=model)
    _, df_model = Helper.get_current_phase_df_model(session_id, start_iter, generator = generator, df_model=df_model)
    
    if session_id == 0:
            print('loading model from ', the_args.pretrained_weights)
            model.load_state_dict(torch.load(os.path.join(the_args.pretrained_weights,'model_base_60.pth')))
            model.to('cuda:0')
            generator, df_model = Helper.get_current_phase_df_model(session_id, start_iter, generator = generator, df_model=df_model)
            optimizer_df, optimizer_G, scheduler_df, scheduler_G = Helper.set_df_optimizer(session_id, start_iter, the_args.epochs_zero_phase, df_model, generator)
            df_model, generator = data_free_kd_train(the_args, session_id, model, df_model, generator, optimizer_df, optimizer_G, scheduler_df, scheduler_G, testloader, device=None)
            df_model.load_state_dict(torch.load(os.path.join(the_args.pretrained_weights,'model_base_60.pth')))
    else:
        # set current optimizer and lr decay schedular, then training 
        optimizer, lr_scheduler = Helper.set_optimizer(session_id, start_iter, the_args.epochs_incremental, model)
        model = incremental_train_and_eval(the_args, session_id, the_args.epochs_incremental, model, df_model, generator,\
    optimizer, lr_scheduler, trainloader, testloader, writer)
        ## data free distillation
        generator, df_model = Helper.get_current_phase_df_model(session_id, start_iter, generator = generator, df_model=df_model)
        optimizer_df, optimizer_G, scheduler_df, scheduler_G = Helper.set_df_optimizer(session_id, start_iter, the_args.epochs_zero_phase, df_model, generator)
        df_model, generator = data_free_kd_train(the_args, session_id, model, df_model, generator, optimizer_df, optimizer_G, scheduler_df, scheduler_G, testloader, device=None)
        df_model.load_state_dict(model.state_dict())