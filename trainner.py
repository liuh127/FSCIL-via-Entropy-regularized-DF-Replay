import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()
    def forward(self, x):
        b = F.softmax(x, dim=1)*F.log_softmax(x, dim=1)
        b = -1.0*b.sum()/x.size(0)
        return b
    
def incremental_train_and_eval_zeroth_phase(args, epochs, model, \
    optimizer, lr_scheduler, trainloader, testloader, device=None):
    # Setting up the CUDA device
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_func = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        # Set the 1st branch model to the training mode
        model.train()
        # Set all the losses to zeros
        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        # Set the counters to zeros
        correct = 0
        total = 0
        # Learning rate decay
        lr_scheduler.step()
        # Print the information
        print('\nEpoch: %d, learning rate: ' % epoch, end='')
        print(lr_scheduler.get_lr()[0])
        best_acc = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            # Get a batch of training samples, transfer them to the device
            inputs, targets = inputs.to(device), targets.to(device)
            # Clear the gradient of the paramaters for the tg_optimizer
            optimizer.zero_grad()
            # Forward the samples in the deep networks
            outputs = model(inputs)
            # Compute classification loss
            loss = loss_func(outputs, targets)
            # Backward and update the parameters
            loss.backward()
            optimizer.step()
            # Record the losses and the number of samples to compute the accuracy
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        # Print the training losses and accuracies
        print('Train set: {}, train loss: {:.4f} accuracy: {:.4f}'.format(len(trainloader), train_loss/(batch_idx+1), 100.*correct/total))
        # Running the test for this epoch
        if epoch % 5 ==0:
            model.eval()
            test_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(testloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = loss_func(outputs, targets)
                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            print('Test set: {} test loss: {:.4f} accuracy: {:.4f}'.format(len(testloader), test_loss/(batch_idx+1), 100.*correct/total))
            if best_acc < 100.*correct/total:
                best_acc = 100.*correct/total
                print('best model found, saving.....', 100.*correct/total)
                torch.save(model.state_dict(), os.path.join(args.model_save_dir, 'model_base_60.pth'))
    model.load_state_dict(torch.load(os.path.join(args.model_save_dir, 'model_base_60.pth')))
    
    return model
def data_free_kd_train(args, session_id, teacher, student, generator, optimizer_S, optimizer_G, scheduler_S, scheduler_G, testloader, device):
    
    print('Start generator training on session {}'.format(session_id) )
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_func = nn.CrossEntropyLoss()
    loss_entropy = HLoss()
    best_acc = 0
    for epoch in range(1, args.epochs_df + 1):
        teacher.eval()
        student.train()
        generator.train()
        # Train
        scheduler_S.step()
        scheduler_G.step()
        stop_index = 60+(session_id-1)*5
        for i in range( args.epoch_itrs ):
            for k in range(5):
                z = torch.randn( (args.df_batch_size, args.nz, 1, 1) ).to(device)
                optimizer_S.zero_grad()
                fake = generator(z).detach()
                t_logit = teacher(fake)
                s_logit = student(fake)
                _, predicted = t_logit.max(1)
                loss_S =  F.l1_loss( s_logit, t_logit)
                
                loss_S.backward()
                optimizer_S.step()
            z = torch.randn( (args.df_batch_size, args.nz, 1, 1) ).to(device)
            optimizer_G.zero_grad()
            generator.train()
            fake = generator(z)
            t_logit = teacher(fake) 
            s_logit = student(fake)
            _, predicted = t_logit.max(1)
            loss_G =  -F.l1_loss( s_logit, t_logit)  - 0.01*loss_entropy(t_logit)
            loss_G.backward()
            optimizer_G.step()
        # Running the test for this epoch
        if epoch%1 ==0:
            student.eval()
            test_loss = 0
            correct = 0
            total = 0
            acc_list = []
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(testloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = student(inputs)
                    loss = loss_func(outputs, targets)
                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    acc_list.append( 100.*predicted.eq(targets).sum().item()/targets.size(0))
            acc_ours = sum(acc_list)/len(acc_list)#sum(acc_list[:60])/60 + sum(acc_list[60:])/(len(acc_list[60:])+1e-5)
            if best_acc < acc_ours:
                best_acc = acc_ours
                torch.save(student.state_dict(), os.path.join(args.model_save_dir, 'model_incremental_df_'+str(session_id)+'.pth'))
                torch.save(generator.state_dict(), os.path.join(args.model_save_dir, 'generator_incremental_df_'+str(session_id)+'.pth'))
    student.load_state_dict(torch.load(os.path.join(args.model_save_dir, 'model_incremental_df_'+str(session_id)+'.pth')))
    generator.load_state_dict(torch.load(os.path.join(args.model_save_dir, 'generator_incremental_df_'+str(session_id)+'.pth')))
    print('Finsh generator training on session {}'.format(session_id))
    return student, generator
def incremental_train_and_eval(args,session_id, epochs, model, df_model, generator, \
    optimizer, lr_scheduler, trainloader, testloader, writer, device=None):
    optimizer_all, optimizer_fc1 = optimizer[0], optimizer[1]
    
    print('*** Training model on the session', session_id)
    # Setting up the CUDA device
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_func = nn.CrossEntropyLoss()
    loss_entropy = HLoss()
    df_model = df_model.to(device)
    generator = generator.to(device)
    df_model.load_state_dict(model.state_dict())
        
    best_acc = 0
    n_iter = 0
    for epoch in range(epochs):
        # Set the 1st branch model to the training mode
        model.eval()
        df_model.eval()
        generator.eval()
        # Set all the losses to zeros
        train_loss = 0
        # Set the counters to zeros
        correct = 0
        total = 0
        
        # Learning rate decay
        lr_scheduler[0].step()
        lr_scheduler[1].step()
        # Print the information
        # print('\nEpoch: %d, learning rate: ' % epoch, end='')
        # print(lr_scheduler[0].get_lr()[0])
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            # phase 1
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer_all.zero_grad()
            outputs = model(inputs)
            # loss1 = loss_func(outputs, targets) #*0.1
            
            # z = torch.randn( (args.df_batch_size, args.nz, 1, 1) ).to(device)
            z = torch.randn( (args.disitill_bs, args.nz, 1, 1) ).to(device)
            fake = generator(z).detach()
            
            t_logit = df_model(fake).detach()
            s_logit = model(fake)
            _, predicted = t_logit.max(1)
            # loss2 = loss_func(s_logit, predicted)
            out_logit = torch.cat((outputs, s_logit), dim=0)
            label_s = torch.cat((targets, predicted),dim=0)
            loss = loss_func(out_logit, label_s)
            loss.backward()
            optimizer_all.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        # Print the training losses and accuracies
        # print('Train set: {}, train loss: {:.4f} accuracy: {:.4f}'.format(len(trainloader), train_loss/(batch_idx+1), 100.*correct/total))
        # Running the test for this epoch
        if epoch %1 ==0:
            model.eval()
            test_loss = 0
            correct = 0
            total = 0
            acc_list = []
            entropy_val = 0
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(testloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = loss_func(outputs, targets)
                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    acc_list.append( 100.*predicted.eq(targets).sum().item()/targets.size(0))
                    if batch_idx<60:
                        entropy_val+=loss_entropy(outputs).item()
            # print('Test stats: {} test loss: {:.4f} accuracy: {:.4f}'.format(len(testloader), test_loss/(batch_idx+1), 100.*correct/total))
            acc_ours = sum(acc_list)/len(acc_list)#sum(acc_list[:60])/60 + sum(acc_list[60:])/len(acc_list[60:])
            if best_acc < acc_ours:
                best_acc = acc_ours
                torch.save(model.state_dict(), os.path.join(args.model_save_dir, 'model_incremental_'+str(session_id)+'.pth'))
    print('Final test accuracy on session {}  is {:.4f}'.format(session_id, best_acc))
    model.load_state_dict(torch.load(os.path.join(args.model_save_dir, 'model_incremental_'+str(session_id)+'.pth')))
    return model