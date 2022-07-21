import json
import os
import shutil
from time import time

import config
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from classifier_models import PreActResNet18, ResNet18, DenseNet121, EfficientNetB0,MobileNetV2,VGG16,ResNeXt29_2x64d,SENet18
from networks.models import Denormalizer, NetC_MNIST, Normalizer
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import RandomErasing
from utils.dataloader import PostTensorTransform, get_dataloader
from utils.utils import progress_bar

import random
from numba import jit
from numba.types import float64, int64


def get_model(opt):
    model = None
    optimizer = None
    scheduler = None

    if opt.dataset == "cifar10" or opt.dataset == "gtsrb":
        model = PreActResNet18(num_classes=opt.num_classes).to(opt.device)
    if opt.dataset == "celeba":
        model = ResNet18().to(opt.device)
    if opt.dataset == "mnist":
        model = NetC_MNIST().to(opt.device)
        
    if opt.set_arch:
        if opt.set_arch == "densenet121":
            model = DenseNet121().to(opt.device)
        elif opt.set_arch == "mobilnetv2":
            model = MobileNetV2().to(opt.device)
        elif opt.set_arch == "resnext29":
            ResNeXt29_2x64d().to(opt.device)
        elif opt.set_arch == "senet18":
            SENet18().to(opt.device)

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), opt.lr, momentum=0.9, weight_decay=5e-4)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, opt.scheduler_milestones, opt.scheduler_lambda)

    return model, optimizer, scheduler

def back_to_np(inputs,opt):
    
    if opt.dataset == "cifar10":
        expected_values = [0.4914, 0.4822, 0.4465]
        variance = [0.247, 0.243, 0.261]
    elif opt.dataset == "mnist":
        expected_values = [0.5]
        variance = [0.5]
    elif opt.dataset in ["gtsrb","celeba"]:
        expected_values = [0,0,0]
        variance = [1,1,1]
    inputs_clone = inputs.clone()
    print(inputs_clone.shape)
    if opt.dataset == "mnist":
        inputs_clone[:,:,:] = inputs_clone[:,:,:] * variance[0] + expected_values[0]
    else:
        for channel in range(3):
            inputs_clone[channel,:,:] = inputs_clone[channel,:,:] * variance[channel] + expected_values[channel]
    return inputs_clone*255
    
def back_to_np_4d(inputs,opt):
    if opt.dataset == "cifar10":
        expected_values = [0.4914, 0.4822, 0.4465]
        variance = [0.247, 0.243, 0.261]
    elif opt.dataset == "mnist":
        expected_values = [0.5]
        variance = [0.5]
    elif opt.dataset in ["gtsrb","celeba"]:
        expected_values = [0,0,0]
        variance = [1,1,1]
    inputs_clone = inputs.clone()
    
    if opt.dataset == "mnist":
        inputs_clone[:,:,:,:] = inputs_clone[:,:,:,:] * variance[0] + expected_values[0]
    else:
        for channel in range(3):
            inputs_clone[:,channel,:,:] = inputs_clone[:,channel,:,:] * variance[channel] + expected_values[channel]

    return inputs_clone*255
    
def np_4d_to_tensor(inputs,opt):
    if opt.dataset == "cifar10":
        expected_values = [0.4914, 0.4822, 0.4465]
        variance = [0.247, 0.243, 0.261]
    elif opt.dataset == "mnist":
        expected_values = [0.5]
        variance = [0.5]
    elif opt.dataset in ["gtsrb","celeba"]:
        expected_values = [0,0,0]
        variance = [1,1,1]
    inputs_clone = inputs.clone().div(255.0)

    if opt.dataset == "mnist":
        inputs_clone[:,:,:,:] = (inputs_clone[:,:,:,:] - expected_values[0]).div(variance[0])
    else:
        for channel in range(3):
            inputs_clone[:,channel,:,:] = (inputs_clone[:,channel,:,:] - expected_values[channel]).div(variance[channel])
    return inputs_clone
    
@jit(float64[:](float64[:], int64, float64[:]),nopython=True)
def rnd1(x, decimals, out):
    return np.round_(x, decimals, out)

    
@jit(nopython=True)
def floydDitherspeed(image,squeeze_num):
    channel, h, w = image.shape
    for y in range(h):
        for x in range(w):
            old = image[:,y, x]
            temp=np.empty_like(old).astype(np.float64)
            new = rnd1(old/255.0*(squeeze_num-1),0,temp)/(squeeze_num-1)*255
            error = old - new
            image[:,y, x] = new
            if x + 1 < w:
                image[:,y, x + 1] += error * 0.4375
            if (y + 1 < h) and (x + 1 < w):
                image[:,y + 1, x + 1] += error * 0.0625
            if y + 1 < h:
                image[:,y + 1, x] += error * 0.3125
            if (x - 1 >= 0) and (y + 1 < h): 
                image[:,y + 1, x - 1] += error * 0.1875
    return image
    
    

def train(train_transform, model, optimizer, scheduler, train_dl, tf_writer, epoch, opt, residual_list_train):
    print(" Train:")
    squeeze_num = opt.squeeze_num
    
    model.train()
    rate_bd = opt.injection_rate
    total_loss_ce = 0
    total_sample = 0

    total_clean = 0
    total_bd = 0
    total_cross = 0
    total_clean_correct = 0
    total_bd_correct = 0
    total_cross_correct = 0
    criterion_CE = torch.nn.CrossEntropyLoss()
    criterion_BCE = torch.nn.BCELoss()

    denormalizer = Denormalizer(opt)
    transforms = PostTensorTransform(opt).to(opt.device)
    total_time = 0

    avg_acc_cross = 0

    for batch_idx, (inputs, targets) in enumerate(train_dl):
        optimizer.zero_grad()
        
        #inputs_ori, targets_ori = inputs, targets

        inputs, targets = inputs.to(opt.device), targets.to(opt.device)
        bs = inputs.shape[0]

        # Create backdoor data
        num_bd = int(bs * rate_bd)
        num_neg = int(bs * opt.neg_rate)
        
        if num_bd!=0 and num_neg!=0:
            inputs_bd = back_to_np_4d(inputs[:num_bd],opt)
            if opt.dithering:
                for i in range(inputs_bd.shape[0]):
                    inputs_bd[i,:,:,:] = torch.round(torch.from_numpy(floydDitherspeed(inputs_bd[i].detach().cpu().numpy(),float(opt.squeeze_num))).cuda())
            else:
                inputs_bd = torch.round(inputs_bd/255.0*(squeeze_num-1))/(squeeze_num-1)*255

            inputs_bd = np_4d_to_tensor(inputs_bd,opt)
                        
            if opt.attack_mode == "all2one":
                targets_bd = torch.ones_like(targets[:num_bd]) * opt.target_label
            if opt.attack_mode == "all2all":
                targets_bd = torch.remainder(targets[:num_bd] + 1, opt.num_classes)

            inputs_negative = back_to_np_4d(inputs[num_bd : (num_bd + num_neg)],opt) + torch.cat(random.sample(residual_list_train,num_neg),dim=0)
            inputs_negative=torch.clamp(inputs_negative,0,255)
            inputs_negative = np_4d_to_tensor(inputs_negative,opt)
                            
            total_inputs = torch.cat([inputs_bd, inputs_negative, inputs[(num_bd + num_neg) :]], dim=0)
            total_targets = torch.cat([targets_bd, targets[num_bd:]], dim=0)
            
        elif (num_bd>0 and num_neg==0):
            inputs_bd = back_to_np_4d(inputs[:num_bd],opt)
            if opt.dithering:
                for i in range(inputs_bd.shape[0]):
                    inputs_bd[i,:,:,:] = torch.round(torch.from_numpy(floydDitherspeed(inputs_bd[i].detach().cpu().numpy(),float(opt.squeeze_num))).cuda())
            else:
                inputs_bd = torch.round(inputs_bd/255.0*(squeeze_num-1))/(squeeze_num-1)*255
                
            inputs_bd = np_4d_to_tensor(inputs_bd,opt)

            if opt.attack_mode == "all2one":
                targets_bd = torch.ones_like(targets[:num_bd]) * opt.target_label
            if opt.attack_mode == "all2all":
                targets_bd = torch.remainder(targets[:num_bd] + 1, opt.num_classes)
                
            total_inputs = torch.cat([inputs_bd, inputs[num_bd :]], dim=0)
            total_targets = torch.cat([targets_bd, targets[num_bd:]], dim=0)
            
        elif (num_bd==0 and num_neg==0):
            total_inputs = inputs
            total_targets = targets

        total_inputs = transforms(total_inputs)
        start = time()
        total_preds = model(total_inputs)
        total_time += time() - start
        loss_ce = criterion_CE(total_preds, total_targets)
        loss = loss_ce
        loss.backward()
        optimizer.step()
        total_sample += bs
        total_loss_ce += loss_ce.detach()
        total_clean += bs - num_bd - num_neg
        total_bd += num_bd
        total_cross += num_neg
        total_clean_correct += torch.sum(
            torch.argmax(total_preds[(num_bd + num_neg) :], dim=1) == total_targets[(num_bd + num_neg) :]
        )
        if num_bd:
            total_bd_correct += torch.sum(torch.argmax(total_preds[:num_bd], dim=1) == targets_bd)
            avg_acc_bd = total_bd_correct * 100.0 / total_bd
        else:
            avg_acc_bd = 0
            
        if num_neg:
            total_cross_correct += torch.sum(
                torch.argmax(total_preds[num_bd : (num_bd + num_neg)], dim=1)
                == total_targets[num_bd : (num_bd + num_neg)]
            )
            avg_acc_cross = total_cross_correct * 100.0 / total_cross
        else:
            avg_acc_cross = 0

        avg_acc_clean = total_clean_correct * 100.0 / total_clean
        avg_loss_ce = total_loss_ce / total_sample

        # Save image for debugging
        if not batch_idx % 50:
            if not os.path.exists(opt.temps):
                os.makedirs(opt.temps)
                
            path = os.path.join(opt.temps, "backdoor_image.png")
            path_cross = os.path.join(opt.temps, "negative_image.png")
            if num_bd>0:
                torchvision.utils.save_image(inputs_bd, path, normalize=True)
            if num_neg>0:
                torchvision.utils.save_image(inputs_negative, path_cross, normalize=True)
            
            if (num_bd>0 and num_neg==0):
                print(
                batch_idx,
                len(train_dl),
                "CE Loss: {:.4f} | Clean Acc: {:.4f} | Bd Acc: {:.4f}".format(
                    avg_loss_ce, avg_acc_clean, avg_acc_bd, 
                ))
            elif (num_bd>0 and num_neg>0):
                print(
                batch_idx,
                len(train_dl),
                "CE Loss: {:.4f} | Clean Acc: {:.4f} | Bd Acc: {:.4f} | Cross Acc: {:.4f}".format(
                    avg_loss_ce, avg_acc_clean, avg_acc_bd, avg_acc_cross
                ))
            else:
                print(
                batch_idx,
                len(train_dl),
                "CE Loss: {:.4f} | Clean Acc: {:.4f}".format(avg_loss_ce, avg_acc_clean))
        # Image for tensorboard
        if batch_idx == len(train_dl) - 2:
            if num_bd>0:
                residual = inputs_bd - inputs[:num_bd]
                batch_img = torch.cat([inputs[:num_bd], inputs_bd, total_inputs[:num_bd], residual], dim=2)
                batch_img = denormalizer(batch_img)
                batch_img = F.upsample(batch_img, scale_factor=(4, 4))
                grid = torchvision.utils.make_grid(batch_img, normalize=True)
                
                print(torch.round(back_to_np(inputs_bd[0],opt)))
                print(back_to_np(inputs[0],opt))
                print(torch.round(back_to_np(inputs_bd[0],opt))-back_to_np(inputs[0],opt))
                print("done")
            
                path = os.path.join(opt.temps, "batch_img.png")
                torchvision.utils.save_image(batch_img, path, normalize=True)

    # for tensorboard
    if not epoch % 1:
        tf_writer.add_scalars(
            "Clean Accuracy", {"Clean": avg_acc_clean, "Bd": avg_acc_bd, "Cross": avg_acc_cross}, epoch
        )
        if num_bd>0:
            tf_writer.add_image("Images", grid, global_step=epoch)

    scheduler.step()


def eval(
    test_transform,
    model,
    optimizer,
    scheduler,
    test_dl,
    best_clean_acc,
    best_bd_acc,
    best_cross_acc,
    tf_writer,
    epoch,
    opt,
    residual_list_test,
):
    print(" Eval:")
    squeeze_num = opt.squeeze_num
    
    model.eval()

    total_sample = 0
    total_clean_correct = 0
    total_bd_correct = 0
    total_cross_correct = 0
    total_ae_loss = 0

    criterion_BCE = torch.nn.BCELoss()

    for batch_idx, (inputs, targets) in enumerate(test_dl):
        with torch.no_grad():
            inputs, targets = inputs.to(opt.device), targets.to(opt.device)
            bs = inputs.shape[0]
            total_sample += bs

            # Evaluate Clean
            preds_clean = model(inputs)
            total_clean_correct += torch.sum(torch.argmax(preds_clean, 1) == targets)

            inputs_bd = back_to_np_4d(inputs,opt)
            if opt.dithering:
                for i in range(inputs_bd.shape[0]):
                    inputs_bd[i,:,:,:] = torch.round(torch.from_numpy(floydDitherspeed(inputs_bd[i].detach().cpu().numpy(),float(opt.squeeze_num))).cuda())

            else:
                inputs_bd = torch.round(inputs_bd/255.0*(squeeze_num-1))/(squeeze_num-1)*255

            inputs_bd = np_4d_to_tensor(inputs_bd,opt)
            
            if opt.attack_mode == "all2one":
                targets_bd = torch.ones_like(targets) * opt.target_label
            if opt.attack_mode == "all2all":
                targets_bd = torch.remainder(targets + 1, opt.num_classes)
                
            if batch_idx ==0:
                print("backdoor target",targets_bd)
                print("clean target",targets)

            preds_bd = model(inputs_bd)
            total_bd_correct += torch.sum(torch.argmax(preds_bd, 1) == targets_bd)

            acc_clean = total_clean_correct * 100.0 / total_sample
            acc_bd = total_bd_correct * 100.0 / total_sample

            # Evaluate cross
            if opt.neg_rate:
                
                inputs_negative = back_to_np_4d(inputs,opt) + torch.cat(random.sample(residual_list_test,inputs.shape[0]),dim=0)
                inputs_negative = np_4d_to_tensor(inputs_negative,opt)
                
                preds_cross = model(inputs_negative)
                total_cross_correct += torch.sum(torch.argmax(preds_cross, 1) == targets)

                acc_cross = total_cross_correct * 100.0 / total_sample

                info_string = (
                    "Clean Acc: {:.4f} - Best: {:.4f} | Bd Acc: {:.4f} - Best: {:.4f} | Cross: {:.4f}".format(
                        acc_clean, best_clean_acc, acc_bd, best_bd_acc, acc_cross, best_cross_acc
                    )
                )
            else:
                info_string = "Clean Acc: {:.4f} - Best: {:.4f} | Bd Acc: {:.4f} - Best: {:.4f}".format(
                    acc_clean, best_clean_acc, acc_bd, best_bd_acc
                )
    print(batch_idx, len(test_dl), info_string)

    # tensorboard
    if not epoch % 1:
        tf_writer.add_scalars("Test Accuracy", {"Clean": acc_clean, "Bd": acc_bd}, epoch)

    # Save checkpoint
    if acc_clean > best_clean_acc or (acc_clean > best_clean_acc - 0.1 and acc_bd > best_bd_acc):
        print(" Saving...")
        best_clean_acc = acc_clean
        best_bd_acc = acc_bd
        if opt.neg_rate:
            best_cross_acc = acc_cross
        else:
            best_cross_acc = torch.tensor([0])
        state_dict = {
            "model": model.state_dict(),
            "scheduler": scheduler.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_clean_acc": best_clean_acc,
            "best_bd_acc": best_bd_acc,
            "best_neg_acc": best_cross_acc,
            "epoch_current": epoch,
        }
        torch.save(state_dict, opt.ckpt_path)
        with open(os.path.join(opt.ckpt_folder, "results.txt"), "w+") as f:
            results_dict = {
                "clean_acc": best_clean_acc.item(),
                "bd_acc": best_bd_acc.item(),
                "cross_acc": best_cross_acc.item(),
            }
            json.dump(results_dict, f, indent=2)

    return best_clean_acc, best_bd_acc, best_cross_acc

def main():
    opt = config.get_arguments().parse_args()

    if opt.dataset in ["mnist", "cifar10"]:
        opt.num_classes = 10
    elif opt.dataset == "gtsrb":
        opt.num_classes = 43
    elif opt.dataset == "celeba":
        opt.num_classes = 8
    else:
        raise Exception("Invalid Dataset")

    if opt.dataset == "cifar10":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    elif opt.dataset == "gtsrb":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    elif opt.dataset == "mnist":
        opt.input_height = 28
        opt.input_width = 28
        opt.input_channel = 1
    elif opt.dataset == "celeba":
        opt.input_height = 64
        opt.input_width = 64
        opt.input_channel = 3
    else:
        raise Exception("Invalid Dataset")

    # Dataset
    train_dl, train_transform = get_dataloader(opt, True)
    test_dl, test_transform = get_dataloader(opt, False)

    # prepare model
    model, optimizer, scheduler = get_model(opt)

    # Load pretrained model
    mode = opt.attack_mode
    opt.ckpt_folder = os.path.join(opt.checkpoints, opt.dataset)
    opt.ckpt_path = os.path.join(opt.ckpt_folder, "{}_{}.pth.tar".format(opt.dataset, mode))
    opt.log_dir = os.path.join(opt.ckpt_folder, "log_dir")
    if not os.path.exists(opt.log_dir):
        os.makedirs(opt.log_dir)

    if opt.continue_training:
        if os.path.exists(opt.ckpt_path):
            print("Continue training!!")
            state_dict = torch.load(opt.ckpt_path)
            model.load_state_dict(state_dict["model"])
            optimizer.load_state_dict(state_dict["optimizer"])
            scheduler.load_state_dict(state_dict["scheduler"])
            best_clean_acc = state_dict["best_clean_acc"]
            best_bd_acc = state_dict["best_bd_acc"]
            best_cross_acc = state_dict["best_cross_acc"]
            epoch_current = state_dict["epoch_current"]
            tf_writer = SummaryWriter(log_dir=opt.log_dir)
        else:
            print("Pretrained model doesnt exist")
            exit()
    else:
        print("Train from scratch!!!")
        best_clean_acc = 0.0
        best_bd_acc = 0.0
        best_cross_acc = 0.0
        epoch_current = 0
        shutil.rmtree(opt.ckpt_folder, ignore_errors=True)
        os.makedirs(opt.log_dir)
        with open(os.path.join(opt.ckpt_folder, "opt.json"), "w+") as f:
            json.dump(opt.__dict__, f, indent=2)
        tf_writer = SummaryWriter(log_dir=opt.log_dir)
        
    residual_list_train = []
    count = 0
    
    if opt.dataset == "celeba":
        n = 1
    else:
        n = 5
        
    for j in range(n):
        for batch_idx, (inputs, targets) in enumerate(train_dl):
            print(batch_idx)
            temp_negetive = back_to_np_4d(inputs,opt)
            
            temp_negetive_modified = back_to_np_4d(inputs,opt)
            if opt.dithering:
                for i in range(temp_negetive_modified.shape[0]):
                    temp_negetive_modified[i,:,:,:] = torch.round(torch.from_numpy(floydDitherspeed(temp_negetive_modified[i].detach().cpu().numpy(),float(opt.squeeze_num))))
            else:
                temp_negetive_modified = torch.round(temp_negetive_modified/255.0*(opt.squeeze_num-1))/(opt.squeeze_num-1)*255

            residual = temp_negetive_modified - temp_negetive
            for i in range(residual.shape[0]):
                residual_list_train.append(residual[i].unsqueeze(0).cuda())
                count = count + 1
    #print(count)
    
    residual_list_test = []
    count = 0
    for batch_idx, (inputs, targets) in enumerate(test_dl):
        temp_negetive = back_to_np_4d(inputs,opt)
        residual = torch.round(temp_negetive/255.0*(opt.squeeze_num-1))/(opt.squeeze_num-1)*255 - temp_negetive
        for i in range(residual.shape[0]):
            residual_list_test.append(residual[i].unsqueeze(0).cuda())
            count = count + 1

    for epoch in range(epoch_current, opt.n_iters):
        print("Epoch {}:".format(epoch + 1))
        train(train_transform,model, optimizer, scheduler, train_dl, tf_writer, epoch, opt, residual_list_train)
        best_clean_acc, best_bd_acc, best_cross_acc = eval(
            test_transform,
            model,
            optimizer,
            scheduler,
            test_dl,
            best_clean_acc,
            best_bd_acc,
            best_cross_acc,
            tf_writer,
            epoch,
            opt,
            residual_list_test,
        )
        
        if opt.save_all:
            if (epoch)%opt.save_freq == 0:
                state_dict = {
                    "model": model.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch_current": epoch,
                }
                epoch_path = os.path.join(opt.ckpt_folder, "{}_{}_epoch{}.pth.tar".format(opt.dataset, mode,epoch))
                torch.save(state_dict, epoch_path)

if __name__ == "__main__":
    main()
