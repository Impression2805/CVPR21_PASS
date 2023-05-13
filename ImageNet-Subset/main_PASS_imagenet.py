import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
import os
import sys
import copy
import argparse
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import MultiStepLR
from ResNet_imagenet import resnet18_ImageNet
from myNetwork_imagenet import network
from data_manager_imagenet import *


torch.multiprocessing.set_sharing_strategy('file_system')
parser = argparse.ArgumentParser(description='Prototype Augmentation and Self-Supervision for Incremental Learning')
parser.add_argument('--epochs', default=101, type=int, help='Total number of epochs to run')
parser.add_argument('--batch_size', default=64, type=int, help='Batch size for training')
parser.add_argument('--print_freq', default=10, type=int, help='print frequency (default: 10)')
parser.add_argument('--data_name', default='imagenet_subset', type=str, help='Dataset name to use')
parser.add_argument('--total_nc', default=100, type=int, help='class number for the dataset')
parser.add_argument('--fg_nc', default=50, type=int, help='the number of classes in first task')
parser.add_argument('--task_num', default=10, type=int, help='the number of incremental steps')
parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--protoAug_weight', default=10.0, type=float, help='protoAug loss weight')
parser.add_argument('--kd_weight', default=10.0, type=float, help='knowledge distillation loss weight')
parser.add_argument('--temp', default=0.1, type=float, help='trianing time temperature')
parser.add_argument('--gpu', default='1', type=str, help='GPU id to use')
parser.add_argument('--save_path', default='../model_saved_check/', type=str, help='save files directory')

args = parser.parse_args()
print(args)


def main():
    cuda_index = 'cuda:' + args.gpu
    device = torch.device(cuda_index if torch.cuda.is_available() else "cpu")
    task_size = int((args.total_nc - args.fg_nc) / args.task_num)  # number of classes in each incremental step
    file_name = args.data_name + '_' + str(args.fg_nc) + '_' + str(args.task_num) + '*' + str(task_size) + 'debug'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    feature_extractor = resnet18_ImageNet()
    data_manager = DataManager()
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    train_loader = None
    test_loader = None
    old_model = None
    class_mean = None
    class_label = None
    radius = 0

    model = nn.DataParallel(network(4*args.total_nc, feature_extractor)).cuda()
    class_set = list(range(args.total_nc))
    numclass = args.fg_nc

    ###################################### train stage ###########################################
    for i in range(args.task_num + 1):
        current_task = i
        if i == 0:
            class_index = 0
        else:
            class_index = class_set[:args.fg_nc + (i - 1) * task_size]

        ############## before training ###############
        if current_task == 0:
            classes = class_set[:numclass]
        else:
            classes = class_set[(numclass - task_size):numclass]
        print(classes)

        trainfolder = data_manager.get_dataset(train_transform, index=classes, train=True)
        testfolder = data_manager.get_dataset(test_transform, index=class_set[:numclass], train=False)
        train_loader = torch.utils.data.DataLoader(
            trainfolder, batch_size=args.batch_size, shuffle=True,
            drop_last=True, num_workers=8)
        test_loader = torch.utils.data.DataLoader(
            testfolder, batch_size=args.batch_size, shuffle=False,
            drop_last=False, num_workers=8)

        if i == 0:
            base_lr = 0.1  # Initial learning rate
            lr_strat = [80, 120, 150]  # Epochs where learning rate gets decreased
            lr_factor = 0.1  # Learning rate decrease factor
            custom_weight_decay = 5e-4  # Weight Decay
            custom_momentum = 0.9  # Momentum
            opt = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=custom_momentum,
                                  weight_decay=custom_weight_decay)
            scheduler = MultiStepLR(opt, milestones=lr_strat, gamma=lr_factor)
            args.epochs = 160
            # args.epochs = 1
        else:
            opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=2e-4)
            scheduler = StepLR(opt, step_size=45, gamma=0.1)
            args.epochs = 100
            # args.epochs = 1

        ################## training ###############
        accuracy = 0
        for epoch in range(args.epochs):
            for step, data in enumerate(train_loader):
                images, target = data
                images, target = images.cuda(), target.cuda()

                images = torch.stack([torch.rot90(images, k, (2, 3)) for k in range(4)], 1)
                images = images.view(-1, 3, 224, 224)
                target = torch.stack([target * 4 + k for k in range(4)], 1).view(-1)

                model.train()
                output, feature = model(images)
                output = output[:, :4*numclass]
                # output, target = output.cuda(), target.cuda()

                loss_cls = nn.CrossEntropyLoss()(output, target)
                if old_model == None:
                    loss = loss_cls
                else:
                    _, feature_old = old_model(images)
                    loss_kd = torch.dist(feature, feature_old, 2)

                    proto_aug = []
                    proto_aug_label = []
                    index = list(range(len(class_index)))
                    for _ in range(args.batch_size):
                        np.random.shuffle(index)
                        temp = class_mean[index[0]] + np.random.normal(0, 1, 512) * radius
                        proto_aug.append(temp)
                        proto_aug_label.append(4 * class_label[index[0]])

                    proto_aug = torch.from_numpy(np.asarray(proto_aug)).cuda()
                    proto_aug_label = torch.from_numpy(np.asarray(proto_aug_label)).cuda()
                    soft_feat_aug = model.module.fc(proto_aug.float())
                    soft_feat_aug = soft_feat_aug[:, :4 * numclass]
                    loss_protoAug = nn.CrossEntropyLoss()(soft_feat_aug, proto_aug_label)
                    loss = loss_cls + args.protoAug_weight*loss_protoAug + args.kd_weight*loss_kd

                opt.zero_grad()
                loss.backward()
                opt.step()
            scheduler.step()
            if epoch % args.print_freq == 0:
                model.eval()
                correct, total = 0.0, 0.0
                for setp, data in enumerate(test_loader):
                    imgs, labels = data
                    imgs, labels = imgs.cuda(), labels.cuda()
                    with torch.no_grad():
                        outputs, _ = model(imgs)
                        outputs = outputs[:, :4 * numclass]
                        outputs = outputs[:, ::4]
                    predicts = torch.max(outputs, dim=1)[1]
                    correct += (predicts.cpu() == labels.cpu()).sum()
                    total += len(labels)
                accuracy = correct.item() / total
                print('epoch:%d,accuracy:%.5f' % (epoch, accuracy))

        ################## save prototype #####################
        model.eval()
        features = []
        features_labels = []

        max_num = min(len(train_loader), 200)
        with torch.no_grad():
            for i, data in enumerate(train_loader):
                inputs, labels = data
                if i > max_num:
                    break
                inputs = Variable(inputs.cuda())
                _, feature = model(inputs)
                if feature.shape[0] == args.batch_size:
                    features_labels.append(labels.numpy())
                    features.append(feature.cpu().numpy())

        features_labels = np.asarray(features_labels)
        features_labels = np.reshape(features_labels, features_labels.shape[0] * features_labels.shape[1])
        labels_set = np.unique(features_labels)
        features = np.asarray(features)
        features = np.reshape(features, (features.shape[0] * features.shape[1], features.shape[2]))

        class_mean_local = []
        cov_local = []
        class_label_local = []
        for item in labels_set:
            index = np.where(item == features_labels)[0]
            class_label_local.append(item)
            class_mean_local.append(np.mean(features[index], axis=0))
            cov_local.append(np.std(features[index], axis=0))

        if current_task == 0:
            radius = np.mean(cov_local)
            class_mean = class_mean_local
            class_label = class_label_local
            print(radius)
        else:
            # temp = (numclass - task_size) * cov + task_size * np.mean(cov_local)
            # cov = temp / numclass
            print(radius)
            class_mean = np.concatenate((class_mean_local, class_mean), axis=0)
            class_label = np.concatenate((class_label_local, class_label), axis=0)

        ################## after train ##################
        path = args.save_path + file_name + '/'
        if os.path.isdir(path) == False: os.makedirs(path)
        numclass += task_size
        filename = path + '%d_model.pkl' % (numclass - task_size)
        torch.save(model, filename)

        old_model = copy.deepcopy(model)
        old_model = nn.DataParallel(old_model.module).cuda()
        model = nn.DataParallel(model.module).cuda()



    ###################################### test stage ###########################################
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    print("############# Test for each Task #############")
    class_set = list(range(100))
    acc_all = []
    for current_task in range(args.task_num + 1):
        class_index = args.fg_nc + current_task * task_size
        filename = args.save_path + file_name + '/' + '%d_model.pkl' % (class_index)
        model = torch.load(filename)
        acc_up2now = []
        for i in range(current_task + 1):
            if i == 0:
                classes = class_set[:args.fg_nc]
            else:
                classes = class_set[(args.fg_nc + (i - 1) * task_size):(args.fg_nc + i * task_size)]
            testfolder = data_manager.get_dataset(test_transform, index=classes, train=False)
            test_loader = torch.utils.data.DataLoader(
                testfolder, batch_size=100,
                shuffle=False,
                drop_last=True, num_workers=4)
            correct, total = 0.0, 0.0
            for setp, data in enumerate(test_loader):
                imgs, labels = data
                imgs, labels = imgs.cuda(), labels.cuda()
                with torch.no_grad():
                    outputs, _ = model(imgs)
                    outputs = outputs[:, :4 * class_index]
                outputs = outputs[:, ::4]
                predicts = torch.max(outputs, dim=1)[1]
                correct += (predicts.cpu() == labels.cpu()).sum()
                total += len(labels)
            accuracy = correct.item() / total
            acc_up2now.append(accuracy)
        if current_task < args.task_num:
            acc_up2now.extend((args.task_num - current_task) * [0])
        acc_all.append(acc_up2now)
        print(acc_up2now)
    print(acc_all)
    a = np.array(acc_all)
    result = []
    for i in range(args.task_num + 1):
        if i == 0:
            result.append(0)
        else:
            res = 0
            for j in range(i + 1):
                res += (np.max(a[:, j]) - a[i][j])
            res = res / i
            result.append(100 * res)
    print(50 * '#')
    print('Forgetting result:')
    print(result)

    print("############# Test for up2now Task #############")
    for current_task in range(args.task_num + 1):
        class_index = args.fg_nc + current_task * task_size
        filename = args.save_path + file_name + '/' + '%d_model.pkl' % (class_index)
        model = torch.load(filename)
        model.eval()
        classes = class_set[:(args.fg_nc + current_task * task_size)]
        testfolder = data_manager.get_dataset(test_transform, index=classes, train=False)
        test_loader = torch.utils.data.DataLoader(
            testfolder, batch_size=100,
            shuffle=False,
            drop_last=False, num_workers=4)
        correct, total = 0.0, 0.0

        for setp, data in enumerate(test_loader):
            imgs, labels = data
            imgs, labels = imgs.cuda(), labels.cuda()
            with torch.no_grad():
                outputs, _ = model(imgs)
                outputs = outputs[:, :4 * class_index]
                outputs = outputs[:, ::4]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == labels.cpu()).sum()
            total += len(labels)
        accuracy = correct.item() / total
        print('incremental acc: ')
        print(accuracy)


if __name__ == "__main__":
    main()
