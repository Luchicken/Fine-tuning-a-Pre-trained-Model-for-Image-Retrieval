import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import alexnet, resnet50
from data_loader import load_train
import time
import os
import numpy as np
import pickle
from args import get_args


args = get_args()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def load_model(model_name='alexnet'):
    """加载模型"""
    if model_name == 'alexnet':
        model = alexnet(pretrained=True)
        n_features = model.classifier[6].in_features  # 4096
        if args.latent_layer:
            Latent_Layer = torch.nn.Linear(n_features, args.latent_size)
            fc = torch.nn.Linear(args.latent_size, args.num_classes)
            model.classifier = nn.Sequential(*list(model.classifier.children())[:-1], Latent_Layer, nn.Sigmoid(), fc)
        else:
            fc = torch.nn.Linear(n_features, args.num_classes)
            model.classifier[6] = fc  # 替换最后一层全连接层
    elif model_name == 'resnet':
        model = resnet50(pretrained=True)
        n_features = model.fc.in_features  # 2048
        fc = torch.nn.Linear(n_features, args.num_classes)
        model.fc = fc  # 替换最后一层全连接层
        model.fc.weight.data.normal_(0, 0.005)  # 初始化权重
        model.fc.bias.data.fill_(0.1)  # 初始化偏置
    return model


def get_optimizer(model_name='alexnet'):
    """获取优化器"""
    learning_rate = args.lr
    if model_name == 'alexnet':
        param_group = [{'params': model.features.parameters(), 'lr': learning_rate}]
        for i in range(6):
            param_group += [{'params': model.classifier[i].parameters(), 'lr': learning_rate}]  # 前6层学习率
        if args.latent_layer:
            for i in range(6, 9):
                param_group += [{'params': model.classifier[i].parameters(), 'lr': learning_rate * 10}]  # 后3层学习率 * 10
        else:
            param_group += [{'params': model.classifier[6].parameters(), 'lr': learning_rate * 10}]  # 最后1层学习率 * 10
    elif model_name == 'resnet':
        param_group = []
        for k, v in model.named_parameters():
            if not k.__contains__('fc'):
                param_group += [{'params': v, 'lr': learning_rate}]  # 非全连接层学习率
            else:
                param_group += [{'params': v, 'lr': learning_rate * 10}]  # 全连接层学习率 * 10
    optimizer = optim.SGD(param_group, momentum=args.momentum)
    return optimizer


def finetune(model, dataloaders, optimizer):
    """微调模型"""
    since = time.time()  # 开始时间
    best_acc = 0  # 最佳准确率
    criterion = nn.CrossEntropyLoss()  # 损失函数
    stop = 0  # 早停
    save_model = f'model_{args.model}_{time.strftime("%Y%m%d-%H%M%S", time.localtime(since))}.pkl'  # 模型保存名称
    ckpt = './ckpt'  # 模型保存路径
    os.makedirs(ckpt, exist_ok=True)
    log_file = f'model_{args.model}_{time.strftime("%Y%m%d-%H%M%S", time.localtime(since))}.pkl'  # 模型保存名称
    hist = './hist'  # 损失、准确率保存路径
    os.makedirs(hist, exist_ok=True)

    loss_hist = []  # 损失历史
    acc_hist = []  # 准确率历史
    for epoch in range(1, args.num_epochs + 1):
        stop += 1
        # You can uncomment this line for scheduling learning rate
        # lr_schedule(optimizer, epoch)
        for phase in ['train']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            total_loss, correct = 0., 0.  # 总损失、正确数
            for images, labels in dataloaders[phase]:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images)  # 前向传播
                    loss = criterion(outputs, labels)  # 计算损失
                predictions = outputs.max(1)[1]  # 预测
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item() * images.size(0)
                correct += torch.sum(predictions == labels).item()
            epoch_loss = total_loss / len(dataloaders[phase].dataset)
            epoch_acc = correct / len(dataloaders[phase].dataset)
            loss_hist.append(epoch_loss)  # 记录损失
            acc_hist.append(epoch_acc)  # 记录准确率
            print('Epoch: [{:02d}/{:02d}]---{}, loss: {:.6f}, cls acc: {:.4f}'.format(
                epoch, args.num_epochs, phase, epoch_loss, epoch_acc))
            if phase == 'train' and epoch_acc > best_acc:
                stop = 0
                best_acc = epoch_acc  # 更新最佳准确率
                torch.save(model.state_dict(), os.path.join(ckpt, save_model))  # 保存模型
        if stop >= args.early_stop:
            break  # 早停
        print()
    time_pass = time.time() - since  # 训练时间
    print('Training complete in {:.0f}m {:.0f}s'.format(time_pass // 60, time_pass % 60))
    # 保存list变量到文件
    with open(os.path.join(hist, log_file), 'wb') as f:
        pickle.dump({
            'train_loss': loss_hist,
            'train_acc': acc_hist,
            'time_cost': time_pass
        }, f)
    return model


if __name__ == '__main__':
    # 设置随机种子
    set_seed(args.seed)
    # 加载数据
    dataloaders = {}
    # dataloaders['train'], dataloaders['val'] = load_train(args.data, 'base', args.batchsize)
    dataloaders['train'] = load_train(args.data, 'base', args.batchsize)
    # 加载模型 (AlexNet / ResNet-50)
    model_name = args.model
    model = load_model(model_name).to(DEVICE)
    # 定义优化器
    optimizer = get_optimizer(model_name)
    print('Train: {}, model: {}'.format(len(dataloaders['train'].dataset), model_name))
    model_best = finetune(model, dataloaders, optimizer)
