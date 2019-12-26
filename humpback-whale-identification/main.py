import torch
import torch.nn as nn
from torch import optim
from torchvision import transforms
from torchvision import datasets

import MobileNetV2

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

input_size = 224
n_worker = 4
batch_size = 64

train_dataset = datasets.ImageFolder(
    "./train/",
    transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    num_workers=n_worker, pin_memory=True)


# val_loader = torch.utils.data.DataLoader(
#     datasets.ImageFolder(valdir, transforms.Compose([
#         transforms.Resize(int(input_size/0.875)),
#         transforms.CenterCrop(input_size),
#         transforms.ToTensor(),
#         normalize,
#     ])),
#     batch_size=batch_size, shuffle=False,
#     num_workers=n_worker, pin_memory=True)


# 开始训练，共训练 args.epochs 周期
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        #
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            criterion = nn.CrossEntropyLoss()
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == "__main__":

    epochs = 50
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    model = MobileNetV2.MobileNetV2(width_mult=1, n_class=5005).to(device)
    save_model = True
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # load params
    pretrained_dict = torch.load('mobilenetv2_1.0-f2a8633.pth.tar')

    # 获取当前网络的dict
    net_state_dict = model.state_dict()
    # print(net_state_dict)
    # 剔除不匹配的权值参数
    pretrained_dict_1 = {k: v for k, v in pretrained_dict.items() if k in net_state_dict and "classifier" not in k}

    # 更新新模型参数字典
    net_state_dict.update(pretrained_dict_1)

    # 将包含预训练模型参数的字典"放"到新模型中
    model.load_state_dict(net_state_dict)

    for epoch in range(epochs):
        # 训练
        train(model, device, train_loader, optimizer, epoch)
        # 测试代码
        # test(model, device, test_loader)

        # 如果设置保存模型，就开始保存模型，保存为 mnist_cnn.pt
        if save_model:
            torch.save(model.state_dict(), "model.pt")
