# Bölüm 1 - Başlangıç
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# CIFAR-10 veri setini yükleyin
def load_cifar10_data():
    data = {}
    for batch in range(1, 6):
        with open(f'cifar-10-batches-py/data_batch_{batch}', 'rb') as file:
            batch_data = pickle.load(file, encoding='bytes')
        if batch == 1:
            data['images'] = batch_data[b'data']
            data['labels'] = batch_data[b'labels']
        else:
            data['images'] = np.vstack((data['images'], batch_data[b'data']))
            data['labels'].extend(batch_data[b'labels'])
    with open('cifar-10-batches-py/test_batch', 'rb') as file:
        test_data = pickle.load(file, encoding='bytes')
    data['test_images'] = test_data[b'data']
    data['test_labels'] = test_data[b'labels']
    return data

# CIFAR-10 veri setini yükle
cifar10_data = load_cifar10_data()

# Sınıf etiketleri
class_names = ['Uçak', 'Otobüs', 'Kuş', 'Kedi', 'Geyik', 'Köpek', 'Kurbağa', 'At', 'Gemi', 'Kamyon']

# Örnek görüntülerin gösterilmesi
plt.figure(figsize=(10, 5))
for i in range(10):
    image = cifar10_data['images'][i]
    image = np.transpose(np.reshape(image, (3, 32, 32)), (1, 2, 0))
    label = cifar10_data['labels'][i]
    plt.subplot(2, 5, i+1)
    plt.imshow(image)
    plt.title(class_names[label])
    plt.axis('off')
plt.tight_layout()
# plt.show()

# Bölüm 1 - Bitiş

# Bölüm 2 - Başlangıç


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms



# CIFAR-10 veri kümesini yükleyin ve ön işleyin
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=2)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Ağ modelini tanımlayın
class Network(nn.Module):
    def init(self, use_batch_norm):
        super(Network, self).init()
        self.use_batch_norm = use_batch_norm
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.relu = nn.ReLU()
        if self.use_batch_norm:
            self.batch_norm1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        if self.use_batch_norm:
            self.batch_norm2 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        if self.use_batch_norm:
            x = self.batch_norm1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        if self.use_batch_norm:
            x = self.batch_norm2(x)
        x = self.pool(x)
        x = x.view(-1, 128 * 8 * 8)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Ağları eğitin ve doğruluk sonuçlarını kaydedin
def train_network(use_batch_norm):
    network = Network(use_batch_norm)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.9)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network.to(device)


accuracies = []
    for epoch in range(50):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        accuracy = evaluate_accuracy(network, testloader)
        accuracies.append(accuracy)
        print(f"Epoch: {epoch+1}, Loss: {running_loss/len(trainloader):.3f}, Accuracy: {accuracy:.2f}%")
    
    return accuracies

# Doğruluk hesaplaması
def evaluate_accuracy(network, dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = network(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

if name == 'main':
    # Ağları eğit ve doğruluk sonuçlarını karşılaştır
    network_with_batch_norm_acc = train_network(use_batch_norm=True)
    network_without_batch_norm_acc = train_network(use_batch_norm=False)

    # Doğruluk sonuçlarını görselleştir
    plt.plot(range(1, 51), network_with_batch_norm_acc, label='With Batch Normalization')
    plt.plot(range(1, 51), network_without_batch_norm_acc, label='Without Batch Normalization')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Comparison')
    plt.legend()
    # plt.show()
# Bölüm 2 - Bitiş



# Bölüm 3 - Başlangıç

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# CIFAR-10 veri kümesini yükleyin ve ön işleyin
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=2)

# DenseNet modeli
class DenseNet(nn.Module):
    def init(self):
        super(DenseNet, self).init()
        self.features = nn.Sequential(
            nn.Linear(32*32*3, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.features(x)
        return x


# VGGNet modeli
class VGGNet(nn.Module):
    def init(self):
        super(VGGNet, self).init()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ResNet modeli
class ResNet(nn.Module):
    def init(self):
        super(ResNet, self).init()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.block1 = self._make_block(64, 64, 2)
        self.block2 = self._make_block(64, 128, 2)
        self.block3 = self._make_block(128, 256, 2)
        self.block4 = self._make_block(256, 512, 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 10)

    def _make_block(self, in_channels, out_channels, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(ResidualBlock(in_channels, out_channels))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class ResidualBlock(nn.Module):
    def init(self, in_channels, out_channels):
        super(ResidualBlock, self).init()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += self.shortcut(residual)
        out = self.relu(out)
        return out


# InceptionNet modeli
class InceptionNet(nn.Module):
    def init(self):
        super(InceptionNet, self).init()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception1 = InceptionModule(64, 64, 64, 64, 96, 32)
        self.inception2 = InceptionModule(256, 64, 64, 96, 128, 32)
        self.inception3 = InceptionModule(480, 128, 64, 96, 128, 64)
        self.fc = nn.Linear(768, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)
        x = torch.mean(x, dim=(2, 3))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class InceptionModule(nn.Module):
    def init(self, in_channels, conv1x1_out, conv3x3_reduce_out, conv3x3_out, conv5x5_reduce_out, conv5x5_out):
        super(InceptionModule, self).init()
        self.conv1x1 = nn.Conv2d(in_channels, conv1x1_out, kernel_size=1)
        self.conv3x3_reduce = nn.Conv2d(in_channels, conv3x3_reduce_out, kernel_size=1)
        self.conv3x3 = nn.Conv2d(conv3x3_reduce_out, conv3x3_out, kernel_size=3, padding=1)
        self.conv5x5_reduce = nn.Conv2d(in_channels, conv5x5_reduce_out, kernel_size=1)
        self.conv5x5 = nn.Conv2d(conv5x5_reduce_out, conv5x5_out, kernel_size=5, padding=2)
        self.conv_pool = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        out1 = self.conv1x1(x)
        out2 = self.conv3x3_reduce(x)
        out2 = self.conv3x3(out2)
        out3 = self.conv5x5_reduce(x)
        out3 = self.conv5x5(out3)
        out4 = self.conv_pool(x)
        out4 = nn.functional.max_pool2d(out4, kernel_size=3, stride=1, padding=1)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out

# Ağları eğitin ve doğruluk sonuçlarını karşılaştırın
def train_network(network, num_epochs=50):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.9)

    network.to(device)

    accuracies = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        accuracy = evaluate_accuracy(network, testloader)
        accuracies.append(accuracy)
        print(f"Epoch: {epoch+1}, Loss: {running_loss/len(trainloader):.3f}, Accuracy: {accuracy:.2f}%")

    return accuracies

def evaluate_accuracy(network, dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = network(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

if name == 'main':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ağları oluşturun
    dense_net = DenseNet()
    vgg_net = VGGNet()
    res_net = ResNet()
    inception_net = InceptionNet()

    # Ağları eğitin ve doğruluk sonuçlarını alın
    dense_net_acc = train_network(dense_net)
    vgg_net_acc = train_network(vgg_net)
    res_net_acc = train_network(res_net)
    inception_net_acc = train_network(inception_net)


# Sonuçları görselleştirin
    plt.plot(range(1, 51), dense_net_acc, label='DenseNet')
    plt.plot(range(1, 51), vgg_net_acc, label='VGGNet')
    plt.plot(range(1, 51), res_net_acc, label='ResNet')
    plt.plot(range(1, 51), inception_net_acc, label='InceptionNet')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Comparison of Different Network Architectures')
    plt.legend()
    plt.show()


# Bölüm 3 - Bitiş