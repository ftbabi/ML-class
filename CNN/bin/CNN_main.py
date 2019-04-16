import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

import numpy as np
import logging
import time
import argparse
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

from CNN.bin.dataconfig import DataConfig


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # input 1 channel
        # output 10 channels
        # kernel 5*5
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=16,
                      kernel_size=5,
                      stride=1,
                      padding=2
                      ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=5,
                      stride=1,
                      padding=2
                      ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc = nn.Linear(32*7*7, 10)

    def forward(self, x):
        # batch*28*28
        in_size = x.size(0)
        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.maxpool(F.relu(self.conv1(x)))  # batch * 14*14*16
        # x = self.maxpool(F.relu(self.conv2(x))) # batch * 7*7*32
        x = x.view(in_size, -1)
        output = self.fc(x)

        return output, x


class NetPlus(nn.Module):
    def __init__(self):
        super(NetPlus, self).__init__()
        self.layer1 = nn.Sequential(
            # 28, 28, 1
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                stride=1,
            ),
            nn.ReLU(),
            # 26, 26, 32
            nn.BatchNorm2d(32),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,  # 卷积filter, 移动块长
                stride=1,  # filter的每次移动步长
            ),
            # 24, 24, 32
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=4,  # 卷积filter, 移动块长
                stride=2,  # filter的每次移动步长
                padding=1,
            ),
            # 12, 12, 32
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.4),

        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,  # 卷积filter, 移动块长
                stride=1,  # filter的每次移动步长
            ),
            # 10, 10, 64
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,  # 卷积filter, 移动块长
                stride=1,  # filter的每次移动步长
            ),
            # 8, 8, 64
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=4,  # 卷积filter, 移动块长
                stride=2,  # filter的每次移动步长
                padding=1,
            ),
            # 4, 4, 64
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.4),
        )
        self.layer3 = nn.Linear(64 * 4 * 4, 10)

    def forward(self, x):
        # print(x.shape)
        x = self.layer1(x)
        # print(x.shape)
        x = self.layer2(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        output = self.layer3(x)
        # print(x.shape)
        return output, x


class NetPlusPlus(nn.Module):
    def __init__(self):
        super(NetPlusPlus, self).__init__()
        # 28, 28
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=64,
                      kernel_size=4,
                      stride=1,
                      padding=2
                      ),
            # 28, 28
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2)
            # 14, 14, 64
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=3,
                      stride=1
                      ),
            # 12, 12, 128
            nn.ReLU(),
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=1,
                      stride=1
                      ),
            # 12, 12, 128
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=5,
                      stride=1
                      ),
            # 8, 8, 128
            nn.ReLU(),
            nn.MaxPool2d(2)
            # 4, 4, 128
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128 * 4 * 4, 128*4*8),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128*4*8, 128*4*8),
            nn.ReLU(inplace=True),
            nn.Linear(128*4*8, 10)
        )


    def forward(self, x):
        # batch*28*28
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        output = self.classifier(x)

        return output, x


class Handler:
    def __init__(self):
        self.dataConfig = DataConfig()

        # init log
        logging.basicConfig(level=logging.DEBUG,  # 定义输出到文件的log级别，
                            format='%(asctime)s  %(filename)s : %(levelname)s  %(message)s',  # 定义输出log的格式
                            datefmt='%Y-%m-%d %A %H:%M:%S',  # 时间
                            filename=self.dataConfig.getLogFile(),  # log文件名
                            filemode='a')
        logging.info("=================================*=================================")
        logging.info("Create Handler")

        # init train data set
        self.train_dataset = datasets.MNIST(root=self.dataConfig.getTestRoot(),
                                            train=True,
                                            transform=transforms.ToTensor(),
                                            download=self.dataConfig.getDownLoadMnist())

        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=self.dataConfig.getBatchSize(),
                                           shuffle=True)

        # init test data set
        self.test_dataset = datasets.MNIST(root=self.dataConfig.getTestRoot(),
                                           train=False,
                                           transform=transforms.ToTensor(),
                                           download=self.dataConfig.getDownLoadMnist())

        self.test_dataloader = DataLoader(dataset=self.test_dataset,
                                          batch_size=self.dataConfig.getBatchSize(),
                                          shuffle=False)


        # init net
        self.cnn = NetPlusPlus()
        print(self.cnn)
        # use GPUs
        if self.dataConfig.getUseGPU():
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            if torch.cuda.device_count() > 1:
                print("Use ", torch.cuda.device_count(), " GPUs!")
                self.cnn = nn.DataParallel(self.cnn)
        else:
            self.device = torch.device("cpu")

        self.cnn.to(self.device)

        logging.info("==> Using device: %8s" % self.device)

        # init optimizer
        self.optimizer = optim.Adam(self.cnn.parameters(), lr=self.dataConfig.getLearningRate())
        # self.optimizer = optim.SGD(self.cnn.parameters(), lr=self.dataConfig.getLearningRate())
        # self.optimizer = optim.Adagrad(self.cnn.parameters(), lr=self.dataConfig.getLearningRate())
        # self.optimizer = optim.Adadelta(self.cnn.parameters(), lr=self.dataConfig.getLearningRate())
        # self.optimizer = optim.Adamax(self.cnn.parameters(), lr=self.dataConfig.getLearningRate())
        # self.optimizer = optim.ASGD(self.cnn.parameters(), lr=self.dataConfig.getLearningRate())

        # for var_name in self.optimizer.state_dict():
        #     print(var_name, "\t", self.optimizer.state_dict()[var_name])


        # init loss function
        self.loss_func = nn.CrossEntropyLoss()

    def parse_args(self):
        parser = argparse.ArgumentParser('CNN on MNIST')
        model_settings = parser.add_argument_group('model settings')
        model_settings.add_argument('--savepath', type=str, default="", help='save your model in savepath')
        model_settings.add_argument('--loadpath', type=str, default="", help='load your model in loadpath')
        model_settings.add_argument('--train', type=bool, default=False, help='train or not')
        model_settings.add_argument('--test', type=bool, default=True, help='test or not')
        model_settings.add_argument('--gpu', type=bool, default=True, help='use GPUs or not')
        model_settings.add_argument('--epoch', type=int, default=1, help='train epoch')

        return parser.parse_args()

    def plot(self, epoch, train_loss, acc, test_loss):
        plt.figure(num=1)
        plt.plot(epoch, train_loss)
        plt.figure(num=2)
        plt.plot(epoch, acc)
        plt.figure(num=3)
        plt.plot(epoch, test_loss)
        plt.show()

    def train(self):
        plt_train_epoch = []
        plt_train_loss = []
        plt_test_acc = []
        plt_test_loss = []

        print('Begin training......')
        logging.info("Begin training")
        start = time.time()
        training_loss = 0.0

        for epoch in range(self.dataConfig.getEpoch()):
            for step, data in enumerate(self.train_dataloader, 0):
                # get the inputs
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                # forward
                outputs, _ = self.cnn(inputs)
                loss = self.loss_func(outputs, labels)
                loss.backward()
                self.optimizer.step()

                training_loss = loss.item()
                if step % 50 == 49:
                    print('[%d, %5d] loss: %.6f' %
                          (epoch + 1, step + 1, training_loss / 2000))
                    acc, test_loss = self.test()
                    plt_train_epoch.append(step+epoch*1000)
                    plt_train_loss.append(training_loss/2000)
                    plt_test_acc.append(acc)
                    plt_test_loss.append(test_loss/2000)

                    # print("Outside: input size", inputs.size(),
                    #       "output_size", outputs.size())

        dur = time.time() - start
        print('Finished Training, Using time: %d' % dur)
        logging.info("==> Time cost: %d, loss: %6f, epoch: %d" % (dur, training_loss, self.dataConfig.getEpoch()))

        if self.dataConfig.getDraw():
            self.savePltFile(plt_train_epoch, plt_train_loss, plt_test_loss, plt_test_acc)
            self.plot(plt_train_epoch, plt_train_loss, plt_test_acc, plt_test_loss)


    def test(self):
        test_loss = 0
        step = 0
        print("Test begin...")
        logging.info("Test begin")

        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        # graph_data = []

        with torch.no_grad():
            # without derivative

            # dataiter = iter(testloader)
            # images, labels = dataiter.next()
            for data in self.test_dataloader:
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)

                # graph_data.append(images)

                outputs, _ = self.cnn(images)
                loss = self.loss_func(outputs, labels)
                test_loss += loss.item()
                step += 1

                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                size = labels.size(0)
                for i in range(size):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        total_c = np.sum(np.array(class_correct))
        total_t = np.sum(np.array(class_total))


        # for i in range(10):
        #     print('Accuracy of %5s : %2f %%' % (
        #         str(i), 100 * float(class_correct[i]) / class_total[i]))

        acc = 100 * float(total_c) / total_t
        print('Accuracy of the network on the 10000 test images: %f %%' % acc)
        logging.info("==> Total acc: %6f %%" % acc)

        return acc, float(test_loss)/step

    def saveModel(self):
        print("Save model: ", self.dataConfig.getModelFile())
        logging.info("Save model: %s" % self.dataConfig.getModelFile())
        torch.save({'model_state_dict': self.cnn.module.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()
                    }
                   , self.dataConfig.getModelFile())

    def savePltFile(self, ep, trainl, testl, acc):
        with open(self.dataConfig.getPltFile(), 'w') as f:
            for i in ep:
                f.write(str(i))
                f.write('\t')
            f.write('\n')
            for i in trainl:
                f.write(str(i))
                f.write('\t')
            f.write('\n')
            for i in testl:
                f.write(str(i))
                f.write('\t')
            f.write('\n')
            for i in acc:
                f.write(str(i))
                f.write('\t')
            f.write('\n')


    def loadModel(self):
        print("Load model: ", self.dataConfig.getModelFile())
        logging.info("Load model: %s" % self.dataConfig.getModelFile())
        modelfile = torch.load(self.dataConfig.getModelFile(), map_location=self.device)
        self.cnn.module.load_state_dict(modelfile['model_state_dict'])
        self.optimizer.load_state_dict(modelfile['optimizer_state_dict'])

        self.cnn.to(self.device)

    def run(self):
        args = self.parse_args()

        if args.train:
            self.train()
        if args.loadpath:
            self.loadModel()
        if args.savepath:
            self.saveModel()
        if args.test:
            self.test()

    def showGraph(self):
        with SummaryWriter(comment='CNN') as w:
            cnn = Net()
            for data in self.test_dataloader:
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                break
            w.add_graph(cnn, images)


if __name__ == '__main__':
    dummy_input = Variable(torch.rand(13, 1, 28, 28))

    model = Net()
    with SummaryWriter(comment='Net1') as w:
        w.add_graph(model, (dummy_input,))
    # handler = Handler()
    # handler.train()
    # # handler.test()
    # handler.saveModel()

    # handler.loadModel()
    # handler.test()