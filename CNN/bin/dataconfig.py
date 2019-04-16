import yaml
import os
import numpy as np

class DataConfig:
    def __init__(self):
        f = open('../conf/config.yaml')
        content = yaml.load(f)
        f.close()

        self.test_root = content['test_root']
        self.train_root = content['train_root']
        self.log_file = content['log_file']
        self.model_file = content['model_file']

        self.download_mnist = bool(content['download_mnist'])

        self.batch_size = content['batch_size']
        self.learning_rate = content['learning_rate']
        self.epoch = content['epoch']
        self.use_gpu = bool(content['use_gpu'])
        self.draw = bool(content['draw'])
        self.plt_file = content['plt_file']


        if not os.path.isdir(self.test_root):
            print("Test root directory not exist: ", self.test_root)
            exit(-1)

        if not os.path.isdir(self.train_root):
            print("Train file not exist: ", self.train_root)
            exit(-1)

        if not os.path.isfile(self.log_file):
            with open(self.log_file, 'w') as f:
                f.write('\n')


    def getTrainRoot(self):
        return self.train_root

    def getTestRoot(self):
        return self.test_root

    def getBatchSize(self):
        return self.batch_size

    def getDownLoadMnist(self):
        return self.download_mnist

    def getLearningRate(self):
        return self.learning_rate

    def getEpoch(self):
        return self.epoch

    def getUseGPU(self):
        return self.use_gpu

    def getLogFile(self):
        return self.log_file

    def getModelFile(self):
        return self.model_file

    def getDraw(self):
        return self.draw

    def getPltFile(self):
        return self.plt_file

if __name__ == '__main__':
    row = 'a,s,b'
    a = np.array([1,2,3,2])
    b = np.array([1,0,1,0])
    print(b>0)
    a[np.where(a == 2)] = 4
    print(a)
    print(row.split(','))
    print(a[-1])
    print(a)
    print(np.sum(a, axis=0))
    c = np.array([[1,2], [2,4]])

    # print(np.argmax(c,axis=0))
    # print(c[np.argmax(c, axis=0)])
    print(c)
    c = np.delete(c, -1, axis=1)
    print(c)