import numpy as np

class DecisionTree:
    def __init__(self, mat, label):
        self.mat = mat
        self.label = label

    def buildTree(self):
        label = self.label
        tmp = np.sort(self.mat, axis=1)
        original = np.copy(tmp)
        for i in range(self.mat.shape[1]-1):
            tmp[:, i] += tmp[:, i+1]
        tmp = tmp / 2
        gain_max = -1
        tar_idx = -1
        whole_gain = -1
        whole_idx = np.zeros((1, 2))
        for i in range(self.mat.shape[0]):
            gain_max = -1
            for j in range(self.mat.shape[1] - 1):
                gain = self.gain(self.mat[i], label, tmp[i, j])
                if gain > gain_max:
                    gain_max = gain
                    tar_idx = j
            if whole_gain < gain_max:
                whole_gain = gain_max
                whole_idx = np.array([i, tar_idx])

        print(whole_idx)



    def crossEntropy(self, vec):
        vec[np.where(vec == 0)] = 1
        ans = -np.sum(vec*np.log(vec))

        return ans

    def getPossibility(self, label):
        pos = np.sum(label[np.where(label == 1)])
        if len(label.shape) > 1:
            all =label.shape[1]
        else:
            all = label.shape[0]
        neg = all - pos
        p = pos/all
        ans = np.array([p, 1-p])

        return ans


    def gain(self, cur_data, cur_label, split_point):
        ent_all = self.crossEntropy(self.getPossibility(cur_label))
        partone = cur_label[np.where(cur_data <= split_point)]
        parttwo = cur_label[np.where(cur_data > split_point)]
        ent_po = self.crossEntropy(self.getPossibility(partone))
        ent_pt = self.crossEntropy(self.getPossibility(parttwo))
        d = cur_label.shape[0]
        tst = np.where(cur_data <= split_point)[0]
        dt = tst.shape[0]
        gain = ent_all - dt/d*ent_po - (d-dt)/d*ent_pt

        return gain

if __name__ == '__main__':
    data = np.array([[24, 53, 23, 25, 32, 52, 22, 43, 52, 48], [40, 52, 25, 77, 48, 110, 38, 44, 27, 65]], dtype=float)
    label = np.array([1, 0, 0, 1, 1, 1, 1, 0, 0, 1], dtype=int)
    label2 = np.array([1, 0, 1, 1, 1, 1, 0, 1], dtype=int)
    data2 = np.array([[24, 53, 25, 32, 52, 22, 43, 48], [40, 52, 77, 48, 110, 38, 44, 65]], dtype=float)
    label3 = np.array([0, 1, 0, 1], dtype=int)
    data3 = np.array([[53, 52, 43, 48], [52, 110, 44, 65]], dtype=float)
    label4 = np.array([1, 1], dtype=int)
    data4 = np.array([[52, 48], [110, 65]], dtype=float)
    handler = DecisionTree(data3, label3)
    handler.buildTree()