from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from scipy import io
import numpy as np
import torch.utils.data as data_utils
import torch
import matplotlib.pyplot as plt
from six.moves import cPickle as Pk
import warnings
import os
import gc


def make_data_loader(inp_mat_file, tar_mat_file, key, batch_size, shuffle=True):
    dict1 = io.loadmat(inp_mat_file)
    features = dict1.get(key)
    features = np.float32(features)
    dict2 = io.loadmat(tar_mat_file)
    targets = dict2.get(key+'_la')
    targets = np.float32(targets)
    dat = data_utils.TensorDataset(torch.Tensor(features), torch.Tensor(targets))
    data_loader = data_utils.DataLoader(dat, batch_size=batch_size, shuffle=shuffle)
    return data_loader


def load_mat(inp_mat_file, tar_mat_file, key):
    dict1 = io.loadmat(inp_mat_file)
    features = dict1.get(key)
    dict2 = io.loadmat(tar_mat_file)
    targets = dict2.get('val_lab')
    return features, targets


def to_dense(sparse, numclass):
    sparse = np.squeeze(sparse)
    for c in range(numclass):
        s = sparse[c, :, :]
        s[s > 0] = c
        sparse[c, :, :] = s
    return np.sum(sparse, 0)


def convert_target(targets, numclass):
    sh = targets.shape
    sparse_target = np.zeros([sh[0], numclass, sh[1], sh[2]])


def to_variable(x):
        """Convert tensor to variable."""
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x)


def iouloss(outs, labs):
    outs = outs.view([-1])
    labs = labs.view([-1])
    inter = torch.sum(outs*labs)
    union = torch.sum(((outs + labs) - (outs*labs)))
    ls = 1.0 - (inter/union)
    return ls


def compute_pixel_accuracy(targets, predicts):
    assert predicts.shape == targets.shape

    sub = np.subtract(np.reshape(predicts, [-1]), np.reshape(targets, [-1]))
    num_err = np.count_nonzero(sub)

    error = 100.00*(num_err/(len(sub)))
    return error


def compute_iou(pred, lab, c):
    indlab = np.where(lab.flatten() == c)[0]
    if len(indlab) == 0:
        return -1

    indpred = np.where(pred.flatten() == c)[0]
    tp = np.intersect1d(indlab, indpred)
    fp = np.setdiff1d(indlab, indpred)
    fn = np.setdiff1d(indpred, indlab)

    iou = len(tp)/(len(fp)+len(tp)+len(fn))

    return iou


def cal_miou(preds, labs, num_class):
    assert preds.shape == labs.shape
    ps = np.reshape(preds, [-1])
    ls = np.reshape(labs, [-1])

    miou = 0
    n = 0
    for i in range(0, num_class):
        iou = compute_iou(ps, ls, i)
        if iou >= 0:
            miou += iou
            n += 1
    return 100.00*(miou/n)


def load():
    net = torch.load('./model/FCN8/FCN8')

    try:
        with open('./model/FCN8/'+'info.pk', 'rb') as f:
            info = Pk.load(f)
            net.set_cepoch(info[0])
    except FileNotFoundError:
        print('Cannot load info file')
    print('Resumed')
    return net


class FCN8(nn.Module):

    def __init__(self, numclass):
        self.__cepoch = 0
        super(FCN8, self).__init__()
        self.conv11 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv21 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv22 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv31 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv32 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv33 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv41 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv42 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv43 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        self.conv51 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv52 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv53 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv6 = nn.Conv2d(in_channels=512, out_channels=4096, kernel_size=7, padding=3)
        self.dropout1 = nn.Dropout(0.85)

        self.conv7 = nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1)
        self.dropout2 = nn.Dropout(0.85)

        self.conv8 = nn.Conv2d(in_channels=4096, out_channels=numclass, kernel_size=1)

        self.tranconv1 = nn.ConvTranspose2d(in_channels=numclass, out_channels=512, kernel_size=4, stride=2, padding=1)

        self.tranconv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=2, output_padding=1)

        self.tranconv3 = nn.ConvTranspose2d(in_channels=256, out_channels=numclass, kernel_size=16, stride=8, padding=4)

    def set_cepoch(self, ce):
        self.__cepoch = ce

    def forward(self, x):
        x = self.conv12(self.conv11(x))
        x = self.pool1(x)
        x = self.pool2(self.conv22(self.conv21(x)))
        x1 = self.pool3(self.conv33(self.conv32(self.conv31(x))))
        x2 = self.pool4(self.conv43(self.conv42(self.conv41(x1))))
        x = self.pool5(self.conv53(self.conv52(self.conv51(x2))))
        x = self.dropout1(self.conv6(x))
        x = self.dropout2(self.conv7(x))
        x = self.conv8(x)
        x = self.tranconv1(x)
        x = x2 + x
        x = self.tranconv2(x)
        x = x1 + x
        x = self.tranconv3(x)
        return x

    def save(self, info):
        if not os.path.exists('./model/'+self.__class__.__name__):
            os.makedirs('./model/'+self.__class__.__name__)

        torch.save(self, './model/'+self.__class__.__name__+'/'+self.__class__.__name__)
        with open('./model/'+self.__class__.__name__+'/'+'info.pk', 'wb') as f:
            Pk.dump(info, f, Pk.HIGHEST_PROTOCOL)
        print('     Saved model...')

    def test(self):
        testloader = make_data_loader('./matfiles/va.mat', './matfiles/va_la.mat', 'va', 80, shuffle=False)
        _, labs = load_mat('./matfiles/va.mat', './matfiles/val_lab.mat', 'va')
        preds = None
        for j, data in enumerate(testloader, 0):
            inps, _ = data

            inps = to_variable(inps)
            _,  ps = torch.max(self(inps), 1)
            ps = ps.data.cpu().numpy()
            if preds is None:
                preds = ps
            else:
                preds = np.vstack((preds, ps))

        print('Pixel error is: ', compute_pixel_accuracy(labs, preds))
        print('Mean Intersection over Union: ', cal_miou(preds, labs, 21))

    def train(self, lr=0.00001, epoch=10000):
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        trainloader = make_data_loader('./matfiles/tr.mat', './matfiles/tr_la.mat', 'tr', 120)

        for ep in range(self.__cepoch, epoch):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                inputs = to_variable(inputs)
                labels = to_variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self(inputs)

                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.data[0]
                if i % 10 == 0:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %f' %
                          (ep + 1, i + 1, loss.data[0]))

                    _,  preds = torch.max(outputs, 1)

                    plt.ion()
                    plt.subplot(1, 3, 1)
                    img = np.swapaxes(np.swapaxes(np.uint8(np.squeeze(inputs[0, :, :, :].data.cpu().numpy())), 0, 2), 0, 1)
                    plt.imshow(img)
                    plt.title('Image')
                    plt.axis('off')
                    plt.subplot(1, 3, 2)
                    plt.title('Ground Truth')
                    gt = to_dense(np.squeeze(labels[0, :, :, :].data.cpu().numpy()), 21)
                    plt.imshow(gt)
                    plt.axis('off')
                    plt.subplot(1, 3, 3)
                    plt.title('Prediction')
                    p = np.squeeze(preds[0, :, :].data.cpu().numpy())
                    plt.imshow(p)
                    plt.axis('off')
                    plt.savefig('./samples/' + str(ep) + '_' + str(i))
                    plt.pause(0.05)
            self.save([ep+1])
            gc.collect()
        print('Finished Training')


def main():
    warnings.filterwarnings("ignore", ".*GUI is implemented.*")

    if not os.path.exists('./model/FCN8'):
        if torch.cuda.is_available():
            net = FCN8(21).cuda()
        else:
            net = FCN8(21)
    else:
        net = load()

    # net.train()
    net.test()


if __name__ == "__main__":
    main()
