import numpy as np
import torch
import os
import torch.nn as nn
from glob import glob
from shutil import rmtree
import torchvision

from torchvision import transforms
from sklearn.metrics import average_precision_score
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from kaggle_submission import output_submission_csv
from classifier import SimpleClassifier, Classifier
from voc_dataloader import VocDataset, VOC_CLASSES

from tqdm import tqdm
import argparse

device = 'cuda:0'


def train_classifier(train_loader, classifier, criterion, optimizer):
    classifier.train()
    losses = []
    for i, (images, labels, _) in enumerate(tqdm(train_loader, leave=False, desc='Epoch', position=1)):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = classifier(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss)
    return torch.stack(losses).mean().item()


def test_classifier(test_loader, classifier, criterion, epoch=None, f=None):
    classifier.eval()
    losses = []
    y_true = np.zeros((0, 21))
    with torch.no_grad():
        y_score = np.zeros((0, 21))
        for i, (images, labels, _) in enumerate(tqdm(test_loader, desc="Test", position=2, leave=False)):
            images, labels = images.to(device), labels.to(device)
            logits = classifier(images)
            y_true = np.concatenate((y_true, labels.cpu().numpy()), axis=0)
            y_score = np.concatenate((y_score, logits.cpu().numpy()), axis=0)
            loss = criterion(logits, labels)
            losses.append(loss.item())
        aps = []
        data = ''
        # ignore first class which is background
        for i in range(1, y_true.shape[1]):
            ap = average_precision_score(y_true[:, i], y_score[:, i])
            data += '-------  Class: {:<12}     AP: {:>8.4f}  -------\n'.format(VOC_CLASSES[i], ap)
            aps.append(ap)

        mAP = np.mean(aps)
        dir_name = None
        if f is None:
            dir_name = f"stored/epoch{epoch}_{round(mAP, 3)}"
            os.mkdir(dir_name)
            f = open(os.path.join(dir_name, "train.txt"), 'w')
        f.write(data)
        test_loss = np.mean(losses)
        f.write('mAP: {0:.4f}\n'.format(mAP))
        f.write('Avg loss: {}\n'.format(test_loss))
        f.close()

    return dir_name, mAP, test_loss, aps


def plot_losses(train, val, test_frequency, num_epochs):
    plt.plot(train, label="train")
    indices = [i for i in range(num_epochs) if ((i + 1) % test_frequency == 0 or i == 0)]
    plt.plot(indices, val, label="val")
    plt.title("Loss Plot")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()


def plot_mAP(train, val, test_frequency, num_epochs):
    indices = [i for i in range(num_epochs) if ((i + 1) % test_frequency == 0 or i == 0)]
    plt.plot(indices, train, label="train")
    plt.plot(indices, val, label="val")
    plt.title("mAP Plot")
    plt.ylabel("mAP")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()


def train(classifier, num_epochs, train_loader, val_loader, criterion, optimizer, test_frequency=5):
    train_losses = []
    train_mAPs = []
    val_losses = []
    val_mAPs = []
    for file in glob('stored/*'):
        rmtree(file)
    with tqdm(range(1, num_epochs + 1), desc="Overall", position=0, leave=False) as t:
        mAP_val, val_loss, mAP_train = None, None, None
        for epoch in t:
            train_loss = train_classifier(train_loader, classifier, criterion, optimizer)
            train_losses.append(train_loss)
            t.set_postfix(refresh=False, train_loss=train_loss, train_mAP=mAP_val, val_loss=val_loss, val_mAP=mAP_val)
            if epoch % test_frequency == 0 or epoch == 1:
                dir_name, mAP_train, _, _ = test_classifier(train_loader, classifier, criterion, epoch)
                train_mAPs.append(mAP_train)
                f = open(os.path.join(dir_name, "validation.txt"), 'w')
                _, mAP_val, val_loss, _ = test_classifier(val_loader, classifier, criterion, f=f)
                val_losses.append(val_loss)
                val_mAPs.append(mAP_val)
                torch.save(classifier.state_dict(), os.path.join(dir_name, "model.pth"))

    return classifier, train_losses, val_losses, train_mAPs, val_mAPs


def main(flags):
    global device
    num_epochs, test_frequency, batch_size, device, voc_path, adam = \
        flags.epochs, flags.frequency, flags.batches, flags.device, flags.path, flags.adam
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.Resize(227),
        transforms.CenterCrop(227),
        transforms.ToTensor(),
        normalize
    ])
    test_transform = transforms.Compose([
        transforms.Resize(227),
        transforms.CenterCrop(227),
        transforms.ToTensor(),
        normalize,
    ])
    ds_train = VocDataset(f'{voc_path}/VOC2007/', 'train', train_transform)
    ds_val = VocDataset(f'{voc_path}/VOC2007/', 'val', test_transform)
    ds_test = VocDataset(f'{voc_path}/VOC2007test/', 'test', test_transform)

    train_loader = torch.utils.data.DataLoader(dataset=ds_train,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=1)

    val_loader = torch.utils.data.DataLoader(dataset=ds_val,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=1)

    test_loader = torch.utils.data.DataLoader(dataset=ds_test,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=1)

    torch.cuda.empty_cache()
    classifier = Classifier().to(device)
    # classifier = SimpleClassifier().to(device)

    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-4) if adam else torch.optim.SGD(classifier.parameters(),
                                                                                                lr=0.01, momentum=0.9)

    classifier, train_losses, val_losses, train_mAPs, val_mAPs = train(classifier, num_epochs, train_loader, val_loader,
                                                                       criterion, optimizer, test_frequency)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path')
    parser.add_argument('-d', '--device', default='cuda:0')
    parser.add_argument('-e', '--epochs', type=int, default=20)
    parser.add_argument('-b', '--batches', type=int, default=50)
    parser.add_argument('--adam', type=bool, default=False)
    parser.add_argument('-f', '--frequency', type=int, default=4)
    args = parser.parse_args()
