import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

from asn.utils.comm import tensor_to_np
from asn.utils.log import log
from asn.utils.sampler import SkillViewPairSequenceSampler


def accuracy_batch(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k
    from https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Classification/RN50v1.5/main.py"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def accuracy(data_loader, model_forward, criterion, img_keys, lable_keys, n_batch=None, use_cuda=True, writer=None,
             step_writer=None, task_names=None, plot_name="acc"):
    """ percent """
    acc_loss_top_1 = []
    criterion_loss = []

    class_predictions = {}
    num_domain_frames = 1
    repeat_cnt = 1
    if isinstance(data_loader.sampler, SkillViewPairSequenceSampler):
        num_domain_frames = data_loader.sampler._sequence_length
        repeat_cnt = 20
    data_cnt = 0
    for _ in range(repeat_cnt):
        for batch_iter, samples in enumerate(data_loader):
            inputs = torch.cat([samples[k] for k in img_keys])
            targets = torch.cat([samples[k] for k in lable_keys])
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
            outputs = model_forward(inputs)
            if num_domain_frames != 1:
                # mak out multi frames labels
                bl = inputs.size(0)
                mask = torch.ByteTensor([i % num_domain_frames == 0 for i in range(bl)]).cuda()
                targets = targets.masked_select(mask)
            loss = criterion(outputs, targets)
            # measure accuracy and record loss
            prec1 = accuracy_batch(outputs.data, targets.data, topk=(1,))[0]
            if writer is not None:
                so = F.softmax(outputs, dim=1)
                # so=outputs
                for c, p in zip(tensor_to_np(targets.detach()), tensor_to_np(so.detach())):
                    # p=p/np.mean(p)
                    data_cnt += 1
                    if c not in class_predictions:
                        class_predictions[c] = p
                    else:
                        class_predictions[c] += p

            acc_loss_top_1.append(tensor_to_np(prec1))
            criterion_loss.append(tensor_to_np(loss.detach()))
            if n_batch is not None and batch_iter > n_batch:
                break
    assert len(criterion_loss), "not data found"
    if writer is not None:
        tb_write_class_dist(writer, class_predictions, data_cnt, step_writer, task_names, plot_name)
    return np.mean(acc_loss_top_1), np.mean(criterion_loss)


def tb_write_class_dist(writer, class_predictions, data_cnt, step_writer, task_names, plot_name):
    if 0 not in class_predictions:
        log.warn("distri class not starts with label zero, missing classe in dataloader!")
    cm = [d / data_cnt for c, d in sorted(class_predictions.items())]
    cm = np.vstack(cm)
    fig, ax = plt.subplots()
    if task_names is None:
        class_names = [str(c) for c in class_predictions.keys()]
    else:
        class_names = task_names
        print('class_names: {}'.format(class_names))

    plot_confusion_matrix(ax, cm, class_names=class_names)
    fig.tight_layout()
    writer.add_figure(plot_name, fig, step_writer)


def plot_confusion_matrix(ax, confusion_matrix,
                          class_names=None,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Oranges):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    based on : https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if not title:
        if normalize:
            title = 'Normalized class distribustion'
        else:
            title = 'class distribustion, without normalization'
    if confusion_matrix.shape[0] != confusion_matrix.shape[1]:
        log.warn("foun classes not same as predictions")
    # Compute confusion matrix
    cm = confusion_matrix
    # Only use the labels that appear in the data
    if class_names is None:
        classes = [str(i) for i in range(confusion_matrix.shape[0])]
    else:
        classes = class_names
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    # fmt = '.2f' if normalize else 'd'
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    return ax
