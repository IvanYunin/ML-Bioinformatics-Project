import argparse
from collections import Counter
import numpy as np
import random
import mxnet as mx
import logging
from sklearn import decomposition


logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout


def load_data(annotation, attributes, metilation, batch_size, train_val_ratio):
    print('1. Reading metilation file')
    file = open(metilation, 'r')
    header = file.readline()
    metilation_table_attributes = header.split('\t')
    metilation_table_attributes.pop(0)
    cpg_sites = []
    metilation = [ [] for i in range(len(metilation_table_attributes)) ]
    for line in file:
        values = line.split('\t')
        print('\t CpG site: {}'.format(values[0]))
        cpg_sites.append(values.pop(0))

        for i in range(len(values)):
            metilation[i].append(float(values[i]))
    file.close()

    print('2. Reading attributes file')
    file = open(attributes, 'r')
    header = file.readline()
    labels_table_attributes = header.split(' ')
    ages = []
    for line in file:
        values = line.split(' ')
        ages.append(int(values[2])) # age column
    file.close()

    # a set of possible ages
    cdages = list(Counter(ages).keys())
    cdages.sort()

    print('3. Generating partitioning of train and validation datasets')
    train_indices = random.sample(range(len(ages)),
        int(train_val_ratio * len(ages)))
    val_indices = [i for i in range(len(ages)) if i not in train_indices]

    print('4. Generating train dataset')
    train_data = [ metilation[train_indices[idx]] for idx in range(len(train_indices)) ]
    train_labels = []
    for idx in range(len(train_indices)):
        train_labels.append(cdages.index(ages[train_indices[idx]]))

    print('5. Generating validation dataset')
    val_data = [ metilation[val_indices[idx]] for idx in range(len(val_indices)) ]
    val_labels = []
    val_ages = []
    for idx in range(len(val_indices)):
        val_labels.append(cdages.index(ages[val_indices[idx]]))
        val_ages.append(ages[val_indices[idx]])

    print('6. Creating iterators for train and validation datasets')
    train_iter = mx.io.NDArrayIter(
        data = np.array(train_data, dtype = np.float32),
        label = np.array(train_labels, dtype = np.float32),
        batch_size = batch_size, shuffle = True,
        data_name = 'data', label_name = 'labels')
    print(train_iter.provide_data)
    print(train_iter.provide_label)
    
    val_iter = mx.io.NDArrayIter(
        data = np.array(val_data, dtype = np.float32),
        label = np.array(val_labels, dtype = np.float32),
        batch_size = batch_size, data_name = 'data', label_name = 'labels')
    print(val_iter.provide_data)
    print(val_iter.provide_label)

    return [train_iter, val_iter, cdages, val_ages]


def create_symbol_fcnn(num_classes):
    # 450 000 CpG sites
    data = mx.sym.var('data')
    # one-hot vector (74 elements)
    labels = mx.sym.var('labels')
    # Normalization layer (by instance)
    norm_data = mx.sym.L2Normalization(data = data, name = 'norm_data')
    # 5000 neurons
    fc1 = mx.sym.FullyConnected(data = norm_data, num_hidden = 5000)
    smfc1 = mx.sym.SoftmaxActivation(data = fc1, name = 'smfc1')
    # 1000 neurons
    fc2 = mx.sym.FullyConnected(data = smfc1, num_hidden = 1000)
    smfc2 = mx.sym.SoftmaxActivation(data = fc2, name = 'smfc2')
    # age
    fcc = mx.sym.FullyConnected(data = smfc2, num_hidden = num_classes)
    smce = mx.sym.SoftmaxOutput(data = fcc, label = labels, name = 'smce')

    fc_model = mx.mod.Module(symbol = smce, data_names = ['data'],
        label_names = ['labels'], context = mx.cpu())
    return fc_model


def save_predictions(filename, val_labels, classes, predictions):
    nppred = predictions.asnumpy()
    f = open(filename, 'w')
    for i in range(len(val_labels)):
        f.write('{0} {1}\n'.format(val_labels[i],
            classes[list(nppred[i]).index(max(nppred[i]))]))
    f.close()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-an', '--annotations', help = 'annotation file name',
        default = 'annotations.txt')
    parser.add_argument('-at', '--attributes', help = 'attributes file name',
        default = 'attributes.txt')
    parser.add_argument('-mt', '--metilation', help = 'metilation file name',
        default = 'GSE40279_average_beta.txt')
    parser.add_argument('-bs', '--batch_size', help = 'batch size', type = int,
        default = 1)
    parser.add_argument('-lr', '--learning_rate', help = 'learning rate',
        type = float, default = 0.1)
    parser.add_argument('-ne', '--num_epoch', help = 'epoch count',
        type = int, default = 1)
    parser.add_argument('-r', '--ratio', help = 'train/val datasets ratio',
        type = float, default = 0.8)
    parser.add_argument('-p', '--predictions',
        help = 'file name for saving predictions', default = 'predictions.csv')
    args = parser.parse_args()

    print('Loading train/val data')
    [train_iter, val_iter, classes, val_labels] = load_data(args.annotations,
        args.attributes, args.metilation, args.batch_size, args.ratio)

    print('Constructing network model')
    fc_model = create_symbol_fcnn(len(classes))

    print('Training network')
    model_prefix = 'fcnn_cl_5000_1000_74'
    # epoch_end_callback = mx.callback.do_checkpoint(model_prefix, period = 2),
    accuracy = mx.metric.Accuracy()
    ce_loss = mx.metric.CrossEntropy()
    fc_model.fit(train_iter, eval_data = val_iter, optimizer = 'sgd',
        optimizer_params = {'learning_rate': args.learning_rate},
        eval_metric = mx.metric.CompositeEvalMetric([accuracy, ce_loss]),
        batch_end_callback = mx.callback.Speedometer(args.batch_size),
        num_epoch = args.num_epoch)

    print('Testing network')
    predictions = fc_model.predict(val_iter)
    save_predictions(args.predictions, val_labels, classes, predictions)
    print('Predictions shape: {}'.format(predictions.shape))
    metric = mx.metric.Accuracy()
    fc_model.score(val_iter, metric)
    print('Accuracy: {}'.format(metric))
