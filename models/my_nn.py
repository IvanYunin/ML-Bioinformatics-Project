import mxnet as mx
import numpy as np
import math

def split_up(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return np.asarray(out)

def data_10_classes(data, batch_size):
    train_label_new =  data['train_label']
    test_label_new = data['test_label']
    
    for idx in range(len(train_label_new)):
        if train_label_new[idx] <= 100:
            train_label_new[idx] = int(train_label_new[idx] / 10)
        else:
            train_label_new[idx] = 9
    
    for idx in range(len(test_label_new)):
        if test_label_new[idx] <= 100:
            test_label_new[idx] = int(test_label_new[idx] / 10)
        else:
            test_label_new[idx] = 9
            
    train_iter = mx.io.NDArrayIter(data['train_data'], train_label_new,
                               batch_size, shuffle = True)
    val_iter = mx.io.NDArrayIter(data['test_data'],
                             test_label_new, batch_size)
    return [train_iter, val_iter]


def create_fcnn(num_classes):
    data = mx.sym.var('data')
    fc1 = mx.sym.FullyConnected(data = data, num_hidden = 10000)
    reLU_activate=mx.sym.relu(data=fc1, name='relu')
    fc2=mx.sym.FullyConnected(data = reLU_activate, num_hidden = num_classes)
    output_layer = mx.sym.SoftmaxOutput(data = fc2, name = 'softmax')
    model = mx.mod.Module(symbol = output_layer, data_names = ['data'], context = mx.cpu())
    return model


def create_log_reg_fcnn():
    data = mx.sym.Variable('data')
    target = mx.sym.Variable('label')
    fc1 = mx.sym.FullyConnected(data = data, num_hidden = 10000, name ='fc1')
    reLU_activate=mx.sym.relu(data=fc1, name='relu')
    fc2=mx.sym.FullyConnected(data = reLU_activate, num_hidden = 5000)
    reLU_activate2=mx.sym.relu(data=fc2, name='relu')
    fc2=mx.sym.FullyConnected(data = reLU_activate2, num_hidden = 1)
    output_data = mx.sym.LogisticRegressionOutput(data=fc2, label=target)
    mod = mx.mod.Module(symbol=output_data, data_names=['data'], label_names=['label'], context=mx.cpu())
    return mod


def cross_val(model, data, batch_size, learning_rate,num_epoch, ratio):
    acc_list = []
    raw_data = np.concatenate((data['test_data'],data['train_data']))
    raw_labels = np.concatenate((data['test_label'],data['train_label']))
    
    num_part = math.floor(1.0/(1.0-ratio))
    split_data = split_up(raw_data, num_part)
    split_labels = split_up(raw_labels, num_part)
    for i in range(len(split_data)):
        temp_model = model
        print("Part ", i ," of ", num_part)
        test_data = split_data[i]
        test_label = split_labels[i]
        train_data = np.delete(split_data,i,0)
        train_label = np.delete(split_labels,i,0)
        train_iter = mx.io.NDArrayIter(train_data,
                                       train_label,
                                       batch_size, shuffle = True)
        val_iter = mx.io.NDArrayIter(test_data,
                                     test_label, batch_size)
        temp_model.fit(train_iter,eval_data = val_iter,optimizer = 'sgd',
              optimizer_params = {'learning_rate':learning_rate},
              eval_metric = 'acc', num_epoch = num_epoch)
        acc = mx.metric.Accuracy()
        temp_model.score(test_iter, acc)
        print("Accuracy on ", i, " part: ", acc)
        acc_list.append(acc)
    
    
    print("Average accuracy: ", sum(acc_list)/(float(num_part)))
    