import mxnet as mx
import pickle
import logging
import my_nn
import argparse
import sys
import os
import math
import numpy as np


def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

        
def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
       
    
if __name__ == '__main__':   
    data = load_obj('gse_data_1')
    gse_data = dict.fromkeys(['train_data', 'train_label',
                              'test_data', 'test_label'])
    test_size = math.ceil((0.2)*len(data['label']))

    #split data into test and train
    gse_data['test_data'],gse_data['train_data'],_= np.vsplit(
        data['data'], (test_size, len(data['label'])))
    gse_data['test_label'],gse_data['train_label'],_= np.split(
        data['label'],(test_size, len(data['label'])))
    
    train_iter = mx.io.NDArrayIter(gse_data['train_data'],
                                       gse_data['train_label'],1 , shuffle = True)
    val_iter = mx.io.NDArrayIter(gse_data['test_data'],
                                     gse_data['test_label'], 1)

    model = my_nn.create_CNN(74)
    accuracy = mx.metric.Accuracy()
    progress_bar = mx.callback.ProgressBar(total=2)
    model.fit(train_iter,eval_data = val_iter,optimizer = 'sgd',
              optimizer_params = {'learning_rate': 0.1},
              eval_metric = 'acc',
              batch_end_callback = progress_bar,
              num_epoch = 10)
    logging.info('Training network DONE!')

    #start testing
    logging.info('Testing network')
    model.score(val_iter, accuracy)
    print('Accuracy: {}'.format(accuracy))
    prdct = model.predict(eval_data = val_iter)
    predict_file = open('predictions.txt', 'w') #need name from cmd
    for i in range(len(prdct)):
        max = prdct[i][0]
        pos = 0 
        for j in range(len(prdct[i])):
            if prdct[i][j]>max: max=prdct[i][j];pos=j
        predict_file.write('predict = '+str(pos)+' val = '
                           +str(gse_data['test_label'][i])+'\n')
    predict_file.close()
    print('Testing network DONE')
    
