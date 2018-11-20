import mxnet as mx
import pickle
import logging
import my_nn
import argparse
import sys

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

        
def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
       
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', '--data', help = 'data file name',
        default = 'gse40249_data')
    parser.add_argument('-bs', '--batch_size', help = 'batch size', type = int,
        default = 1)
    parser.add_argument('-lr', '--learning_rate', help = 'learning rate',
        type = float, default = 0.1)
    parser.add_argument('-ne', '--num_epoch', help = 'epoch count',
        type = int, default = 1)
    parser.add_argument('-r', '--ratio', help = 'train/val datasets ratio',
        type = float, default = 0.8)
    parser.add_argument('-model_name', '--model_name', 
                        help = 'model name Available: fcnn_class_age, fcnn_10_classes, log_reg',
                        default = 'fcnn_class_age')

    args = parser.parse_args()
    logging.getLogger().setLevel(logging.DEBUG)
    model = mx.mod.BaseModule()
    print('Loading train/val data and Constructing network model')
    gse_data = load_obj("../../data/transform_data/"+args.data)
    
    if (args.model_name == 'fcnn_class_age'):
        train_iter = mx.io.NDArrayIter(gse_data['train_data'],
                                       gse_data['train_label'],
                                       args.batch_size, shuffle = True)
        val_iter = mx.io.NDArrayIter(gse_data['test_data'],
                                     gse_data['test_label'], args.batch_size)
        model = my_nn.create_fcnn(74)
        
    elif (args.model_name == 'fcnn_10_classes'):
        [train_iter, val_iter] = my_nn.data_10_classes(gse_data, 
                                                       args.batch_size)
        model = my_nn.create_fcnn(10)
        
    elif (args.model_name == 'log_reg'):
        train_iter = mx.io.NDArrayIter(gse_data['train_data'],
                                       gse_data['train_label'],
                                       args.batch_size, shuffle = True, 
                                       label_name='label')
        val_iter = mx.io.NDArrayIter(gse_data['test_data'],
                                     gse_data['test_label'], args.batch_size)
        model = my_nn.create_log_reg_fcnn()
    else:
        print('Error(model name), check arguments. Model name Available: fcnn_class_age, fcnn_10_classes, log_reg')
        sys.exit(0)
    print('Loading train/val data and Constructing network model DONE!')
    
   #Some errors on Cross validation
   #print('Cross validation')
   #my_nn.cross_val(model, gse_data, args.batch_size, args.learning_rate, 
   #                args.num_epoch, args.ratio)
   #print('Cross validation DONE!')
    print('Training network')
    accuracy = mx.metric.Accuracy()
    
    model.fit(train_iter,eval_data = val_iter,optimizer = 'sgd',
              optimizer_params = {'learning_rate':args.learning_rate},
              eval_metric = 'acc',
              batch_end_callback = mx.callback.Speedometer(args.batch_size, 100),
              num_epoch = args.num_epoch)
    print('Training network DONE!')
    print('Testing network')
    model.score(val_iter, accuracy)
    print('Accuracy: {}'.format(accuracy))
    print('Testing network DONE')
                
    
                
                
    
    
        