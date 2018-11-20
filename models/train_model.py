import mxnet as mx
import pickle
import logging
import my_nn 

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

        
def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
    

def data_10_classes(data):
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
    parser.add_argument('-m_n', '--model_name', help = 'modelname', default = 'fcnn_class_age')
  
    
    args = parser.parse_args()
    
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
        [train_iter, val_iter] = data_10_classes(gse_data)
        model = my_nn.create_fcnn(10)
        
    else:
        print('Error, check arguments')
        sys.exit(0)
    
    print('Training network')
    accuracy = mx.metric.Accuracy()
    ce_loss = mx.metric.CrossEntropy()
    model.fit(train_iter, eval_data = val_iter, optimizer = 'sgd',
        optimizer_params = {'learning_rate': args.learning_rate},
        eval_metric = mx.metric.CompositeEvalMetric([accuracy, ce_loss]),
        batch_end_callback = mx.callback.Speedometer(args.batch_size,100),
        num_epoch = args.num_epoch)

    print('Testing network')
    
    predictions = fc_model.predict(val_iter)
    print('Predictions shape: {}'.format(predictions.shape))
    metric = mx.metric.Accuracy()
    fc_model.score(val_iter, metric)
    print('Accuracy: {}'.format(metric))
                
    
                
                
    
    
        