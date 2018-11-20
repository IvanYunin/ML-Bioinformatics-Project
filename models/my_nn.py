import mxnet as mx

def create_fcnn(num_classes):
	data = mx.sym.var('data')
	labels = mx.sym.var('labels')
	
    fc1 = mx.sym.FullyConnected(data = data, num_hidden = 10000)
    reLU_activate=mx.sym.relu(data=fc1, name='relu')
	
    fc2=mx.sym.FullyConnected(data = reLU_activate, num_hidden = num_classes)
    
	output_layer = mx.sym.SoftmaxOutput(data = fc2, label = labels,
										name = 'softmax')

	model = mx.mod.Module(symbol = output_layer, data_names = ['data'],
						  label_names = ['labels'], context = mx.cpu())
	
	return model

