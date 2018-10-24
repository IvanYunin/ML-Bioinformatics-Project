import numpy as np
import pickle

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

path ="../Data/GSE40249/"
flag = 0

res = np.empty(shape=(473035,656),dtype=np.float32) #memory for data

#geting data from GSE40279_average_beta
textFile = open(path+"GSE40279_average_beta.txt",'r')
for line in textFile:
	if flag:
		split_line = line.split('	')		
		float_row=[]
		for i in range(1,len(split_line)):
			float_row.append(float(split_line[i]))
		res[flag-1]=np.asarray(float_row)
	flag=flag+1

textFile.close()

flag = 0
#getting labels from attributes
label = []
attributes = open(path+"attributes.txt",'r')
for line in attributes:
	if flag:
		split_line = line.split(' ')
		label.append(int(split_line[2]))
	flag = flag + 1
attributes.close()

np_label = np.asarray(label)
	
res = res.transpose()

data = dict.fromkeys(['train_data', 'train_label','test_data','test_label'])

#division into train and test
data['test_data'],data['train_data'],_=np.vsplit(res,(100,656))
data['test_label'],data['train_label'],_=np.split(np_label,(100,656))
#save dict 

save_obj(data,'gse_data')




