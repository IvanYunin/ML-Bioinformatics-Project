import numpy as np
import pickle

def save_obj(obj, name ):
    with open('../data/transform_data/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

path = "../data/GSE87571/"
flag = 0

#geting data from GSE40279_average_beta
textFile = open(path+"average_beta.txt",'r')
res = np.empty(shape = (485512 , 729), dtype = np.float32)
for line in textFile:
    if flag:
        split_line = line.split('\t')
        float_row = []
        for i in range(1, len(split_line)):
            float_row.append(float(split_line[i]))
        res[flag-1] = np.asarray(float_row)
    flag = flag+1
print('---')
textFile.close()
print(flag)
flag = 0
#getting labels from attributes
label = []
attributes = open(path+"attributes.txt",'r')
for line in attributes:
    if flag:
        split_line = line.split(' ')
        label.append(int(split_line[3]))
        flag = flag + 1
attributes.close()

np_label = np.asarray(label)

res = res.transpose()

data = dict.fromkeys(['data','labels'])

data['data'] = res
data['labels'] = np_label
#division for train and test
#save dict

print(123)
save_obj(data,'gse87571_data')