import pickle
import numpy as np
import os
import urllib.request
import tarfile

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo,encoding='bytes')
    return dict

def maybe_download_and_extract():
    if not os.path.exists('cifar-10-batches-py'):
        #testfile = urllib.URLopener()
        print('downloading data...')
        print('this might take several minutes...Please wait..')
        urllib.request.urlretrieve("https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", "temp_file.gz")
        print('download complete')
        print('extrating data...')
        tar = tarfile.open('temp_file.gz', "r:gz")
        tar.extractall()
        tar.close()
        print('Extract complete..')
        os.remove('temp_file.gz')
    else:
        print('file exists')
        

def load_CIFAR10_data(type = 'training',path_ = os.getcwd()):
    val_images,val_labels = None,None
    if type == 'training':
        files = ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5']
        labels = []
        for idx,batch in enumerate(files):
            path = os.path.join(path_,'cifar-10-batches-py',batch)
            data = unpickle(path)
            if idx == 0:
                images = data[b'data']
                labels = data[b'labels']
            else:
                images = np.append(images,data[b'data'],axis=0)
                #labels = labels + data['labels']
                labels = np.append(labels,data[b'labels'])
        idx = np.random.choice(50000, 50000, replace=False)
        images = images[idx]
        labels = labels[idx]
        targets = np.zeros((images.shape[0], 10))
        targets[np.arange(images.shape[0]), labels] = 1
        
        val_images = images[40000:50000]
        val_labels = labels[40000:50000]
        val_targets = np.zeros((val_images.shape[0], 10))
        val_targets[np.arange(val_images.shape[0]), val_labels] = 1
        
        images = images[0:40000]
        targets = targets[0:40000]
    elif type == 'testing':
        path = os.path.join(path_,'cifar-10-batches-py','test_batch')
        data = unpickle(path)
        images = data[b'data']
        labels = data[b'labels']
    else:
        raise ValueError("type_of_data must be 'testing' or 'training'")
    return (images,targets,val_images,val_labels)