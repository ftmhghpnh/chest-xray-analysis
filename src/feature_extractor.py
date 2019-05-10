import pandas as pd
import os
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
from src.models import VGG19Bottleneck

tf.enable_eager_execution()

base_path = '/home/chavosh/chest-xray-analysis'
train_table = pd.read_csv(os.path.join(base_path, 'train.csv'))
train_table = train_table.loc[train_table['Frontal/Lateral'] == 'Frontal']
test_table = pd.read_csv(os.path.join(base_path, 'valid.csv'))
test_table = test_table.loc[test_table['Frontal/Lateral'] == 'Frontal']

case_array = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
ans = [-1, 1]


def create_dataset_images(file_paths):
    # Reads an image from a file, decodes it into a dense tensor, and resizes it
    # to a fixed shape. Read more here: https://www.tensorflow.org/guide/datasets#decoding_image_data_and_resizing_it
    def _parse_function(filename):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        image_resized = tf.image.resize_images(image_decoded, [200, 200])
        return image_resized

    file_paths = tf.constant(file_paths)
    
    dataset = tf.data.Dataset.from_tensor_slices((file_paths))
    dataset = dataset.map(_parse_function)

    return dataset


def cache_bottleneck_layers(file_paths, batch_size, model):

    bottle_necks = []
    dataset = create_dataset_images(file_paths).batch(batch_size)
    n_samples = len(file_paths)

    device = "gpu:0" if tfe.num_gpus() else "cpu:0"
    
    with tf.device(device):
        for batch_num, image in enumerate(dataset):
            print('\rComputing bottle neck layers... batch {} of {}'.format(batch_num+1, n_samples//batch_size), end="")
            
            # Compute bottle necks layer for image batch convert to numpy and append to bottle_necks
            # ...
            # ...
            result = model(image)
            result = result.numpy()
            bottle_necks.append(result)
            
    return np.vstack(bottle_necks)


train_selected = train_table.loc[(train_table[case_array[0]].isin(ans) | train_table[case_array[1]].isin(ans) | train_table[case_array[2]].isin(ans) | train_table[case_array[3]].isin(ans) | train_table[case_array[4]].isin(ans))]
train_file_paths = [os.path.join(base_path, '/'.join(path.split('/')[1:])) for path in train_selected['Path'].tolist()]
test_file_paths = [os.path.join(base_path, '/'.join(path.split('/')[1:])) for path in test_table['Path'].tolist()]

bottle_necks = cache_bottleneck_layers(train_file_paths, batch_size=64, model=VGG19Bottleneck())
bottle_necks_test = cache_bottleneck_layers(test_file_paths, batch_size=64, model=VGG19Bottleneck())

np.savez(os.path.join(base_path, 'VGG19_bottle_neck.npz'), bottle_necks=bottle_necks, indexes=train_selected.index.values)
np.savez(os.path.join(base_path, 'VGG19_bottle_neck_test.npz'), bottle_necks=bottle_necks_test, indexes=test_table.index.values)
