import pandas as pd
import os
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
from sklearn.model_selection import train_test_split
from evaluation_utils import accuracy_precision_recall_fscore, save_confusion_matrix, roc_auc, plot_loss_curve
from models import XceptionEnd2End, MobileNetEnd2End

tf.enable_eager_execution()

base_path = '/home/chavosh/chest-xray-analysis'
train_table = pd.read_csv(os.path.join(base_path, 'train.csv'))
device = "gpu:0" if tfe.num_gpus() else "cpu:0"

case_array = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
ans = [-1, 1]
batch_size = 64
n_epochs = 1
learning_rate = 0.0001
train_loss_iteration = []
train_loss_epoch = []
val_loss_epoch = []


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


train_selected = train_table.loc[(
            train_table[case_array[0]].isin(ans) | train_table[case_array[1]].isin(ans) | train_table[
        case_array[2]].isin(ans) | train_table[case_array[3]].isin(ans) | train_table[case_array[4]].isin(ans))]
train_file_paths = [os.path.join(base_path, '/'.join(path.split('/')[:1])) for path in train_selected['Path'].tolist()]

X_train, index_train = train_file_paths, train_selected.index.values
X_train, X_val, index_train, index_val = train_test_split(X_train, index_train, test_size=0.2, random_state=40)
Y_train = train_table.loc[index_train, case_array].values

Y_train[Y_train == -1] = 0
Y_train[np.isnan(Y_train)] = -1
Y_val = train_table.loc[index_val, case_array].values
Y_val[Y_val == -1] = 0
Y_val[np.isnan(Y_val)] = -1

train_images_dataset = create_dataset_images(X_train)
train_label_dataset = tf.data.Dataset.from_tensor_slices(Y_train)
train_dataset = tf.data.Dataset.zip((train_images_dataset, train_label_dataset))
train_dataset = train_dataset.batch(batch_size)

classifier = MobileNetEnd2End(len(case_array))
optimizer = tf.train.AdamOptimizer(learning_rate)

val_images_dataset = create_dataset_images(X_val)


def loss_calculator(inp, targ):
    logits = classifier(inp)
    mask = tf.math.logical_not(tf.equal(targ, -1))
    return tf.reduce_mean(tf.losses.sigmoid_cross_entropy(multi_class_labels=targ, logits=logits, weights=mask))


with tf.device(device):
    for epoch in range(n_epochs):
        for batch, (images, labels) in enumerate(train_dataset):

            with tf.GradientTape() as tape:
                loss = loss_calculator(images, labels)

            train_loss_iteration.append(loss.numpy())
            grads = tape.gradient(loss, classifier.variables)
            optimizer.apply_gradients(zip(grads, classifier.variables),
                                      global_step=tf.train.get_or_create_global_step())

            if batch % 100 == 0:
                print('\rEpoch: {}, Batch: {}, Loss: {}'.format(epoch, batch, train_loss_iteration[-1]), end='')
        train_loss_epoch.append(loss_calculator(tf.constant(X_train), tf.constant(Y_train)))
        val_loss_epoch.append(loss_calculator(tf.constant(X_val), tf.constant(val_images_dataset)))

logits = classifier(tf.constant(val_images_dataset))
Y_pred = tf.nn.sigmoid(logits).numpy()
acc_overall, accs, precision_overall, precisions, recall_overall, recalls, fscore_overall, fscores = \
    accuracy_precision_recall_fscore(Y_pred, Y_val, len(case_array))
roc_auc_overall, roc_aucs = roc_auc(Y_pred, Y_val, len(case_array))
save_confusion_matrix(Y_pred, Y_val, case_array, os.path.join(base_path, 'conf_mat_{}'))
plot_loss_curve([train_loss_epoch, val_loss_epoch], ['Train', 'Validation'], 'Loss curve per epoch',
                os.path.join(base_path, 'loss_curve_epoch'), 'Epoch')
plot_loss_curve([train_loss_iteration], ['Train'], 'Train loss curve per iteration',
                os.path.join(base_path, 'loss_curve_iter'), 'Iteration')
print()
print(acc_overall, accs, precision_overall, precisions, recall_overall, recalls, fscore_overall, fscores)
print(roc_auc_overall, roc_aucs)