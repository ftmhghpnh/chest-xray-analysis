import os
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from evaluation_utils import accuracy_precision_recall_fscore, save_confusion_matrix, roc_auc, plot_loss_curve
from models import FeedForwardClassifier

tf.enable_eager_execution()
base_path = '/home/chavosh/chest-xray-analysis'
train_table = pd.read_csv(os.path.join(base_path, 'train.csv'))
case_array = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
device = "gpu:0" if tfe.num_gpus() else "cpu:0"

batch_size = 64
n_epochs = 1
learning_rate = 0.0001
train_loss_iteration = []
train_loss_epoch = []
val_loss_epoch = []

data = np.load(os.path.join(base_path, 'Xception_bottle_neck.npz'))
X_train, index_train = data['bottle_necks'],  data['indexes']
X_train, X_val, index_train, index_val = train_test_split(X_train, index_train, test_size=0.2, random_state=40)
Y_train = train_table.loc[index_train, case_array].values

Y_train[Y_train == -1] = 0
Y_train[np.isnan(Y_train)] = -1
Y_val = train_table.loc[index_val, case_array].values
Y_val[Y_val == -1] = 0
Y_val[np.isnan(Y_val)] = -1

train_features_dataset = tf.data.Dataset.from_tensor_slices(X_train)
train_label_dataset = tf.data.Dataset.from_tensor_slices(Y_train)
train_dataset = tf.data.Dataset.zip((train_features_dataset, train_label_dataset))
train_dataset = train_dataset.batch(batch_size) 

classifier = FeedForwardClassifier(n_classes=len(case_array), layer_dims=[20 * len(case_array), 5 * len(case_array)],
                                   activations=[tf.keras.activations.sigmoid, tf.keras.activations.sigmoid],
                                   drop_out_flag=True, drop_out_rate=0.5)
optimizer = tf.train.AdamOptimizer(learning_rate)


def loss_calculator(inp, targ):
    logits = classifier(inp)
    mask = tf.math.logical_not(tf.equal(targ, -1))
    return tf.reduce_mean(tf.losses.sigmoid_cross_entropy(multi_class_labels=targ, logits=logits, weights=mask))


with tf.device(device):
    for epoch in range(n_epochs):
        for batch, (features, labels) in enumerate(train_dataset):

            with tf.GradientTape() as tape:
                loss = loss_calculator(features, labels)

            train_loss_iteration.append(loss.numpy())
            grads = tape.gradient(loss, classifier.variables)
            optimizer.apply_gradients(zip(grads, classifier.variables), global_step=tf.train.get_or_create_global_step())
            
            if batch % 100 == 0:
                print('\rEpoch: {}, Batch: {}, Loss: {}'.format(epoch, batch, train_loss_iteration[-1]), end='')
        train_loss_epoch.append(loss_calculator(tf.constant(X_train), tf.constant(Y_train)))
        val_loss_epoch.append(loss_calculator(tf.constant(X_val), tf.constant(Y_val)))


logits = classifier(tf.constant(X_val))
Y_pred = tf.nn.sigmoid(logits).numpy()
acc_overall, accs, precision_overall, precisions, recall_overall, recalls, fscore_overall, fscores = \
    accuracy_precision_recall_fscore(Y_pred, Y_val, len(case_array))
roc_auc_overall, roc_aucs = roc_auc(Y_pred, Y_val, len(case_array))
save_confusion_matrix(Y_pred, Y_val, case_array, os.path.join(base_path, 'conf_mat_{}'))
plot_loss_curve([train_loss_epoch, val_loss_epoch], ['Train', 'Validation'], 'Loss curve per epoch',
                os.path.join(base_path, 'loss_curve_epoch'), 'Epoch')
plot_loss_curve([train_loss_iteration], ['Train'], 'Train loss curve per iteration',
                os.path.join(base_path, 'loss_curve_iter'), 'Iteration')

