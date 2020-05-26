import sys,os
import time
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # __file__获取执行文件相对路径，整行为取上一级的上一级目录
sys.path.append(BASE_DIR)
 

from GCN.data import *
from GCN.model import MLP

import tensorflow as tf

if __name__ == "__main__":
    # Settings
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
    flags.DEFINE_string('model', 'dense', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
    flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
    flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
    flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
    flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
    flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
    flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
    flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)

    features = normalize_features(features)
    print(features[2])
    if FLAGS.model == 'gcn':
        support = [normal_and_selfconnct(adj)]
        num_supports = 1
        model_func = GCN
    elif FLAGS.model == 'dense':
        support = [normal_and_selfconnct(adj)]
        num_supports = 1
        model_func = MLP
    else:
        raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

    # Define placeholders
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
    }

    model = model_func(placeholders,input_dim=features[2][1],logging=True)

    with tf.Session() as sess:
        #初始化变量
        sess.run(tf.global_variables_initializer())

        cost_val = []

        def evaluate(features, support, labels, mask, placeholders):
            t_test = time.time()
            feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
            outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
            return outs_val[0], outs_val[1], (time.time() - t_test)

        #train
        for epoch in range(FLAGS.epochs):
            t = time.time()

            feed_dict = construct_feed_dict(features,support,y_train,train_mask,placeholders)
            feed_dict.update({placeholders['dropout']:FLAGS.dropout})

            outs = sess.run([model.opt_op,model.loss,model.accuracy],feed_dict=feed_dict)

            cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
            cost_val.append(cost)

            # Print results
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
                "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
                "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

            if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
                print("Early stopping...")
                break



    

