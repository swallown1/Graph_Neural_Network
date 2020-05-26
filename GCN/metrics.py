import tensorflow as tf 

def masked_softmax_cross_entropy(logits,labels,mask):
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels)
    mask = tf.cast(mask,dtype=tf.float32)
    mask /=tf.reduce_mean(mask)
    loss *=mask
    return tf.reduce_mean(loss)

def masked_accuracy(preds,labels,mask):
    correct_prediction = tf.equal(tf.argmax(preds,1),tf.argmax(labels,1))
    acc_all = tf.cast(correct_prediction,tf.float32)
    mask = tf.cast(mask,dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    acc_all *=mask
    return tf.reduce_mean(acc_all)