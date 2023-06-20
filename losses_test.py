import tensorflow as tf
import numpy as np

def arcface_loss(embeddings_anchor, embeddings_positive, angular_margin = 0.5):
    embeddings_anchor = tf.nn.l2_normalize(embeddings_anchor, axis=-1)
    embeddings_positive = tf.nn.l2_normalize(embeddings_positive, axis=-1)
    similarity = tf.reduce_sum(tf.multiply(embeddings_anchor, embeddings_positive), axis=-1)

    theta = tf.acos(similarity)
    target_similarity = tf.cos(theta + angular_margin)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=target_similarity, logits=similarity))

    return np.abs(loss.numpy())



