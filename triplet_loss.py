import tensorflow as tf
import numpy as np
from typing import List, Type
from sklearn.neighbors import KDTree


class KDTreeTripletLoss():
    def __init__(self, tree: KDTree, true_labels: List[Type[int]]):
        self.tree = tree
        self.max_dist = tree.node_data[0]['radius']
        self.true_labels = true_labels

    def get_dists(self, labels: tf.Tensor, embeddings: tf.Tensor):
        print(embeddings.numpy().shape)
        dist, idx = self.tree.query(embeddings.numpy(), 50)
        pos_dists = []
        neg_dists = []
        for i, i_idx in enumerate(idx):
            pos_dist = dist[i][-1]**2
            for di in i_idx:
                if self.true_labels[di] == labels[i]:
                    pos_dist = dist[di]**2
            pos_dists.append(pos_dist)

            neg_dist = dist[i][-1]
            for di in i_idx:
                if self.true_labels[di] != labels[i]:
                    neg_dist = dist[di]**2
                    break
            neg_dists.append(neg_dist)
        return pos_dists, neg_dists

    @tf.function
    def kdtree_triplet_loss(self, y_true, y_pred, margin=1.0):
        """Computes the triplet loss with semi-hard negative mining.
        MODIFIED FROM tensorflow_addons.losses.triplet_semihard_loss

        Args:
        y_true: 1-D integer `Tensor` with shape [batch_size] of
            multiclass integer labels.
        y_pred: 2-D float `Tensor` of embedding vectors. Embeddings should
            be l2 normalized.
        margin: Float, margin term in the loss definition.
        """
        labels, embeddings = y_true, y_pred
        # Reshape [batch_size] label tensor to a [batch_size, 1] label tensor.
        lshape = tf.shape(labels)
        print('step1')
        assert lshape.shape == 1 or lshape.shape == (2,)
        print('step2')
        labels = tf.reshape(labels, [-1])
        print('step3')

        pos_dists, neg_dists = tf.py_function(self.get_dists, inp=[labels, embeddings], Tout=[tf.float64, tf.float64])
        print('step4')
        print(pos_dists, neg_dists)
        pos_diffs = tf.minimum(pos_dists - neg_dists, [0])
        print('step5')
        print(pos_diffs)
        zero_idx = tf.cast(tf.where(pos_diffs == 0, -tf.ones_like(pos_diffs), pos_diffs), tf.int64)
        print(zero_idx)
        print(pos_dists[zero_idx])
        norm_diffs = tf.tensor_scatter_nd_update(pos_diffs, zero_idx, pos_dists[zero_idx])
        print(norm_diffs)
        # pos_diffs[[1] == 0] = pos_dists[[1] == 0] / 2
        print('step6')

        triplet_loss = tf.cast(tf.reduce_mean(norm_diffs), tf.float32)
        print('step7')
        print(triplet_loss)

        return triplet_loss
