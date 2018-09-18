#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 18-9-18 下午3:26 
# @Author : ywl
# @File : mnist_class.py 
# @Software: PyCharm

'''
创建一个类似于mnist的类，实现以下方法
'epochs_completed',
 'data',
 'labels',
  next_batch(self, batch_size, shuffle=True)
 'num_examples'
'''
import numpy as np


class mnist(object):
    def __init__(self, X, y):

        self._X = X
        self._y = y
        if len(self._X) != len(self._y):
            assert 'len(X)!=len(labels)'
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = len(self.data)

    @property
    def data(self):
        return self._X

    @property
    def labels(self):
        return self._y

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""

        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._X = self.data[perm0]
            self._y = self.labels[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self._X[start:self._num_examples]
            labels_rest_part = self._y[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._X = self.data[perm]
                self._y = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._X[start:end]
            labels_new_part = self._y[start:end]
            return np.concatenate(
                (images_rest_part, images_new_part), axis=0), np.concatenate(
                (labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._X[start:end], self._y[start:end]


if __name__ == '__main__':
    x = np.arange(1000).reshape(-1, 100)
    y = np.arange(len(x))
    dataset = mnist(x, y)
