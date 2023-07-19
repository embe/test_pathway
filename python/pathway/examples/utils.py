# Copyright (c) 2022 NavAlgo
#
# Proprietary and confidential.

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

import pathway as pw
from pathway.debug import table_from_pandas


class DataPoint(pw.Schema):
    data: np.ndarray


def load_mnist_sample(sample_size=70000):
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    X = X / 255.0
    X_sample = X[:70000]
    y_sample = y[:70000]
    X_train_total = X_sample[:60000]
    X_test_total = X_sample[60000:70000]
    y_train_total = y_sample[:60000]
    y_test_total = y_sample[60000:70000]
    train_size = int((sample_size * 6) / 7)
    test_size = int((sample_size) / 7)
    X_train = X_train_total[:train_size]
    y_train = y_train_total[:train_size]
    X_test = X_test_total[:test_size]
    y_test = y_test_total[:test_size]
    X_train_table = table_from_pandas(
        pd.DataFrame(
            {"data": [np.array(pixels_list) for pixels_list in X_train.tolist()]}
        )
    )
    y_train_table = table_from_pandas(pd.DataFrame({"label": y_train.tolist()}))
    X_test_table = table_from_pandas(
        pd.DataFrame(
            {"data": [np.array(pixels_list) for pixels_list in X_test.tolist()]}
        )
    )
    y_test_table = table_from_pandas(pd.DataFrame({"label": y_test.tolist()}))
    return X_train_table, y_train_table, X_test_table, y_test_table


def classifier_accuracy(predicted_labels, exact_labels):
    pw.universes.promise_is_subset_of(predicted_labels, exact_labels)
    comparative_results = predicted_labels.select(
        predicted_label=predicted_labels.predicted_label, label=exact_labels.label
    )
    comparative_results = comparative_results + comparative_results.select(
        match=comparative_results.label == comparative_results.predicted_label
    )
    accuracy = comparative_results.groupby(comparative_results.match).reduce(
        cnt=pw.reducers.count(comparative_results.match),
        value=comparative_results.match,
    )
    pw.universes.promise_is_subset_of(predicted_labels, accuracy)
    return accuracy
