import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import zscore
from keras.layers import Dense, Dropout, Conv1D, Flatten
import keras_tuner as kt
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import keras.backend as K
import smogn
from math import ceil
from config_file import *
from graph_model import *
from ann_visualizer.visualize import ann_viz

def soft_acc(y_true, y_pred):
    return K.mean(K.equal(K.round(y_true), K.round(y_pred)))

def build_and_compile_model(lr, model, inputs, targets, splits):
    acc_per_fold = []
    loss_per_fold = []
    histories = []
    fold_no = 1
    kf = KFold(n_splits = splits)
    for train, test in kf.split(inputs, targets):
        model.compile(
            # Test different learning rates and print results
            optimizer=tf.optimizers.Adam(learning_rate=lr),
            loss="mean_squared_error",
            metrics=[soft_acc]
        )
        histories.append(model.fit(inputs[train], targets[train], epochs=50, verbose=0))
        scores = model.evaluate(inputs[test], targets[test], verbose=0)
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])
        fold_no = fold_no + 1
    return (histories, loss_per_fold, acc_per_fold)

def compile_model(model,inputs,targets, lr_list, splits = 8, best_lr = None):
    results = []
    for lr in lr_list:
        if best_lr is not None:
            results.append(build_and_compile_model(best_lr, model, inputs, targets))
            print(model.summary())
            scores_per_fold(results)
            break
        print(f"Current lr: {round(lr,10)}")
        results.append(build_and_compile_model(round(lr,10), model, inputs, targets, splits))
    return results