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
            results.append(build_and_compile_model(best_lr, model, inputs, targets, splits))
            print(model.summary())
            scores_per_fold(results, lr)
            break
        print(f"Current lr: {round(lr,10)}")
        results.append(build_and_compile_model(round(lr,10), model, inputs, targets, splits))
    return results

def scores_per_fold(results, _best_lr = None):
    if _best_lr is not None:
        print('------------------------------------------------------------------------')
        print('Score per fold')
        for i in range(len(results[0][2])):
            print('------------------------------------------------------------------------')
            print(f'> Fold {i+1} - Loss: {results[0][1][i]} - Accuracy: {results[0][2][i]}%')
        print('------------------------------------------------------------------------')
        print('Average scores for all folds:')
        print(f'> Accuracy: {np.mean(results[0][2])} (+- {np.std(results[0][2])})')
        print(f'> Loss: {np.mean(results[0][1])}')
        print('------------------------------------------------------------------------')
        

def show_nine_losses(results, lr_list, title=""):
    fig, axs = plt.subplots(3,3, figsize=(10, 6))
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle(title, fontsize=16, y=1.05)
    axs = np.ravel(axs)
    for i,lr in enumerate(lr_list):
        avg_loss_per_epoch = [0 for i in range(len(results[0][0][0].history['loss']))] #epochs
        for hist in results[i][0]:
            avg_loss_per_epoch= [x + y for x, y in zip(avg_loss_per_epoch, hist.history['loss'])]
        avg_loss_per_epoch = [x/len(results[0][0]) for x in avg_loss_per_epoch]
        axs[i].plot(avg_loss_per_epoch, label='loss')
        axs[i].set_title(f'lr={round(lr,8)}')
        axs[i].set_xlabel('Epoch')
        axs[i].set_ylabel('Loss')
        axs[i].legend()
        axs[i].grid(True)
    plt.show()

def show_single_loss(results, lr, title="", legend='loss'):
    fig, ax = plt.subplots(figsize=(10, 6))
    avg_loss_per_epoch = [0 for i in range(len(results[0][0][0].history['loss']))]
    for hist in results[0][0]:
        avg_loss_per_epoch= [x + y for x, y in zip(avg_loss_per_epoch, hist.history['loss'])]
    avg_loss_per_epoch = [x/len(results[0][0]) for x in avg_loss_per_epoch]
    plt.tight_layout()
    fig.suptitle(title, fontsize=16, y=1.05)
    ax.plot(avg_loss_per_epoch, label='loss')
    ax.set_title(f'lr={lr}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend(title=legend)
    ax.grid(True)
    plt.show()