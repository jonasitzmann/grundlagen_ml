from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.linalg import inv
plt.style.use('fivethirtyeight')

def load_dataset():
    return pd.read_csv('dataset.csv')

def pseudoinverse(A):
    return inv(A.T @ A) @ A.T

def design_mat_polynomial(x, m):
    phi_x_T = np.array([
        np.power(x.ravel(), order) for order in range(m+1)])
    return phi_x_T.T

def get_y(w, x):
    w = np.array(w)
    m = w.shape[0] - 1
    phi_x = design_mat_polynomial(x, m)
    y = phi_x @ w
    return y

def plot_weights(weights, label, x_min, x_max, ax, **kwargs):
    n_samples = 200
    xs = np.linspace(x_min, x_max, n_samples)
    ys = get_y(weights, xs)
    ax.plot(xs, ys, label=label, **kwargs)

def plot(x, y=None, coeff_dict={}, **kwargs):
    n_items = len(coeff_dict.items())
    ax = plt.gca()
    cm = plt.get_cmap('plasma')
    if not y is None:
        ax.scatter(x, y, label='samples', marker='X', zorder=10, **kwargs)
    if not type(coeff_dict) == dict:
        coeff_dict = {'' : coeff_dict}
    for idx, (label, w) in enumerate(coeff_dict.items()):
        plot_weights(w, label, min(x), max(x), ax, c=cm(idx/n_items), **kwargs)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

def plot_df(df, coeff_dict={}):
    y = df.y if 'y' in df.columns else None
    plot(df.x, y, coeff_dict)

def log_plot(*args, **kwargs):
    fig, ax = plt.subplots()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.plot(*args, **kwargs)