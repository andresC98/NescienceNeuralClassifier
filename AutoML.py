#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 14:07:07 2018

@author: rleiva
"""

# Generic
import random
import time

# Bokeh
from tornado        import gen
from bokeh.models   import ColumnDataSource, GraphRenderer, StaticLayoutProvider, Circle

from bokeh.plotting import curdoc, figure
from bokeh.layouts  import row, layout

# Treading
from functools      import partial
from threading      import Thread

# ML
import numpy as np
from sklearn.datasets import load_digits
from NNClassifier.NescienceNeuralNetworkClassifierV211 import *

count = 1

@gen.coroutine
def stream_update(x, vals):

    global f0
    
    nsc = vals["nescience"]
    mis = vals["miscoding"]
    sur = vals["surfeit"]
    ina = vals["inaccuracy"]
    scr = vals["score"]
    
    source_nsc.stream(dict(x=[x], y=[nsc]), rollover=100)
    source_mis.stream(dict(x=[x], y=[mis]), rollover=100)
    source_sur.stream(dict(x=[x], y=[sur]), rollover=100)
    source_ina.stream(dict(x=[x], y=[ina]), rollover=100)
    source_scr.stream(dict(x=[x], y=[scr]), rollover=100)

    # Display neural network
     
    canvas_size = 1
    top    = .9 * canvas_size
    bottom = .1 * canvas_size
    left   = .1 * canvas_size
    right  = .9 * canvas_size

    layer_sizes = vals["layer_sizes"]
    N = np.sum(layer_sizes)

    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)

    node_indices = list(range(N))

    graph = GraphRenderer()

    graph.node_renderer.data_source.add(node_indices, 'index')

    size = int(v_spacing * 300)
    graph.node_renderer.glyph = Circle(size=size, fill_color='white')

    # Plot edges
    
    start = list()
    end  = list()

    count = 0
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
           
        for m in np.arange(layer_size_a):
            for o in np.arange(layer_size_b):
                start.append(count + m)
                end.append(count + layer_size_a + o)
    
        count = count + layer_size_a

    graph.edge_renderer.data_source.data = dict(start=start, end=end)
             
    # Nodes

    x = list()
    y = list()

    for n, layer_size in enumerate(layer_sizes):
    
        layer_top = v_spacing * (layer_size - 1)/2. + (top + bottom)/2.
        
        for m in np.arange(layer_size):

            x.append(n * h_spacing + left)
            y.append(layer_top - m * v_spacing)

    graph_layout = dict(zip(node_indices, zip(x, y)))
    graph.layout_provider = StaticLayoutProvider(graph_layout=graph_layout)

    if len(f0.renderers) != 0:
        f0.renderers.pop()
        
    f0.renderers.append(graph) 
     

def update_plot():
    
    global queue, count

    while True:
        
        vals = queue.get()
        queue.task_done()
        
        count = count + 1
                    
        doc.add_next_tick_callback(partial(stream_update, x=count, vals=vals))


# Data source for Bokeh
source_nsc = ColumnDataSource(data=dict(x=[0], y=[1]))
source_mis = ColumnDataSource(data=dict(x=[0], y=[1]))
source_sur = ColumnDataSource(data=dict(x=[0], y=[1]))
source_ina = ColumnDataSource(data=dict(x=[0], y=[1]))
source_scr = ColumnDataSource(data=dict(x=[0], y=[1]))

# Bokeh current session
doc = curdoc()
                
# Start web
f0 = figure(y_range=[0, 1], x_range=[0, 1], title="Neural Network", plot_width=300, plot_height=300, toolbar_location=None)

f1 = figure(y_range=[0,1], title="Nescience", plot_width=300, plot_height=300, toolbar_location=None)
f1.line(x='x', y='y', source=source_nsc)

f2 = figure(y_range=[0,1], title="Miscoding", plot_width=300, plot_height=300, toolbar_location=None)
f2.line(x='x', y='y', source=source_mis)

f3 = figure(y_range=[0,1], title="Inaccuracy", plot_width=300, plot_height=300, toolbar_location=None)
f3.line(x='x', y='y', source=source_ina)

f4 = figure(y_range=[0,1], title="Surfeit", plot_width=300, plot_height=300, toolbar_location=None)
f4.line(x='x', y='y', source=source_sur)

f5 = figure(y_range=[0,1], title="Validation Score", plot_width=300, plot_height=300, toolbar_location=None)
f5.line(x='x', y='y', source=source_scr)

l = layout([
    [f0],
    [f1, f2, f3, f4, f5]
    ], sizing_mode='stretch_both')

# doc.add_root(row(f1, f2, f3, f4, f5))
doc.add_root(l)

# Start web page thread
thread_consumer = Thread(target=update_plot)
thread_consumer.start()

# Prepare data and model

# scikit - Digits <----
# data = load_digits()
# X = data.data
# y = data.target
#alternative MNIST from keras:
# from keras.datasets import mnist
# (X_train, y_train), (X_test, y_test) = mnist.load_data()

#Pulsar Star Classification from Kaggle
from numpy import genfromtxt
data = genfromtxt('pulsar_stars.csv', delimiter=',')
data = np.delete(data, 0, 0)
X = data[:,:8]
y = data[:,8]

#Breast cancer
# from sklearn.datasets import load_breast_cancer
# data = load_breast_cancer()
# X = data.data
# y = data.target

# fashion_mnist = keras.datasets.fashion_mnist
# (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


# Numerai
# train = pd.read_csv('../data/numerai_training_data.csv', header=0)
# tournament = pd.read_csv('../data/numerai_tournament_data.csv', header=0)
# validation = tournament[tournament['data_type']=='validation']
# train_bernie = train.drop([
#     'id', 'era', 'data_type',
#     'target_charles', 'target_elizabeth',
#     'target_jordan', 'target_ken'], axis=1)
# features = [f for f in list(train_bernie) if "feature" in f]
# X = train_bernie[features]
# Y = train_bernie['target_bernie']
# X = np.array(X)
# y = np.array(Y)

# CIFAR-10
#import pickle
#
#with open("/data/CIFAR-10/cifar-10-batches-py/data_batch_1", 'rb') as fo:
#    data = pickle.load(fo, encoding='bytes')
#
#X = data[b"data"]
#y = data[b"labels"]

# Train the model

model = NescienceNeuralNetworkClassifier(backward=False, verbose=True)

thread_producer = Thread(target=model.fit, args=(X, y))
thread_producer.start()

