# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 21:46:34 2023

@author: athen
"""
from tkinter import *
from tkinter import filedialog
from tkinter import ttk
from scipy.io import loadmat
import pickle

import tensorflow as tf
import numpy as np
import pandas as pd
import io
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from keras.models import load_model
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LeakyReLU, Reshape
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
import math
pi = math.pi
import datetime, time
from tensorflow import keras

#starting file directory, this will change depending on the computer running the file
starting_directory = "X:\\Nano-Photonics and Quantum Optics Lab!\\ML Project" # please change as needed

#Start of GUI
win = Tk()

#Scroll bar vertical
wrapper1 = LabelFrame(win)
mycanvas = Canvas(wrapper1)
mycanvas.pack(side=LEFT, fill="both", expand="yes")
yscrollbar = ttk.Scrollbar(wrapper1, orient="vertical", command=mycanvas.yview)
yscrollbar.pack(side=RIGHT, fill="y")
mycanvas.configure(yscrollcommand=yscrollbar.set)
mycanvas.bind('<Configure>', lambda e: mycanvas.configure(scrollregion = mycanvas.bbox('all')))

root = Frame(mycanvas)
mycanvas.create_window((0,0), window=root, anchor="nw")
wrapper1.pack(fill="both", expand="yes", padx=10, pady=10)

win.title("Models Training Page")

def open_training_dataset():
    global dimensions_array_shuffle, responce_array_shuffle
    root.filename = filedialog.askopenfilename(initialdir =starting_directory, title="Select a file", filetypes=(("matlab files", "*.mat"), ("all files", "*.*")))
    my_label = Label(root, text=root.filename).pack() #print the directory of image
    mat_1 = loadmat(root.filename)
    mat_dict_1 = mat_1.items()#items: to return a group of the key-value pairs in the dictionary
    mat_list1 = list(mat_dict_1)#convert object to a list
    
    dimensions_array = np.asarray([mat_list1[7][1],mat_list1[8][1]])
    dimensions_array = np.squeeze(dimensions_array)
    dimensions_array = np.transpose(dimensions_array, (1,0)) # reshape before slicing
    response_array = np.concatenate((mat_list1[3][1], mat_list1[4][1], mat_list1[5][1], mat_list1[6][1]), axis=1)
    
    #shuffle data
    tf.random.set_seed(27)
    dimensions_array_shuffle = tf.random.shuffle(dimensions_array, seed=27)
    tf.random.set_seed(27)
    responce_array_shuffle = tf.random.shuffle(response_array, seed=27)
  
def format_output_weighted(data):
      output = []
      for i in range(28):
        if i == 0:
          output.append(data[:,i]*float(Weight_Px.get())) 
        elif i == 1:
          output.append(data[:,i]*float(Weight_Py.get()))
        elif i>1 and i<15:
          output.append(data[:,i]*float(Weight_Tx.get()))
        else:
          output.append(data[:,i]*float(Weight_Ty.get()))
      return tuple(output)
  
def format_output_P_weighted(data):
      output = []
      for i in range(2):
        if i == 0:
          output.append(data[:,i]*float(Weight_Px.get())) 
        elif i == 1:
          output.append(data[:,i]*float(Weight_Py.get()))
      return tuple(output)
  
def format_output_T_weighted(data):
      output = []
      for i in range(26):
        if i <13: 
          output.append(data[:,i+2]*float(Weight_Tx.get()))
        else:
          output.append(data[:,i+2]*float(Weight_Ty.get()))
      return tuple(output)
    
def process_data():
    global train_X_forward, val_X_forward, train_Y_forward_weighted, val_Y_forward_weighted, train_Y_forward_Phase_weighted, val_Y_forward_Phase_weighted, train_Y_forward_Trans_weighted, val_Y_forward_Trans_weighted
    global train_Y_reverse, val_Y_reverse, train_X_reverse_weighted, val_X_reverse_weighted
    
    # Split the data into train and validation sets
    train_dime, test_dime = train_test_split(np.asarray(dimensions_array_shuffle), test_size=float(Val_fraction.get()), random_state=2)
    train_resp, test_resp = train_test_split(np.asarray(responce_array_shuffle), test_size=float(Val_fraction.get()), random_state=2)

    #I/O DATA PROCESSING
    #Forward ------------------------------------------------------
    train_X_forward = train_dime/2.000000e-07
    val_X_forward = test_dime/2.000000e-07
    
    train_Y_forward_weighted = format_output_weighted(train_resp)
    val_Y_forward_weighted = format_output_weighted(test_resp)
    
    train_Y_forward_Phase_weighted = format_output_P_weighted(train_resp)
    val_Y_forward_Phase_weighted = format_output_P_weighted(test_resp)
  
    train_Y_forward_Trans_weighted = format_output_T_weighted(train_resp)
    val_Y_forward_Trans_weighted = format_output_T_weighted(test_resp)
    
    #Reverse ------------------------------------------------------
    train_Y_reverse = tuple(np.transpose(train_dime/2.000000e-07))
    val_Y_reverse = tuple(np.transpose(test_dime/2.000000e-07))
    
    train_X_reverse_weighted = train_resp
    train_X_reverse_weighted[:,0:1] = train_resp[:,0:1]*float(Weight_Px.get()) 
    train_X_reverse_weighted[:,1:2] = train_resp[:,1:2]*float(Weight_Py.get())
    train_X_reverse_weighted[:,2:15] = train_resp[:,2:15]*float(Weight_Tx.get())
    train_X_reverse_weighted[:,15:28] = train_resp[:,15:28]*float(Weight_Ty.get())
    
    val_X_reverse_weighted = test_resp
    val_X_reverse_weighted[:,0:1] = test_resp[:,0:1]*float(Weight_Px.get()) 
    val_X_reverse_weighted[:,1:2] = test_resp[:,1:2]*float(Weight_Py.get())
    val_X_reverse_weighted[:,2:15] = test_resp[:,2:15]*float(Weight_Tx.get())
    val_X_reverse_weighted[:,15:28] = test_resp[:,15:28]*float(Weight_Ty.get())

def build_phase_model93():
    input_layer = Input(shape = (2,))
    outputs=[]
    dense_1 = Dense(units='1024', activation='relu')(input_layer)

    dense_2 = Dense(units='1024', activation='relu')(dense_1)

    dense_3 = Dense(units='512', activation='relu')(dense_2)

    dense_4 = Dense(units='1024', activation='relu')(dense_3) 

    dense_5 = Dense(units='1024', activation='relu')(dense_4)

    dense_6 = Dense(units='512', activation='relu')(dense_5)

    outputs.append(Dense(units='1', name=("out_Px"))(dense_6))
    outputs.append(Dense(units='1', name=("out_Py"))(dense_6))

    model = tf.keras.Model(inputs=input_layer, outputs=outputs)
    return model

def train_phase_model():
    global history_phase_93, forward_phase_93
    tf.random.set_seed(27)
    forward_phase_93 = build_phase_model93()
    
    M10nm_forw_P_checkpoint = tf.keras.callbacks.ModelCheckpoint("M10nmPhase_h1um_"+str(P_name_textbox.get())+"_40kpts_checkpoint.h5", save_best_only = True, save_weights_only = False, monitor = 'val_loss', mode = 'min', save_freq="epoch", verbose = 1)
    optimizer=tf.keras.optimizers.Adam()
    losses={}
    losses["out_Px"] = 'mse'
    losses["out_Py"] = 'mse'
    forward_phase_93.compile(optimizer=optimizer, loss=losses)

    # Create a learning rate scheduler callback9
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: (float(P_LR_start.get())) * 10**(epoch/-(float(P_LR_decay.get())))) #1e-3 to 9.326e-7

    # Train the model for N epochs
    epochs = int(Epoch_phase.get())
    history_phase_93 = forward_phase_93.fit(train_X_forward, train_Y_forward_Phase_weighted, epochs=epochs, batch_size=int(P_Batch_size.get()), validation_data=(val_X_forward, val_Y_forward_Phase_weighted), callbacks=[lr_scheduler, M10nm_forw_P_checkpoint])
    #forward_phase_93.save("Forward_phase_weighted.h5") #don't seem to need, the model checkpoint automatically saves to computer
    
def save_phase_loss():
    plt.clf()
    plt.rcParams['figure.figsize']=(8,4)
    plt.rcParams['figure.dpi']=100
    plt.plot(pd.DataFrame({"loss": history_phase_93.history['loss']}), label = "Training Loss")
    plt.plot(pd.DataFrame({"val_loss": history_phase_93.history['val_loss']}), label = "Validation Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.title("Loss Plot of Phase Model")
    plt.legend()
    plt.savefig("Loss plots_phase_"+str(P_name_textbox.get())+".png")
    
def phase_model_error_calc():
    print("\nTrain")
    train_eval_P = np.asarray(forward_phase_93.evaluate(x=train_X_forward, y=train_Y_forward_Phase_weighted))
    #All MSE here
    train_eval_Px = train_eval_P[1] #the number 0 is total val
    train_eval_Py = train_eval_P[2]
    print("Total Phase Error:", train_eval_P[0])
    print("Px:",train_eval_Px)
    print("Py:",train_eval_Py)
    
    print("\nVal")
    val_eval_P = np.asarray(forward_phase_93.evaluate(x=val_X_forward, y=val_Y_forward_Phase_weighted))
    val_eval_Px = val_eval_P[1] #the number 0 is total val
    val_eval_Py = val_eval_P[2]
    print("Total Phase Error:", val_eval_P[0])
    print("Px:",val_eval_Px)
    print("Py:",val_eval_Py)
    
def build_trans_model93():
    input_layer = Input(shape = (2,))
    outputs=[]
    dense_1 = Dense(units='1024', activation='relu')(input_layer)

    dense_2 = Dense(units='1024', activation='relu')(dense_1)

    dense_3 = Dense(units='512', activation='relu')(dense_2)

    dense_4 = Dense(units='512', activation='relu')(dense_3)

    dense_5 = Dense(units='256', activation='relu')(dense_4)

    dense_6 = Dense(units='256', activation='relu')(dense_5)

    for i in range(13):
        outputs.append(Dense(units='1', name=("out_Tx_wl"+str(i+1)))(dense_6))
    for i in range(13):
        outputs.append(Dense(units='1', name=("out_Ty_wl"+str(i+1)))(dense_6))
    model = tf.keras.Model(inputs=input_layer, outputs=outputs)
    return model
    
def train_trans_model():
    global history_trans_93, forward_trans_93
    tf.random.set_seed(27)
    forward_trans_93 = build_trans_model93()
    
    M10nm_forw_T_checkpoint = tf.keras.callbacks.ModelCheckpoint("M10nmTrans_h1um_"+str(T_name_textbox.get())+"_40kpts_checkpoint.h5", save_best_only = True, save_weights_only = False, monitor = 'val_loss', mode = 'min', save_freq="epoch", verbose = 1)
    optimizer=tf.keras.optimizers.Adam()
    losses={}
    for i in range(13):
        losses["out_Tx_wl"+str(i+1)] = 'mse'
    for i in range(13):
        losses["out_Ty_wl"+str(i+1)] = 'mse'
    forward_trans_93.compile(optimizer=optimizer, loss=losses)

    # Create a learning rate scheduler callback9
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: (float(T_LR_start.get())) * 10**(epoch/-(float(T_LR_decay.get())))) #1e-3 to 9.326e-7

    # Train the model for N epochs
    epochs = int(Epoch_trans.get())
    history_trans_93 = forward_trans_93.fit(train_X_forward, train_Y_forward_Trans_weighted, epochs=epochs, batch_size=int(P_Batch_size.get()), validation_data=(val_X_forward, val_Y_forward_Trans_weighted), callbacks=[lr_scheduler, M10nm_forw_T_checkpoint])

def save_trans_loss():
    plt.clf()
    plt.rcParams['figure.figsize']=(8,4)
    plt.rcParams['figure.dpi']=100
    plt.plot(pd.DataFrame({"loss": history_trans_93.history['loss']}), label = "Training Loss")
    plt.plot(pd.DataFrame({"val_loss": history_trans_93.history['val_loss']}), label = "Validation Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.title("Loss Plot of Transmission Model")
    plt.legend()
    plt.savefig("Loss plots_trans_"+str(T_name_textbox.get())+".png")
    
def trans_model_error_calc():   
    print("\nTrain")
    train_eval_T = np.asarray(forward_trans_93.evaluate(x=train_X_forward, y=train_Y_forward_Trans_weighted))
    #All MSE here
    train_eval_Tx = train_eval_T[1:14] #the number 0 is total val
    train_eval_Ty = train_eval_T[14:27]
    train_eval_Tx_mean = sum(train_eval_Tx)/13
    train_eval_Ty_mean = sum(train_eval_Ty)/13
    
    print("Total Trans Error:", train_eval_T[0])
    print("Mean Tx:",train_eval_Tx_mean)
    print("Mean Ty:",train_eval_Ty_mean)
    
    print("\nVal")
    val_eval_T = np.asarray(forward_trans_93.evaluate(x=val_X_forward, y=val_Y_forward_Trans_weighted))
    val_eval_Tx = val_eval_T[1:14] #the number 0 is total val
    val_eval_Ty = val_eval_T[14:27]
    val_eval_Tx_mean = sum(val_eval_Tx)/13
    val_eval_Ty_mean = sum(val_eval_Ty)/13
    
    print("Total Trans Error:", val_eval_T[0])
    print("Mean Tx:",val_eval_Tx_mean)
    print("Mean Ty:",val_eval_Ty_mean)
    
def build_rev_model02():
    # Reverse model --------------------------------------------
    input_Total = Input(shape=(28,), name='input_Total')
    input_P, input_T = tf.split(input_Total, [2, 26], axis=1)

    layer_P = Dense(units='512', activation='relu', name='layer_P')(input_P) 
    layer_T = Dense(units='512', activation='relu', name='layer_T')(input_T)

    mergedTP = layers.concatenate([layer_P, layer_T], axis=1)

    dense_2 = Dense(units='1024', activation='relu', name='layer_TP2')(mergedTP)

    dense_3 = Dense(units='512', activation='relu', name='layer_TP3')(dense_2)

    dense_4 = Dense(units='512', activation='relu', name='layer_TP4')(dense_3)

    dense_5 = Dense(units='256', activation='relu', name='layer_TP5')(dense_4)

    dense_6 = Dense(units='256', activation='relu', name='layer_TP6')(dense_5)

    Dx_output = Dense(units='1', name='Dx_output')(dense_6)
    Dy_output = Dense(units='1', name='Dy_output')(dense_6)

    model = Model(inputs=input_Total, outputs=[Dx_output, Dy_output])
    return model

@tf.function
def train_step_tandem(X_rev, Y_forw):
    # Train the reverse model (note that we should *not* update the weights of the forward model)
    with tf.GradientTape() as tape:
        rev_output_2 = reverse_02(X_rev)
        rev_output_1 = tf.concat([rev_output_2[0], rev_output_2[1]], 1)

        condition1 = tf.greater(rev_output_1, 1)
        result1 = tf.where(condition1, tf.multiply(rev_output_1, 100), rev_output_1)
        condition2 = tf.less(result1, 0.2)
        result2 = tf.where(condition2, tf.multiply(tf.subtract(result1, 1.2), 100), result1)

        #tf.print(result2)
        predictions_Phase = forward_loaded_Phase(result2)
        predictions_Trans = forward_loaded_Trans(result2)
        predictions = tf.concat([tf.squeeze(predictions_Phase), tf.squeeze(predictions_Trans)], 0)
        reverse_loss = loss_fn(Y_forw, tf.squeeze(predictions))

    grads = tape.gradient(reverse_loss, reverse_02.trainable_weights)
    optimizer.apply_gradients(zip(grads, reverse_02.trainable_weights))
    train_acc_metric.update_state(Y_forw, tf.squeeze(predictions))
    train_Px_metric.update_state(Y_forw[0], tf.squeeze(predictions)[0])
    train_Py_metric.update_state(Y_forw[1], tf.squeeze(predictions)[1])

    return reverse_loss

@tf.function
def test_step_tandem(X_rev, Y_forw):
    rev_output_2 = reverse_02(X_rev, training=False)
    rev_output_1 = tf.concat([rev_output_2[0], rev_output_2[1]], 1)

    condition1 = tf.greater(rev_output_1, 1)
    result1 = tf.where(condition1, tf.multiply(rev_output_1, 100), rev_output_1)
    condition2 = tf.less(result1, 0)
    result2 = tf.where(condition2, tf.multiply(tf.subtract(result1, 1.2), 100), result1)

    predictions_Phase = forward_loaded_Phase(result2)
    predictions_Trans = forward_loaded_Trans(result2)
    predictions = tf.concat([tf.squeeze(predictions_Phase), tf.squeeze(predictions_Trans)], 0)

    reverse_loss = loss_fn(Y_forw, tf.squeeze(predictions))
    val_acc_metric.update_state(Y_forw, tf.squeeze(predictions))
    val_Px_metric.update_state(Y_forw[0], tf.squeeze(predictions)[0])
    val_Py_metric.update_state(Y_forw[1], tf.squeeze(predictions)[1])
    
def train_rev_model():
    global reverse_02, forward_loaded_Phase, forward_loaded_Trans
    global loss_fn, optimizer, train_Tx_metric, train_Ty_metric, train_acc_metric, train_Px_metric, train_Py_metric, val_acc_metric, val_Px_metric,val_Py_metric
    forward_loaded_Phase = forward_phase_93 #connect names to forward model from previous code
    forward_loaded_Trans = forward_trans_93
    
    tf.random.set_seed(27)
    reverse_02 = build_rev_model02()
    
    train_Tx_metric = {}
    train_Ty_metric = {}
    train_acc_metric = keras.metrics.MeanSquaredError()
    for i in range(13):
        train_Tx_metric[i+1] = keras.metrics.MeanSquaredError()
        train_Ty_metric[i+1] = keras.metrics.MeanSquaredError()
    train_Px_metric = keras.metrics.MeanSquaredError()
    train_Py_metric = keras.metrics.MeanSquaredError()
    
    #logging loss curves
    train_writer = tf.summary.create_file_writer("logs10nm/train/")
    val_writer = tf.summary.create_file_writer("logs10nm/val/")
    
    # optimiser, loss, compile
    optimizer = keras.optimizers.Adam(learning_rate=float(R_LR_start.get()))
    loss_fn = keras.losses.MeanSquaredError()
    reverse_02.compile(optimizer=optimizer, loss=loss_fn)
    
    # Prepare the training dataset.
    batch_size = int(R_Batch_size.get()) 
    
    # Prepare the training dataset.
    train_dataset = tf.data.Dataset.from_tensor_slices((train_X_reverse_weighted, train_Y_forward_weighted))
    train_dataset = train_dataset.batch(batch_size)
    
    # Prepare the validation dataset.
    val_dataset = tf.data.Dataset.from_tensor_slices((val_X_reverse_weighted, val_Y_forward_weighted))
    val_dataset = val_dataset.batch(batch_size)
    
    # Prepare the metrics.
    train_Tx_metric = {}
    train_Ty_metric = {}
    train_acc_metric = keras.metrics.MeanSquaredError()
    for i in range(13):
        train_Tx_metric[i+1] = keras.metrics.MeanSquaredError()
        train_Ty_metric[i+1] = keras.metrics.MeanSquaredError()
    train_Px_metric = keras.metrics.MeanSquaredError()
    train_Py_metric = keras.metrics.MeanSquaredError()
    
    val_Tx_metric = {}
    val_Ty_metric = {}
    val_acc_metric = keras.metrics.MeanSquaredError()
    for i in range(13):
        val_Tx_metric[i+1] = keras.metrics.MeanSquaredError()
        val_Ty_metric[i+1] = keras.metrics.MeanSquaredError()
    val_Px_metric = keras.metrics.MeanSquaredError()
    val_Py_metric = keras.metrics.MeanSquaredError()
    
    tf.random.set_seed(27)

    #Cascade Training For Loop
    epochs = int(Epoch_rev.get())
    best_validation_loss = float('inf') #starting loss as larger number
    for epoch in range(epochs):
        start_time = time.time()
        print("\nStart of epoch %d" % (epoch+1,))
        lr = (float(R_LR_start.get())) * 10**(epoch/-(float(R_LR_decay.get()))) #from 0.0018 to 5.6e-4
        optimizer = keras.optimizers.Adam(learning_rate=lr)
        reverse_02.compile(optimizer=optimizer, loss=loss_fn)
    
        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            loss_value = train_step_tandem(train_X_reverse_weighted, train_Y_forward_weighted)
    
        # Display metrics at the end of each epoch.
        print("Mean Training loss: %.20f" % (float(train_acc_metric.result()),), "Px loss: %.10f" % (float(train_Px_metric.result())), "Py loss: %.10f" % (float(train_Py_metric.result())), "LR: %.10f" % optimizer.lr.numpy())
        with train_writer.as_default():
           tf.summary.scalar("Mean Training Loss", train_acc_metric.result(), step=epoch+1)
        train_acc_metric.reset_states()
        train_Px_metric.reset_states()
        train_Py_metric.reset_states()
    
        # Run a validation loop at the end of each epoch.
        for y_batch_val, y_batch_val in val_dataset:
            test_step_tandem(val_X_reverse_weighted, val_Y_forward_weighted)
        print("Mean Validation loss: %.20f" % (float(val_acc_metric.result()),), "val_Px loss: %.10f" % (float(val_Px_metric.result())), "val_Py loss: %.10f" % (float(val_Py_metric.result())))
    
        #Tracking loss
        with val_writer.as_default():
          tf.summary.scalar("Mean Val Loss", val_acc_metric.result(), step=epoch+1)
          tf.summary.scalar("Learning Rate", float(optimizer.lr.numpy()), step=epoch+1)
    
        #Save improved models
        if float(val_acc_metric.result()) < best_validation_loss:
            best_validation_loss = float(val_acc_metric.result())
            print("Checkpoint: mean val loss improved to %.10f" % (float(val_acc_metric.result()),))
            reverse_02.save('tan_rev'+str(T_name_textbox.get())+'checkpoint.h5', overwrite=True)
        val_acc_metric.reset_states()
        val_Px_metric.reset_states()
        val_Py_metric.reset_states()
        print("Time taken: %.3fs" % (time.time() - start_time))
        
def save_rev_loss():
    pass

def upload_pretrained_model_FP():
    root.filename = filedialog.askopenfilename(initialdir =starting_directory, title="Select a file", filetypes=(("h5 files", "*.h5"), ("all files", "*.*")))
    my_label = Label(root, text=root.filename).pack()
    global forward_phase_93
    train_phase_model_button['state'] = 'disabled'
    save_phase_loss_button['state'] = 'disabled'
    forward_phase_93 = tf.keras.models.load_model(root.filename)
    
def upload_pretrained_model_FT():
    root.filename = filedialog.askopenfilename(initialdir =starting_directory, title="Select a file", filetypes=(("h5 files", "*.h5"), ("all files", "*.*")))
    my_label = Label(root, text=root.filename).pack()
    global forward_trans_93
    train_trans_model_button['state'] = 'disabled'
    save_trans_loss_button['state'] = 'disabled'
    forward_trans_93 = tf.keras.models.load_model(root.filename)
        
def download_rev_model():
    reverse_02.save('tan_rev'+str(T_name_textbox.get())+'.h5') #don't seem to need, the model checkpoint automatically saves

#button states
train_FP_state = True
train_FT_state = True

#upload training data file
part_1_label = Label(root, text="Part 1: Data Processing", font=('Arial', 18)).pack()
import_ideal_label = Label(root, text="Please upload the training dataset", font=('Arial', 12)).pack()
load_training_dataset = Button(root, text="Upload", command=open_training_dataset).pack()

# Split the data into train and test
weights_input_label = Label(root, text="\nPlease specify the fraction of data to become the validation set", font=('Arial', 12)).pack()
Val_fraction_label = Label(root, text="Default: 80% Training and 20% Validation", font=('Arial', 10)).pack()
Val_fraction = Entry(root, width = 50, borderwidth=5)
Val_fraction.insert(0, "0.2") #default, can change
Val_fraction.pack()

#specify weights
weights_input_label = Label(root, text="\nPlease specify the weights (numbers) of the reverse model", font=('Arial', 12)).pack()
Weight_Px_label = Label(root, text="Px weight:", font=('Arial', 10)).pack()
Weight_Px = Entry(root, width = 50, borderwidth=5)
Weight_Px.insert(0, "1") #default weight
Weight_Px.pack()
Weight_Py_label = Label(root, text="Py weight:", font=('Arial', 10)).pack()
Weight_Py = Entry(root, width = 50, borderwidth=5)
Weight_Py.insert(0, "1")
Weight_Py.pack()
Weight_Tx_label = Label(root, text="Tx weight:", font=('Arial', 10)).pack()
Weight_Tx = Entry(root, width = 50, borderwidth=5)
Weight_Tx.insert(0, "1")
Weight_Tx.pack()
Weight_Ty_label = Label(root, text="Ty weight:", font=('Arial', 10)).pack()
Weight_Ty = Entry(root, width = 50, borderwidth=5)
Weight_Ty.insert(0, "1")
Weight_Ty.pack()

#data processing button
data_processing_label = Label(root, text="\nClick to process data before training:", font=('Arial', 12)).pack()
data_processing_button = Button(root, text="Process data", command=process_data).pack()
separator1 = ttk.Separator(root, orient='horizontal').pack(fill=X)



###Train Phase Model############################################################
part_2_label = Label(root, text="\nPart 2: Training of Phase Model", font=('Arial', 18)).pack()

Label(root, text="\n", font=('Arial', 12)).pack()
upload_pretrained_FP = Button(root, text="Upload a pretrained model checkpoint instead of traning new model", command=upload_pretrained_model_FP).pack()

P_name_input_label = Label(root, text="\nOptional: Provide a name for the files in this section:", font=('Arial', 12)).pack()
P_name_textbox = Entry(root, width = 50, borderwidth=5)
P_name_textbox.pack()

#specify: model arch, epochs, starting LR, decay rate
customisable_parameters_label = Label(root, text="\nCustomizable training parameters", font=('Arial', 12)).pack()
Epoch_label = Label(root, text="Number of epochs (training cycles):", font=('Arial', 10)).pack()
Epoch_label2 = Label(root, text="*For quick tests, reduce this number", font=('Arial', 8)).pack()
Epoch_phase = Entry(root, width = 50, borderwidth=5)
Epoch_phase.insert(0, "1000") #default
Epoch_phase.pack()

Batch_size_label = Label(root, text="Batch size:", font=('Arial', 10)).pack()
P_Batch_size = Entry(root, width = 50, borderwidth=5)
P_Batch_size.insert(0, "80") #default 
P_Batch_size.pack()

LR_start_label = Label(root, text="Starting learning rate value:", font=('Arial', 10)).pack()
P_LR_start = Entry(root, width = 50, borderwidth=5)
P_LR_start.insert(0, "0.001") #default 
P_LR_start.pack()

LR_decay_label = Label(root, text="Decay rate of learning rate (exponenetial decay):", font=('Arial', 10)).pack()
P_LR_decay = Entry(root, width = 50, borderwidth=5)
P_LR_decay.insert(0, "330") #default 
P_LR_decay.pack()

#train
train_phase_model_button = Button(root, text="Train the Phase Model", command=train_phase_model)
if train_FP_state == "disabled":
    train_phase_model_button .config(state = 'disabled')
train_phase_model_button.pack()

#save loss plots
save_phase_loss_button = Button(root, text="Save loss plots", command=save_phase_loss)
if train_FP_state == "disabled":
    save_phase_loss_button .config(state = 'disabled')
save_phase_loss_button.pack()

#quick validation with dataset
phase_model_error_calc_button = Button(root, text="Calculate errors", command=phase_model_error_calc).pack()
separator2 = ttk.Separator(root, orient='horizontal').pack(fill=X)



###Train Transmission Model#######################################################
part_3_label = Label(root, text="\nPart 3: Training of Transmission Model", font=('Arial', 18)).pack()

Label(root, text="\n", font=('Arial', 12)).pack()
upload_pretrained_FP = Button(root, text="Upload a pretrained model checkpoint instead of traning new model", command=upload_pretrained_model_FT).pack()

T_name_input_label = Label(root, text="\nOptional: Provide a name for the files in this section:", font=('Arial', 12)).pack()
T_name_textbox = Entry(root, width = 50, borderwidth=5)
T_name_textbox.pack()

#specify: model arch, epochs, starting LR, decay rate
customisable_parameters_label = Label(root, text="\nCustomizable training parameters", font=('Arial', 12)).pack()
Epoch_label = Label(root, text="Number of epochs (training cycles):", font=('Arial', 10)).pack()
Epoch_label2 = Label(root, text="*For quick tests, reduce this number", font=('Arial', 8)).pack()
Epoch_trans = Entry(root, width = 50, borderwidth=5)
Epoch_trans.insert(0, "700") #default
Epoch_trans.pack()

Batch_size_label = Label(root, text="Batch size:", font=('Arial', 10)).pack()
T_Batch_size = Entry(root, width = 50, borderwidth=5)
T_Batch_size.insert(0, "80") #default 
T_Batch_size.pack()

LR_start_label = Label(root, text="Starting learning rate value:", font=('Arial', 10)).pack()
T_LR_start = Entry(root, width = 50, borderwidth=5)
T_LR_start.insert(0, "0.001") #default 
T_LR_start.pack()

LR_decay_label = Label(root, text="Decay rate of learning rate (exponenetial decay):", font=('Arial', 10)).pack()
T_LR_decay = Entry(root, width = 50, borderwidth=5)
T_LR_decay.insert(0, "330") #default 
T_LR_decay.pack()

#train
train_trans_model_button = Button(root, text="Train the Transmission Model", command=train_trans_model)
if train_FT_state == "disabled":
    train_trans_model_button .config(state = 'disabled')
train_trans_model_button.pack()

#save loss plots
save_trans_loss_button = Button(root, text="Save loss plots", command=save_trans_loss)
if train_FT_state == "disabled":
    save_trans_loss_button .config(state = 'disabled')
save_trans_loss_button.pack()

#quick validation with dataset
trans_model_error_calc_button = Button(root, text="Calculate errors", command=trans_model_error_calc).pack()
separator3 = ttk.Separator(root, orient='horizontal').pack(fill=X)



###Train tandem reverse model####################################################
part_4_label = Label(root, text="\nPart 4: Training of Reverse Model", font=('Arial', 18)).pack()
part_4_label2 = Label(root, text="(Cascaded training configuration)", font=('Arial', 12)).pack()

R_name_input_label = Label(root, text="\nOptional: Provide a name for the files in this section:", font=('Arial', 12)).pack()
R_name_textbox = Entry(root, width = 50, borderwidth=5)
R_name_textbox.pack()

#specify: model arch, epochs, starting LR, decay rate
customisable_parameters_label = Label(root, text="\nCustomizable training parameters", font=('Arial', 12)).pack()
Epoch_label = Label(root, text="Number of epochs (training cycles):", font=('Arial', 10)).pack()
Epoch_label2 = Label(root, text="*For quick tests, reduce this number", font=('Arial', 8)).pack()
Epoch_rev = Entry(root, width = 50, borderwidth=5)
Epoch_rev.insert(0, "100") #default
Epoch_rev.pack()

Batch_size_label = Label(root, text="Batch size:", font=('Arial', 10)).pack()
R_Batch_size = Entry(root, width = 50, borderwidth=5)
R_Batch_size.insert(0, "80") #default 
R_Batch_size.pack()

LR_start_label = Label(root, text="Starting learning rate value:", font=('Arial', 10)).pack()
R_LR_start = Entry(root, width = 50, borderwidth=5)
R_LR_start.insert(0, "0.0018") #default 
R_LR_start.pack()

LR_decay_label = Label(root, text="Decay rate of learning rate (exponenetial decay):", font=('Arial', 10)).pack()
R_LR_decay = Entry(root, width = 50, borderwidth=5)
R_LR_decay.insert(0, "500") #default 
R_LR_decay.pack()

#train
train_rev_model_button = Button(root, text="Train the Reverse Model", command=train_rev_model).pack()
train_rev_lable = Label(root, text="*Note: this would take longer to load due to its use of a for loop training loop.\n", font=('Arial', 8)).pack()

#save loss plots - yet to create
save_rev_loss_button = Button(root, text="Save loss plots", command=save_rev_loss).pack()

#download model (checkpoint file should be also automatically downloaded to the computer)
download_rev_model_button = Button(root, text="Download Reverse Model (checkpoint file should be also automatically downloaded to the computer)", command=download_rev_model).pack()

root.mainloop()
































