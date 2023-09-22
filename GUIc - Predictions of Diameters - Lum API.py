# -*- coding: utf-8 -*-

#for GUI
from tkinter import *
from tkinter import filedialog
from tkinter import ttk
from scipy.io import loadmat
import pickle

#for tensorflow code
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp2d
from keras.models import load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
import math
pi = math.pi
from tensorflow import keras

#for Lumerical api
import imp
from scipy.io import savemat

#starting file directory, this will change depending on the computer running the file
#starting_directory = "X:\\Nano-Photonics and Quantum Optics Lab!\\ML Project" # please change as needed
starting_directory = "Desktop/Athena" # please change as needed
data_size = 200 #default until changed
half_size = 100

#Start of GUI
win = Tk()
win.minsize(600,300) # set minimum window size value (starting size)

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

win.title("Dimensions Prediction Page")
#root.geometry("500x1500") #specify window size

def open_ideal_Px_response():
    root.filename = filedialog.askopenfilename(initialdir =starting_directory, title="Select a file", filetypes=(("matlab files", "*.mat"), ("all files", "*.*")))
    # used // to fix error
    my_label = Label(root, text=root.filename).pack() #print the directory of image
    mat_ideal = loadmat(root.filename)
    mat_dict_ideal = mat_ideal.items()#items: to return a group of the key-value pairs in the dictionary
    mat_Px_list_ideal = list(mat_dict_ideal)#convert object to a list
    global Px_ideal
    Px_ideal = mat_Px_list_ideal[3][1]
    
def open_ideal_Py_response():
    root.filename = filedialog.askopenfilename(initialdir =starting_directory, title="Select a file", filetypes=(("matlab files", "*.mat"), ("all files", "*.*")))
    # used // to fix error
    my_label = Label(root, text=root.filename).pack() #print the directory of image
    mat_ideal = loadmat(root.filename)
    mat_dict_ideal = mat_ideal.items()#items: to return a group of the key-value pairs in the dictionary
    mat_Py_list_ideal = list(mat_dict_ideal)#convert object to a list
    global Py_ideal
    Py_ideal = mat_Py_list_ideal[3][1]
    
def open_ideal_Tx_response():
    root.filename = filedialog.askopenfilename(initialdir =starting_directory, title="Select a file", filetypes=(("matlab files", "*.mat"), ("all files", "*.*")))
    # used // to fix error
    my_label = Label(root, text=root.filename).pack() #print the directory of image
    mat_ideal = loadmat(root.filename)
    mat_dict_ideal = mat_ideal.items()#items: to return a group of the key-value pairs in the dictionary
    mat_Tx_list_ideal = list(mat_dict_ideal)#convert object to a list
    global Tx_ideal
    Tx_ideal = mat_Tx_list_ideal[3][1]
    
def open_ideal_Ty_response():
    root.filename = filedialog.askopenfilename(initialdir =starting_directory, title="Select a file", filetypes=(("matlab files", "*.mat"), ("all files", "*.*")))
    # used // to fix error
    my_label = Label(root, text=root.filename).pack() #print the directory of image
    mat_ideal = loadmat(root.filename)
    mat_dict_ideal = mat_ideal.items()#items: to return a group of the key-value pairs in the dictionary
    mat_Ty_list_ideal = list(mat_dict_ideal)#convert object to a list
    global Ty_ideal
    Ty_ideal = mat_Ty_list_ideal[3][1]

def open_reverse_model():
    root.filename = filedialog.askopenfilename(initialdir =starting_directory, title="Select a file", filetypes=(("h5 files", "*.h5"), ("all files", "*.*")))
    my_label = Label(root, text=root.filename).pack()
    global reverse_loaded
    reverse_loaded = tf.keras.models.load_model(root.filename)
    
def interpolate_data():
    global Px_ideal, Py_ideal, Tx_ideal, Ty_ideal #overwrite these variables if the buttom was clicked
    
    x = np.linspace(0,1,200) #original data size, can change to variable
    y = np.linspace(0,1,200)
    inter_x = np.linspace(0,1, int(interpolation_size.get())) #wanted data size
    inter_y = np.linspace(0,1, int(interpolation_size.get()))
    
    interpol_Px_func = interp2d(x,y,np.reshape(np.asarray(Px_ideal), (200,200), order='C'), kind = 'cubic') #cubic spline
    interpol_Py_func = interp2d(x,y,np.reshape(np.asarray(Py_ideal), (200,200), order='C'), kind = 'cubic')
    Px_interpol_res = interpol_Px_func(inter_x, inter_y)
    Py_interpol_res = interpol_Py_func(inter_x, inter_y)
    
    Tx_interpol_res = np.zeros([int(interpolation_size.get()),int(interpolation_size.get()),13]) #here use interpol size that we want
    Ty_interpol_res = np.zeros([int(interpolation_size.get()),int(interpolation_size.get()),13]) #here use interpol size that we want
    for n in range(13):
        interpol_Tx_func = interp2d(x,y,np.reshape(np.asarray(Tx_ideal), (200,200,13), order='C')[:,:,n], kind = 'cubic')
        Tx_interpol_res[:,:,n] = interpol_Tx_func(inter_x, inter_y)
        interpol_Ty_func = interp2d(x,y,np.reshape(np.asarray(Ty_ideal), (200,200,13), order='C')[:,:,n], kind = 'cubic')
        Ty_interpol_res[:,:,n] = interpol_Ty_func(inter_x, inter_y)
    #flatten before done
    Px_ideal = np.reshape(Px_interpol_res, (int(interpolation_size.get())**2,1), order='C')
    Py_ideal = np.reshape(Py_interpol_res, (int(interpolation_size.get())**2,1), order='C')
    Tx_ideal = np.reshape(Tx_interpol_res, (int(interpolation_size.get())**2,13), order='C')
    Ty_ideal = np.reshape(Ty_interpol_res, (int(interpolation_size.get())**2,13), order='C')
    
def get_file_name():
    global prediction_name
    prediction_name = name_textbox.get('1.0', END)

def predict_dimensions():
    global data_size, half_size,small_size
    data_size = int(interpolation_size.get())
    half_size = data_size//2
    small_size = half_size//2
    response_array = np.concatenate((Px_ideal*float(Weight_Px.get()), Py_ideal*float(Weight_Py.get()), Tx_ideal*float(Weight_Tx.get()), Ty_ideal*float(Weight_Ty.get())), axis=1)
    rev_pred = reverse_loaded.predict(response_array) #weighted

    global pred_Rx, pred_Ry
    pred_Rx = np.reshape(np.asarray(rev_pred)[0]*2e-7, (int(interpolation_size.get()),int(interpolation_size.get())), order='C')
    pred_Ry = np.reshape(np.asarray(rev_pred)[1]*2e-7, (int(interpolation_size.get()),int(interpolation_size.get())), order='C')

    global quad_pred_Rx, quad_pred_Ry
    quad_pred_Rx = pred_Rx[0:half_size,0:half_size]
    quad_pred_Ry = pred_Ry[0:half_size,0:half_size]

    global small_quad_Rx,small_quad_Ry
    small_quad_Rx = quad_pred_Rx[0::2,0::2]#every 2 items
    small_quad_Ry = quad_pred_Ry[0::2,0::2]
    
def download_full_predictions_pickle():
    with open('Prediction Rx_10nm_LS'+str(name_textbox.get())+'_full.pkl', 'wb') as pred_Rx_pkl: #wb=write binary
        pickle.dump(pred_Rx, pred_Rx_pkl)
    with open('Prediction Ry_10nm_LS'+str(name_textbox.get())+'_full.pkl', 'wb') as pred_Ry_pkl: #wb=write binary
        pickle.dump(pred_Ry, pred_Ry_pkl)
       
def download_quadrant_predictions_pickle():
    with open('Prediction Rx_10nm_LS'+str(name_textbox.get())+'_quad.pkl', 'wb') as pred_Rx_pkl: #wb=write binary
        pickle.dump(quad_pred_Rx, pred_Rx_pkl)
    with open('Prediction Ry_10nm_LS'+str(name_textbox.get())+'_quad.pkl', 'wb') as pred_Ry_pkl: #wb=write binary
        pickle.dump(quad_pred_Ry, pred_Ry_pkl)

def download_50x50_predictions_pickle():
    with open('Prediction Rx_10nm_LS'+str(name_textbox.get())+'_50x50.pkl', 'wb') as pred_Rx_pkl: #wb=write binary
        pickle.dump(small_quad_Rx, pred_Rx_pkl)
    with open('Prediction Ry_10nm_LS'+str(name_textbox.get())+'_50x50.pkl', 'wb') as pred_Ry_pkl: #wb=write binary
        pickle.dump(small_quad_Ry, pred_Ry_pkl)
        
def download_full_predictions_csv():
    np.savetxt('Prediction Rx_10nm_LS'+str(name_textbox.get())+'.csv',pred_Rx, fmt='%1.20e', delimiter=', ')
    np.savetxt('Prediction Ry_10nm_LS'+str(name_textbox.get())+'.csv',pred_Ry, fmt='%1.20e', delimiter=', ')
    
def download_quadrant_predictions_csv():
    np.savetxt('Prediction Rx_10nm_LS'+str(name_textbox.get())+'_quad.csv',quad_pred_Rx, fmt='%1.20e', delimiter=', ')
    np.savetxt('Prediction Ry_10nm_LS'+str(name_textbox.get())+'_quad.csv',quad_pred_Ry, fmt='%1.20e', delimiter=', ')

def download_50x50_predictions_csv():
    np.savetxt('Prediction Rx_10nm_LS'+str(name_textbox.get())+'_50x50.csv',small_quad_Rx, fmt='%1.20e', delimiter=', ')
    np.savetxt('Prediction Ry_10nm_LS'+str(name_textbox.get())+'_50x50.csv',small_quad_Ry, fmt='%1.20e', delimiter=', ')

def download_dim_mat():
    m_dictionary = {"Rx_array": small_quad_Rx, "Ry_array": small_quad_Ry, "size_num":small_size, "file_save_name":str("lum_H1um_"+str(small_size)+'x'+str(small_size)+"_Prediction.mat")}
    savemat("prediction_matrix_"+str(small_size)+'x'+str(small_size)+".mat", m_dictionary, format='4') #format 4 helps to remain arrays and not struct arrays

def run_lum_fdtd():
    lumapi = imp.load_source("lumapi", "/opt/lumerical/v232/api/python/lumapi.py") #this might change depending on verion of Lumerical installed
    
    fdtd = lumapi.FDTD(filename=r"Desktop/Athena/Metasurface_aSi_RCWA.fsp", hide=False) # import project file .fsp
    
    code = open('Desktop/Athena/Metasurface_RCWA_LumAPI.lsf', 'r').read()# import script file .lsf
    fdtd.eval(code)
    #input() #this is to stop the window from closing immediately

def upload_lum_response():
    root.filename = filedialog.askopenfilename(initialdir =starting_directory, title="Select a file (lum_H1um_..._Predictions.mat)", filetypes=(("matlab files", "*.mat"), ("all files", "*.*")))
    my_label = Label(root, text=root.filename).pack()
    mat_Lum = loadmat(root.filename)
    mat_dict_Lum = mat_Lum.items()
    global mat_list_Lum
    mat_list_Lum = list(mat_dict_Lum)

def calculate_errors():
    ideal_Px = np.reshape(Px_ideal,(data_size,data_size), order='C')
    ideal_Py = np.reshape(Py_ideal,(data_size,data_size), order='C')
    ideal_Tx = np.reshape(Tx_ideal,(data_size,data_size,13), order='C')
    ideal_Ty = np.reshape(Ty_ideal,(data_size,data_size,13), order='C')
    quad_ideal_Tx = ideal_Tx[0:half_size,0:half_size,:]
    quad_ideal_Ty = ideal_Ty[0:half_size,0:half_size,:]
    quad_ideal_Px = ideal_Px[0:half_size,0:half_size]
    quad_ideal_Py = ideal_Py[0:half_size,0:half_size]
    
    global small_ideal_Tx,small_ideal_Ty,small_ideal_Px,small_ideal_Py
    small_ideal_Tx = quad_ideal_Tx[0::2,0::2,:]
    small_ideal_Ty = quad_ideal_Ty[0::2,0::2,:]
    small_ideal_Px = quad_ideal_Px[0::2,0::2]
    small_ideal_Py = quad_ideal_Py[0::2,0::2]

    global quad_pred_Tx,quad_pred_Ty,quad_pred_Px,quad_pred_Py
    quad_pred_Tx = np.reshape(np.asarray(mat_list_Lum[5][1]),((half_size//2),(half_size//2),51), order='C')
    quad_pred_Ty = np.reshape(np.asarray(mat_list_Lum[6][1]),((half_size//2),(half_size//2),51), order='C')
    quad_pred_Px = np.reshape(np.asarray(mat_list_Lum[3][1]),((half_size//2),(half_size//2),51), order='C')
    quad_pred_Py = np.reshape(np.asarray(mat_list_Lum[4][1]),((half_size//2),(half_size//2),51), order='C')
    #slice to 20nm spectrum for transmission, and center wavelength for phase
    quad_pred_Tx=quad_pred_Tx[:,:,19:32]
    quad_pred_Ty=quad_pred_Ty[:,:,19:32]
    quad_pred_Px=quad_pred_Px[:,:,25]
    quad_pred_Py=quad_pred_Py[:,:,25]
    quad_pred_Px = (quad_pred_Px+pi)/2/pi #normalize the phase
    quad_pred_Py = (quad_pred_Py+pi)/2/pi

    print('\nMean Absolute Error (MAE)')
    Mean_Abs_Error_Tx = sum(sum(sum(abs(small_ideal_Tx-quad_pred_Tx))))/(13*(half_size//2)**2)
    print('Tx: ',Mean_Abs_Error_Tx)
    Mean_Abs_Error_Ty = sum(sum(sum(abs(small_ideal_Ty-quad_pred_Ty))))/(13*(half_size//2)**2)
    print('Ty: ',Mean_Abs_Error_Ty)
    Mean_Abs_Error_Px = sum(sum(abs(small_ideal_Px-quad_pred_Px)))/((half_size//2)**2)
    print('Px: ',Mean_Abs_Error_Px)
    Mean_Abs_Error_Py = sum(sum(abs(small_ideal_Py-quad_pred_Py)))/((half_size//2)**2)
    print('Py: ',Mean_Abs_Error_Py) #NOTE: THIS ERROR HAS BEEN NORMALIZED
    print('\nMean Squared Error (MSE)')
    Mean_Sq_Error_Tx = Mean_Abs_Error_Tx**2
    print('Tx: ',Mean_Sq_Error_Tx)
    Mean_Sq_Error_Ty = Mean_Abs_Error_Ty**2
    print('Ty: ',Mean_Sq_Error_Ty)
    Mean_Sq_Error_Px = Mean_Abs_Error_Px**2
    print('Px: ',Mean_Sq_Error_Px)
    Mean_Sq_Error_Py = Mean_Abs_Error_Py**2
    print('Py: ',Mean_Sq_Error_Py)

def helper_plot_P_func(ideal_data, predicted_data, variable_name):
    plt.rcParams['figure.figsize']=(8,4)
    plt.rcParams['figure.dpi']=100
    plt.rcParams['font.size'] = 12
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    #plt.rcParams.update({'font.size': 12})
    ax = plt.subplot(1,2,1)
    cm = ideal_data
    cax = ax.matshow(cm, cmap=plt.cm.Blues,vmin=0, vmax=1)
    plt.colorbar(cax,label=variable_name)
    ax = plt.subplot(1,2,2)
    cm = predicted_data
    cax = ax.matshow(cm, cmap=plt.cm.Blues,vmin=0, vmax=1)
    plt.colorbar(cax,label=variable_name)
    plt.savefig(variable_name+".png")
    
def helper_plot_T_func(ideal_data, predicted_data, variable_name, n): #n correspond to wavelength
    plt.rcParams['figure.figsize']=(8,4)
    plt.rcParams['figure.dpi']=100
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    ax = plt.subplot(1,2,1)
    cm = ideal_data[:,:,n]
    cax = ax.matshow(cm, cmap=plt.cm.Blues,vmin=0, vmax=1)
    plt.colorbar(cax,label=variable_name)
    ax = plt.subplot(1,2,2)
    cm = predicted_data[:,:,n]
    cax = ax.matshow(cm, cmap=plt.cm.Blues,vmin=0, vmax=1)
    plt.colorbar(cax,label=variable_name)
    plt.savefig(variable_name+"_CenterWave.png")
    
def plot_comparison_quarter():
    helper_plot_P_func(small_ideal_Px, quad_pred_Px, "Phase X")
    helper_plot_P_func(small_ideal_Py, quad_pred_Py, "Phase Y")
    helper_plot_T_func(small_ideal_Tx, quad_pred_Tx, "Transmission X", 6)
    helper_plot_T_func(small_ideal_Ty, quad_pred_Ty, "Transmission Y", 6)
    
def knit_grid_helper(small_quad):
    quad2 = np.rot90(small_quad, k=1, axes=(0, 1))
    quad3 = np.rot90(small_quad, k=2, axes=(0, 1))
    quad4 = np.rot90(small_quad, k=3, axes=(0, 1))
    quad12 = np.concatenate((small_quad,quad4),axis=1)
    quad34 = np.concatenate((quad2,quad3), axis=1)
    knitted = np.concatenate((quad12,quad34),axis=0)
    return knitted

def plot_comparison_knitted():
    knit_pred_Tx = knit_grid_helper(quad_pred_Tx[:,:,6])
    knit_pred_Ty = knit_grid_helper(quad_pred_Ty[:,:,6])
    knit_pred_Px = knit_grid_helper(quad_pred_Px)
    knit_pred_Py = knit_grid_helper(quad_pred_Py)
    knit_ideal_Tx = knit_grid_helper(small_ideal_Tx[:,:,6])
    knit_ideal_Ty = knit_grid_helper(small_ideal_Ty[:,:,6])
    knit_ideal_Px = knit_grid_helper(small_ideal_Px)
    knit_ideal_Py = knit_grid_helper(small_ideal_Py)
    
    helper_plot_P_func(knit_ideal_Px, knit_pred_Px, "Phase X - Knitted")
    helper_plot_P_func(knit_ideal_Py, knit_pred_Py, "Phase Y - Knitted")
    helper_plot_P_func(knit_ideal_Tx, knit_pred_Tx, "Transmission X - Knitted") #using P function here since already center wavelength
    helper_plot_P_func(knit_ideal_Ty, knit_pred_Ty, "Transmission Y - Knitted")

def flatten_data (data):
    a,b,c = (data).shape
    return np.reshape(data,(((a)**2),13), order='C')
    
def plot_spectrum_pred_small (n, x_axis, y_ideal, y_pred, name): #where n is r1 r2 values in single vector
  #transmission subplots
    plt.rcParams['figure.figsize']=(5,3)
    plt.plot(x_axis, y_pred[n], c="r", label="Predictions", marker='o')
    plt.plot(x_axis, y_ideal[n], c="g", label="Validation", marker='o')
    plt.xlabel("Wavelength (nm)")
    plt.ylabel(name)
    plt.ylim([0,1.1])
    #plt.tight_layout(pad=2, h_pad=2, w_pad=2)

def plot_trans_spectrum():
    lambda_c=0.85e-6 #central wavelength
    c = 299792458 #speed of light
    frequency_list =np.linspace(0.95, 1.05, num=51)*c/lambda_c #list of frequency spectrum
    wavelength_list = c/frequency_list
    wavelength_list_nm = wavelength_list*1e9
    wavelist_20nm_range = wavelength_list_nm[19:32]
    
    flat_pred_Tx = flatten_data(quad_pred_Tx)
    flat_pred_Ty = flatten_data(quad_pred_Ty)
    flat_ideal_Tx = flatten_data(small_ideal_Tx)
    flat_ideal_Ty = flatten_data(small_ideal_Ty)
    
    plt.clf() #clears plot first
    plt.figure(figsize=(24, 20))
    plt.subplots_adjust(hspace=0.3)
    for n in range(1,26):
        plt.subplot(5,5,n)
        plot_spectrum_pred_small((n*100)-1, wavelist_20nm_range, flat_ideal_Tx, flat_pred_Tx, "Transmission X")
    plt.savefig("Transmission X - 20 nm Spectrum.png")

    plt.clf() #clears plot first
    plt.figure(figsize=(24, 20))
    plt.subplots_adjust(hspace=0.3)
    for n in range(1,26):
        plt.subplot(5,5,n)
        plot_spectrum_pred_small((n*100)-1, wavelist_20nm_range, flat_ideal_Ty, flat_pred_Ty, "Transmission Y")
    plt.savefig("Transmission Y - 20 nm Spectrum.png")

#part 1
part_1_label = Label(root, text="Step 1: Predict Data and Download Dimensions", font=('Arial', 15)).pack()
#import ideal data
import_ideal_label = Label(root, text="Please upload the ideal responses:", font=('Arial', 12)).pack()
#load_ideal_btn = Button(root, text="Load ideal responce", command=open_ideal_response).pack()

top_frame  =  Frame(root,  width=50,  height=20)
top_frame.pack(side='top',  padx=10,  pady=5)

load_px_btn = Button(top_frame, text="Ideal Px", command=open_ideal_Px_response).pack(side = LEFT)
load_py_btn = Button(top_frame, text="Ideal Py", command=open_ideal_Py_response).pack(side = LEFT)
load_tx_btn = Button(top_frame, text="Ideal Tx", command=open_ideal_Tx_response).pack(side = LEFT)
load_ty_btn = Button(top_frame, text="Ideal Ty", command=open_ideal_Ty_response).pack(side = LEFT)

#interpolation
interpolation_label = Label(root, text="\nWould you like to interpolate? If so, enter new data size below:", font=('Arial', 12)).pack()
explanation_label = Label(root, text="The default size is 200x200", font=('Arial',8)).pack()
interpolation_size = Entry(root, width =50, borderwidth=5)
interpolation_size.insert(0, "200") #default size
interpolation_size.pack()
interpolate_btn =  Button(root, text="Interpolate ideal data", command=interpolate_data).pack()

#import reverse model
import_ideal_label = Label(root, text="\nPlease upload the reverse model:", font=('Arial', 12)).pack()
load_rev_btn = Button(root, text="Import reverse model", command=open_reverse_model).pack()
#name_input_label = Label(root, text="\nOptional: Provide a name for the prediciton files:", font=('Arial', 12)).pack()
#name_textbox = Entry(root, width = 50, borderwidth=5)
#name_textbox.pack()
#separator1 = ttk.Separator(root, orient='horizontal').pack(fill=X)

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
#separator2 = ttk.Separator(root, orient='horizontal').pack(fill=X)

#prediction
run_pred_label = Label(root, text="\nClick to run model predictions:", font=('Arial', 12)).pack()
download_pred_button = Button(root, text="Predict", command=predict_dimensions).pack()
#separator3 = ttk.Separator(root, orient='horizontal').pack(fill=X)

#downloads
import_ideal_label = Label(root, text="\nDownload full size predicted Dx and Dy", font=('Arial', 12)).pack()
download_pred_button = Button(root, text="Download full dimensions (.pkl file)", command = download_full_predictions_pickle).pack()
download_pred_button2 = Button(root, text="Download full dimensions (.csv file)", command = download_full_predictions_csv).pack()

#import_ideal_label = Label(root, text="\nDownload quadrant Dx and Dy", font=('Arial', 12)).pack()
#download_pred_button = Button(root, text="Download quadrant dimensions (.pkl file)", command = download_quadrant_predictions_pickle).pack()
#download_pred_button2 = Button(root, text="Download quadrant dimensions (.csv file)", command = download_quadrant_predictions_csv).pack()

import_ideal_label_50x50 = Label(root, text="\nDownload small test size Dx and Dy (half instances quadrant)", font=('Arial', 12)).pack()
download_pred_button_50x50 = Button(root, text="Download small sample dimensions (.pkl file)", command = download_50x50_predictions_pickle).pack()
download_pred_button_50x50_2 = Button(root, text="Download small sample dimensions (.csv file)", command = download_50x50_predictions_csv).pack()
separator5 = ttk.Separator(root, orient='horizontal').pack(fill=X)

#part 2
step_2_label = Label(root, text="\nStep 2: Lumerical Simulation", font=('Arial', 15)).pack()
download_new_btn_mat = Button(root, text="Download file for Lumerical (.mat)", font=('Arial', 10), command=download_dim_mat)
download_new_btn_mat.pack()

#run lumerical simulation
run_lum_fdtd_label = Label(root, text="Click the buttom below to run lumerical FDTD with generated data", font=('Arial', 12)).pack()
run_lum_fdtd_btn = Button(root, text="Run FDTD Simulation", font=('Arial', 10), command=run_lum_fdtd)
run_lum_fdtd_btn.pack()

#upload lumerical responses
upload_lum_response_label = Label(root, text="\nUpload responce from Lumerical (.mat file)", font=('Arial', 12)).pack()
upload_lum_response_label2 = Label(root, text="This file has been automatically saved to computer after running Lumerical", font=('Arial', 8)).pack()
upload_lum_response_btn = Button(root, text="Upload", font=('Arial', 10), command=upload_lum_response)
upload_lum_response_btn.pack()
separator6 = ttk.Separator(root, orient='horizontal').pack(fill=X)

#part 3
part_3_label = Label(root, text="\nStep 3: Evaluation", font=('Arial', 15)).pack()

#print out MAE and MSE losses
calc_errors_label = Label(root, text="\nClick to view error values on console:", font=('Arial', 12)).pack()
calc_errors_button = Button(root, text="Calculate Errors", command=calculate_errors).pack()
#separator7 = ttk.Separator(root, orient='horizontal').pack(fill=X)

#plot phase and transmission - quarter & knitted plots
plot_compare_label = Label(root, text="\nPlot phase and transmission comparison graphs:", font=('Arial', 12)).pack()
plot_compare_button = Button(root, text="Generate quarter plots", command=plot_comparison_quarter).pack()
plot_compare_button = Button(root, text="Generate knitted plots", command=plot_comparison_knitted).pack()

#plot transmission spectrum
spectrum_plot_label = Label(root, text="\nPlot transmission 20 nm spectrum:", font=('Arial', 12)).pack()
spectrum_plot_button = Button(root, text="Generate plots", command=plot_trans_spectrum).pack()
separator8 = ttk.Separator(root, orient='horizontal').pack(fill=X)

root.mainloop()

























