# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 13:29:35 2023

@author: athen
"""
#for GUI
from tkinter import *
from tkinter import filedialog
from tkinter import ttk
import io
import pickle

#for tensorflow code
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from keras.models import load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from scipy.io import loadmat
import math
pi = math.pi
from tensorflow import keras

#for Lumerical api
import imp
from scipy.io import savemat

#starting file directory, this will change depending on the computer running the file
starting_directory = "Desktop/Athena" # please change as needed

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

win.title("Cross Validation of Forward Model")

def generate_random_dim():
    #Generate random samples from a uniform distribution
    global xi, yi
    xi = np.random.uniform(4e-8, 2e-7, size=(int(Test_size.get()),int(Test_size.get())))
    yi = np.random.uniform(4e-8, 2e-7, size=(int(Test_size.get()),int(Test_size.get())))
    
def download_dim_csv():
    np.savetxt('Forward_Eval_H1um_Rand_'+str(Test_size.get())+'x'+str(Test_size.get())+'_Dx.csv',xi, fmt='%1.64e', delimiter=', ')
    np.savetxt('Forward_Eval_H1um_Rand_'+str(Test_size.get())+'x'+str(Test_size.get())+'_Dy.csv',yi, fmt='%1.64e', delimiter=', ')
    
def download_dim_pkl():
    with open('Forward_Eval_H1um_Rand_'+str(Test_size.get())+'x'+str(Test_size.get())+'_Dx.pkl', 'wb') as xi_pkl: #wb=write binary
        pickle.dump(xi, xi_pkl)
    with open('Forward_Eval_H1um_Rand_'+str(Test_size.get())+'x'+str(Test_size.get())+'_Dy.pkl', 'wb') as yi_pkl: #wb=write binary
        pickle.dump(yi, yi_pkl)
        
def upload_dim_Dx():
    root.filename = filedialog.askopenfilename(initialdir =starting_directory, title="Select a file", filetypes=(("csv files", "*.csv"), ("all files", "*.*")))
    my_label = Label(root, text=root.filename).pack() #print the directory of image, not required
    global xi
    xi=pd.read_csv(root.filename,sep=",", header=None)
    
def upload_dim_Dy():
    root.filename = filedialog.askopenfilename(initialdir =starting_directory, title="Select a file", filetypes=(("csv files", "*.csv"), ("all files", "*.*")))
    my_label = Label(root, text=root.filename).pack() #print the directory of image, not required
    global yi
    yi=pd.read_csv(root.filename,sep=",", header=None)

def download_dim_mat():
    file_save_name = str("lum_H1um_"+str(Test_size.get())+'x'+str(Test_size.get())+"_RandomVal.mat")
    m_dictionary = {"Rx_array": xi, "Ry_array": yi, "size_num":int(Test_size.get()), "file_save_name":file_save_name}
    savemat("prediction_matrix_"+str(Test_size.get())+'x'+str(Test_size.get())+".mat", m_dictionary,format='4') #format 4 helps to remain arrays and not struct arrays

def run_lum_fdtd():
    lumapi = imp.load_source("lumapi", "/opt/lumerical/v232/api/python/lumapi.py") #this might change depending on verion of Lumerical installed
    
    fdtd = lumapi.FDTD(filename=r"Desktop/Athena/Metasurface_aSi_RCWA.fsp", hide=False) # import project file .fsp
    
    code = open('Desktop/Athena/Metasurface_RCWA_LumAPI.lsf', 'r').read()# import script file .lsf
    fdtd.eval(code)
    #input() #this is to stop the window from closing immediately

def upload_lum_response():
    root.filename = filedialog.askopenfilename(initialdir =starting_directory, title="Select a file (lum_H1um_..._RandomVal.mat)", filetypes=(("mat files", "*.mat"), ("all files", "*.*")))
    my_label = Label(root, text=root.filename).pack() #print the directory of image
    mat_5 = loadmat(root.filename)
    mat_dict5 = mat_5.items()
    mat_list5 = list(mat_dict5)
    global Lum_Tx, Lum_Ty, Lum_Px, Lum_Py
    Lum_Tx_full = np.reshape(np.asarray(mat_list5[5][1]),(int(Test_size.get()),int(Test_size.get()),51), order='C')
    Lum_Ty_full = np.reshape(np.asarray(mat_list5[6][1]),(int(Test_size.get()),int(Test_size.get()),51), order='C')
    Lum_Px_full = np.reshape(np.asarray(mat_list5[3][1]),(int(Test_size.get()),int(Test_size.get()),51), order='C')
    Lum_Py_full = np.reshape(np.asarray(mat_list5[4][1]),(int(Test_size.get()),int(Test_size.get()),51), order='C')
    #slice to 20nm spectrum for transmission, and center wavelength for phase
    Lum_Tx = Lum_Tx_full[:,:,19:32]
    Lum_Ty = Lum_Ty_full[:,:,19:32]
    Lum_Px = Lum_Px_full[:,:,25]
    Lum_Py = Lum_Py_full[:,:,25]
    Lum_Px = (Lum_Px+pi)/2/pi #normalize phase
    Lum_Py = (Lum_Py+pi)/2/pi
    
def upload_phase_model():
    root.filename = filedialog.askopenfilename(initialdir =starting_directory, title="Select a file", filetypes=(("h5 files", "*.h5"), ("all files", "*.*")))
    my_label = Label(root, text=root.filename).pack()
    global forward_loaded_Phase
    forward_loaded_Phase = tf.keras.models.load_model(root.filename)
    
def upload_trans_model():
    root.filename = filedialog.askopenfilename(initialdir =starting_directory, title="Select a file", filetypes=(("h5 files", "*.h5"), ("all files", "*.*")))
    my_label = Label(root, text=root.filename).pack()
    global forward_loaded_Trans
    forward_loaded_Trans = tf.keras.models.load_model(root.filename)
    
def predict_responses():
    flat_xi = np.reshape(np.asarray(xi),(int(Test_size.get())**2,1), order='C')/2e-7 #flatten
    flat_yi = np.reshape(np.asarray(yi),(int(Test_size.get())**2,1), order='C')/2e-7
    input_dim = np.transpose(np.squeeze(np.asarray([flat_xi, flat_yi])))
    pred_responce_Phase = np.squeeze(forward_loaded_Phase.predict(input_dim))
    pred_responce_Trans = np.squeeze(forward_loaded_Trans.predict(input_dim))
    
    global Pred_Px, Pred_Py, Pred_Tx, Pred_Ty
    Pred_Px = np.reshape(np.asarray(pred_responce_Phase[0]),(int(Test_size.get()),int(Test_size.get())), order='C')
    Pred_Py = np.reshape(np.asarray(pred_responce_Phase[1]),(int(Test_size.get()),int(Test_size.get())), order='C')
    Pred_Tx = np.reshape(np.transpose(np.asarray(pred_responce_Trans[0:13])),(int(Test_size.get()),int(Test_size.get()),13), order='C')
    Pred_Ty = np.reshape(np.transpose(np.asarray(pred_responce_Trans[13:27])),(int(Test_size.get()),int(Test_size.get()),13), order='C')

    #undoing weights
    Pred_Px = Pred_Px/float(Weight_Px.get())
    Pred_Py = Pred_Py/float(Weight_Py.get())
    Pred_Tx = Pred_Tx/float(Weight_Tx.get())
    Pred_Ty = Pred_Ty/float(Weight_Ty.get())

def calculate_errors():
    print('\nMean Absolute Error (MAE)')
    Mean_Abs_Error_Tx = sum(sum(sum(abs(Lum_Tx-Pred_Tx))))/((int(Test_size.get())**2)*13)
    print('Tx: ',Mean_Abs_Error_Tx)
    Mean_Abs_Error_Ty = sum(sum(sum(abs(Lum_Ty-Pred_Ty))))/((int(Test_size.get())**2)*13)
    print('Ty: ',Mean_Abs_Error_Ty)
    Mean_Abs_Error_Px = sum(sum(abs(Lum_Px-Pred_Px)))/(int(Test_size.get())**2)
    print('Px: ',Mean_Abs_Error_Px)
    Mean_Abs_Error_Py = sum(sum(abs(Lum_Py-Pred_Py)))/(int(Test_size.get())**2)
    print('Py: ',Mean_Abs_Error_Py)
    print('\nMean Squared Error (MSE)')
    Mean_Sq_Error_Tx = Mean_Abs_Error_Tx**2
    print('Tx: ',Mean_Sq_Error_Tx)
    Mean_Sq_Error_Ty = Mean_Abs_Error_Ty**2
    print('Ty: ',Mean_Sq_Error_Ty)
    Mean_Sq_Error_Px = Mean_Abs_Error_Px**2
    print('Px: ',Mean_Sq_Error_Px)
    Mean_Sq_Error_Py = Mean_Abs_Error_Py**2
    print('Py: ',Mean_Sq_Error_Py)

def helper_plot_P_func(from_lum, predicted, variable_name):
    plt.rcParams['figure.figsize']=(8,4)
    plt.rcParams['figure.dpi']=100
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    ax = plt.subplot(1,2,1)
    cm = from_lum
    cax = ax.matshow(cm, cmap=plt.cm.Blues,vmin=0, vmax=1)
    plt.colorbar(cax,label=variable_name)
    #ax.remove()

    ax = plt.subplot(1,2,2)
    cm = predicted
    cax = ax.matshow(cm, cmap=plt.cm.Blues,vmin=0, vmax=1)
    plt.colorbar(cax,label=variable_name)
    plt.savefig(variable_name+".png")
    
def helper_plot_T_func(from_lum, predicted, variable_name, n): #n correspond to wavelength
    plt.rcParams['figure.figsize']=(8,4)
    plt.rcParams['figure.dpi']=100
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    ax = plt.subplot(1,2,1)
    cm = from_lum[:,:,n]
    cax = ax.matshow(cm, cmap=plt.cm.Blues,vmin=0, vmax=1)
    plt.colorbar(cax,label=variable_name)
    ax = plt.subplot(1,2,2)
    cm = predicted[:,:,n]
    cax = ax.matshow(cm, cmap=plt.cm.Blues,vmin=0, vmax=1)
    plt.colorbar(cax,label=variable_name)
    plt.savefig(variable_name+"_CenterWave.png")
    
def plot_comparison():
    helper_plot_P_func(Lum_Px, Pred_Px, "Phase X")
    helper_plot_P_func(Lum_Py, Pred_Py, "Phase Y")
    helper_plot_T_func(Lum_Tx, Pred_Tx, "Transmission X", 6)
    helper_plot_T_func(Lum_Ty, Pred_Ty, "Transmission Y", 6)
    
def flatten_data (data):
    return np.reshape(data,((int(Test_size.get())**2),13), order='C')

def plot_spectrum_pred (n, x_axis, y_ideal, y_pred, name): #where n is r1 r2 values in single vector
    plt.clf() #clears plot first
    plt.rcParams['figure.figsize']=(6,4)
    plt.plot(x_axis, y_pred[n], c="r", label="Predictions", marker='o')
    plt.plot(x_axis, y_ideal[n], c="g", label="Validation", marker='o')
    plt.xlabel("Wavelength (nm)")
    plt.ylabel(name)
    plt.ylim([0,1.1])
    plt.legend(loc='upper right', borderaxespad=-3.5)
  
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
    
    flat_pred_Tx = flatten_data(Pred_Tx)
    flat_pred_Ty = flatten_data(Pred_Ty)
    flat_ideal_Tx = flatten_data(Lum_Tx)
    flat_ideal_Ty = flatten_data(Lum_Ty)
    
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


#Specify size of the testing dataset (defualt = 50)
Test_size_label = Label(root, text="Specify the size of the testing dataset (defualt = 50):", font=('Arial', 10)).pack()
Test_size = Entry(root, width = 50, borderwidth=5)
Test_size.insert(0, "50") #default size
Test_size.pack()
separator0 = ttk.Separator(root, orient='horizontal').pack(fill=X)
    
#option: would you like to generate a new random dimensions dataset?
#yes = generate new dimensions from uniform distribution
#no = upload previously generated dimensions
step_1_label = Label(root, text="\nStep 1: Dimensions Dataset", font=('Arial', 15)).pack()
generate_or_not_label = Label(root, text="\nWould you like to generate a new random dataset or upload existing ones?", font=('Arial', 12)).pack()
generate_new_label = Label(root, text="Option 1: Generate a new random dimensions dataset", font=('Arial', 12)).pack()
generate_new_btn = Button(root, text="Generate", font=('Arial', 10), command=generate_random_dim)
generate_new_btn.pack()

download_new_btn_csv = Button(root, text="Download random dimenisons (.csv)", font=('Arial', 8), command=download_dim_csv)
download_new_btn_csv.pack()

download_new_btn_pkl = Button(root, text="Download random dimenisons (.pkl)", font=('Arial', 8), command=download_dim_pkl)
download_new_btn_pkl.pack()

upload_previous_label = Label(root, text="Option 2: Upload previously generated dataset (.csv file)", font=('Arial', 12)).pack()
upload_Dx_btn = Button(root, text="Upload Dx", font=('Arial', 10), command=upload_dim_Dx)
upload_Dx_btn.pack()
upload_Dy_btn = Button(root, text="Upload Dy", font=('Arial', 10), command=upload_dim_Dy)
upload_Dy_btn.pack()

separator1 = ttk.Separator(root, orient='horizontal').pack(fill=X)

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
separator2 = ttk.Separator(root, orient='horizontal').pack(fill=X)

#upload forward models
step_3_label = Label(root, text="\nStep 3: Upload forward models (.h5 file)", font=('Arial', 15)).pack()
upload_forw_phase_btn = Button(root, text="Upload phase model", font=('Arial', 10), command=upload_phase_model)
upload_forw_phase_btn.pack()
upload_forw_trans_btn = Button(root, text="Upload transmission model", font=('Arial', 10), command=upload_trans_model)
upload_forw_trans_btn.pack()

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
separator3 = ttk.Separator(root, orient='horizontal').pack(fill=X)

#run forward model predicitons
step_4_label = Label(root, text="\nStep 4: Evaluation", font=('Arial', 15)).pack()
run_pred_label = Label(root, text="\nClick to run forward model predictions:", font=('Arial', 12)).pack()
run_pred_button = Button(root, text="Predict", command=predict_responses).pack()

#print out MAE and MSE losses
calc_errors_label = Label(root, text="\nClick to view error values on console:", font=('Arial', 12)).pack()
calc_errors_button = Button(root, text="Calculate Errors", command=calculate_errors).pack()

#plot phase and transmission
plot_compare_label = Label(root, text="\nPlot phase and transmission comparison graphs:", font=('Arial', 12)).pack()
plot_compare_button = Button(root, text="Generate plots", command=plot_comparison).pack()

#plot transmission spectrum
spectrum_plot_label = Label(root, text="\nPlot transmission 20 nm spectrum:", font=('Arial', 12)).pack()
spectrum_plot_button = Button(root, text="Generate plots", command=plot_trans_spectrum).pack()
separator4 = ttk.Separator(root, orient='horizontal').pack(fill=X)

root.mainloop()














