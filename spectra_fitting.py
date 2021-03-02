# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 23:51:31 2020

@author: Roman
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 17:35:19 2020

Desired functions:
    
    - Intelligent peak recognition
    - Peak position extraction
    - Impedance value calculation
    - simultaneous fitting of multiple curves with boundary conditions
    - intelligent test group recognition via NLP
            - providing statistical tests


@author: Administratio"""


import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import random as rd
import seaborn as sns

import os, fnmatch
from fnmatch import fnmatch
from os import path
import log_functions as lg
import inspect
import sys

import scipy as scipy

from scipy import interpolate
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d


import scipy.cluster.hierarchy as sch



def cluster_distance(data, f_threshold=0.5): #Done
    
    """
    Rearranges the distance/similarity matrix, corr_array, so that groups of low distance/ 
    high similarty are next to eachother 
    
    Parameters
    ----------
    data : pandas.DataFrame or numpy.ndarray
        a NxN distance or similarty matrix 
        
    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        a NxN distance or similarty matrix with the columns and rows rearranged
    """
    
    pairwise_distances = sch.distance.pdist(data)

    linkage = sch.linkage(pairwise_distances, method='complete')
    
    cluster_distance_threshold = pairwise_distances.max()*f_threshold
    
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold, 
                                        criterion='distance')  
    
    
    group_df = pd.DataFrame({'group':idx_to_cluster_array,'words':data.columns.values})
    group_keys = list(set(idx_to_cluster_array))
    
    groups = []
    
    for key in group_keys:
        groups.append(group_df[group_df['group'].isin([key])]['words'].tolist())
        
    
    
    idx = np.argsort(idx_to_cluster_array)
    
    if isinstance(data, pd.DataFrame):
        return data.iloc[idx, :].T.iloc[idx, :], groups
    
    return data[idx, :][:, idx], idx, idx_to_cluster_array

def get_file_list(pattern, avoid, root): #Done
    """
    

    Parameters
    ----------
    pattern : list of strings
        contains the text pattern that should be in the file name.
    avoid : list of strings
        contains the text parttern that must not be in the file name.
    root : string
        file path containing the data. Subdirectrories are also taken into account

    Returns
    -------
    result : TYPE
        DESCRIPTION.

    """
    root = root.replace('\\','/')
    
    file_list = []
    name_list = []
    final_list= []
        
    """Get list of all files in directory and the subdirectories with *.txt pattern"""
    for path, subdirs, files in os.walk(root):
        for name in files:
            if fnmatch(name,'*'+pattern+'*.txt'):
                spec_file = os.path.join(path, name)
                name_list.append(name)
                file_list.append(spec_file)
    
    
    for file in file_list:
        
        a_count = 0
        for item in avoid:
            if fnmatch(file, '*'+item+'*') == True:
                a_count = a_count+1
        
        if a_count == 0:
            final_list.append(file)
    
    #return file list (paths) as dataframe
    result = pd.DataFrame({'file_path':final_list})
    
    os.chdir(root)
            
    return  result

def load_clean_data(factor, file_path): #Done
    """
    

    Parameters
    ----------
    factor : float
        measurement area correction factor.
    file_path : string
        file path of impedance data.

    Returns
    -------
    result : pandas DataFrame
        contains log10(frequency), real and negative imaginary part of impedance.

    """
    table = pd.read_csv(file_path, sep="\t",skiprows=1, header=None)
    table = table.sort_values([2])
    table = table.reset_index()
    
    f_raw = np.log10(table[2])
    real_z_raw = np.multiply(factor,table[0])
    imag_z_raw = np.multiply(factor,table[1])
    
    positive_list = []
    
    for i in range(0, len(imag_z_raw)):
        if imag_z_raw[i] >= 0:
            positive_list.append(i)
    
    p_min = min(positive_list)
    p_max = max(positive_list)+1

    
    f = f_raw[p_min:p_max]
    real_z = real_z_raw[p_min:p_max]
    imag_z = imag_z_raw[p_min:p_max]
    
    result = pd.DataFrame({'frequency':f, 'real_z': real_z, 'imag_z': imag_z})
    result = result.reset_index()
    
    return result              

def interp_derivative(x,y,start, end, increment, window, p_order, d_order, output):  
    lg.function_log()
    x_list = x
    y_list = y
    
    for n in range(0,d_order):
        
        if  window > len(y_list):
            window = 11
        
        filter_curve = savgol_filter(y_list, window, p_order)
        spline = UnivariateSpline(x_list,filter_curve, k=4)
        spline_der = spline.derivative()
        
        x_list = np.arange(start, end, increment)
        y_list = spline_der(x_list)
        
    roots_list = spline_der.roots()
    
    filter_curve = savgol_filter(y_list, window, p_order)
    spline = UnivariateSpline(x_list,filter_curve, k=4)
    spline_der2 = spline.derivative()
    
    roots_vals = spline_der2(roots_list)
    
    peak_x = []
    
    for k in range(0,len(roots_vals)):
        if roots_vals[k] < 0:
            peak_x.append(roots_list[k])
    
    filter_curve = savgol_filter(y, window, p_order)
    spline = UnivariateSpline(x,filter_curve, k=4)
    peak_y = spline(peak_x)
    peak_y = peak_y.tolist()
    
    if output == "data":
        result_data = {'x':x_list, 'y':y_list}
        
    if output == "peaks":
        result_data = {'peak_x':peak_x ,'peak_y':peak_y}
        
    result = pd.DataFrame(result_data)
    
    return result


def multiple_gauss(x, *args):
    
    gauss = 0
     
    
    if len(args) == 1:
        arg_list = args[0]
    else:
        arg_list = list(args)
    
    k_temp = len(arg_list)/3   
    k = int(k_temp)
    
    amp = arg_list[0:k]
    cen = arg_list[1*k:1*k+k]
    sigma = arg_list[2*k:2*k+k]
    

    for n in range(0,len(amp)):
        if sigma[n] == 0:
            value = 0*x
        else:
            value = amp[n]*(np.exp((-1.0/2.0)*(((x-cen[n])/sigma[n])**2)))
        
        gauss = gauss + value
        #gauss = gauss + amp[n]*(1/(sigma[n]*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-cen[n])/sigma[n])**2)))
        
    return gauss

def generate_bounds(params, error):
    lg.function_log()
    bounds_min = []
    bounds_max = []
    
    for i in params:
        bounds_min.append(i-error)
        bounds_max.append(i+error)
               
    bnds = ((*bounds_min,), (*bounds_max,))
    
    return bnds
            
def bounds_merge(merged_bounds):
    lg.function_log()
    
    bounds_min = []
    bounds_max = []
    
    for i in merged_bounds:
        bounds_min = bounds_min + (list(i[0]))
        bounds_max = bounds_max + (list(i[1]))
               
    bnds = ((*bounds_min,), (*bounds_max,))
    
    return bnds
    
def peak_extract(sg_window, sg_poly, dev_order, data, plot):
    lg.function_log()
    
      
    f_min_list = []
    f_max_list = []
    
    f_inc_list = []
    
    peaks_array = []
    
    
    file_list = []
    
    if isinstance(data, str):
        file_list.append(data)
    else:
        if isinstance(data, list):
            file_list = data
        else:
            file_list = data['file_path'].tolist()
    
    """create list of all peaks found in the datasets"""
    for i in range(0,len(file_list)):

        file = load_clean_data(1,file_list[i])
        
        f = file['frequency']
        imag_z = file['imag_z']
            
        min_f = round(min(f),1)
        max_f = round(max(f),1)
    
        f_min_list.append(min_f)
        f_max_list.append(max_f)
  
        f_inc = round(abs((f[1])-(f[0])),2)
        
        f_inc_list.append(f_inc)
            
        """extraction of peaks from interpolation curve"""   
        #d_Z = interp_derivative(f, imag_z, min_f, max_f, f_inc, sg_window, sg_poly, dev_order, "data")
        peaks_d1 = interp_derivative(f, imag_z, min_f, max_f, f_inc, sg_window, sg_poly, 1, "peaks")
        peaks_dn = interp_derivative(f, imag_z, min_f, max_f, f_inc, sg_window, sg_poly, dev_order, "peaks")
        
        peaks_d1_list = (peaks_d1["peak_x"].values.tolist())
        peaks_dn_list = (peaks_dn["peak_x"].values.tolist())
        peaks_array = peaks_array+peaks_d1_list+peaks_dn_list
        
    hist_start = min(f_min_list)-0.5
    hist_end = max(f_max_list)+1
    hist_bins = max(f_inc_list)


    """create histogram"""
    peak_hist, peak_bins = np.histogram(peaks_array, bins = np.arange(hist_start,hist_end,hist_bins))
    plot_bins = (peak_bins[:len(peak_bins)-1])

    """smoothing of histogram via gaussian filter"""

    hist_filter_1 = savgol_filter(peak_hist,7,5)
    hist_filter = gaussian_filter1d(hist_filter_1,2.5)

    """interpolation of histogram"""
    hist_spline = UnivariateSpline(plot_bins, hist_filter, k=4, s=0)

    """higher number of bins for analysis of interpolation curve"""
    new_bins = np.arange(0,hist_end,f_inc*0.1)

    """calculation of derivatives to find local maxima"""
    d_hist_spline = hist_spline.derivative()
    d2_hist_spline = hist_spline.derivative(2)
    d_hist_roots = d_hist_spline.roots()


    """only extract roots with positive values for d2/d2x(root)"""
    find_peaks_result = []

    for n in range(0,len(d_hist_roots)):
        if d_hist_roots[n]>min(new_bins):
            if d_hist_roots[n]<max(new_bins):
                if d2_hist_spline(d_hist_roots[n])<0:
                    find_peaks_result.append(d_hist_roots[n])
    
    cen_list = []
    
    for n in range(0, len(find_peaks_result)):
        if hist_spline(find_peaks_result[n])>0:
            cen_list.append(find_peaks_result[n])
           
    """refinement with gaussian peak fitting"""
    sigma_peak_find = 0.2
    params = []
    
    amp_list = list(hist_spline(cen_list))
    
    #sigma_list = list([sigma_peak_find])*len(find_peaks_result)
    """
    print('detected peaks')
    print(find_peaks_result)
    print('these are the amplitudes')
    print(hist_spline(find_peaks_result))
    """
    if len(cen_list)>0:
        params, bnds = fit_params(amp_list, cen_list, hist_start, hist_end, 0.1, 1.25, 0.5, 2)
        
    
        hist_gauss, errs_gauss = scipy.optimize.curve_fit(multiple_gauss, plot_bins, hist_filter, p0=params, bounds = bnds)
        g_hist = multiple_gauss(new_bins, hist_gauss)
        
        """plot peak search result"""
    
        if plot == True:
            plt.plot(plot_bins, peak_hist, label = 'peak_hist')
            plt.plot(plot_bins, hist_filter, label = 'hist_filter')
            plt.plot(new_bins, d_hist_spline(new_bins), label = 'd_hist_spline')
            #plt.plot(new_bins, multiple_gauss(new_bins, *params), label = 'gauss')
            plt.plot(new_bins,g_hist)
        
            folder_text, sample_text = get_sample_name(file_list[i])
            
            plt.title(sample_text)
            plt.legend()
            plt.show()
    
        n_peaks = len(cen_list)
        gauss_peaks = []
        gauss_sigma = []
    
        errs_peaks = []
        errs_sigma = []

        for a in range(0,n_peaks):
            if hist_gauss[a]>(0.2*np.mean(hist_gauss[0:n_peaks])):
                gauss_peaks.append(hist_gauss[a+n_peaks])
                gauss_sigma.append(abs(hist_gauss[a+n_peaks*2]))
            
                errs_peaks.append(errs_gauss[a+n_peaks])
                errs_sigma.append(abs(errs_gauss[a+n_peaks*2]))

    else:
        print('no local peaks found')
        gauss_peaks = []
        gauss_sigma = []
    
    
    return gauss_peaks, gauss_sigma                

def histogram_maxima(peak_hist, plot_bins, hist_sigma, plot):
    lg.function_log()

    #hist_filter_1 = savgol_filter(peak_hist,5,3)
    #hist_filter = gaussian_filter1d(hist_filter_1,0.1)
    
    gauss_sigma = max(plot_bins)*hist_sigma
    
    """interpolate histogram for higher resolution"""
    
    interp_hist = interpolate.interp1d(plot_bins, peak_hist)
    
    new_inc = (plot_bins[1]-plot_bins[0])/10
    
    new_bins = np.arange(min(plot_bins), max(plot_bins), new_inc)
    
    
    
    hist_filter = gaussian_filter1d(interp_hist(new_bins),gauss_sigma)

    """interpolation of histogram"""
    hist_spline = UnivariateSpline(new_bins, hist_filter, k=4, s=0)

    """higher number of bins for analysis of interpolation curve"""
    #new_bins = np.arange(0,max(peak_hist),0.1)

    """calculation of derivatives to find local maxima"""
    d_hist_spline = hist_spline.derivative()
    d2_hist_spline = hist_spline.derivative(2)
    d_hist_roots = d_hist_spline.roots()


    """only extract roots with positive values for d2/d2x(root)"""
    find_peaks_result = []

    for n in range(0,len(d_hist_roots)):
        if d_hist_roots[n]>min(new_bins):
            if d_hist_roots[n]<max(new_bins):
                if d2_hist_spline(d_hist_roots[n])<0:
                    find_peaks_result.append(d_hist_roots[n])
    
        
    if plot == True:
        plt.plot(plot_bins, peak_hist, label = 'center of mass histogram')
        #plt.plot(plot_bins, hist_filter_1, label = 'savgol_filter')
        plt.plot(new_bins, hist_filter, label = 'gauss_filter')
        plt.plot(new_bins, d_hist_spline(new_bins), label = 'derivative')

        plt.legend()
        plt.show()

    
    cen_list = []
    
    for n in range(0, len(find_peaks_result)):
        if hist_spline(find_peaks_result[n])>0:
            cen_list.append(np.round(find_peaks_result[n],0))
            
    return cen_list


def _1gaussian(x, amp1,cen1,sigma1):
    lg.function_log()
    if sigma1 == 0:
        gauss = 0*x
    else:
        gauss = amp1*(np.exp((-1.0/2.0)*(((x-cen1)/sigma1)**2)))
    return gauss


def fit_params(amp_list, cen_list, min_x, max_x, sigma_min, sigma_max, u_fit, div_cen): 
    lg.function_log()

    bnds_amp_min = []
    bnds_cen_min = []
    bnds_sigma_min = []
    
    bnds_amp_max = []
    bnds_cen_max = []
    bnds_sigma_max = []
    
    
    fit_amp = []
    fit_cen = []
    fit_sigma = []
        
    
    
    
    d_cen_list = []
    
    for j in range(0,len(amp_list)):

        f_amp = amp_list[j]
        f_cen = cen_list[j]
            
        fit_amp.append(f_amp)
        fit_cen.append(f_cen)
        
        if j == 0:
            d_min = abs(min_x-cen_list[j])
            d_cen_list.append(d_min)
                       
        if j == (len(amp_list)-1):
            d_cen = abs(max_x-cen_list[j])/2
            d_cen_list.append(d_cen)
        else:
            d_int = abs(cen_list[j]-cen_list[j+1])/div_cen
            d_cen_list.append(d_int)


        d_amp = f_amp*u_fit            
        bnds_amp_min.append(f_amp-d_amp)
        bnds_amp_max.append(f_amp+d_amp)
    
    for i in range(0,len(fit_cen)):
        bnds_cen_min.append(fit_cen[i]-d_cen_list[i])
        bnds_cen_max.append(fit_cen[i]+d_cen_list[i+1])
        
        f_sigma = min(max((d_cen_list[i]+d_cen_list[i+1])/2,0.5),1.24)
        #d_sigma = f_sigma*u_fit
        
        
        
        bnds_sigma_min.append(sigma_min)
        bnds_sigma_max.append(sigma_max)
        
        
        
        fit_sigma.append(f_sigma)


    if sigma_max == sigma_min:
        bounds_min = bnds_amp_min+bnds_cen_min  
        bounds_max = bnds_amp_max+bnds_cen_max
        params = fit_amp+fit_cen
    else:
        bounds_min = bnds_amp_min+bnds_cen_min+bnds_sigma_min   
        bounds_max = bnds_amp_max+bnds_cen_max+bnds_sigma_max
        params = fit_amp+fit_cen+fit_sigma
                       
    bnds = ((*bounds_min,), (*bounds_max,))
    

    return params, bnds 
    
def max_peaks(peak_list, n_peaks, data):
    lg.function_log()
    
    peaks = np.array(peak_list)
    
    f = data['frequency']
    imag_z = data['imag_z']
    
    y_int = UnivariateSpline(f, imag_z, k=4)
    
    y_peaks = y_int(peaks)
    
    df = pd.DataFrame({'f': peaks, 'y': y_peaks})
    
    df = df.sort_values(['y'],ascending=False)
    df = df.reset_index()
    
    
    result = df[0:(n_peaks)]
    
    return result
    
def peak_filter(global_peaks, f_min, f_max):
    lg.function_log()
    
    global_peaks = sorted(global_peaks)
    
    peak_list = []
    peak_index = []
    
    for i in range(0,len(global_peaks)):
        if f_min <= global_peaks[i] <= f_max:
            peak_list.append(global_peaks[i])
            peak_index.append(i)
            
    return peak_list, peak_index

def df_append(path, name, f_min, f_max, im_max, real_max, exp_params_list, global_peaks, error, loop_index, df):
    lg.function_log()
    amp_txt = []
    cen_txt = []
    sigma_txt = []
    fit_values = params_align(global_peaks, exp_params_list)
    
    df_values = list([path])+list([name])+list([f_min])+list([f_max])+list([im_max])+list([real_max])+list([error])+fit_values
    

    if loop_index == 0:
        
        for i in range(0,len(global_peaks)):
        
            amp_txt.append('im_Z_'+str(i+1))
            cen_txt.append('freq_'+str(i+1))
            sigma_txt.append('sigma_'+str(i+1))
            
            column_list = amp_txt+cen_txt+sigma_txt
        
        
        df_list = ['path', 'name', 'f_min', 'f_max', 'im_max', 'real_max', 'error']+column_list
        
        for j in range(0,len(df_list)):
            df.insert(j, df_list[j], df_values[j],True)
            
        
        df.loc[loop_index] = df_values
    else:
        df.loc[loop_index] = df_values

def get_sample_name(file_path):
    lg.function_log()
    
    file_path = file_path.replace('\\', '/')
    
    file_path_list = file_path.split('/')
    
    parent_folder = file_path_list[len(file_path_list)-2]
    sample_name = file_path_list[len(file_path_list)-1]
    
    block_end = sample_name.find("_")
    sample_name = sample_name[block_end+1:]
    sample_name = sample_name.replace(".txt","")
    
    result = parent_folder, sample_name

    
    return result

def peaks_align(global_peaks, local_peaks, fill_zero):
    lg.function_log()
    
    d_n = abs(len(global_peaks)-len(local_peaks))
    
    score_list = []
    
    for i in range(0, (d_n+1)):
        
        score = 0
        
        for j in range(0, len(local_peaks)):
            score = score+abs(global_peaks[i+j]-local_peaks[j])
            
        score_list.append(score)
        
    max_score = min(score_list)
    max_index = score_list.index(max_score)
    
    zero = [0]*len(global_peaks)
    
    for i in range(0, len(global_peaks)):
        if i == max_index:
            for j in range(0, len(local_peaks)):
                if fill_zero == True:
                    zero[i+j] = local_peaks[j]
                else:
                    global_peaks[i+j] = local_peaks[j]
   
    if fill_zero == True:
        result = zero
    else:
        result = global_peaks
        
    return result

def params_align(global_peaks, exp_fit_params):
    lg.function_log()
         
    n_peaks = len(global_peaks)    
    n_params = round(len(exp_fit_params)/3)
    
    amp_list = exp_fit_params[0:n_params]
    cen_list = exp_fit_params[n_params:n_params*2]
    sigma_list = exp_fit_params[n_params*2:n_params*3]
    
    amp_new = [0]*n_peaks
    cen_new = [0]*n_peaks
    sigma_new = [0]*n_peaks
    
    local_peaks = cen_list
    d_n = abs(len(global_peaks)-len(local_peaks))
    
    score_list = []
    
    for i in range(0, (d_n+1)):
        
        score = 0
        
        for j in range(0, len(local_peaks)):
            score = score+abs(global_peaks[i+j]-local_peaks[j])
            
        score_list.append(score)
        
    max_score = min(score_list)
    max_index = score_list.index(max_score)
    
    zero = [0]*len(global_peaks)
    
    for i in range(0, len(global_peaks)):
        if i == max_index:
            for j in range(0, len(local_peaks)):
                    amp_new[i+j] = amp_list[j]
                    cen_new[i+j] = cen_list[j]
                    sigma_new[i+j] = sigma_list[j]
                    
                    
    result = amp_new+cen_new+sigma_new
        
    return result
    
def generate_peaks(gauss_peaks, f, imag_z):
    lg.function_log()
    
    
    peak_df = pd.DataFrame()
    
    y_int = UnivariateSpline(f, imag_z, k=4, s=0)
    
    sigma_list = list([0.5])*len(gauss_peaks)
    cen_list = gauss_peaks
    amp_list = y_int(gauss_peaks)
    
    gauss_params = list(amp_list)+list(cen_list)+list(sigma_list)
    
    n_peaks = len(gauss_peaks)
    
    for n in range(0, n_peaks):
        amp1 = gauss_params[n]
        cen1 = gauss_params[n+n_peaks]
        sigma1 = gauss_params[n+n_peaks*2]
        
        gauss_curve = _1gaussian(f, amp1, cen1, sigma1)
        peak_df.insert(n, str(n), gauss_curve)
    
    return peak_df

def peaks_assign(global_peaks, exp_fit_params, f_min, f_max):
    lg.function_log()
    
    peak_ranges = []
    
    n_peaks = len(global_peaks)
    
    n_params = round(len(exp_fit_params)/3)
    
    amp_list = exp_fit_params[0:n_params]
    cen_list = exp_fit_params[n_params:n_params*2]
    sigma_list = exp_fit_params[n_params*2:n_params*3]
    
    
    print(amp_list)
    print(cen_list)
    print(sigma_list)
    

    p_list = []
    d_list = []
    
    for d in range(0, n_peaks-1):
        d_f = global_peaks[d+1]-global_peaks[d]
        
        f_x = global_peaks[d]+d_f
        
        d_list.append(f_x)
    
    
    
    p_list.append(f_min)
    p_list = p_list + d_list
    p_list.append(f_max)
    
    print(p_list)
    
    amp_new = [0]*n_peaks
    cen_new = [0]*n_peaks
    sigma_new = [0]*n_peaks
    
    for j in range(0, len(cen_list)):
        
        for i in range(0,n_peaks):
        
            if p_list[i] < cen_list[j] <= p_list[i+1]:
                amp_new[i] = amp_list[j]
                cen_new[i] = cen_list[j]
                sigma_new[i] = sigma_list[j]
                
    result = amp_new+cen_new+sigma_new
       
            
    return result

def fit_spectrum(sg_global, sg_local, sg_poly, dev_global, dev_local, u_fit, div_cen, data, n_peaks, peaks):
    lg.function_log()
    
    fail_df = pd.DataFrame()
    fail_df.insert(0,'fail_path', 'dummy')
    file_list = []

    n_success = 0    

    if isinstance(data, str):
        file_list.append(data)
    else:
        if isinstance(data, list):
            file_list = data
        else:
            file_list = data['file_path'].tolist()

    
    output_df = pd.DataFrame()
    
    
    gauss_peaks, gauss_sigma = peak_extract(sg_global, sg_poly, dev_global, data, True)
    global_peaks = gauss_peaks

    
    for i in range(0,len(file_list)):
        
        peak_inc = 0
        
        if peaks == 'single':
            gauss_peaks, gauss_sigma = peak_extract(sg_local, sg_poly, dev_local, file_list[i], True)
            if len(gauss_peaks) > len(global_peaks):
                while len(gauss_peaks) > len(global_peaks):
                    gauss_peaks, gauss_sigma = peak_extract(sg_local+peak_inc, sg_poly, dev_local, file_list[i], True)
                    peak_inc = peak_inc + 2
        
        if  1 < len(gauss_peaks) <= len(global_peaks):
            print(str(len(gauss_peaks))+'local peaks used')
            #print(gauss_peaks)
        else:
            print(str(len(global_peaks))+' global peaks used')
            gauss_peaks = global_peaks
            #print(gauss_peaks)
        
        
        folder_name, sample_name = get_sample_name(file_list[i])
               
        print(sample_name)
        
        test_file = load_clean_data(1, file_list[i])    
        
        f = test_file['frequency']
        imag_z = test_file['imag_z']
        real_z = test_file['real_z']
                
        """Filtern der Peaks über den Frequenzbereich"""
        
        gauss_peaks = peaks_align(global_peaks, gauss_peaks, False)
        
        gauss_peaks, peak_index = peak_filter(gauss_peaks, min(f), max(f))
        
        

        imag_int = UnivariateSpline(f, imag_z, k=4, s=0)
        
        amp_list = list(imag_int(gauss_peaks))
        cen_list = gauss_peaks
        
        print(str(len(cen_list))+' peaks used for fitting')
        #print(cen_list)
        
        sigma_min = 0.5
        sigma_max = 0.8
        
        p_init, bnds = fit_params(amp_list, cen_list, min(f), max(f), sigma_min, sigma_max, u_fit, div_cen)
        

        """
        exp_fit_params, exp_fit_errs = scipy.optimize.curve_fit(multiple_gauss, f, imag_z, p0=p_init, bounds = bnds)
        df_append(file_list[i], sample_list[i], min(f), max(f), exp_fit_params, global_peaks, i, output_df)
        
        """
        ff = 0
        
        
        
        try:
            
            if sigma_min == sigma_max:
                
                exp_fit_params, exp_fit_errs = scipy.optimize.curve_fit(multiple_gauss_fix, f, imag_z, p0=p_init, bounds = bnds)
                success = True
                exp_fit_params = exp_fit_params.tolist()+len(cen_list)*[0.5]
                exp_fit_params = np.array(exp_fit_params)
            else:
                exp_fit_params, exp_fit_errs = scipy.optimize.curve_fit(multiple_gauss, f, imag_z, p0=p_init, bounds = bnds)
                success = True
            
            
            

            
            #df_append(file_list[i], sample_name, min(f), max(f), exp_fit_params, global_peaks, i, output_df)
        except:
            success = False
            print(sample_name+' failed')
            
            fail_df.loc[ff] = file_list[i]
            
            #exp_fit_param = p_init
            #df_append(sample_list[i], min(f), max(f), exp_fit_params, global_peaks, i, output_df)
            pass
        
        if success == True:
            error = total_fit_error(f, imag_z, multiple_gauss(f, *exp_fit_params))
            df_append(file_list[i], sample_name, min(f), max(f), max(imag_z), max(real_z), exp_fit_params, global_peaks, error, n_success, output_df)
            n_success = n_success+1
            
    print(str(len(file_list)-n_success)+' curves could not be fitted')
    return output_df

def total_fit_error(x, y_real, y_fit):
    lg.function_log()
    err = 0
    
    
    for i in range(0,len(x)):
        err = err+(y_real[i]-y_fit[i])**2
    
    return err    

def Z_R_CPE(log_f,n,R,Q):
    lg.function_log()
    
    f = 10**log_f
    
    omega = 2*np.pi*f
    
    Z_CPE = 1/((1j*omega)**n*Q)
    
    Z_R = complex(R,0)

    
    Z_res = 1/(1/Z_R+1/Z_CPE)
     
    Z_imag = -Z_res.imag
    Z_real = Z_res.real
    
    return Z_imag    

def R_CPE_fit(x, amp1, cen1, sigma1):
    lg.function_log()
        
    y = _1gaussian(x, amp1,cen1,sigma1)
    
    params = [1, amp1*2, 0.000001]
    
    bnds = ((0.6,1,1E-14),(1,1E10, 1E-2))
    
    values, errors = scipy.optimize.curve_fit(Z_R_CPE, x, y, p0 = params, bounds=bnds)
        
    return values

def get_Im_Z(data):
    lg.function_log()
    
    data = data['file_path']
    
    df = pd.DataFrame()
    n = 0
    
    for i in data:
        raw = load_clean_data(1, i)
        frequency = raw['imag_z']
        
        df.insert(n, i, frequency, True)
        
        n = n+1
        
    return df

        
def get_sample_dataset(data):
    lg.function_log()
    file_list = []

    
    if isinstance(data, list):
        file_list = data
    
    if isinstance(data, pd.DataFrame):
        file_list = list(data['file_path'])
        
    if isinstance(data, str):
        file_list.append(data)
    
    if len(file_list)>1:
        random_index = rd.randint(0,len(file_list)-1)
    else:
        random_index = 0       
    

    test_file = load_clean_data(1, file_list[random_index])    
    f = test_file['frequency']
    imag_z = test_file['imag_z']
    name = file_list[random_index]
        
    return name, f, imag_z
        

"""/////////////////////////////////
///////////////////////////////////
//////////////////////////////////
/////////////////////////////////
////////////////////////////////
///////////////////////////////
//////////////////////////////

    test script starting form here"""

def main():
    lg.function_log()
    
    
    
    
    """get data"""
    
    root = r'E:\PhD\Finale Übergabe\Derma Imp\Analyse\237_Derma Imp\raw curves'
    pattern = 'Block'
    avoid = []
    
    data = get_file_list(pattern, avoid, root)
    
    n_samples = len(data)
    print(str(n_samples)+' datasets will be analysed')
    
    
    
    
    
    """set parameters for peak extraction"""
    
    sg_global = 11
    sg_local = 7
    
    sg_poly = 3
    
    dev_global = 3
    dev_local = 1
    
    u_fit = 0.95
    
    div_cen = 3
    
    """choose if global peaks should be adjusted"""
    
    #peak_select = 'all'
    peak_select = 'single'
    
    """plot random data set"""
    plt.rcParams.update({'font.size': 14})
       
    width = 14
    height = 3
    scaling = 1
    fig = plt.figure(figsize=(width*scaling, height*scaling*2))
    
    
    
    print(data)
    
    """2. fit all curves - use globally extracted peaks"""
    
    print('start curve analysis?')
    
    if input() == 'y':
        #threshold for correlation cluster binarization
        threshold = 0.97
        
        try:
            test_f = get_Im_Z(data)
            corr_map = test_f.corr()
            corr_map = cluster_corr(corr_map)
            
            data_groups = corr_divide(corr_map, threshold)
            data = data_groups
        except:
            data_groups = []
            data_groups.append(data['file_path'].tolist())
            pass
        
        
        
        
        result_list = [] 
        result_names = []
        
        for i in range(0,len(data_groups)):
            
            data = data_groups[i]
            test_name, f, imag_z = get_sample_dataset(data)
            
            print(test_name)
            
            proceed = False
            
            while proceed == False:
                """1. Check peak extraciont"""
                gauss_peaks, gauss_sigma = peak_extract(sg_global, sg_poly, dev_global, data, True)
                print(str(len(gauss_peaks))+' used for fitting')
                
                local_peaks, local_sigma = peak_extract(sg_local, sg_poly, dev_local, test_name, True)
                
                
                peak_df = generate_peaks(local_peaks, f, imag_z)
                
                """display peaks"""
                plt.plot(f, imag_z, label = 'sample dataset')
                for j in range(0, len(local_peaks)):
                    plt.plot(f, peak_df[str(j)],label = 'peak '+str(j))
                plt.legend()
                plt.show()
                
                print('peak extraction ok?')
                
                if input() == 'y':
                    break
                else:
                    print('enter global filter window (odd, greater then 7)')
                    sg_global = int(input())
                    print('enter local filter window (odd, greater then 5)')
                    sg_local = int(input())
                    print(sg_global)
            
            print('proceed?')
            
            if input() != 'y':
                sys.exit()
            
            data = data_groups[i]
            print('Fitting group number '+str(i))
            result = fit_spectrum(sg_global, sg_local, sg_poly, dev_global, dev_local, u_fit, div_cen, data, 3, peak_select)
            result_names.append(r'\pattern '+pattern+'_'+str(i)+'.csv')
            result_list.append(result)
    else:
        sys.exit()
    
    
    print(len(result_list))
    
    for i in range(0,len(result_list)):    
        #write results to file
        file_name = result_names[i]
        result = result_list[i]
        result.to_csv(root+file_name)
        
        #plot result
        result_shape = result.shape
        
        result = result.reset_index()
        
        n_samples = result_shape[0]
        n_peaks = round((result_shape[1]-6)/3)
        
        
        
        plt.rcParams.update({'font.size': 14})
           
        width = 14
        height = 3
        scaling = 1
        
        
        
        for k in range(0,n_samples):
            
            
            fit_data = pd.DataFrame()
            
            fig = plt.figure(figsize=(width*scaling, height*scaling*2))
            
            names = result['name']
            paths = result['path']
            
            row = result.loc[k]
            gauss_params = row[8:len(row)]
     
        
            
            legend_temp = names[k]
            print(legend_temp)
            legend_text = legend_temp.replace('.','_')
            spectrum = load_clean_data(1, paths[k])
            
            x = spectrum['frequency']
            y = spectrum['imag_z']
            
            fit_data.insert(fit_data.shape[1], 'frequency', x, True)
            fit_data.insert(fit_data.shape[1], 'imag_z', y, True)
             
        
            plt.plot(x, y, '-',label=legend_text)
            
            amp_list = []
            cen_list = []
            sigma_list = []
            
            R_CPE_list = []
            
            n_index = 3
            
            start = 0
            end = n_peaks
            
            for n in range(start, end):
                
                amp1 = gauss_params[n]
                cen1 = gauss_params[n+n_peaks]
                sigma1 = gauss_params[n+n_peaks*2]
                
                amp_list.append(amp1)
                cen_list.append(cen1)
                sigma_list.append(sigma1)
                
        
                
                plt.plot(x, _1gaussian(x, amp1, cen1, sigma1), label = 'f_'+str(n+1)+', sigma'+str(sigma1))
                fit_data.insert(fit_data.shape[1], 'f_'+str(n+1), _1gaussian(x, amp1, cen1, sigma1), True)
            
            
            
            sum_gauss = amp_list+cen_list+sigma_list
        
            plt.plot(x, multiple_gauss(x, *sum_gauss), label = 'fit')
            fit_data.insert(fit_data.shape[1], 'fit', multiple_gauss(x, *sum_gauss), True)
            
            plt.title(legend_text)
            plt.legend()
            plt.grid()
            plt.show()
             
            
            #save figures
            
            fig.savefig(legend_text, dpi=600)
            
            print(fit_data)
            fit_data.to_csv(root+'\\'+legend_text+'.csv')
            print(root+legend_text)

#lg.log_init()
main()


