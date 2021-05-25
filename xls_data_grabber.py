import pandas as pd
import numpy as np
import os, fnmatch
from os import path


def extract_data(c_dir = ''):

    if c_dir == '':
        c_dir = os.getcwd()
        
    os.chdir(c_dir)
    
    
    xls_files = []
    
    listOfFiles = os.listdir(c_dir)
    pattern = "*.xls"
    for entry in listOfFiles:
        if fnmatch.fnmatch(entry, pattern):
                xls_files.append(entry)
    
    print(xls_files)
    print(len(xls_files))
    
    n_values = 48;
    
    for k in range(len(xls_files)):
        file_name_temp = xls_files[k]
        n_len = len(file_name_temp)-4
        folder_name = file_name_temp[:n_len]
        txt_path = c_dir+'/'+folder_name
        
    
        if not os.path.isdir(txt_path):
            os.makedirs(txt_path)
            
    
        data = pd.read_excel (c_dir+'/'+file_name_temp, sheet_name='Blocks') #for an earlier version of Excel, you may need to use the file extension of 'xls'
    
        df = pd.DataFrame(data, columns= ['Block 1'])
        data = data.fillna(0)
    
        
        rows = len(df.index)
        i = n_values

        
        while i < rows:
            freq = data.iloc[i,[0]].values[0]
            file_name = str(data.iloc[i,[1]].values[0])
            
            if '/' in file_name:
                file_name = file_name.replace('/','_per_')
                
            if '?' in file_name:
                file_name = file_name.replace('?','check')
            
            block_name = str(data.iloc[i,[0]].values[0])
           
            file_name = block_name+"_"+file_name+".txt"
            
            os.chdir(c_dir)
            
            if freq == 0.0:
                break
            
            os.chdir(txt_path)

            
            f= open(file_name,"a+")
            f.write("48\n")
            
            
            for x in range(0, n_values):
                j = i+n_values-x
                omega = str((data.iloc[j,[0]].values[0]))
                Re = str(data.iloc[j,[1]].values[0])
                m_Im = str(data.iloc[j,[2]].values[0]*(-1))
    
                f.write(Re+"\t"+m_Im+"\t"+omega+"\n")
            
            f.close()
            
            i += n_values+1


extract_data()