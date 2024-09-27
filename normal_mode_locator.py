#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 09:29:26 2022

@author: csloaner
"""

import numpy as np

temp = 300
resolution = 100
supercell_size = 4    #will generate 64 qpoints
par_dir = 'C:/Users/biknb/Downloads/Tantalum/GP/HELD_QMD_'+str(temp)+'K'

#%%
# dont change anything below this line (make sure to check line 551-554)

#%%
supercell = supercell_size * np.array([[1.,0.,0.], [0.,1.,0.], [0.,0.,1.]])
axis = np.linspace(0, supercell_size, num=supercell_size*resolution, endpoint=False)
axis = np.linspace(0, 1, num=resolution, endpoint=False)

# print(supercell)

q_points_arr = np.zeros((resolution**3,3))

# print(q_points_arr)

normal_modes = []
normal_modes = np.zeros((supercell_size**3,3))
count = 0
for qx in axis:
    for qy in axis:
        for qz in axis:
            q_point = np.array([[qx], [qy], [qz]])
            res = np.matmul(supercell,q_point)
            
            if res[0,0].is_integer():
                if res[1,0].is_integer():
                    if res[2,0].is_integer():
                        normal_mode = [qx, qy, qz]
                        normal_modes[count] =  normal_mode
                        count = count + 1
                        
if count != supercell_size**3:
    print('Error')
    
else:
    np.savetxt(par_dir + '/files/QPOINTS', normal_modes, fmt=('%10.5f'), header=str(count), comments='')
    print('QPOINTS written succesfully')