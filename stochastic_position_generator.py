#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 15:39:36 2022

@author: csloaner
"""
import yaml, sys, os   
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

temp = 300
alat = 3.319
root = 'Ta_'+str(alat)+'_'+str(temp)+'K'
stage_number = 2    #use current stage_number in normal modes (1 at start, so on)
natoms = 2
atoms = ['Ta', 'Ta']
masses = np.array([180.95, 180.95])

## The code needs the information from SPOSCAR & 'phonopy.yalm' generated in previous steps
#%%         
supercell_length = 4
hbar = 6.582e-16 #eV s
kB = 8.617e-5 #eV/K
par_dir = 'C:/Users/bkc/Downloads/Tantalum_recover/Ta_QMD/'
in_path = par_dir + 'normal_modes/stage_'+ str(stage_number) +'/'
in_filename = 'qpoints.yaml'

with open(in_path+in_filename) as file:
    yaml_file = yaml.full_load(file)
    
phonons = yaml_file['phonon']
nphonons = len(phonons)

nconfigurations = 20
nqpoints = nphonons

#%%
# dont change anything below this line

#%%

def thermal_occupation(angular_frequency, temp):
    denominator = np.exp((hbar*angular_frequency)/(kB*temp)) -1 
    return 1 / denominator


q_points_arr = np.zeros((nphonons, 3))
angular_frequency_arr = np.zeros((nphonons, 6, 1))
Q_arr = np.zeros((nphonons, 6, 1))
eigenvectors_arr = np.zeros((nphonons, 6, 2, 3))

# frequencies = []
for i in range(len(phonons)):
    # print(len(phonons))
    # break 
    band_contents = phonons[i]['band']
    q_point = phonons[i]['q-position']
    q_points_arr[i] = q_point
    # print(q_point)
    for j in range(len(band_contents)):
        frequency = band_contents[j]['frequency']
        # print(frequency)
        # for frequency in frequencies:
        angular_frequency = frequency * 2*np.pi*1e12
        occupation = thermal_occupation(angular_frequency, temp)
        # nominator = hbar*angular_frequency*(occupation + 0.5)
        Q = hbar*(occupation + 0.5)/angular_frequency
        Q_arr[i][j] = Q
        # denominator = 
        # print(occupation)
       # frequencies.append(band_contents[j]['frequency'])
        #print(band_contents[j]['eigenvector'])
        eigenvectors = band_contents[j]['eigenvector']
        # print(len(eigenvectors))
        for k in range(len(eigenvectors)):
            # denominator = angular_frequency**2 * (masses[k] * ((931.494e6)/(3e8**2)))
            # print(np.sqrt(nominator/denominator)*1e10) # go back to angstrongs
            eigenvector = [eigenvectors[k][0][0], eigenvectors[k][1][0], eigenvectors[k][2][0]]
            eigenvectors_arr[i][j][k] = eigenvector
            # print(k, eigenvector)
        # print()
        # for k in range
    # print()

    
# frequency_series = pd.Series(frequencies)
# frequency_series.plot(kind='hist')

#%%

in_path = par_dir + '/normal_modes/stage_'+ str(stage_number) +'/'
in_filename = 'SPOSCAR'
ide_lat = supercell_length * np.genfromtxt(in_path+in_filename, skip_header=8)
supercell_size = supercell_length**3 # Number of primitive cells

# print(ide_lat)

# ide_lat[64]

# print(ide_lat[:64], ide_lat[64:])

# There should be a phase, selected randomly, for each q-point and branch
random_phases_arr = 2 * np.pi * np.random.rand(len(q_points_arr),len(Q_arr[0]))
# random_phases_arr = np.zeros((len(q_points_arr),len(Q_arr[0])))
# print(random_phases_arr)

# gaussian_distribution = np.random.normal(size=(nphonons, nqpoints, natoms))

ser = pd.Series(np.reshape(random_phases_arr, len(q_points_arr)*len(Q_arr[0])))
# ser.plot(kind='hist')
factors = []
amplitudes = []
norm_displacements = np.zeros((nconfigurations * len(ide_lat)))

print('temp: ', temp)
#%%
#check if rmsd_results.txt exists? if so delete it.
rmsd_file_path = par_dir + '/normal_modes/stage_' + str(stage_number) + '/rmsd_results.txt'

# Check if the file exists
if os.path.exists(rmsd_file_path):
    # If it exists, delete the file
    os.remove(rmsd_file_path)

# np.random.normal(size=(nconfigurations, nqpoints, natoms))
amplitudes_all = []
configurations = np.zeros((nconfigurations, len(ide_lat), 3))
for c in range(len(configurations)):
    displacements = np.zeros((len(ide_lat), 3))
    # thermal_configuration = np.zeros((len(ide_lat), 3))
    for i in range(len(ide_lat)):
        for j in range(1, len(q_points_arr)): # explodes at Gamma, so ignore it
            # factor = np.cos( 2 * np.pi * (np.dot(q_points_arr[j], ide_lat[i]) - random_phases_arr[j][k]))
            # print(i, j, factor)
            for k in range(len(Q_arr[j])):
                atom_kind = 0 if i < 64 else 1
                #factor = np.cos( 2 * np.pi * (np.dot(q_points_arr[j], ide_lat[i]) - random_phases_arr[j][k])) # More correct way
                factor = np.random.normal() # Hellman's way
                # print(factor)
                # print(np.dot(q_points_arr[j], ide_lat[i]), factor)
                factors.append(factor)
                # factors.append(np.cos( 2 * np.pi * (-1) * random_phases_arr[j][k]))
                # factors.append(np.cos( 2 * np.pi * (np.dot(q_points_arr[j], ide_lat[i]))))
                amplitude = np.sqrt( Q_arr[j][k] / (supercell_size * masses[atom_kind] * ((931.494e6)/(3e8**2))) ) * 1e10 #in Angstrom
                amplitude = amplitude[0]
                amplitudes.append(amplitude)
                # print(amplitude)
                if amplitude > 1:
                    print(i, j, k, amplitude)
                prefactor =  amplitude * eigenvectors_arr[j][k][atom_kind] 
                # print(eigenvectors_arr[j][k][atom_kind] , amplitude * eigenvectors_arr[j][k][atom_kind] )
                result = prefactor * factor 
                # result = np.dot(result, [1, 1, 1] / np.sqrt(3))
                # if 0.1 < result.any():
                #     print(i, j, k, prefactor * factor )
                # print(i, j, k, result)
                displacements[i] = displacements[i] + result # prefactor * factor 
                # print(displacements[i], result)
                # print(displacements[i])
        #     print()
        # print()
        # displacements[i] = displacements[i]
        # thermal_configuration[i] = 2.77 * 4 * ide_lat[i] + displacements[i]
        configurations[c][i] = alat * ide_lat[i] + displacements[i]
        configurations[c][i] = configurations[c][i] / (alat * 4) 
    amplitudes_all.extend(amplitudes)
    
    
    norm_displacements[c * len(ide_lat) : c * len(ide_lat) + len(ide_lat)] = np.linalg.norm(displacements, axis=1)
    # print("Processing configuration", c)
    RMSD = np.sqrt(np.sum(np.square(np.linalg.norm(displacements, axis=1)))/float(len(np.linalg.norm(displacements, axis=1))))
    print('RMSD', RMSD) 
    # print('RMSD: ', np.sqrt(np.sum(np.square(np.linalg.norm(displacements, axis=1)))/float(len(np.linalg.norm(displacements, axis=1)))))
    # print('MSD: ', np.sum(np.square(np.linalg.norm(displacements, axis=1)))/float(len(np.linalg.norm(displacements, axis=1))))    


    with open(par_dir + '/normal_modes/stage_'+ str(stage_number) +'/rmsd_results.txt', 'a') as file: 
        # Write RMSD value for the current configuration
        file.write(str(RMSD) + '\n')
 
print('Aggregate RMSD: ', np.sqrt(np.sum(np.square(norm_displacements))/float(len(norm_displacements))))


#%%
            
        # print(Q_arr[j])
    # print()
    # print(site)

# print(displacements)
# print(np.linalg.norm(displacements, axis=1))
# print(len(np.linalg.norm(displacements, axis=1)))




# pd.Series(np.sqrt(np.sum(np.square(np.linalg.norm(displacements, axis=1)))/float(len(np.linalg.norm(displacements, axis=1))))).plot(kind='hist')
# pd.Series(np.sum(np.square(np.linalg.norm(displacements, axis=1))/float(len(np.linalg.norm(displacements, axis=1))))).plot(kind='hist')

# pd.Series(factors).plot(kind='hist', bins=100)
# pd.Series(amplitudes).plot(kind='hist', bins=100)
# pd.Series(amplitudes_all).plot(kind='hist', bins=100)
# pd.Series(np.linalg.norm(displacements, axis=1)).plot(kind='hist', bins=50)
pd.Series(norm_displacements).plot(kind='hist', bins=100)
plt.savefig(par_dir + '/normal_modes/stage_'+ str(stage_number) +'/histogram_displacements_stage_'+ str(stage_number) +'.png')

# sys.exit()
#%%
out_path = par_dir + '/normal_modes/stage_'+ str(stage_number) +'/'
out_filename = ''+root+'_QE_in_stage_' + str(stage_number) +'_instance_'
for c in range(len(configurations)):
    with open(out_path+out_filename+str(c), 'w') as file:
        
        file.write('ATOMIC_POSITIONS (crystal) \n')
        
        for i in range(len(ide_lat)):
            
            atom_kind = 0 if i < 64 else 1
            line = [atoms[atom_kind], ' ']
            
            for d in range(3):
                if configurations[c][i][d] > 1.0:
                    configurations[c][i][d] = configurations[c][i][d] - 1.0
                if configurations[c][i][d] < 0.0:
                    configurations[c][i][d] = configurations[c][i][d] + 1.0 
                    
                line.append('%10.8f' % configurations[c][i][d])
                line.append(' ')
                # line.append(str.format('{%10.5f}', configurations[c][i][d]))
            
            line.append('\n')
            file.writelines(line)
        file.write('\n')
        file.write('K_POINTS {gamma} \n')
        file.write('1 1 1 0 0 0 \n')
        file.write('\n')
        file.write('CELL_PARAMETERS {angstrom} \n')
        
        file.writelines(['%10.8f' % (alat * 4), ' ',  '%10.8f' % (0), ' ', '%10.8f' % (0), '\n'])
        file.writelines(['%10.8f' % (0), ' ', '%10.8f' % (alat * 4), ' ', '%10.8f' % (0), '\n'])
        file.writelines(['%10.8f' % (0), ' ', '%10.8f' % (0), ' ', '%10.8f' % (alat * 4), '\n'])
        
        
            # print(line)
                
        # print(configurations[c])
        # print()
    
    
    # with open(out_path+out_filename+str(c), 'w') as file:
    #     print(str(n) + ' ' + str(n), file=file)
    #     for i in range(n):
    #         for j in range(n):
    #             print(str(i+1) + ' ' + str(j+1), file=file)
    #             np.savetxt(file, fc_mat[i, j, :, :])


