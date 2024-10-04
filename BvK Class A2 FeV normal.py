import numpy as np
from math import sqrt, isclose
from scipy.stats import norm
from matplotlib import pyplot as plt
import time, sys, os
from matplotlib.ticker import FuncFormatter

start_time = time.time()                #line 609, 624

temp = 300
alat = 3.308
root = 'Ta_'+str(alat)+'_'+str(temp)+'K'
supercell_length = 4
natoms = 128
stage_number = 0   
par_dir = 'C:/Users/bkc/Downloads/Tantalum_recover/Ta_QMD/319/'+str(alat)+'/'

class Force_Constants:
    #Values of Born-von Kármán model for up to fifth nearest neighbors of a bcc structure
    #for a hdf5 file. args = np array[n, 3, t], np array[n, 3, t], np array[n, 3],  float, np array[n, 3], int[1, 5]
    def __init__(self, positions, forces, alat, ideal_lattice, under_lattice=np.zeros(0), NN=5):
        if NN < 1 or NN > 5 or not isinstance(NN, int):
            raise TypeError('Nearest neighbors argument must be integer in range [1, 5].')
            
            
        self.out_path = par_dir + 'force_constants/stage_'+ str(stage_number) +'/'
        self.positions = positions
        self.forces = forces
        self.alat = alat
        self.ide_lat = ideal_lattice
        self.nn = NN
        self.n = positions.shape[0] #Number of atoms in supercell
        self.n_steps = positions.shape[2] #Number of time steps
        self.steps = range(self.n_steps)
        self.a_val = self.alat / ((self.n / 2.)**(1./3.)) #Side length of each cube in bcc
        
        self.fc = [3, 5, 8, 12, 14][self.nn-1] #Amount of constants according to NN
        self.index = ['c0xx', 'c1xx', 'c1xy', 'c2xx', 'c2yy', 'c3xx', 'c3yy', 'c3yz', 'c4xx', 'c4yy', 'c4xy', 'c4yz', 'c5xx', 'c5xy'] #Index of force constants
        self.sm = np.array([1., 8., 0., 2., 4., 4., 8., 0., 8., 16., 0., 0., 8., 0.]) #Instances of each constant in [alpha alpha]
        self.disp_mat = np.zeros((3 * self.n, self.fc, self.n_steps)) #Matrices of displacements from equilibrium by time step
        self.phiv_steps = np.zeros((self.fc, self.n_steps)) #Vector of all the unique values of phi

    #Calculate distance between atoms in the ideal lattice
    def ideal_lat_dist(self):
        ideal_distances = np.zeros((self.n, 3, self.n))
        ideal_dist_sca = np.zeros((self.n, self.n))
        
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    ideal_distances[i, :, j] = self.ide_lat[j, :] - self.ide_lat[i, :] #Ideal distance vector from i to j

                    #Apply mic:
                    for d in range(3):
                        if ideal_distances[i, d, j] > (self.alat / 2):
                            ideal_distances[i, d, j] -= self.alat

                        elif ideal_distances[i, d, j] <= (-self.alat / 2):
                            ideal_distances[i, d, j] += self.alat

                    ideal_dist_sca[i, j] = np.linalg.norm(ideal_distances[i, :, j]) #Distance scalar

        return ideal_distances, ideal_dist_sca

    #Identify up to fifth nearest neighbors for every atom
    def nearest_neighbors(self):
        if not hasattr(self, 'ideal_dist_sca'):
            self.ideal_distances, self.ideal_dist_sca = self.ideal_lat_dist()
        
        neighbors = [] #List of dictionaries holding a list of the IDs for all types of nn

        #Distances to each type of neighbor:
        first = (sqrt(3) * self.a_val) / 2
        second = self.a_val
        third = sqrt(2) * self.a_val
        fourth = (sqrt(11) * self.a_val) / 2
        fifth = sqrt(3) * self.a_val
        self.nn_dist = [first, second, third, fourth, fifth]

        for i in range(self.n):
            i_neigh = [[i]] 
            for prox in range(self.nn):
                i_neigh.append([]) #Empty placeholder lists for neighbors of i
            
            #Add each list of nn to the placeholder dictionary, append the dictionary to the list:
            for j in range(self.n):
                if i != j:
                    for prox in range(1,self.nn+1):
                        if isclose(self.ideal_dist_sca[i, j], self.nn_dist[prox-1], rel_tol=.001):
                            i_neigh[prox].append(j)

            neighbors.append(i_neigh)
            
        return neighbors

    #Arrange part of U corresponding to time t in F=U*phi
    def BvK_matrix(self, t):
        if not hasattr(self, 'neighbors'):
            self.neighbors = self.nearest_neighbors()
            
        disp_mat = np.zeros((3 * self.n, self.fc)) #Empty matrix U of displacements from equilibrium
        for i in range(self.n):
            for prox in range(self.nn+1):
                for j in self.neighbors[i][prox]:
                    displacements = self.positions[j, :, t] - self.ide_lat[j, :] #Distance vector from i to j at t

                    #Apply mic:
                    for d in range(3):
                        if displacements[d] > (self.alat / 2):
                            displacements[d] -= self.alat

                        elif displacements[d] <= (-self.alat / 2):
                            displacements[d] += self.alat

                    #Arrange matrix U according to similar terms:
                    m1 = np.array([[displacements[0], 0., 0.],
                                   [0., displacements[1], 0.],
                                   [0., 0., displacements[2]]])
                    m2 = np.array([[0., displacements[1], displacements[2]],
                                   [displacements[0], 0., displacements[2]],
                                   [displacements[0], displacements[1], 0.]])
                    for a in range(3):
                        if self.ideal_distances[i, a, j] < 0:
                            m2[a, :] *= -1
                            m2[:, a] *= -1

                    if prox == 0:
                        m1r = np.diag(m1)
                        disp_mat[3 * i: 3 * (i+1), 0] += m1r

                    if prox == 1 or prox == 5:
                        m1r = np.diag(m1)
                        m2r = np.sum(m2, axis=1)
                        if prox == 1:
                            disp_mat[3 * i: 3 * (i+1), 1] += m1r
                            disp_mat[3 * i: 3 * (i+1), 2] += m2r

                        if prox == 5:
                            disp_mat[3 * i: 3 * (i+1), 12] += m1r
                            disp_mat[3 * i: 3 * (i+1), 13] += m2r

                    if prox == 2:
                        m1r = np.zeros((3, 2))
                        for a in range(3):
                            if self.ideal_distances[i, a, j] == 0.:
                                m1r[:, 1] += m1[:, a]

                            else:
                                m1r[:, 0] = m1[:, a]

                        disp_mat[3 * i: 3 * (i+1), 3:5] += m1r

                    if prox == 3:
                        m1r = np.zeros((3, 2))
                        for a in range(3):
                            if self.ideal_distances[i, a, j] == 0.:
                                m1r[:, 0] = m1[:, a]

                            else:
                                m1r[:, 1] += m1[:, a]

                            if self.ideal_distances[i, a, j] == 0.:
                                m2[a, :] = np.zeros(3)
                                m2[:, a] = np.zeros(3)

                        m2r = np.sum(m2, axis = 1)
                        disp_mat[3 * i: 3 * (i+1), 5:7] += m1r
                        disp_mat[3 * i: 3 * (i+1), 7] += m2r

                    if prox == 4:
                        m1r = np.zeros((3, 2))
                        m2r = np.zeros((3, 2))
                        for a in range(3):
                            if abs(self.ideal_distances[i, a, j]) == 1.5 * self.a_val:
                                m1r[:, 0] = m1[:, a]
                                m2r[a, 0] = np.sum(m2[a, :])
                                m2r[(a+1) % 3, 0] = m2[(a+1) % 3, a]
                                m2r[(a+2) % 3, 0] = m2[(a+2) % 3, a]
                                m2r[(a+1) % 3, 1] = m2[(a+1) % 3, (a+2) % 3]
                                m2r[(a+2) % 3, 1] = m2[(a+2) % 3, (a+1) % 3]

                            else:
                                m1r[:, 1] += m1[:, a]

                        disp_mat[3 * i: 3 * (i+1), 8:10] += m1r
                        disp_mat[3 * i: 3 * (i+1), 10:12] += m2r
        
        return disp_mat #Matrix of sums of displacements from equilibrium

    def FC_file_steps(self, fpath, steps_list=[]):
        if not steps_list:
            steps_list = self.steps
            
        for t in steps_list:
            fc = []
            for tpe in range(self.fc):
                fc.append(self.phiv_steps[tpe, t])
                
            fc.append(0.)
            fc_mat = np.zeros((self.n, self.n, 3, 3))
            base_mat = np.zeros((6, 3, 3))
            nn_matrix = .5 * self.a_val * np.array([[0.,0.,0.],
                                                    [1.,1.,1.],
                                                    [2.,0.,0.],
                                                    [0.,2.,2.],
                                                    [3.,1.,1.],
                                                    [2.,2.,2.]])
            
            fc_arr = np.array([[0, 0, 14, 14],
                               [1, 1, 2, 2],
                               [3, 4, 14, 14],
                               [5, 6, 14, 7],
                               [8, 9, 10, 11],
                               [12, 12, 13, 13]])
            
            for prox in range(6):
                base_mat[prox, :, :] = np.array([[fc[fc_arr[prox, 0]], fc[fc_arr[prox, 2]], fc[fc_arr[prox, 2]]],
                                                 [fc[fc_arr[prox, 2]], fc[fc_arr[prox, 1]], fc[fc_arr[prox, 3]]],
                                                 [fc[fc_arr[prox, 2]], fc[fc_arr[prox, 3]], fc[fc_arr[prox, 1]]]])
            
            for i in range(self.n):
                for prox in range(6):
                    for j in self.neighbors[i][prox]:
                        fc_mat[i, j, :, :] = base_mat[prox, :, :]
                        if prox in [2, 3, 4]:
                            for r in range(1,3):
                                if isclose(abs(self.ideal_distances[i, r, j]), nn_matrix[prox, 0], rel_tol=.001):
                                    fc_mat[i, j, [r, 0], :] = fc_mat[i, j, [0, r], :]
                                    fc_mat[i, j, :, [r, 0]] = fc_mat[i, j, :, [0, r]]
                                    
                        for r in range(3):
                            if self.ideal_distances[i, r, j] < 0:
                                fc_mat[i, j, r, :] = -fc_mat[i, j, r, :]
                                fc_mat[i, j, :, r] = -fc_mat[i, j, :, r]
                                
            print(t)
            np.around(fc_mat, decimals=5)
            if not os.path.isfile(fpath + '/' + str(t)):
                os.mkdir(fpath + '/' + str(t))
            
            with open(fpath + '/' + str(t) + '/FORCE_CONSTANTS', 'w') as file:
                print(str(self.n) + ' ' + str(self.n), file=file)
                for i in range(self.n):
                    for j in range(self.n):
                        print(str(i+1) + ' ' + str(j+1), file=file)
                        np.savetxt(file, fc_mat[i, j, :, :])
                        
    def FC_file(self, fpath, phiv):
        fc = []
        for tpe in range(self.fc):
            fc.append(phiv[tpe])
            
        fc.append(0.)
        fc_mat = np.zeros((self.n, self.n, 3, 3))
        base_mat = np.zeros((6, 3, 3))
        nn_matrix = .5 * self.a_val * np.array([[0.,0.,0.],
                                                [1.,1.,1.],
                                                [2.,0.,0.],
                                                [0.,2.,2.],
                                                [3.,1.,1.],
                                                [2.,2.,2.]])
        
        fc_arr = np.array([[0, 0, 14, 14],
                           [1, 1, 2, 2],
                           [3, 4, 14, 14],
                           [5, 6, 14, 7],
                           [8, 9, 10, 11],
                           [12, 12, 13, 13]])
        
        for prox in range(6):
            base_mat[prox, :, :] = np.array([[fc[fc_arr[prox, 0]], fc[fc_arr[prox, 2]], fc[fc_arr[prox, 2]]],
                                             [fc[fc_arr[prox, 2]], fc[fc_arr[prox, 1]], fc[fc_arr[prox, 3]]],
                                             [fc[fc_arr[prox, 2]], fc[fc_arr[prox, 3]], fc[fc_arr[prox, 1]]]])
        
        for i in range(self.n):
            for prox in range(6):
                for j in self.neighbors[i][prox]:
                    fc_mat[i, j, :, :] = base_mat[prox, :, :]
                    if prox in [2, 3, 4]:
                        for r in range(1,3):
                            if isclose(abs(self.ideal_distances[i, r, j]), nn_matrix[prox, 0], rel_tol=.001):
                                fc_mat[i, j, [r, 0], :] = fc_mat[i, j, [0, r], :]
                                fc_mat[i, j, :, [r, 0]] = fc_mat[i, j, :, [0, r]]
                                
                    for r in range(3):
                        if self.ideal_distances[i, r, j] < 0:
                            fc_mat[i, j, r, :] = -fc_mat[i, j, r, :]
                            fc_mat[i, j, :, r] = -fc_mat[i, j, :, r]
                            
        np.around(fc_mat, decimals=5)
        if not os.path.isfile(fpath):
            os.mkdir(fpath)
        
        with open(fpath + '/FORCE_CONSTANTS', 'w') as file:
            print(str(self.n) + ' ' + str(self.n), file=file)
            for i in range(self.n):
                for j in range(self.n):
                    print(str(i+1) + ' ' + str(j+1), file=file)
                    np.savetxt(file, fc_mat[i, j, :, :])

    #Calculate force constant values for list of steps, normal_fit defines wether the mean of the distribution is given 
    #instead of all time steps
    def get_fc_steps(self, steps_list=[], normal_fit=False, FC_file=False, fpath='FC_files'):
        if not steps_list:
            steps_list = self.steps
        
        Fv = np.zeros((3 * self.n) + 1) #Vector of all the force vetors at every time step appended
        Disp = np.zeros(((3 * self.n) + 1, self.fc)) #Empty matrix of displacements from equilibrium
        for t in steps_list:
            if self.phiv_steps[:, t].all() == 0.:
                if self.disp_mat[:, :, t].all() == 0.:
                    self.disp_mat[:, :, t] = self.BvK_matrix(t)
                
                Disp[:3 * self.n, :] = self.disp_mat[:, :, t] #Fill equations related to force

                #Append last row so that the sum of all phi_{alfa, alfa} equals 0:
                Disp[3 * self.n, :] = self.sm[:self.fc]

                #Arrange all force vectors at t in a single vector:
                for i in range(self.n):
                    Fv[3 * i: 3 * (i + 1)] = -self.forces[i, :, t]

                #Solve for vector phi from algebra:
                Disp_inv = np.linalg.pinv(Disp)
                self.phiv_steps[:, t] = np.matmul(Disp_inv, Fv)
                
                #Apply correction so that translation constrain holds
                self.phiv_steps[0,t] = -np.matmul(self.sm[1:], self.phiv_steps[1:,t])
                
            print(t)

        if not normal_fit:
            dicts_list = {}
            for t in steps_list:
                phiv_dict = {}
                for tpe in range(self.fc):
                    phiv_dict[self.index[tpe]] = self.phiv_steps[tpe, t]

                dicts_list[str(t)] = phiv_dict
                
            if FC_file:
                self.FC_file_steps(fpath, steps_list=steps_list)

            return dicts_list #Dictionary of dictionaries with unique force constant values at each time step

        else:
            phiv_fit = np.zeros((self.fc, 2)) #Fit to be made for the vector of phi
            for tpe in range(self.fc):
                phiv_fit[tpe, 0], phiv_fit[tpe, 1] = norm.fit(np.take(self.phiv_steps[tpe, :], steps_list)) #Fit the Gaussian
                plt.hist(np.take(self.phiv_steps[tpe, :], steps_list), bins=40, density=True)
                xmin, xmax = plt.xlim()
                x = np.linspace(xmin, xmax, 40)
                pdf = norm.pdf(x, phiv_fit[tpe, 0], phiv_fit[tpe, 1])
                plt.plot(x, pdf, 'r')
                plt.title('Distribution of ' + str(self.index[tpe]), fontsize=14)
                plt.xlabel('Force constant value', fontsize=14)
                plt.ylabel('Density', fontsize=14)
                plt.xticks(fontsize=14)  # Increase x-axis tick font size
                plt.yticks(fontsize=14)  # Increase y-axis tick font size
                # Function to truncate x values to 2 decimal places
                def truncate_x(x, pos):
                    return f'{x:.2f}'
                
                # Set formatter for x-axis to truncate to 2 decimal places
                plt.gca().xaxis.set_major_formatter(FuncFormatter(truncate_x))
                
                plt.savefig('Normal fit ' + self.index[tpe] + '.png')
                plt.show()
                plt.close()

            fit_dict = {}
            for tpe in range(self.fc):
                fit_dict[self.index[tpe]] = phiv_fit[tpe, 0]
                
            if FC_file:
                self.FC_file(fpath, phiv_fit[:, 0])

            return fit_dict #Dictionary with unique force constant values

    #Calculate force constant values with a matrix of all time steps in list simultaneously
    def get_fc_sim(self, steps_list=[], FC_file=False, fpath=''):
        if not steps_list:
            steps_list = self.steps
        
        len_steps = len(steps_list)
        Fv = np.zeros(3 * self.n) #Vector of all the force vetors at every time step appended
        Fv_tot = np.zeros((len_steps * (3 * self.n)) + 1) #Appending of all Fv of t
        Disp_tot = np.zeros(((len_steps * (3 * self.n)) + 1, self.fc)) # Appending of all U of t
        it = 0
        for t in steps_list:
            if self.disp_mat[:, :, t].all() == 0.:
                self.disp_mat[:, :, t] = self.BvK_matrix(t)
                
            Disp_tot[(3 * self.n) * it:(3 * self.n) * (it+1), :] = self.disp_mat[:, :, t] #Fill equations related to force
            
            #Arrange all force vectors at t in a single vector:
            for i in range(self.n):
                Fv[3 * i: 3 * (i+1)] = -self.forces[i, :, t]

            #Arrange all force vectors at t in a single vector:
            Fv_tot[(3 * self.n) * it:(3 * self.n) * (it+1)] = Fv
            it += 1
            print(t)
            
        #Append last row so that the sum of all phi_{alfa, alfa} equals 0:
        Disp_tot[len_steps * (3 * self.n), :] = self.sm[:self.fc]
        
        #Solve for vector phi from algebra and arrange in matrix form:
        Disp_inv = np.linalg.pinv(Disp_tot)
        phiv = np.matmul(Disp_inv, Fv_tot)
        
        
        #Apply correction so that translation constrain holds
        Disp_inv = np.linalg.pinv(Disp_tot)
        phiv[0] = np.matmul(self.sm[1:], phiv[1:])

        phiv_dict = {}
        for tpe in range(self.fc):
            phiv_dict[self.index[tpe]] = phiv[tpe]
            
        if FC_file:
            self.FC_file(fpath, phiv)

        return phiv_dict #Dictionary with unique force constant values
    
    def get_fc_error(self, steps_list=[]):
        if not steps_list:
            steps_list = self.steps
        
        empty_steps = []
        for t in steps_list:
            if self.phiv_steps[:, t].all() == 0.:
                empty_steps.append(t)
                
        if empty_steps:
            self.get_fc_steps(steps_list=empty_steps)
        
        sigma_rsq = {} #Dictionary of sum of the square errors per time step
        for t in steps_list:
            error = np.zeros((3 * self.n))
            force_fit = np.matmul(self.disp_mat[:, :, t], -self.phiv_steps[:, t]) #Matrix multiplication to get vector of all forces
            for i in range(self.n):
                error[3 * i:3 * (i+1)] = force_fit[3 * i:3 * (i+1)] - self.forces[i, :, t] #Force of the fit - Force by md
                
            err_mean, err_dev = norm.fit(error) #Fit the Gaussian
            plt.hist(error, bins=20, density=True)
            xmin, xmax = plt.xlim()
            x = np.linspace(xmin, xmax, 20)
            pdf = norm.pdf(x, err_mean, err_dev)
            plt.plot(x, pdf, 'r')
            plt.title('Error distribution for step #' + str(t))
            plt.xlabel('Error')
            plt.ylabel('Density')
            plt.savefig('Error/Error step ' + str(t) + '.png')
            plt.show()
            plt.close()
            sigma_rsq[str(t)] = np.sum(np.square(error)) #Sum of the squared errors
            print(t)
            
        return sigma_rsq
    
    def get_longitudinal_fc(self, steps_list=[]):
        if not steps_list:
            steps_list = self.steps
            
        if not hasattr(self, 'phi'):
            self.phi = np.zeros((self.nn, 3, 3, self.n_steps))
        
        #Check for empty steps and fill:
        empty_steps = []
        for t in steps_list:
            if self.phiv_steps[:, t].all() == 0.:
                empty_steps.append(t)
                
        if empty_steps:
            self.get_fc_steps(steps_list=empty_steps)
        
        if not hasattr(self, 'ihat'):
            self.long_fc = np.zeros((self.nn, self.n_steps))
            nn_matrix = np.array([[1.,1.,1.],
                                  [2.,0.,0.],
                                  [0.,2.,2.],
                                  [3.,1.,1.],
                                  [2.,2.,2.]])
            self.ihat = .5 * self.a_val * nn_matrix[:self.nn, :] #Unit vectors in direction of each NN
            for prox in range(self.nn):
                self.ihat[prox, :] = self.ihat[prox, :] / self.nn_dist[prox]
        
        fc_arr = np.array([[1, 1, 2, 2],
                           [3, 4, self.fc, self.fc],
                           [5, 6, self.fc, 7],
                           [8, 9, 10, 11],
                           [12, 12, 13, 13]])
        long_dict = {} #Dictionary of lists holding longitudinal fc values per time step
        for t in steps_list:
            long_dict[str(t)] = []
            if self.phi[:, :, :, t].all() == 0.:
                phiv = np.zeros((self.fc+1)) #Vector of all fc values for NN with an appended 0
                phiv[:self.fc] = self.phiv_steps[:, t]
                for prox in range(self.nn):
                    #Arrange force constant matrix:
                    self.phi[prox, :, :, t] = np.array([[phiv[fc_arr[prox, 0]], phiv[fc_arr[prox, 2]], phiv[fc_arr[prox, 2]]],
                                                        [phiv[fc_arr[prox, 2]], phiv[fc_arr[prox, 1]], phiv[fc_arr[prox, 3]]],
                                                        [phiv[fc_arr[prox, 2]], phiv[fc_arr[prox, 3]], phiv[fc_arr[prox, 1]]]])
            
            for prox in range(self.nn):
                self.long_fc[prox, t] = np.matmul(self.ihat[prox, :].reshape(1, 3), np.matmul(self.phi[prox, :, :, t], self.ihat[prox, :])).item()
                long_dict[str(t)].append(self.long_fc[prox, t])
            
            print(t)
            
        return long_dict
        
    def get_transversal_fc(self, steps_list=[]):
        if not steps_list:
            steps_list = self.steps
            
        if not hasattr(self, 'phi'):
            self.phi = np.zeros((self.nn, 3, 3, self.n_steps))
        
        #Check for empty steps and fill:
        empty_steps = []
        for t in steps_list:
            if self.phiv_steps[:, t].all() == 0.:
                empty_steps.append(t)
                
        if empty_steps:
            self.get_fc_steps(steps_list=empty_steps)
        
        if not hasattr(self, 'ihat'):
            nn_matrix = np.array([[1.,1.,1.],
                                  [2.,0.,0.],
                                  [0.,2.,2.],
                                  [3.,1.,1.],
                                  [2.,2.,2.]])
            self.ihat = .5 * self.a_val * nn_matrix[:self.nn, :] #Unit vectors in direction of each NN
            for prox in range(self.nn):
                self.ihat[prox, :] = self.ihat[prox, :] / self.nn_dist[prox]
        
        if not hasattr(self, 'jhat'):
            self.khat = self.jhat = np.zeros((self.nn, 3)) #Unit vectors orthogonal to each NN pair
            for prox in range(self.nn):
                self.khat[prox, :] = [0.,0.,1.] - (self.ihat[prox, 2] * self.ihat[prox, :]) #Gram Schmidt
                self.khat[prox, :] = self.khat[prox, :] / np.linalg.norm(self.khat[prox, :])
                self.jhat[prox, :] = np.cross(self.khat[prox, :], self.ihat[prox, :])
                 
        fc_arr = np.array([[1, 1, 2, 2],
                           [3, 4, self.fc, self.fc],
                           [5, 6, self.fc, 7],
                           [8, 9, 10, 11],
                           [12, 12, 13, 13]])
        trans_dict = {} #Dictionary of lists holding transversal fc values per time step
        for t in steps_list:
            trans_dict[str(t)] = []
            if self.phi[:, :, :, t].all() == 0.:
                phiv = np.zeros((self.fc+1)) #Vector of all fc values for NN with an appended 0
                phiv[:self.fc] = self.phiv_steps[:, t]
                for prox in range(self.nn):
                    #Arrange force constant matrix:
                    self.phi[prox, :, :, t] = np.array([[phiv[fc_arr[prox, 0]], phiv[fc_arr[prox, 2]], phiv[fc_arr[prox, 2]]],
                                                        [phiv[fc_arr[prox, 2]], phiv[fc_arr[prox, 1]], phiv[fc_arr[prox, 3]]],
                                                        [phiv[fc_arr[prox, 2]], phiv[fc_arr[prox, 3]], phiv[fc_arr[prox, 1]]]])
            
            for prox in range(self.nn):
                #Check for uniqueness of transversal values:
                jfc = np.matmul(self.jhat[prox, :].reshape(1, 3), np.matmul(self.phi[prox, :, :, t], self.jhat[prox, :])).item()
                kfc = np.matmul(self.khat[prox, :].reshape(1, 3), np.matmul(self.phi[prox, :, :, t], self.khat[prox, :])).item()
                if isclose(jfc, kfc, rel_tol=.01):
                    trans_fc = jfc
                    
                else:
                    trans_fc = [jfc, kfc]
                
                trans_dict[str(t)].append(trans_fc)
                
            print(t)
        
        return trans_dict
        
    def get_trans_fc_eig(self, steps_list=[]):
        if not steps_list:
            steps_list = self.steps
        
        #Check for empty steps and fill:
        empty_steps = []
        for t in steps_list:
            if self.phiv_steps[:, t].all() == 0.:
                empty_steps.append(t)
                
        if empty_steps:
            self.get_fc_steps(steps_list=empty_steps)
            
        if not hasattr(self, 'long_fc'):
            self.get_longitudinal_fc(steps_list=steps_list)
            
        trans_dict = {} #Dictionary of lists holding transversal fc values per time step
        for t in steps_list:
            trans_dict[str(t)] = []
            for prox in range(self.nn):
                lst = []
                val, vec = np.linalg.eig(self.phi[prox, :, :, t]) #Get eigenvalues
                for a in range(3):
                    #Compare eigenvalues to longitudinal values and remove it:
                    if not isclose(val[a], self.long_fc[prox, t], rel_tol=.01):
                        lst.append(val[a])
                        
                #Check for uniqueness of transversal values:
                if isclose(lst[0], lst[1], rel_tol=.01):
                    lst = lst[0]
            
                trans_dict[str(t)].append(lst)
                
            print(t)
                
        return trans_dict

#Example for 20 instances
# ntsteps = 11 * 20  # 17 files, 10 timesteps each, plus the original (11 timesteps total). Maybe.
ntsteps = 11 * 1     #1 instances and 10 timesteps in QMD run
# in_path = '/Users/csloaner/Documents/FeV_lammps/normal_modes/B2_2.77_900K/stage_1/'
in_path = par_dir + '/normal_modes/stage_'+ str(stage_number) +'/'
box_length = alat*supercell_length
dump_file_root = root + '_QE_in_stage_' + str(stage_number) + '_instance_' # 'FeVmd_'
# dump_file_root = 'nm_' + root + '_QE_in_stage_' + str(stage_number) + '_instance_' # 'FeVmd_'
#pos_file = 'FeV_B2_2.78.pos'
# fc_filename = 'fc_B2_2.77_900K_QE_stage_1.csv'
fc_filename = 'fc_'+root+'_QE_in_stage_' + str(stage_number) + '.csv'
# fc_filename = 'nm_fc_' + root + '_QE_in_stage_' + str(stage_number) + '.csv'

positions = np.zeros((natoms, 3, ntsteps))
forces = np.zeros((natoms, 3, ntsteps))

big_ntstep = 0
for file_number in [0]: #[0,1,2,3,4,5,6,8,9,10,11,12,15,16,17,18]: #select individual instance files
# for file_number in range(1):   #20 for 20 instances, 1 for first instances...
    qe_output_file_name = dump_file_root + str(file_number)
    positions_start_lines = []
    forces_start_lines = []
    with open(in_path+qe_output_file_name+".out") as qe_file:
        lines = qe_file.readlines()
        # lines = [line.split() for line in lines]

    # print(lines)
    
    # sys.exit() 
    for line_number, line in enumerate(lines):
        # print(line_number, line)
        if 'Cartesian axes' in line:
            if 0 == len(positions_start_lines):
                positions_start_lines.append(line_number + 3)
            else:
                print('Error, should not ge here')
        if 'Forces acting on atoms' in line:
            forces_start_lines.append(line_number + 2)
        if 'ATOMIC_POSITIONS' in line:
            if 0 != len(positions_start_lines):
                positions_start_lines.append(line_number + 1)
            else:
                print('Error, should not ge here')
                
    
                
    for ntstep, line_index in enumerate(positions_start_lines):
        # print(ntstep, line_index)
        for atom_number in range(natoms):
            positions_strings = lines[line_index + atom_number].split()
            
            if 0 == ntstep:
                positions[atom_number, 0, ntstep + big_ntstep] = float(positions_strings[6]) * box_length
                positions[atom_number, 1, ntstep + big_ntstep] = float(positions_strings[7]) * box_length
                positions[atom_number, 2, ntstep + big_ntstep] = float(positions_strings[8]) * box_length
                
            else:
                positions[atom_number, 0, ntstep + big_ntstep] = float(positions_strings[1]) * box_length
                positions[atom_number, 1, ntstep + big_ntstep] = float(positions_strings[2]) * box_length
                positions[atom_number, 2, ntstep + big_ntstep] = float(positions_strings[3]) * box_length
    
    for ntstep, line_index in enumerate(forces_start_lines):
        # print(ntstep, line_index)
        for atom_number in range(natoms):
            forces_strings = lines[line_index + atom_number].split()
            forces[atom_number, :, ntstep + big_ntstep] = [float(i) * 25.71 for i in forces_strings[-3:]] #1 Ry/Bohr = 25.71104309541616 eV/Angstrom
            
    big_ntstep = big_ntstep + 10

# root = 'B2_2.77_900K'        
in_path = par_dir + 'normal_modes/stage_'+ str(stage_number) +'/' 
in_filename = 'SPOSCAR'
id_lattice = supercell_length * np.genfromtxt(in_path+in_filename, skip_header=8) * alat


# print(positions)
# print()
# print(forces)
# print('here')

# sys.exit()

# id_lattice = np.zeros((250, 3)) # id_lattice is the ideal underlying lattice
# data = np.genfromtxt(in_path+pos_file, skip_header=11)#, delimiter=' ', skip_header=11)
# for row in data:
#     id_lattice[int(row[0])-1, :] = row[2:5]
    
# id_lattice = positions[:, :, 0]

# under_lattice = positions[:, :, 0] # under_lattice is the relaxed structure

# print(id_lattice)
# sys.exit()

# print(id_lattice)
# print('----')
# print(positions[:, :, 5])
# sys.exit()

bvk = Force_Constants(positions, forces, box_length, id_lattice)#, under_lattice=under_lattice)
# print(bvk.nearest_neighbors())
dictn = bvk.get_fc_steps(normal_fit=True)

# out_path = '/Users/csloaner/Documents/FeV_lammps/force_constants/B2_2.77_900K/'
out_path = par_dir + '/force_constants/stage_'+ str(stage_number) +'/'

with open(out_path+fc_filename, 'w') as f:
    for c in range(bvk.fc):
        print(bvk.index[c] + ', ', end='', file=f)
        
    print('\n', file=f)
    
    for c in range(bvk.fc):
        print(str(dictn[bvk.index[c]]) + ', ', end='', file=f)
        
    print('\n', file=f)
    
    for t in range(ntsteps):
        for c in range(bvk.fc):
            print(str(bvk.phiv_steps[c, t]) + ', ', end='', file=f)
            
        print('', file=f)
# print(bvk.nearest_neighbors())
# print(bvk.get_fc_steps(normal_fit=True))
# print(bvk.get_fc_error(steps_list=range(300,310)))
# print(bvk.BvK_matrix(500))

# lon = bvk.get_longitudinal_fc()
# print(bvk.get_transversal_fc(steps_list=[300]))
# print(bvk.get_longitudinal_fc(steps_list=[300]))
# print(bvk.get_trans_fc_eig(steps_list=[300]))
# print()
# print(bvk.get_longitudinal_fc(steps_list=[400]))
# print(bvk.get_trans_fc_eig(steps_list=[400]))
# print()
# print(bvk.get_longitudinal_fc(steps_list=[519]))
# print(bvk.get_trans_fc_eig(steps_list=[519]))

# lfc = np.zeros((5, 520))
# for t in range(520):
#     lfc[:, t] = np.asarray(lon[str(t)])
    
# import seaborn as sns
# import pandas as pd
# hm = pd.DataFrame(data=lfc, index=['1nn', '2nn', '3nn', '4nn', '5nn'])
# sns.heatmap(hm, center=0.)

end_time = time.time()
mnt = int((end_time - start_time) / 60)
if mnt >= 60:
    hr = int(mnt / 60)
    mnt = int(mnt % 60)
    
else:
    hr = 0
    
scn = int((end_time - start_time) % 60.)
print(str(hr).zfill(2) + ':' + str(mnt).zfill(2) + ':' + str(scn).zfill(2))

print(fc_filename)

sys.exit()

