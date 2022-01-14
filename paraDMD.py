import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.interpolate
import time
#import vtk
import os
#from vtk.numpy_interface import dataset_adapter as dsa
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge, Lasso
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
import warnings
warnings.filterwarnings("ignore")


class paraDMD:

  def __init__(self, rank=0, rbf='linear', Urepsilon=0, Atilepsilon=0, omegasepsilon=0, Wepsilon=0, dataepsilon=0):
    # input:
    #   rank : snapshots for all parameters: n_param x n_dof x steps
    #   rbf  : parameter values at which snapshots have been obtained
    #   *epsilon: epsilon value corresponding to the interpolation of the prefix with rbf function
    self.rank          = rank
    self.function      = rbf
    self.Urepsilon     = Urepsilon
    self.Atilepsilon   = Atilepsilon
    self.omegasepsilon = omegasepsilon
    self.Wepsilon      = Wepsilon
    self.dataepsilon   = dataepsilon

  def build_reduced_koopman(self, snapshots, screePlotting=False):
    # Compute lagged and forward data
    X = snapshots[:,:-1] + 0j
    self.Y = snapshots[:,1:]  + 0j

    # Perform SVD on the lagged snapshot matrix 
    U,Sig,Vh = np.linalg.svd(X, False) 
    if screePlotting:
        plt.figure()
        plt.semilogy(Sig,'o')
        plt.xlabel('Rank')
        plt.ylabel('Singular Values')
        plt.grid()
        plt.savefig('scree.jpg')
    
    # Truncate the SVD data
    print(self.rank,' modes retained for DMD')
    Ur = U[:,:self.rank]
    Sigr = np.diag(Sig)[:self.rank,:self.rank]
    Vr = Vh.conj().T[:,:self.rank]
    
    Atil = Ur.conj().T @ self.Y @ Vr @ np.linalg.inv(Sigr)
    return Ur, Atil

  def build_DMD(self, data, mus):
    # input:
    #   data : snapshots for all parameters: n_param x n_dof x steps
    #   mus  : parameter values at which snapshots have been obtained

    # Initialize matrices
    n_param     = len(mus)
    self.Ur          = np.zeros([n_param, np.shape(data)[1], self.rank        ])     
    self.initialData = np.zeros([n_param, np.shape(data)[1], 1                ])
    self.Atil        = np.zeros([n_param, self.rank        , self.rank        ])

    for i in range(n_param):
      self.initialData[i,:,:] = np.expand_dims(data[i,:,0], axis=1)
      self.Ur[i,:,:], self.Atil[i,:,:] = self.build_reduced_koopman(data[i])

  def evaluate_time_sequence(self,mus,mu,final_time,steps):
    self.Urx = self.interpolate_2D_param(self.Ur, mus, mu, self.function)
    Atilx = self.interpolate_2D_param(self.Atil, mus, mu, self.function)
    initialDatax = self.interpolate_1D_param(self.initialData, mus, mu, self.function)
    
    self.omegasx, self.Wx = np.linalg.eig(Atilx)
    phi = self.Urx @ self.Wx
    self.x0 = np.linalg.pinv(phi) @ initialDatax

    # Compute DMD evolution
    dt = final_time/steps
    timet = 0
    solutions = []
    for _ in range(steps):
      tilde_solution = phi @ np.diag(self.omegasx**(timet/dt)) @ self.x0
      solutions.append(tilde_solution)
      timet += dt
    self.solutions = np.real(np.array(solutions)).T
    return self.solutions

  def evaluate_QOI_time_sequence(self,mus,mu,final_time,steps):
    self.Urx = self.interpolate_2D_param(self.Ur, mus, mu, self.function)
    Atilx = self.interpolate_2D_param(self.Atil, mus, mu, self.function)
    initialDatax = self.interpolate_1D_param(self.initialData, mus, mu, self.function)
        
    self.omegasx, self.Wx = np.linalg.eig(Atilx)
    phi = self.Urx @ self.Wx
    self.x0 = np.linalg.pinv(phi) @ initialDatax

    # Compute DMD evolution
    dt = final_time/steps
    timet = 0
    solutions = []
    for _ in range(steps):
      tilde_solution = phi @ np.diag(self.omegasx**(timet/dt)) @ self.x0
      solutions.append(tilde_solution[:len(tilde_solution)//2].reshape((33,33))[7:14,19:26].flatten())
      timet += dt
      self.solutions = np.real(np.array(solutions)).T
    return self.solutions

  def interpolate_2D_param(self, A, points, point, interpolationMode = 'linear'):

    num_of_param = np.shape(A)[0]
    shape_data_1 = np.shape(A)[1] # Warning: this is a general interpolation function for Reduced Koopman Operator (RKO) and U
    shape_data_2 = np.shape(A)[2] # RKO shape_1 = rank,shape_2 = rank: U shape_1 = dof,shape_2 = rank

    # Interpolates the set of matrices A in terms of xs at value x
    real_interpolant = np.zeros((shape_data_1,shape_data_2))
    imag_interpolant = np.zeros((shape_data_1,shape_data_2))

    A_real = np.real(A)
    A_imag = np.imag(A)
    
    for i in range(np.shape(A)[1]):
      for j in range(np.shape(A)[2]):

        # Linear intepolation -- maybe remove
        if interpolationMode == 'linear':
          real_interpolant[i,j] = scipy.interpolate.griddata(points,A_real[:,i,j],point,method=interpolationMode)
          imag_interpolant[i,j] = scipy.interpolate.griddata(points,A_imag[:,i,j],point,method=interpolationMode)

        # Polynomial interpolation
        elif interpolationMode == 'poly':
          model_r, model_i = Ridge(), Ridge()
          poly = PolynomialFeatures(2)
          points_base = points
          X = poly.fit_transform(points_base)
          model_r.fit(X, A_real[:,i,j])
          model_i.fit(X, A_imag[:,i,j])
          real_interpolant[i,j] = model_r.predict(poly.fit_transform(np.asarray(point).reshape(1, -1)))
          imag_interpolant[i,j] = model_i.predict(poly.fit_transform(np.asarray(point).reshape(1, -1)))

        # Gaussian Process interpolation
        elif interpolationMode == 'gp':
          global_max = np.max([np.max(A_real), np.max(A_imag)])
          current_std_real, current_mean_real = np.std(A_real[:,i,j]), np.mean(A_real[:,i,j])
          current_std_imag, current_mean_imag = np.std(A_imag[:,i,j]), np.mean(A_imag[:,i,j])
          kernel_r = ConstantKernel(constant_value=abs(current_mean_real), constant_value_bounds=(0, global_max*2)) *\
                   RBF(length_scale=current_std_real, length_scale_bounds=(current_std_real/10, current_std_real*10))
          kernel_i = ConstantKernel(constant_value=abs(current_mean_imag), constant_value_bounds=(0, global_max*2)) *\
                   RBF(length_scale=1, length_scale_bounds=(1e-5, 10))
          model_r = GaussianProcessRegressor(kernel=kernel_r, random_state=0).fit(points, A_real[:,i,j])
          model_i = GaussianProcessRegressor(kernel=kernel_i, random_state=0).fit(points, A_imag[:,i,j])
          real_interpolant[i,j] = model_r.predict(np.asarray(point).reshape(1, -1))
          imag_interpolant[i,j] = model_i.predict(np.asarray(point).reshape(1, -1))

        # Multilayer perceptron interpolation
        elif interpolationMode == 'mlp':
          regr_r = MLPRegressor(hidden_layer_sizes=[2,2], max_iter=100, solver='lbfgs').fit(points, A_real[:,i,j])
          real_interpolant[i,j] = regr_r.predict(np.asarray(point).reshape(1, -1))
          regr_i = MLPRegressor(hidden_layer_sizes=[2,2], max_iter=100, solver='lbfgs').fit(points, A_imag[:,i,j])
          imag_interpolant[i,j] = regr_i.predict(np.asarray(point).reshape(1, -1))
        
        # Adaboost Regression interpolation
        elif interpolationMode == 'ada':
          regr_r = AdaBoostRegressor(DecisionTreeRegressor(max_depth=10),n_estimators=12)
          regr_r.fit(points, A_real[:,i,j])
          real_interpolant[i,j] = regr_r.predict(np.asarray(point).reshape(1, -1))
          regr_i = AdaBoostRegressor(DecisionTreeRegressor(max_depth=10),n_estimators=12)
          regr_i.fit(points, A_imag[:,i,j])
          imag_interpolant[i,j] = regr_i.predict(np.asarray(point).reshape(1, -1))

    interpolant = real_interpolant + 1j * imag_interpolant
    return interpolant

  def interpolate_1D_param(self, A, points, point, interpolationMode = 'linear'):
    # Interpolates the set of matrices A in terms of xs at value x
    real_interpolant = np.zeros((np.shape(A)[1]))
    imag_interpolant = np.zeros((np.shape(A)[1]))
    doLinear = False
    A_real = np.real(A)
    A_imag = np.imag(A)
    
    for i in range(np.shape(A)[1]):
      if interpolationMode == 'linear':
        real_interpolant[i] = scipy.interpolate.griddata(points,A_real[:,i],point,method=interpolationMode)[0]
        imag_interpolant[i] = scipy.interpolate.griddata(points,A_imag[:,i],point,method=interpolationMode)[0]
      elif interpolationMode == 'poly':
        model_r, model_i = Ridge(), Ridge()
        poly = PolynomialFeatures(2)
        points_base = points
        X = poly.fit_transform(points_base)
        model_r.fit(X, A_real[:,i])
        model_i.fit(X, A_imag[:,i])
        real_interpolant[i] = model_r.predict(poly.fit_transform(np.asarray(point).reshape(1, -1)))
        imag_interpolant[i] = model_i.predict(poly.fit_transform(np.asarray(point).reshape(1, -1)))
      else:
        doLinear = True
        
    if doLinear == True:
      print('Trained interpolation models use linear for initial data')
      interpolant = self.interpolate_1D_param(A, points, point, 'linear')
      real_interpolant = np.real(interpolant)
      imag_interpolant = np.imag(interpolant)
    interpolant = real_interpolant + 1j * imag_interpolant
    return interpolant