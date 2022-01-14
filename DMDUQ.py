
#!/usr/bin/env python
# coding: utf-8

from cProfile import label
from paraDMD import paraDMD
import numpy as np
import matplotlib.pyplot as plt
import time


ks = [2.5,2.7,2.9,3.1,3.3,3.5]
zs = [5,7,9,11,13,15]
kk,zz = np.meshgrid(ks,zs)
kk,zz = kk.flatten(),zz.flatten()
mus = np.column_stack((kk,zz))

fileks = [25,27,29,31,33,35]
filezs = [50,70,90,110,130,150]
filekk,filezz = np.meshgrid(fileks,filezs)
filekk,filezz = filekk.flatten(),filezz.flatten()
for i, filek in enumerate(filekk):
  if i==0:
    tempdata = np.expand_dims(np.loadtxt('snapshots/TemperatureArray_001_10_{}_25_{}'.format(filezz[i],filek)), axis=0)
    raddata = np.expand_dims(np.loadtxt('snapshots/RadiationArray_001_10_{}_25_{}'.format(filezz[i],filek)), axis=0)
  else:
    tempdata = np.append(tempdata,np.expand_dims(np.loadtxt('snapshots/TemperatureArray_001_10_{}_25_{}'.format(filezz[i],filek)), axis=0), 0)
    raddata = np.append(raddata,np.expand_dims(np.loadtxt('snapshots/RadiationArray_001_10_{}_25_{}'.format(filezz[i],filek)), axis=0) ,0)
data = np.append(tempdata,raddata,axis=1)


alpha_mu = np.random.random(20)+2.5
z_mu = 10 * np.ones_like(alpha_mu)
alpha_mus = np.column_stack((alpha_mu,z_mu))

pDMD = paraDMD(rank=20,rbf='linear')
pDMD.build_DMD(np.asarray(data), mus)
local_error_sum = np.zeros(np.shape(data)[2])
avg_error = 1
error_chg = 1
DMD_solution = []
for i, mu in enumerate(alpha_mus):
  print(i)
  start_time = time.time()
  DMD_solution.append(np.mean(pDMD.evaluate_QOI_time_sequence(mus,mu,2,np.shape(data)[2])))
  print(time.time()-start_time)

plt.plot(alpha_mu,DMD_solution,'o')
plt.xlabel('$\\alpha$')
plt.ylabel('Average Temperature')
plt.grid()
plt.savefig('alphaSens.jpg')


z_mu =  np.linspace(5,15,11)
alpha_mu = 3*np.ones_like(z_mu)
z_mus = np.column_stack((alpha_mu,z_mu))

pDMD = paraDMD(rank=20,rbf='linear')
pDMD.build_DMD(np.asarray(data), mus)
local_error_sum = np.zeros(np.shape(data)[2])
avg_error = 1
error_chg = 1
DMD_solution = []
for i, mu in enumerate(z_mus):
  print(i)
  start_time = time.time()
  DMD_solution.append(np.mean(pDMD.evaluate_QOI_time_sequence(mus,mu,2,np.shape(data)[2])))
  print(time.time()-start_time)

plt.plot(z_mu,DMD_solution,'o')
plt.xlabel('Atomic Number')
plt.ylabel('Average Temperature')
plt.grid()
plt.savefig('zSens.jpg')


mean = [3,10]
cov = [[.5/3,0],[0,5/3]]

sampled_mus = np.random.multivariate_normal(mean,cov,[60])

saved_mus = []
for sample in sampled_mus:
  if 2.5 < sample[0] and sample[0] < 3.5 and 5 < sample[1] and sample[1] < 15:
    saved_mus.append([sample[0],round(sample[1])])
sampled_mus = saved_mus

pDMD = paraDMD(rank=20,rbf='linear')
pDMD.build_DMD(np.asarray(data), mus)
local_error_sum = np.zeros(np.shape(data)[2])
avg_error = 1
error_chg = 1
DMD_solution = []
for i, mu in enumerate(sampled_mus):
  print(i)
  start_time = time.time()
  DMD_solution.append(np.mean(pDMD.evaluate_QOI_time_sequence(mus,mu,2,np.shape(data)[2]),axis=0))
  print(time.time()-start_time)

mean = np.mean(DMD_solution,axis=0)

pl = plt.figure(dpi = 100)
time_points = np.linspace(0.25,2,30)
for solution in DMD_solution:
  plt.plot(time_points,solution, 'b-',alpha=0.3)
plt.plot(time_points,mean, 'r-',label = 'Mean')
plt.legend()
plt.grid()
plt.ylabel('Temperature')
plt.xlabel('Time (s)')
plt.savefig('QOI.png')