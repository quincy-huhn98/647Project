
#!/usr/bin/env python
# coding: utf-8

from paraDMD import paraDMD
import numpy as np
import matplotlib.pyplot as plt


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

tempsolutions = np.loadtxt('snapshots/TemperatureArray_001_10_110_25_295')
radsolutions = np.loadtxt('snapshots/RadiationArray_001_10_110_25_295')
solutions = np.vstack([tempsolutions,radsolutions])

test_ks = [2.6,2.8,3.0,3.2,3.4]
test_zs = [6,8,10,12,14]
test_kk,test_zz = np.meshgrid(test_ks,test_zs)
test_kk,test_zz = test_kk.flatten(),test_zz.flatten()
test_mus = np.column_stack((test_kk,test_zz))

test_fileks = [25,28,30,32,34]
test_filezs = [60,80,100,120,140]
test_filekk,test_filezz = np.meshgrid(test_fileks,test_filezs)
test_filekk,test_filezz = test_filekk.flatten(),test_filezz.flatten()
for i, filek in enumerate(test_filekk):
  if i==0:
    tempdata = np.expand_dims(np.loadtxt('snapshots/TemperatureArray_001_10_{}_25_{}'.format(test_filezz[i],filek)), axis=0)
    raddata = np.expand_dims(np.loadtxt('snapshots/RadiationArray_001_10_{}_25_{}'.format(test_filezz[i],filek)), axis=0)
  else:
    tempdata = np.append(tempdata,np.expand_dims(np.loadtxt('snapshots/TemperatureArray_001_10_{}_25_{}'.format(test_filezz[i],filek)), axis=0), 0)
    raddata = np.append(raddata,np.expand_dims(np.loadtxt('snapshots/RadiationArray_001_10_{}_25_{}'.format(test_filezz[i],filek)), axis=0) ,0)
test_data = np.append(tempdata,raddata,axis=1)


plot_error = []
error = []
dim = 10
ranks = [5,10,15,20,25]

for r in ranks:
  pDMD = paraDMD(rank=r,rbf='linear')
  pDMD.build_DMD(np.asarray(data), mus)
  local_error_sum = np.zeros(np.shape(data)[2])
  avg_error = 1
  error_chg = 1
  for i, mu in enumerate(test_mus):
    DMD_solution = pDMD.evaluate_time_sequence(mus,mu,2,np.shape(data)[2])
    local_error = []
    for j in range(30):
      local_error.append(np.linalg.norm(test_data[i,:,j]-DMD_solution[:,j])/np.linalg.norm(test_data[i,:,j]))
    
    error.append(np.mean(local_error))
    local_error_sum += local_error
    avg_error_new = np.mean(error)
    error_chg = abs(avg_error_new - avg_error)
    avg_error = avg_error_new

  i+=1
  plot_error.append(np.array(local_error_sum).T/i)

plot_error = np.array(plot_error).T

print('2param_error')
pl = plt.figure(dpi = 100)
time_points = np.linspace(0.25,2,len(plot_error[:,0]))
plt.semilogy(time_points,plot_error[:,0], 'b-',label = 'rank = ' + str(ranks[0]))
print(np.mean(plot_error[:,0]))
plt.semilogy(time_points,plot_error[:,1], 'g--', label = 'rank = ' + str(ranks[1]))
print(np.mean(plot_error[:,1]))
plt.semilogy(time_points,plot_error[:,2], 'k:',label = 'rank = ' + str(ranks[2]))
print(np.mean(plot_error[:,2]))
plt.semilogy(time_points,plot_error[:,3], 'r-.',label = 'rank = ' + str(ranks[3]))
print(np.mean(plot_error[:,3]))
plt.semilogy(time_points,plot_error[:,4], 'r-.',label = 'rank = ' + str(ranks[4]))
print(np.mean(plot_error[:,4]))
plt.legend()
plt.grid()
plt.ylabel('L2-Error')
plt.xlabel('Time (s)')
plt.savefig('Validation.png')


plot_error = []
error = []
dim = 10
ranks = [5,10,15,20,25]

for r in ranks:
  pDMD = paraDMD(rank=r,rbf='linear')
  pDMD.build_DMD(np.asarray(data), mus)
  local_error_sum = np.zeros(np.shape(data)[2])
  avg_error = 1
  error_chg = 1
  for i, mu in enumerate(test_mus):
    DMD_solution = pDMD.evaluate_time_sequence(mus,mu,2,np.shape(data)[2])
    local_error = []
    for j in range(30):
      local_error.append(np.linalg.norm(test_data[i,:np.shape(test_data)[1]//2,j].reshape((33,33))[7:14,19:26] - \
        DMD_solution[:np.shape(DMD_solution)[0]//2,j].reshape((33,33))[7:14,19:26])/np.linalg.norm(test_data[i,:np.shape(test_data)[1]//2,j].reshape((33,33))[7:14,19:26]))
    
    error.append(np.mean(local_error))
    local_error_sum += local_error
    avg_error_new = np.mean(error)
    error_chg = abs(avg_error_new - avg_error)
    avg_error = avg_error_new

  i+=1
  plot_error.append(np.array(local_error_sum).T/i)

plot_error = np.array(plot_error).T

print('2param_error')
pl = plt.figure(dpi = 100)
time_points = np.linspace(0.25,2,len(plot_error[:,0]))
plt.semilogy(time_points,plot_error[:,0], 'b-',label = 'rank = ' + str(ranks[0]))
print(np.mean(plot_error[:,0]))
plt.semilogy(time_points,plot_error[:,1], 'g--', label = 'rank = ' + str(ranks[1]))
print(np.mean(plot_error[:,1]))
plt.semilogy(time_points,plot_error[:,2], 'k:',label = 'rank = ' + str(ranks[2]))
print(np.mean(plot_error[:,2]))
plt.semilogy(time_points,plot_error[:,3], 'r-.',label = 'rank = ' + str(ranks[3]))
print(np.mean(plot_error[:,3]))
plt.semilogy(time_points,plot_error[:,4], 'r-.',label = 'rank = ' + str(ranks[4]))
print(np.mean(plot_error[:,4]))
plt.legend()
plt.grid()
plt.ylabel('L2-Error')
plt.xlabel('Time (s)')
plt.savefig('ValidationQOI.png')