#time to run 1k iterations

import matplotlib.pyplot as plt

#time_exp=[18,10,5]
#ngpus_exp=[1,2,4]

#Gaudi1 1.21  (eager mode, default)
time_voy_g1_121_eager=[55,33,19,9,6]
hpus_voy_g1_121_eager=[1,2,4,8,16]

#gaudi1 1.21 (Lazy mode)
time_voy_g1_121_lazy=[35,20,12,8,6]
hpus_voy_g1_121_lazy=[1,2,4,8,16]

#Gaudi1 1.18
time_voy_g1_118=[36,19,12,8,7]
hpus_voy_g1_118=[1,2,4,8,16]

#Gaudi2 1.21 (1-07 node) (Eager mode)
time_voy_g2_121=[140,65,33,24]
hpus_voy_g2_121=[1,2,4,8]

#Gaudi2 1.21 (1-07 node) (Lazy mode)
time_voy_g2_121_lazy=[10,6,4,5.5,7]
hpus_voy_g2_121_lazy=[1,2,4,6,8]

#Gaudi2 1.18 (1-10 node) (Lazy, default)
time_voy_g2_118=[11,7,4,3]
hpu_voy_g2_118=[1,2,4,8]


#plt.plot(ngpus_exp,time_exp,marker='d',label='Expanse (V100)')
plt.plot(hpus_voy_g1_121_eager,time_voy_g1_121_eager,marker='s',label='Gaudi 1.21 (Eager)',color='g')
plt.plot(hpus_voy_g1_121_lazy,time_voy_g1_121_lazy,linestyle=':',marker='d',label='Gaudi 1.21 (Lazy)',color='g')
plt.plot(hpus_voy_g1_118,time_voy_g1_118,linestyle='--',marker='*',label='Gaudi 1.18 (Lazy)',color='g')

plt.plot(hpus_voy_g2_121,time_voy_g2_121,marker='s',label='Gaudi2 1.21 (1-07, Eager)',color='b')
plt.plot(hpus_voy_g2_121_lazy,time_voy_g2_121_lazy,linestyle=':',marker='d',label='Gaudi2 1.21 (1-07, Lazy)',color='b')
plt.plot(hpu_voy_g2_118,time_voy_g2_118,linestyle='--', marker='*',label='Gaudi2 1.18 (1-10,Lazy)',color='b')


plt.yscale('log')
plt.xscale('log')
plt.xlabel('# devices')
plt.ylabel('Time(min)')
plt.title('Training Super-Resolution: 1k images')
plt.legend()
plt.show()
