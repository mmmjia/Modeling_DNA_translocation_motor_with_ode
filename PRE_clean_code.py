# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 12:43:08 2024

@author: user
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import math
import sys

# Define the parameters
gamma = 200 #pN ms/nm
katp0=650 #ms muM
tatp0=28.8 #ms
atp_con=20 #muM atp concentration
tr=1 #ms
datp=90 #free energy pN nm
tmech=gamma*0.85**2/datp
ts0=0.05 #ms
katp=1/(katp0/atp_con+tatp0) #rate for take ireversibly four ATPs
l0=0.85

class runsimulation:

    def __init__(self,ex_force,
                 simulation_cycle=3,sigma=0.1,rtol=1e-4, atol=1e-6,Vb_max=270/2/(0.85/2), VD_max=150*2/0.85,time_interval=100):
        
        self.time_interval=time_interval
        self.sigma=sigma #use guassian variance to control the stochatic effect
        self.atol=atol
        self.rtol=rtol
        self.ex_force=ex_force
        self.VD_max=VD_max
        self.Vb_max=Vb_max
        self.simulation_cycle=simulation_cycle
        self.td_rand=self.get_TD()
        self.ts_rand=self.get_ts()
        
    
    def DisTD(self,t):  ##distribution of dwell time tD
       return katp**4*t**3*np.exp(-katp*t)/3/2

    def Dists(self,t): ##distribution of dwell time ts
        ks=math.exp(self.ex_force/5.87)/ts0
        return ks*np.exp(-ks*t)
 
    def get_TD(self,tdd = np.linspace(0.01, 1000, 2000)):
        dist_tdd=self.DisTD(tdd)
        dist_tdd=dist_tdd/np.sum(dist_tdd)
        tDcycle=np.random.choice(tdd,size=self.simulation_cycle,p=dist_tdd)
        return tDcycle
    
    def get_ts(self,ts = np.linspace(0.01, 150, 2000)):
        dist_ts=self.Dists(ts)
        dist_ts=dist_ts/np.sum(dist_ts)
        tscycle=np.random.choice(ts,size=self.simulation_cycle*4,p=dist_ts)
        return tscycle
        
    def square_wave(self,x, period, Vm): #VD' (after differentiated)
        
        wave = np.where(x % period <= period*0.5, Vm, -Vm)
        
        return wave

    def Vb(self,i,x,period,Vm):#VB' (after differentiated)
        
         wave=self.square_wave(x, period,Vm=Vm)     
         
         wave[np.where((x >= ((i) * period-0.5*period)) & 
                       (x <= (i+1) * period))] = -Vm/3
         
         return wave

    def ode_func(self,t, y):
        random_force = np.random.normal(0, self.sigma)
        dydt = (self.ex_force-self.square_wave(y, 0.85,self.VD_max))/gamma+random_force
        return dydt


    def ode_funcwork(self,k,t, y):
        random_force = np.random.normal(0, self.sigma)
        dydt = (self.ex_force-self.Vb(k,y,0.85,self.Vb_max))/gamma+random_force
        return dydt 
        
    def simulation_function(self):
        print('start simulation')
        time_count=0
        t_all=[]
        traj_all=[]
        y0=l0
        k=0
        for i in range(self.simulation_cycle):
            sol0 = solve_ivp(self.ode_func, [time_count, time_count+self.td_rand[i]], [y0],
                             t_eval=np.linspace(time_count, time_count+self.td_rand[i],self.time_interval),
                             rtol=self.rtol, atol=self.atol)
            t_all.append(sol0.t)
            traj_all.append(sol0.y.squeeze())
            time_count=time_count+self.td_rand[i]
            y0new=sol0.y.squeeze()[-1]    

            for j in range(4):
                
                k0=round(y0new/l0)

                k=k+1
                sol1 = solve_ivp(lambda t,y: self.ode_funcwork(k,t,y), [time_count, time_count+tr+tmech], [y0new],
                                 t_eval=np.linspace(time_count, time_count+tr+tmech, self.time_interval), rtol=self.rtol, atol=self.atol)
                
                t_all.append(sol1.t)
                traj_all.append(sol1.y.squeeze())
                time_count=time_count+tr+tmech
                y1new=sol1.y.squeeze()[-1]
                
                sol2 = solve_ivp(self.ode_func, [time_count, time_count+self.ts_rand[k-1]], [y1new],
                                 t_eval=np.linspace(time_count, time_count+self.ts_rand[k-1], self.time_interval), rtol=self.rtol, atol=self.atol)
                t_all.append(sol2.t)

                traj_all.append(sol2.y.squeeze())
                time_count=time_count+self.ts_rand[k-1]
                y0new=sol2.y.squeeze()[-1]

            y0=y0new.copy()
            print(y0)
                
        t_all = np.stack(t_all, axis=0).flatten()
        traj_all=np.stack(traj_all,axis=0).flatten()
        plt.plot(t_all,traj_all)
        plt.xlabel('t')
        plt.ylabel('x(t)')
        plt.title('Solution to the Differential Equation')
        plt.show()
        return t_all,traj_all
        
        

        
sim1=runsimulation(-35,simulation_cycle=5)

t_all1,traj_all1=sim1.simulation_function()

sim2=runsimulation(-8,simulation_cycle=5)
t_all2,traj_all2=sim2.simulation_function()


plt.figure()
plt.plot(t_all1,traj_all1)
plt.plot(t_all2,traj_all2)
plt.xlabel('t')
plt.ylabel('x(t)')
plt.show()        
        
        
        
        
        
        
        