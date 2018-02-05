# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 22:33:09 2017
Iteration functions for non-LTE calculation
1. read parameters
2. set calculate grids (set layer width)
--> Tk(z), P(z), ni(z)
3. radiative transfer calculation
-->Ji
4. Set matrix of An=f
--> ni_new
5. iterate 3,4
@author: yamada Takayoshi
cd home/yamada/workspace/ATRASU/atrasu
"""
import numpy as np
import pylab as P
from scipy import constants as C
import typhon.nonlte as atrasu

"""
================================================================
1. Read constant parameter
================================================================
"""
"""============================================================================
1.1 Read atmoshpere
============================================================================"""
Radius = 2631.e3  # *1000.  # sphere_GANYMEDE
atmosphere = '/home/yamada/workspace/ATRASU/main_data/JUICE/'#Lat60/'
#from atrasu.setup_atmosphere import *
alt, pre, temp, mole, alt_ref = atrasu.setup_atmosphere.swifile(atmosphere)
# atrasu.setup_atmosphere.plot_alt_temp_mole(atmosphere)

"""============================================================================
1.2 Read molecular parameter
  Einstein A and B coefficients, Collisional transition.
============================================================================"""
molecules_state_consts = '/home/yamada/workspace/data/JUCE/'
#from atrasu.nonltecalc import MolecularConsts, calcu_grid
# RaRaA, RaRaAd, RaRaBd, RaRaB_induced, RaRaB_absorption, Tran_tag, CoRa_block =\
# read_swi_molecules(molecules_state_consts,temp, mole)
# fn = open(RateConstantPath+'oH2O16.lev_7levels', 'r')
MolPara = atrasu.nonltecalc.MolecularConsts(molecules_state_consts)
RaRaA, RaRaAd, RaRaBd, RaRaB_induced, RaRaB_absorption = MolPara.ratematrix()
CoRa_block = MolPara.collision(mole, temp)
#CoRa_block, ETS = MolPara.collision(mole, temp, ETS=True)
Ite_pop = MolPara.population(mole, temp)  # population[Alt, 1, ni, iter]

"""
================================================================
Condition of Radiative Transfer calculation
================================================================
"""

""""Make distance Matrix(deg, NP)"""
mu_weight, PSC2, mu_tangent = atrasu.nonltecalc.calcu_grid(Radius, alt_ref)
mu_weight, PSC2, mu_tangent, speed = \
atrasu.nonltecalc.calcu_grid(Radius, alt_ref, speed=True)

"""abs coeff"""
#from atrasu.spectra.abscoeff import basic
basic= atrasu.spectra.abscoeff.basic

Abs_ite = np.array([basic(np.array(Ite_pop)[:, 0, lower, 0],
                          np.array(Ite_pop)[:, 0, upper, 0],
                          MolPara.blu[upper, lower],
                          MolPara.bul[upper, lower],
                          MolPara.freq_array[hoge]*1.e9)
                    for hoge, upper, lower in MolPara.tran_tag.astype(int)])


"""
setting Frequency range (absoption and induced transition)
====Line shape=====
This completely devidid the line and line.
This frequencies set ignoring mixing line shape!
I must to make flag for convolution of line shape
===================
"""

#from atrasu.spectra.lineshape import DLV
DLV =atrasu.spectra.lineshape.DLV
hwhm = DLV('D',
           temp.max(),
           HWHM=True,
           Para=[MolPara.freq_array*1.e9, 18.0153])[0]

Fre_range_i = ([np.linspace(MolPara.freq_array[x]*1.e9-hwhm[x]*5,
                            MolPara.freq_array[x]*1.e9+hwhm[x]*5, 200)
                for x in range(MolPara.nt)])

F_vl_i = [DLV('D',
              temp.reshape(temp.size, 1),
              Freq=Fre_range_i[x],
              Para=[MolPara.freq_array[x]*1.e9, 18.0153],
              HWHM=False)
          for x in range(MolPara.nt)]

cmZ = MolPara.freq_array[0]*1.e9*speed/c
cmz = np.ma.masked_where(cmZ==0,cmZ)
pcolormesh(alt_ref, mu_tangent, cmz*1.e-6)
cbar = colorbar()
cbar.set_label('|Doppler Shif| [MHz]', fontsize=18)
xlabel('Altitude Grid [km]', fontsize=18)
ylabel('Angle grid [km]', fontsize=18)
P.tick_params(axis='both',labelsize=14,size=5)
savefig('Doppler_Shif_556GHz.png')
 
#F_vl_i_d = Doppler_Expanding(velocity, PSC2, mu_tangent)

"""
non-LTE calculation iteration
"""
import imp
imp.reload(atrasu.nonltecalc)
imp.reload(atrasu.rtc)
#from atrasu.nonltecalc import Calc, CalcSC
import time

calc_count = 0
error = 100.
new_pop = 0  # dammy
Iteration_record = [np.array(Ite_pop)[:,0,:,0]]
#CalcSC is first order sosc
while error > 1.e-1:
#while calc_count < 10:
    A = time.time()
    if calc_count > 0:
        Ite_pop = new_pop
        Abs_ite = np.array([basic(np.array(Ite_pop)[:, 0, lower, 0],
                                  np.array(Ite_pop)[:, 0, upper, 0],
                                  MolPara.blu[upper, lower],
                                  MolPara.bul[upper, lower],
                                  MolPara.freq_array[hoge]*1.e9)
                            for hoge, upper, lower in
                            MolPara.tran_tag.astype(int)])

    new_pop = atrasu.nonltecalc.Calc(np.array(Ite_pop), Abs_ite,
                   PSC2, mu_tangent, mu_weight,
                   alt_ref, temp,
                   Fre_range_i, MolPara.freq_array, F_vl_i,
                   MolPara.b_place,
                   MolPara.nt, MolPara.ni,
                   MolPara.ai, MolPara.bul, MolPara.blu,
                   RaRaB_absorption, RaRaB_induced, RaRaA, RaRaAd, CoRa_block,
                   MolPara.tran_tag, wind_v=speed)
    diff = np.abs((np.array(Ite_pop) - new_pop)/np.array(Ite_pop))*100
    error = diff.max()
    Iteration_record.append(new_pop[:,0,:,0])
    print(calc_count, round(error,3), round((time.time() - A),3))
    calc_count += 1

savename = 'Excitation_Temp_GS_2SC_SSL10_01percent_dppler'
np.save(savename+'.npy', np.array(Iteration_record))


Mu_tangent = mu_tangent
Alt_ref = alt_ref
Tran_tag = MolPara.tran_tag
Ite_pop = np.array(Ite_pop)
Ite_pop = new_pop
Nt = MolPara.nt
Ni = MolPara.ni
Aul = MolPara.ai
Bul = MolPara.bul
Blu = MolPara.blu
Temp = temp
B_place = MolPara.b_place
Freq_array = MolPara.freq_array
import numpy as np
import pylab as plt
from scipy.constants import c, k, h

import numpy as N
plt.figure()
pcdot = ['ro-', 'bo-', 'go-', 'yo-', 'ko-', 'mo-', 'co-', 'rD-', 'bD-']
Trotational = N.zeros((MolPara.nt, alt_ref.size))
for xx in range(Nt):
    _up = int(MolPara.tran_tag[xx][1])
    _low = int(MolPara.tran_tag[xx][2])
    Nup = N.array(new_pop)[:, 0, _up, 0]
    Nlo = N.array(new_pop)[:, 0, _low, 0]
    Gup = MolPara.weighti[_up]
    Glo = MolPara.weighti[_low]
    Ehv = MolPara.freq_array[xx]*1.e9*h
    Trot = -Ehv/(k*N.log(Nup/Nlo*Glo/Gup))
    Trotational[xx] = Trot
    P.plot(Trot, alt_ref, pcdot[xx],
           label=str(round(MolPara.freq_array[xx],3)))
P.plot(temp, alt_ref, 'k-', label='T_k')
P.legend()


N.savetxt(savename +'.txt',N.vstack((alt_ref,Trotational)).T,
          delimiter='\t', header='Alt [km] \t Trot [K] at '
          +str(N.around(MolPara.freq_array,3)))


"""
Calc spectra
"""
Rtcl = atrasu.rtc.CalcSpectra
new_pop = Rtcl(np.array(Ite_pop), Abs_ite,
               PSC2, mu_tangent, mu_weight,
               alt_ref, temp,
               Fre_range_i, MolPara.freq_array, F_vl_i,
               MolPara.b_place,
               MolPara.nt, MolPara.ni,
               MolPara.ai, MolPara.bul, MolPara.blu,
               RaRaB_absorption, RaRaB_induced, RaRaA, RaRaAd, CoRa_block,
               MolPara.tran_tag)


FO = np.load('/home/yamada/workspace/ATRASU/atrasu/7-levels_GS_FOSC_SSL10.npy')
pop = FO[-1,:,:]
Abs_ite2 = np.array([basic(pop[:, lower],
                          pop[:, upper],
                          MolPara.blu[upper, lower],
                          MolPara.bul[upper, lower],
                          MolPara.freq_array[hoge]*1.e9)
                    for hoge, upper, lower in MolPara.tran_tag.astype(int)])
new_pop2 = Rtcl(FO[-1].reshape(101,1,7,1), Abs_ite2,
                PSC2, mu_tangent, mu_weight,
                alt_ref, temp,
                Fre_range_i, MolPara.freq_array, F_vl_i,
                MolPara.b_place,
                MolPara.nt, MolPara.ni,
                MolPara.ai, MolPara.bul, MolPara.blu,
                RaRaB_absorption, RaRaB_induced, RaRaA, RaRaAd, CoRa_block,
                MolPara.tran_tag)

from scipy.constants import c, k, h
vx = Fre_range_i[0]
y = new_pop[0][0]
y2 = new_pop2[0][0]
Tbk = h*vx/(k*np.log(2.*h*vx**3/c**2/y +1. ))
Tbk2 = h*vx/(k*np.log(2.*h*vx**3/c**2/y2 +1. ))
import matplotlib
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
#imshow(Tbk2-Tbk)
C = (Tbk2-Tbk)
C[C!=C] = 0
C_map = np.ma.masked_invalid(C)
#C_map = np.ma.masked_where(C_map==0,C_map)
levels = MaxNLocator(nbins=15).tick_values(C_map.min(), C_map.max())
levels = [  -60,    -50,  -40,           -20,         -5,       -1, 1, 5]
color = ['navy', 'blue', 'deepskyblue', 'lawngreen', 'yellow', 'w', 'red']

#clevel = [0, 1, 10, 100, 200, 300]
#colorle = [str(tt) for tt in np.linspace(0.7,0.1,len(level)-1)]
#colorle.insert(0, '1')
#Z = np.ma.masked_where(mapdata==0,mapdata)
#cs = m.pcolormesh(px, py, Z, cmap=cmap, norm=norm, zorder=1)
cmap, norm = matplotlib.colors.from_levels_and_colors(levels, color)
#cmap = plt.get_cmap('jet')
#norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
PX = (vx*1.e-9 - MolPara.freq_array[0])*1.e3
PY = mu_tangent*1.
PY[mu_tangent<0] = PY[mu_tangent<0]*0.1
Pcol = pcolormesh(PX, PY, C_map,cmap=cmap,norm=norm)
#pcolormesh((vx*1.e-9 - MolPara.freq_array[0])*1.e3, mu_tangent, C_map, vmin=C_map.min(),vmax=C_map.max())
colb = colorbar(Pcol)
colb.set_label(r"NLTE - LTE (K)",fontsize=16)
levels = [  -60,    -50,  -40,           -20,         -5,    0,   -1, 1, 5]
colb.set_ticks(levels)
xlabel('Frequency offset [MHz] from ' + str(round(MolPara.freq_array[0],3))+' GHz',fontsize=16)
xlim(-2,2)
ylabel('Tangent height [km]',fontsize=16)
title('A Tentative plot for spectral difference for each geometry')
PYticks = PY[::25]*1.
#PYticks = np.insert(PYticks,-3,0)
PYticks2 = PYticks*1.
PYticks[PYticks<0] = PYticks[PYticks<0]*10.
yticks(PYticks2, PYticks)
# np.save('7-levels_GS_FOSC_SSL60.npy', Iteration_record)
#np.save('7-levels_GS_SOSC_SSL60.npy', Iteration_record)
# np.save('7-levels_GS_FOSC_SSL10.npy', Iteration_record)
# np.save('7-levels_GS_SOSC_SSL10.npy', Iteration_record)

FO = np.load('/home/yamada/workspace/ATRASU/atrasu/7-levels_GS_FOSC_SSL10.npy')
clf()
imshow(np.array(Iteration_record)[-1,:,:] - FO[-1,:,:])
colorbar()

imshow(((np.array(Iteration_record)[-1,:,:] - FO[-1,:,:])/(FO[-1,:,:])*100))
colorbar()

Mu_tangent = mu_tangent
Alt_ref = alt_ref
Tran_tag = MolPara.tran_tag
Ite_pop = np.array(Ite_pop)
#Ite_pop = new_pop
Nt = MolPara.nt
Ni = MolPara.ni
Aul = MolPara.ai
Bul = MolPara.bul
Blu = MolPara.blu
Temp = temp
B_place = MolPara.b_place
Freq_array = MolPara.freq_array
import numpy as np
import pylab as plt
from scipy.constants import c, k, h
from atrasu.mathmatics import Trapz_inte_edge
# from . import sub
# from ..spectra.lineshape import DLV
from atrasu.spectra.source_function import Bv_T, PopuSource_AB
from atrasu.spectra.abscoeff import basic

import numpy as N
plt.figure()
pcdot = ['ro-', 'bo-', 'go-', 'yo-', 'ko-', 'mo-', 'co-', 'rD-', 'bD-']
Trotational = N.zeros((MolPara.nt, alt_ref.size))
for xx in range(Nt):
    _up = int(MolPara.tran_tag[xx][1])
    _low = int(MolPara.tran_tag[xx][2])
    Nup = N.array(new_pop)[:, 0, _up, 0]
    Nlo = N.array(new_pop)[:, 0, _low, 0]
    Gup = MolPara.weighti[_up]
    Glo = MolPara.weighti[_low]
    Ehv = MolPara.freq_array[xx]*1.e9*h
    Trot = -Ehv/(k*N.log(Nup/Nlo*Glo/Gup))
    Trotational[xx] = Trot
    P.plot(Trot, alt_ref, pcdot[xx],
           label=str(round(MolPara.freq_array[xx],3)))
P.plot(temp, alt_ref, 'k-', label='T_k')

import numpy as N
plt.figure()
pcdot = ['ro-', 'bo-', 'go-', 'yo-', 'ko-', 'mo-', 'co-', 'rD-', 'bD-']
Trotational = N.zeros((MolPara.nt, alt_ref.size))
for xx in range(Ni):
    P.plot(FO[-1,:,xx], alt_ref, pcdot[xx],label='level:'+str(xx+1))
legend()
xscale('log')
tick_params(axis='both', which='both',direction='in')
tick_params(axis='both', which='major',length=5)
xlabel('Population', fontsize=12, weight='bold')
ylabel('Altitude [km]', fontsize=12, weight='bold')
grid(linestyle=':')
ylim(0,450)


import numpy as N
plt.figure()
pcdot = ['ro-', 'bo-', 'go-', 'yo-', 'ko-', 'mo-', 'co-', 'rD-', 'bD-']
Trotational = N.zeros((MolPara.nt, alt_ref.size))
for xx in range(Nt):
    _up = int(MolPara.tran_tag[xx][1])
    _low = int(MolPara.tran_tag[xx][2])
    P.plot(ETS[xx], alt_ref, pcdot[xx],
           label=str(round(MolPara.freq_array[xx],3)))
P.plot(temp, alt_ref, 'k-', label='T_k')



import pylab as P
xx = 2
level = np.arange(-10,3,2)+1
for xx in range(9):
    plt.figure(figsize=(8,8))
    ite_num = np.array(Iteration_record)[:,0,0].size
    xg = np.arange(0, ite_num, 1)

    true_error = np.array(Iteration_record)[-1, :, xx]
    coC = (np.array(Iteration_record)[:, :, xx] - true_error)/true_error*100
    CC = np.abs(coC.T)
    CC[CC == 0] = nan
    plC = np.ma.masked_where(CC!=CC, CC)
    # plt.pcolormesh(xg, alt_ref, np.log10(plC))
    plt.contourf(xg, alt_ref, np.log10(plC), level)
    cbar = P.colorbar()
    cbar.set_label('log10(|$\Delta$n|/n$_{\infty}$) [%] ', fontsize=24, weight='bold')
    # cbar.set_label('Relative change [%] ', fontsize=24, weight='bold')
    P.xlabel('Iteration number', fontsize=24,weight='bold')
    P.ylabel('Altitdue [km]', fontsize=24,weight='bold')
    P.tick_params(axis='both',labelsize=14,size=5)
    P.savefig(str(xx)+'convergence_change_SSL60.png')
    P.close()



P.yticks(N.arange(0,Alt_ref.size,20)[::-1], Alt_ref[N.arange(0,Alt_ref.size,20)])
P.xlim(-0.5,45.5)
P.xlim(-0.5,53.5)
P.ylim(100,0)





CC = (np.array(Iteration_record)[1:, :, xx] -
      np.array(Iteration_record)[:-1, :, xx])/ np.array(Iteration_record)[:-1, :, xx]
imshow(CC.T)

clf()
imshow(np.array(Iteration_record)[:,:,0].T)


PSC2 = ma.masked_invalid(PSC2)
plt.matshow(CC)


Ite_pop = N.load('GSIte99_7-levels_20170522.npy')
Ite_pop = N.load('7-levels_Lat60.npy')
Nlo = N.array(Ite_pop)[:,:,:]
Nlo_last = N.array(Ite_pop)[:,-1,:]
True_error = N.zeros((Nlo.shape[:-1]))
Relartive_error = N.zeros((Nlo.shape[:-1]))
terror = N.zeros((Nlo.shape[1]))
for hoge in range(Nlo.shape[1]-1):
         terror[hoge] = (N.abs(Nlo[:,hoge, :] -
                                         Nlo[:,hoge+1,:])).sum()/(Nlo[:,hoge+1,:].sum())
P.plot(terror*100)
yscale('log')
for hoge in range(Nlo.shape[0]):
    for hoge2 in range(Nlo.shape[1]-1):
        True_error[hoge, hoge2] = (N.abs(Nlo[hoge, hoge2, :] -
                                        Nlo_last[hoge])/Nlo_last[hoge]).sum()
        Relartive_error[hoge, hoge2] = (N.abs(Nlo[hoge, hoge2, :] -
                                        Nlo[hoge, hoge2+1, :])).sum()/(Nlo[hoge,hoge2,:].sum())
Nlo[0,-1,:].sum()
P.matshow(N.log10(N.abs(True_error))[::-1, :])
P.matshow(N.log10(N.abs(Relartive_error*100))[::-3, :])
cbar = P.colorbar()
cbar.set_label('log10(|$\Delta$n|/n$_{\infty}$) ', fontsize=24, weight='bold')
cbar.set_label('Relative change [%] ', fontsize=24, weight='bold')
P.xlabel('Iteration number', fontsize=24,weight='bold')
P.ylabel('Altitdue [km]', fontsize=24,weight='bold')
P.yticks(N.arange(0,Alt_ref.size,20)[::-1], Alt_ref[N.arange(0,Alt_ref.size,20)])
P.xlim(-0.5,45.5)
P.xlim(-0.5,53.5)
P.ylim(100,0)
P.tick_params(axis='both',labelsize=20,size=15)


tr = Trotational*1

0 71.4067783247
1 34.1879190156
2 22.5019614614
3 16.1443980864
4 12.0574818199
5 9.2094969875
6 7.13321664983
7 5.57689270896
8 4.38888473652
9 3.47063450617
10 2.75455354981
11 2.19247107271
12 1.74908221984
13 1.39797673531
14 1.11909655365
15 0.897033769007

vx = Fre_range_i[0]

ji_out_all_nonlte = ji_out_all*1.
y = ji_out_all[0,-1,10,:]
pl_x = (Fre_range_i[0]*1.e-9-Freq_array[0])*1.e3
Tb_y = h*vx/(k*N.log(2.*h*vx**3/c**2/y +1. ))
plt.plot(pl_x, Tb_y, 'k--', label='LTE nadir')


y = ji_out_all_nonlte[0,-1,0,:]
pl_x = (Fre_range_i[0]*1.e-9-Freq_array[0])*1.e3
Tb_y = h*vx/(k*np.log(2.*h*vx**3/c**2/y +1. ))
plt.plot(pl_x, Tb_y, 'k-', label='Non-LTE nadir')

y = ji_out_all[0,-1,-90,:]
pl_x = (Fre_range_i[0]*1.e-9-Freq_array[0])*1.e3
Tb_y = h*vx/(k*N.log(2.*h*vx**3/c**2/y +1. ))
plt.plot(pl_x, Tb_y, 'r--', label='LTE limb(50 km)')


y = ji_out_all_nonlte[0,-1,-90,:]
pl_x = (Fre_range_i[0]*1.e-9-Freq_array[0])*1.e3
Tb_y = h*vx/(k*N.log(2.*h*vx**3/c**2/y +1. ))
plt.plot(pl_x, Tb_y, 'r-', label='Non-LTE limb(50 km)')


y = ji_out_all[0,-1,-45,:]
pl_x = (Fre_range_i[0]*1.e-9-Freq_array[0])*1.e3
Tb_y = h*vx/(k*N.log(2.*h*vx**3/c**2/y +1. ))
plt.plot(pl_x, Tb_y, 'r--', label='LTE limb(250 km)')


y = ji_out_all_nonlte[0,-1,-45,:]
pl_x = (Fre_range_i[0]*1.e-9-Freq_array[0])*1.e3
Tb_y = h*vx/(k*N.log(2.*h*vx**3/c**2/y +1. ))
plt.plot(pl_x, Tb_y, 'r-', label='Non-LTE limb(250 km)')


y = ji_out_all[3,-1,-23,:]
pl_x = (Fre_range_i[0]*1.e-9-Freq_array[0])*1.e3
Tb_y = h*vx/(k*N.log(2.*h*vx**3/c**2/y +1. ))
plt.plot(pl_x, Tb_y, 'b--', label='LTE limb(300 km)')


y = ji_out_all_nonlte[3,-1,-23,:]
pl_x = (Fre_range_i[0]*1.e-9-Freq_array[0])*1.e3
Tb_y = h*vx/(k*N.log(2.*h*vx**3/c**2/y +1. ))
plt.plot(pl_x, Tb_y, 'b-', label='Non-LTE limb(300 km)')

plt.grid(linestyle=':')
plt.tick_params
plt.xlabel('Freqency offset [MHz]',fontsize=20)
plt.ylabel('Brightness Temperature [K]',fontsize=20)
plt.tick_params(labelsize=15)
plt.xlim(-3,3)
plt.legend()




plot(alt_ref, CoRa_block)
[plot(CoRa_block[:,int(x),int(y)], alt_ref) for x, y in MolPara.tran_tag[:,1:]]

[x*y for x, y in MolPara.tran_tag[:,1:]]
"[Nt, Alt, Mu, freq]"
ji_in = N.zeros((Alt_ref.size, Nt))
ji_in_all = N.zeros((Nt, Alt_ref.size, Mu_tangent.size, Fre_range.size))
ji_out_all = N.zeros((Nt, Alt_ref.size, Mu_tangent.size, Fre_range.size))
ji_out = N.zeros((Alt_ref.size, Nt))

Ite_pop = [[Ni_LTE[i].reshape((Ni, 1))] for i in range(Mole.size)]
# for IteNum in range(0, 100):
max_true_error = N.ones(Alt_ref.size)
IteNum = 0
while max_true_error.max() >= 1.e-3:
    print ()
    for ii in range(Alt_ref.size)[::-1]:  # spatial points
        # print ii
        for xx in range(Nt):  # transitions
            Fre_range = Fre_range_i[xx]
            if ii == Alt_ref.size-1:  # upper Boundary(u>0) incoming
                B_v_cosmic = B_v_cosmicint[xx]
                ji_in_all[xx, ii, :, :] = 0.*B_v_cosmic[:, 0].reshape((1,
                                                                      Fre_range.size))*1.
            elif ii == Alt_ref.size-1-1:
                tdu1 = xx_const_ini[xx][ii] * F_vl_i[xx][:, ii]
                tdu2 = xx_const_ini[xx][ii+1] * F_vl_i[xx][:, ii+1]
                tdu2 = calc_abscoeff2(ii+1, xx, IteNum) * F_vl_i[xx][:, ii+1]
                tdu1 = calc_abscoeff2(ii, xx, IteNum) * F_vl_i[xx][:, ii]
                tdu = 0.5*N.abs(tdu1+tdu2).reshape((1, Fre_range.size)) *\
                      PSC2[:, ii].reshape((Mu_tangent.size, 1))
                ydu = tdu-1.+N.exp(-tdu)  # (12.120)
                lambda_dd = ydu/tdu  # (12.117)
                lambda_ddu = -ydu/tdu+1.-N.exp(-tdu)   # (12.118)
                Sd = PopuSource(xx, ii, IteNum).reshape((1, Fre_range.size))*1.
                Sdu1 = PopuSource(xx, ii, IteNum).reshape((1, Fre_range.size))*1.
                Sdu2 = (Sd + Sdu1)/2.
                Idu = ji_in_all[xx, ii+1, :, :]*1.  # !!!CHECK!!!
                Sdu = PopuSource(xx, ii+1, IteNum).reshape((1, Fre_range.size))*1.
                ji_in_all[xx, ii, :, :] = (Idu*N.exp(-tdu) +
                                           lambda_dd*Sd +
                                           lambda_ddu*Sdu)  # (12.113)
                ji_in_all[xx, ii, ((Mu_tangent > Alt_ref[ii])), :] = 0  # mu; 0->i,
            else:
                tdu1 = xx_const_ini[xx][ii] * F_vl_i[xx][:, ii]
                tdu2 = xx_const_ini[xx][ii+1] * F_vl_i[xx][:, ii+1]
                tdu2 = calc_abscoeff2(ii+1, xx, IteNum) * F_vl_i[xx][:, ii+1]
                tdu1 = calc_abscoeff2(ii, xx, IteNum) * F_vl_i[xx][:, ii]
                tdu = 0.5*N.abs(tdu1+tdu2).reshape((1, Fre_range.size)) *\
                     PSC2[:, ii].reshape((Mu_tangent.size, 1))
                ydu = tdu-1.+N.exp(-tdu)  # (12.120)
                lambda_dd = ydu/tdu  # (12.117)
                lambda_ddu = -ydu/tdu+1.-N.exp(-tdu)   # (12.118)
                Sd = PopuSource(xx, ii, IteNum).reshape((1, Fre_range.size))*1.
                Sdu1 = PopuSource(xx, ii+1, IteNum).reshape((1, Fre_range.size))*1.
                Sdu2 = (Sd + Sdu1)/2.
                Idu = ji_in_all[xx, ii+1, :, :]*1.  # !!!CHECK!!!
                Sdu = PopuSource(xx, ii+1, IteNum).reshape((1, Fre_range.size))*1.
                ji_in_all[xx, ii, :, :] = (Idu*N.exp(-tdu) +
                                           lambda_dd*Sd +
                                           lambda_ddu*Sdu)  # (12.113)
                ji_in_all[xx, ii, ((Mu_tangent > Alt_ref[ii])), :] = 0  # mu; 0->i,
    for i in range(Alt_ref.size):  # spatial points
        # print i
        B_int = B_place*0.  # intensity for each transitoin
        for xx in range(Nt):  # transitions
            Fre_range = Fre_range_i[xx]
            if i == 0:  # Lower Boundary(u>0) outgoing
                ji_out_all[xx, i, :, :] = PopuSource(xx,
                                                     i,
                                                     IteNum
                                                     ).reshape((1, Fre_range.size))
                ji_out_all[:, i, (Mu_tangent > Alt[i]), :] = 0
#==============================================================================
#             elif i == Alt_ref.size-1:  # upper boundary
#                 tdl1 = xx_const_ini[xx][i] * F_vl_i[xx][:, i]
#                 tdl2 = xx_const_ini[xx][i-1] * F_vl_i[xx][:, i-1]
#                 tdl2 = calc_abscoeff2(i-1, xx, IteNum+1) * F_vl_i[xx][:, i-1]
#                 tdl1 = calc_abscoeff2(i, xx, IteNum) * F_vl_i[xx][:, i]
#                 tdl = 0.5*N.abs(tdl1+tdl2).reshape((1, NumChan)) *\
#                     PSC2[:, i-1].reshape((Mu_tangent.size, 1))  # (Mu, Fre)
#                 ydl = tdl-1.+N.exp(-tdl)  # (12.120)
#                 lambda_dd = ydl/tdl  # (12.115)
#                 lambda_ddl = -ydl/tdl+1.-N.exp(-tdl)   # (12.116)
#                 Sd = PopuSource(xx, i, IteNum).reshape((1, Fre_range.size))*1.
#                 Idl = ji_out_all[xx, i-1, :, :]*1.  #
#                 Sdl = PopuSource(xx, i-1, IteNum+1).reshape((1, Fre_range.size))*1.
#                 ji_out_all[xx, i, :, :] = (Idl*N.exp(-tdl) +
#                                            lambda_dd*Sd +
#                                            lambda_ddl*Sdl)  # (12.113)
#                 ji_out_all[xx, i, Mu_tangent > Alt_ref[i], :] = 0
#                 #ji_out_all[xx, i, Mu_tangent == Alt_ref[i], :] = Sd*1.
#                 ji_out_all[xx, i,
#                            Mu_tangent == Alt_ref[i],
#                            :] = ji_in_all[xx, i, Mu_tangent == Alt_ref[i], :]
#==============================================================================
            else:
                tdl1 = xx_const_ini[xx][i] * F_vl_i[xx][:, i]
                tdl2 = xx_const_ini[xx][i-1] * F_vl_i[xx][:, i-1]
                tdl1 = calc_abscoeff2(i, xx, IteNum) * F_vl_i[xx][:, i]
                # GS-iteration
                tdl2 = calc_abscoeff2(i-1, xx, IteNum+1) * F_vl_i[xx][:, i-1]
                tdl = 0.5*N.abs(tdl1+tdl2).reshape((1, Fre_range.size)) *\
                     PSC2[:, i-1].reshape((Mu_tangent.size, 1))  # (Mu, Fre)
                ydl = tdl-1.+N.exp(-tdl)  # (12.120)
                lambda_dd = ydl/tdl  # (12.115)
                lambda_ddl = -ydl/tdl+1.-N.exp(-tdl)   # (12.116)
                Sd = PopuSource(xx, i, IteNum).reshape((1, Fre_range.size))*1.
                Idl = ji_out_all[xx, i-1, :, :]*1.  #
                Sdl = PopuSource(xx, i-1, IteNum+1).reshape((1, Fre_range.size))*1.
                ji_out_all[xx, i, :, :] = (Idl*N.exp(-tdl) +
                                           lambda_dd*Sd +
                                           lambda_ddl*Sdl)  # (12.113)
                ji_out_all[xx, i, Mu_tangent > Alt_ref[i], :] = 0
                ji_out_all[xx, i,
                           Mu_tangent == Alt_ref[i],
                           :] = ji_in_all[xx, i, Mu_tangent == Alt_ref[i], :]
        for xx in range(Nt):  # transitions
            Fre_weight = Trapz_inte_edge(F_vl_i[xx][:, i], Fre_range_i[xx])
            weighttemp = mu_weight[:,i].reshape(Mu_point.size, 1)*N.ones((Mu_point.size, Fre_weight.size))
            weighttemp = weighttemp*1./weighttemp[:,0].sum()
            intex = N.zeros((Mu_point.size))
            intex[:-(Alt_ref.size-i)] = Mu_point[:-(Alt_ref.size-i)]
            Intex = N.ones((Mu_point.size, Fre_weight.size)) * intex.reshape(Mu_point.size, 1)
#==============================================================================
#             j_mu_2 = N.sum((ji_out_all[xx, i, :, :] +
#                            ji_in_all[xx, i, :, :]) *
#                           mu_weight[:, i].reshape((Mu_point.size, 1)),
#                           axis=0)
#             j_mu1 = N.sum((ji_out_all[xx, i, :, :] +
#                           ji_in_all[xx, i, :, :]) *
#                          Intex/intex.sum(),
#                          axis=0)*0.5
#==============================================================================
            j_mu = N.sum((ji_out_all[xx, i, :, :] +
                          ji_in_all[xx, i, :, :]) *
                         weighttemp,
                         axis=0)*0.5
#            Freq_weight = F_vl_i[xx][:, i]*1.  # /F_vl_i[xx][:, i].sum()
#            J_mean = (j_mu * FreDelta * Freq_weight).sum()  # * FreDelta test
            J_mean = (j_mu * Fre_weight).sum()  # * FreDelta
#            if Alt_ref[i]==0: J_mean = Bbar[xx, i]
            if Alt_ref[i]==0:
                J_mean = (N.array(B_vl_int)[xx, :, 0, 0] * Fre_weight).sum()
            B_int[B_place == xx] = J_mean*1.
        RaRij = (RaRaB_absorption+RaRaB_induced) * B_int
        RaRii = N.eye(Ni)*(RaRij.sum(axis=0))*-1.
        A_m = (RaRaA+RaRaAd+RaRij+RaRii+CoRa_block[i])*-1.
        # A_m = (RaRaA+RaRaAd+CoRa_block[i])*-1.
        # A_m = (CoRa_block[i])*-1.
        A_m[-1, :] = 1.
        b = N.zeros((Ni, 1))*1.
        # b[-1] = Ni_LTE[i].sum()
        b[-1] = Ite_pop[i][0].sum()
        #Ni_LTE = Mole.reshape((Mole.size, 1))*RoPr
        #n_old = Ni_LTE[i].reshape(Ni, 1)*1.
        n_old = Ite_pop[i][IteNum]*1.
        # P.plot(x, n_old, 'bo')
        # N.dot(A_m,Ni_LTE[0].reshape(Ni,1))
        #n_delta_correct = b - N.dot(A_m, n_old)
        #n_new = n_old - n_delta_correct  # plus or minus?
        # """Inversion method"""
        n_new = N.linalg.inv(A_m).dot(b)
        n_delta = n_new-n_old
        Ite_pop[i].append(n_new)
        # print( i, ((n_delta/n_new)*100).max(), IteNum)
        max_true_error[i] = N.abs((n_delta/n_new)*100).max()
    print('Num:', IteNum, max_true_error.max(), '%',
          'at', Alt_ref[N.argmax(max_true_error)], 'km')
    IteNum += 1


import numpy as N
plt.figure()
pcdot = ['ro-', 'bo-', 'go-', 'yo-', 'ko-', 'mo-', 'co-', 'rD-', 'bD-']
Trotational = N.zeros((MolPara.nt, alt_ref.size))
for xx in range(Nt):
    _up = int(MolPara.tran_tag[xx][1])
    _low = int(MolPara.tran_tag[xx][2])
    Nup = N.array(new_pop)[:, 0, _up, 0]
    Nlo = N.array(new_pop)[:, 0, _low, 0]
    Gup = MolPara.weighti[_up]
    Glo = MolPara.weighti[_low]
    Ehv = MolPara.freq_array[xx]*1.e9*C.h
    Trot = -Ehv/(C.k*N.log(Nup/Nlo*Glo/Gup))
    Trotational[xx] = Trot
    P.plot(Trot, alt_ref, pcdot[xx],
           label=str(round(MolPara.freq_array[xx],3)))
P.plot(temp, alt_ref, 'k-', label='T_k')




SaveName = '_SSL10Jbar0_test_abs'
N.savetxt('Excitation_Temp'+SaveName+'.txt',N.vstack((alt_ref,Trotational)).T,
          delimiter='\t', header='Alt [km] \t Trot [K] at '
          +str(N.around(MolPara.freq_array,3)))
N.save('7-levels'+SaveName+'.npy',N.array(Ite_pop)[:,:,:,0])  #(al,ite,tr)

P.plot(Temp, Alt_ref, 'k-', label='T_k')
#plt.title("Test", fontname="serif")
P.legend(loc=0)
P.grid(linestyle=":")
P.xlim(0,150)
P.xlim(0,30)
P.xlim(20,150)
P.ylim(0, 450)
P.ylabel('Altitdue [km]', fontsize=24,weight='bold')
P.xlabel('Excitation Temperature [K]', fontsize=24,weight='bold')
P.tick_params(axis='both',labelsize=14,size=5)
N.savetxt('GSIte99_7-levels_surface-const.txt',N.vstack((Alt_ref,Trot)).T,
          delimiter='\t', header='Alt [km] \t Trot [K]')



