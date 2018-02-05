# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 17:25:02 2017

@author: yamada
"""
import numpy as np
from scipy.special import wofz
from scipy.constants import c, k, R
# =============================================================================
# h =6.62607004*10**-34  # Js or mmkg/s
# k = 1.38064852*10**-23  # mmkg/ssK  = J/K
# c = 299792458.  # m/s
#ac = 6.02214086e23  # Avogadro constant/M
from scipy.constants import Avogadro as ac
#6.022140857e+23
ac = 6.0220450e26  # Avogadro constant/M
# debye = 3.33564e-30  # 1D = 3.34e-30 [Cm]
# R = 8.3144598  # J/K/mol universal gas constant
# =============================================================================


def DLV(Type, gct, *,Freq=0, gcp=1, gcv=1, Para=1, HWHM=False):
    if Type == 'D':  # dopplar
        u"""#doppler width
        #Para[transient Freq[Hz], relative molecular mass[g/mol]]"""
        # step1 = Para[0]/c*(2.*R*gct/(Para[1]*1.e-3))**0.5
        # outy = np.exp(-(Freq-Para[0])**2/step1**2) / (step1*(np.pi**0.5))
        step1 = Para[0]/c*(2.*R*gct*np.log(2.)/(Para[1]*1.e-3))**0.5  # HWHM
        outy = np.exp(-np.log(2.)*(Freq-Para[0])**2/step1**2) *\
                     (np.log(2.)/np.pi)**0.5/step1
#        GD = np.sqrt(2*k*ac/18.0153*gct)/c*Para[0]
#        step1 = GD
#        outy = wofz((Freq-Para[0])/GD).real / np.sqrt(np.pi) / GD
#        print(outy.shape)
    elif Type == 'L':  # Lorenz
        """#Collisional width
        Temperature [K]
        pressure [atm]: 1atm = 101325Pa
        Number density of a specie [m-3]
        Para[transient Freq,gamm air, gamma self ,air n]
        gamma [cm-1 atm-1]"""
        vmrs = gcv/(gcp/k/gct)  # gamma
        step1 = (Para[1]*gcp/101325.*(1.-vmrs)+Para[2]*gcp/101325.*(vmrs)) *\
                (296./gct)**Para[3]*100*c  # cm-1*100*c ->s-1
        # step1 = (Para[1]*gcp/101325.*(1.-vmrs))*(296./gct)**Para[3]*100*c
        outy = step1/(np.pi*((Freq-Para[0])**2 + step1**2))
    elif Type == 'V':  # Voigt
        """#Para[frequency[Jup]*1.e9,
        gamma_air[Hn],gamma_self[Hn],n_air[Hn],M]#"""
        ad = Para[0]/c*(2.*R*gct*np.log(2.)/(Para[4]*1.e-3))**0.5  # vd
        # fd = np.exp(-np.log(2.)*(Freq-Para[0])**2/ad**2) *\
        #            (np.log(2.)/np.pi)**0.5/ad  # #fd
        vmrs = gcv/(gcp/k/gct)  # gamma
        al = (Para[1]*gcp/101325.*(1.-vmrs)) *\
             (296./gct)**Para[3]*100*c  # alphaL
        # fl = ad/(np.pi*((Freq-Para[0])**2 + ad**2))  #
        # outy = np.cumsum(fv1,axis=1)
        # wofz_y = al/ad  # see (2.46)
        # wofz_x = (Freq-Para[0])/ad  # see (2.46)
        sigma = ad/(2.*np.log(2))**0.5
        gamma = al
        for iii in range(al.size):
            # z = wofz_y[iii] - ( wofz_x[:,iii]*1j )
            # Z = np.array(wofz(z),dtype = complex)/ad[iii]/np.pi**0.5
            # wofz:exp(z**2)*ergc(z)#ergc:error_function
            if iii == 0:
                outy = np.real(wofz((Freq.reshape(Freq.size)-Para[0] +
                                     1j*gamma[iii])/sigma[iii]/2**0.5)) /\
                                     sigma[iii]/(2*np.pi)**0.5
            else:
                outy = np.vstack((outy,
                                  np.real(wofz((Freq.reshape(Freq.size) -
                                                Para[0] + 1j*gamma[iii]) /
                                               sigma[iii]/2**0.5)) /
                                 sigma[iii]/(2*np.pi)**0.5))
        outy = outy.T
        # outy = np.real(Z)
    else:
        print('Do you wanna calculate other shape function?')
    if HWHM is True:
        return step1, outy
    elif HWHM is False:
        return outy


def Linewidth(Type, gct, Para):
    """#doppler width
    #Para[transient Freq[Hz], relative molecular mass[g/mol]]"""
    step1 = Para[0]/c*(2.*R*gct/(Para[1]*1.e-3))**0.5
    step1 = Para[0]/c*(2.*R*gct*np.log(2.)/(Para[1]*1.e-3))**0.5  # HWHM
    return int(step1.max())



def DopplerWind(Temp, FreqGrid, Para, wind_v, shift_direction=False):
    u"""#doppler width
    #Para[transient Freq[Hz], relative molecular mass[g/mol]]"""
    # step1 = Para[0]/c*(2.*R*gct/(Para[1]*1.e-3))**0.5
    # outy = np.exp(-(Freq-Para[0])**2/step1**2) / (step1*(np.pi**0.5))
    #wind_v = speed[:,10] 
    #Temp=temp[10]
    #FreqGrid = Fre_range_i[0]
    wind = wind_v.reshape(wind_v.size, 1)
    FreqGrid = FreqGrid.reshape(1, FreqGrid.size)
    deltav = Para[0]*wind/c
    if shift_direction is 'Red' or 'red' or 'RED':
        D_effect = (deltav)
    elif shift_direction is 'Blue' or 'blue' or 'BLUE':
        D_effect = (-deltav)
    else:print('Set Direction, red shift or blue shift')
#    step1 = Para[0]/c*(2.*R*Temp*np.log(2.)/(Para[1]*1.e-3))**0.5  # HWHM
#    outy = np.exp(-np.log(2.)*(FreqGrid-Para[0])**2/step1**2) *\
#                 (np.log(2.)/np.pi)**0.5/step1
#    outy_d = np.exp(-np.log(2.)*(FreqGrid+D_effect-Para[0])**2/step1**2) *\
#                   (np.log(2.)/np.pi)**0.5/step1
    GD = np.sqrt(2*k*ac/Para[1]*Temp)/c*Para[0]
    step1 = GD
    outy_d = wofz((FreqGrid+D_effect-Para[0])/GD).real / np.sqrt(np.pi) / GD
    #plot(FreqGrid, outy)
    #plot(FreqGrid, outy_d[:,0])
    return outy_d





