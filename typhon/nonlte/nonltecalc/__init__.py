# -*- coding: utf-8 -*-
import logging

import numpy as np
import pylab as plt
from scipy.constants import c, k, h
from ..mathmatics import trapz_inte_edge
# from . import sub
# from ..spectra.lineshape import DLV
from ..spectra.source_function import Bv_T, PopuSource_AB
from ..spectra.abscoeff import basic
from ..rtc import SOSC, FOSC
from ..spectra.lineshape import DopplerWind
from typhon.physics.em import planck, rayleighjeans


logger = logging.getLogger(__name__)


def calcu_grid(radius, alt_ref, angle=False, speed=None):
    """ Calculation grid for given altitude and radius

    Parameters:
        radius: Radius of planet/moon/object [meters]
        alt_ref: Reference altitude grid (1D grid) [kilometers]
        angle: Will return the angle grid of the atmospheric layers
        speed: Using wind-velocities (1D grid) [meters/second]

    Returns:
        Grid point weights, central altitudes, tangent point, [angle at grid points]
    """
    mu_delta = radius/100. # *1000.  # m
    mu_point = np.r_[np.arange(0, radius, mu_delta),
                     alt_ref*1e3 + radius]  # [m]
    mu_tangent = (mu_point - radius) * 1e-3

    PSC = ((alt_ref*1.e3 + radius)**2 -
           mu_point.reshape(mu_point.size, 1)**2)**0.5
    PSC2 = np.zeros((mu_point.size, alt_ref.size-1))  # alt_center

    for xx in range(1, alt_ref.size):  # alt_center
        PSC2[:, -xx] = PSC[:, -xx] - PSC[:, -xx-1]
        
    PSC_sin = mu_point.reshape(mu_point.size, 1)/(alt_ref*1.e3 + radius)
    deg = np.arcsin(PSC_sin)
    PSC_sin[deg != deg] = 0
    deg[deg != deg] = np.pi / 2.
    mu_weight = np.zeros((mu_point.size, alt_ref.size))
    
    for i in range(alt_ref.size):
        mu_weight[:, i] = trapz_inte_edge(PSC_sin[:, i], deg[:, i])
    
    if angle is True:
        return mu_weight, PSC2, mu_tangent, PSC_sin
    elif speed is not None:
        velocity = speed * np.cos(deg)
        velocity[deg == np.pi / 2.] = 0
        return mu_weight, PSC2, mu_tangent, velocity
    else:
        #tete
        return mu_weight, PSC2, mu_tangent


class MolecularConsts:
    """
    From a specific file, read followings;
    Transition frequency
    number of energy levels
    energy levels
    degeneracies
    Einstein's A coeff
    transition tag (upper and lower level for each transition)
    """
    def __init__(self, molecules_state_consts):
        self.filename = molecules_state_consts
        fn = open(molecules_state_consts + 'oH2O16.lev_7levels', 'rb')
        self.fr = fn.readlines()
        fn.close()
        self.ni = int(self.fr[3])  # NUMBER OF ENERGY LEVELS
        self.e_cm1 = np.array([float(self.fr[xx].split()[1])
                               for xx in range(5, 5 + self.ni)])
        self.weighti = np.array([float(self.fr[xx].split()[2])
                                 for xx in range(5, 5 + self.ni)])

        Ai = np.zeros((self.ni, self.ni))
        Freqi = np.zeros((self.ni, self.ni))
        E_Ki = np.zeros((self.ni, self.ni))
        B_place = np.ones((self.ni, self.ni))*-1.
        Tran_tag = []
        Ai_tag = []
        Nt = int(self.fr[3+self.ni+3])  # int(9)
        for xx in range(8+self.ni, 8+self.ni+Nt):  # Nt Number of transition ,24,twolevel ->16):
            """up: temp[1]-1, low: temp[2]-1"""
            temp = np.array(self.fr[xx].split()).astype(np.float64)
            Tran_tag.append(temp[:3]-1)  # for 1... -> 0....
            Ai[int(temp[1]-1), int(temp[2])-1] = temp[3]
            Ai_tag.append(temp[3])
            # Freqi[int(temp[1]-1), int(temp[2])-1] = temp[4]
            _deltaEcm1 = (self.e_cm1[int(temp[1])-1] -
                          self.e_cm1[int(temp[2])-1])
            Freqi[int(temp[1]-1), int(temp[2])-1] = (_deltaEcm1*c *
                                                     100.*1.e-9)  # GHz
            E_Ki[int(temp[1]-1), int(temp[2])-1] = temp[5]
            B_place[int(temp[1]-1), int(temp[2])-1] = int(temp[0]-1)  # induced
            B_place[int(temp[2])-1, int(temp[1]-1)] = int(temp[0]-1)  # abs.ed.
        Tran_tag = np.array(Tran_tag)*1
        Ai_tag = np.array(Ai_tag)*1.
        Nt = len(Tran_tag)  # int(9)
        self.ai = Ai
        self.freqi = Freqi
        self.tran_tag = Tran_tag
        self.b_place = B_place
        self.ai_tag = Ai_tag
        self.nt = Nt
        Freq_array = np.array([])
        for xx in range(Nt):
            _up = int(Tran_tag[xx][1])
            _low = int(Tran_tag[xx][2])
            Freq_array = np.hstack((Freq_array, Freqi[_up, _low]))
        self.freq_array = Freq_array
        Bul = self.ai*c**2/(2*h*(self.freqi*1.e9)**3)
        Blu = Bul*self.weighti.reshape((self.ni, 1))/self.weighti
        Bul[Bul != Bul] = 0  # [i,j]: i->j
        Blu[Blu != Blu] = 0  # [i,j]: j->i
        self.blu = Blu
        self.bul = Bul

    def ratematrix(self):
        """
        Make a matrix of radiation rate
            lower
          |     |  |n1|
        u |     |  |n2|
          |     |  |n3|
        *diagonal is sum of loss rate
        """
        Bul = self.ai*c**2/(2*h*(self.freqi*1.e9)**3)
        Blu = Bul*self.weighti.reshape((self.ni, 1))/self.weighti
        Bul[Bul != Bul] = 0  # [i,j]: i->j
        Blu[Blu != Blu] = 0  # [i,j]: j->i
        # RaRaB_induced = -Bul + Bul.T
        # RaRaB_absorption = Blu-Blu.T
        RaRaB_induced = Bul.T  # [i,j] i<-j
        RaRaB_absorption = Blu  # [i,j] i<-j
        RaRaA = self.ai.T
        # Diagonal
        RaRaAd = np.zeros((self.ni, self.ni))  #
        RaRaBd = np.zeros((self.ni, self.ni))  #
        for xx in range(self.ni):
            cond = self.tran_tag[:, 1] == xx
            RaRaAd[xx, xx] += self.ai_tag[cond].sum()*-1.
            RaRaBd[xx, xx] += (Bul[xx]+Blu.T[xx]).sum()*-1.
        return RaRaA, RaRaAd, RaRaBd, RaRaB_induced, RaRaB_absorption

    def collision(self, Mole, Temp, ETS=False):
        """
        make self collisional rate matrix
        
        input parameter;
        number of Molecules [molecules/m3] collisional parameter
        Temperature [K]
        rate constant is read from filename
        
        If ETS is True, 
        Population of escape to space approximation is plotted
        
        """
        RateConst = np.loadtxt(self.filename+'H2O'+'_col.txt')*1.e-6
        colt = np.array([100,  200,  300,  400,  500,  600,  700,  800])*1.
        coltt = np.around(np.arange(0., 400., 0.01), 2)  # We calculate 0.1 K
        col_tk = []
        for xx in range(self.nt):
            col_tk.append(np.interp(coltt, colt, RateConst[xx]))
        col_tk = np.array(col_tk)
        "set col rate for each atmospheric Temp."
        col_ul, col_lu = [], []
        for xx in range(self.nt):
            _up = int(self.tran_tag[xx][1])
            _low = int(self.tran_tag[xx][2])
            col_ul.append(np.array([col_tk[xx][np.argwhere(coltt ==
                                               round(xxx, 2))][0][0]
                                   for xxx in np.around(Temp, 2)]))
            col_lu.append((col_ul[xx]*self.weighti[_up]/self.weighti[_low]) *
                          np.exp(-h*self.freqi[_up, _low] *
                                 1.e9/k/Temp))
        rate_ul, rate_lu = Mole*col_ul, Mole*col_lu  # molecules/s
        self.Cul = rate_ul
        J_ETS, J_LTE = [], []
        for xx in range(self.nt):
            _up = int(self.tran_tag[xx][1])
            _low = int(self.tran_tag[xx][2])
            J_ETS.append(2.*h*(self.freqi[_up, _low]*1.e9)**3 /
                         c**2 /
                         (((rate_lu[xx])/(rate_ul[xx]) *
                           (rate_ul[xx]/self.ai[_up, _low]) /
                           (1.+rate_ul[xx]/self.ai[_up, _low]))**-1 *
                          self.weighti[_up]/self.weighti[_low] - 1.)
                         )  # See (5,2)(3.35)
            J_LTE.append(2.*h*(self.freqi[_up, _low]*1.e9)**3 /
                         c**2 /
                         ((rate_ul[xx])/(rate_lu[xx]) *
                          self.weighti[_up]/self.weighti[_low] - 1.)
                         )  # See (3.35)
            # P.plot(h*(Freqi[_up, _low]*1.e9/(k*np.log(2.*h*
            #             (Freqi[_up, _low]*1.e9)**3/c**2/J_LTE[xx]+1))))

            # P.plot(h*(Freqi[_up, _low]*1.e9/(k*np.log(2.*h*
            #            (Freqi[_up, _low]*1.e9)**3/c**2/J_ETS[xx] +1)))
            #       , label = str(Freqi[_up, _low]))
        CoRa_block = []  # Collisional Rate matrix
        for hoge in range(Temp.size):
            Cul, Clu = rate_ul.T[hoge], rate_lu.T[hoge]
            CoRa = np.zeros((self.ni, self.ni))
            for xx in range(self.nt):
                _up = int(self.tran_tag[xx][1])
                _low = int(self.tran_tag[xx][2])
                # CoRa[_up, _low] += -Cul[xx]+Clu[xx]
                # CoRa[_low, _up] += Cul[xx]-Clu[xx]
                CoRa[_up, _low] += Clu[xx]
                CoRa[_low, _up] += Cul[xx]
            # Diagonal
            for xx in range(self.ni):
                cond = self.tran_tag[:, 1] == xx
                CoRa[xx, xx] -= Cul[cond].sum()
                cond = self.tran_tag[:, 2] == xx
                CoRa[xx, xx] -= Clu[cond].sum()
            CoRa_block.append(CoRa*1.)
        CoRa_block = np.array(CoRa_block)*1.
        if ETS is True:
            J_ETS = np.array(J_ETS)
            J_ETS_plot = h*(self.freq_array.reshape(self.nt, 1)*1.e9 /
                            (k*np.log(2.*h *
                                      (self.freq_array.reshape(self.nt, 1) *
                                       1.e9)**3/c**2/J_ETS + 1.)))
            plt.figure()
            [plt.plot(J_ETS_plot[xx],
                      label=str(round(self.freq_array[xx],
                                      3))) for xx in range(self.nt)]
            plt.plot(Temp, 'k', label='T$_k$')
            plt.legend(loc=0)
            plt.grid()
        # ======================================================================
            return CoRa_block, J_ETS_plot

        return CoRa_block

    def population(self, Mole, Temp):
        """Calculate population for each level at given temperature"""
        RoTemp = np.reshape(Temp*1., (Temp.size, 1))
        Qr = self.weighti*np.exp(-(self.e_cm1*100*c*h)/(k*RoTemp))
        # RoPr = Qr/Ntotal  # This is for all transitions
        RoPr = Qr/(Qr.sum(axis=1).reshape(RoTemp.size, 1))  # only given trans.
        linet = []
        for xx in range(self.nt):
            gdu, gdl = self.weighti[self.tran_tag[xx][1:].astype(int)]
            _up = int(self.tran_tag[xx][1])
            _low = int(self.tran_tag[xx][2])
            Aei = self.ai[_up, _low]
            line_const = (c*10**2)**2*Aei*(gdu/gdl)*1.e-6*1.e14 /\
                         (8.*np.pi*(self.freq_array[xx]*1.e9)**2)
            # Hz->MHz,cm^2 ->nm^2
            # W = C.h*C.c*E_cm1[_low]*100.  # energy level above ground state
            "This is the function of calculating H2O intensity"
            line = (1.-np.exp(-h*(self.freq_array[xx]*1.e9) /
                              k/RoTemp))*line_const
            linet.append(line[:, 0]*RoPr[:, _low])  # line intensity non-LTE
        Ni_LTE = Mole.reshape((Mole.size, 1))*RoPr  # *0.75  # orth para ratio
        Ite_pop = [[Ni_LTE[i].reshape((self.ni, 1))] for i in range(Mole.size)]
        return Ite_pop

#import numba


"""Population Calculation
"""

""" Takayoshi Yamada:  Work in progress below.  Change sparingly.
"""
def Calc(Ite_pop, Abs_ite,
         PSC2, Mu_tangent, mu_weight,
         Alt_ref, Temp,
         Fre_range_i, Freq_array, F_vl_i,
         B_place,
         Nt, Ni, Aul, Bul, Blu,
         RaRaB_absorption, RaRaB_induced, RaRaA, RaRaAd, CoRa_block,
         Tran_tag,
         wind_v=False,
         update_population=True,
         out_put_spectra=False,
         continuum_surface_temperature_unit='Planck',
         back_ground_radiation='CMB',
         iteration='MUGA'):
#    if iteration == 'MUGA1SC':
#        from ..rtc import SOSCdamy as SOSC
#        print('Hey using FOSC in SOSC')
#        iteration = 'MUGA'
    ji_in_all = np.zeros((Nt,
                          Alt_ref.size,
                          Mu_tangent.size,
                          Fre_range_i[0].size))
    ji_out_all = ji_in_all*0.
    lambda_approx_in = np.zeros((Nt,
                                 Alt_ref.size,
                                 Mu_tangent.size,
                                 Fre_range_i[0].size))
    lambda_approx_out = np.zeros((Nt,
                                  Alt_ref.size,
                                  Mu_tangent.size,
                                  Fre_range_i[0].size))
    # for IteNum in range(0, 100):
    max_true_error = np.ones(Alt_ref.size)*0.
    new_pop = np.array(Ite_pop)*0.+0.
    for ii in range(Alt_ref.size)[::-1]:  # spatial points
        # print(ii)
        for xx in range(Nt):  # transitions
            # Line_Shape = F_vl_i[xx]
            Para=[Freq_array[xx]*1.e9, 18.0153]
            Abs_coeff = Abs_ite[xx]
            Fre_range = Fre_range_i[xx]
            up_tag = int(Tran_tag[xx][1])
            low_tag = int(Tran_tag[xx][2])
            # Up_pop = Ite_pop[ii, 0, up_tag, 0]
            # Low_pop = Ite_pop[ii, 0, low_tag, 0]
            if ii == Alt_ref.size-1:  # upper Boundary(u>0) incoming
                if back_ground_radiation == 'CMB':
                    B_v_cosmic = planck(Fre_range, 2.725)
                else:
                    B_v_cosmic = 0
                ji_in_all[xx, ii, :, :] = B_v_cosmic  # Ladi comparison
            elif ii == Alt_ref.size-1-1:
                tdu2 = Abs_coeff[ii+1] * \
                DopplerWind(Temp[ii+1], Fre_range_i[xx], Para, wind_v[:,ii+1], shift_direction='Red')#F_vl_i[xx][ii+1]
                tdu1 = Abs_coeff[ii] * \
                DopplerWind(Temp[ii], Fre_range_i[xx], Para, wind_v[:,ii], shift_direction='Red')#F_vl_i[xx][ii]
#                tdu = (0.5*np.abs(tdu1+tdu2).reshape((1, Fre_range.size)) *
#                       PSC2[:, ii].reshape((Mu_tangent.size, 1)))
                tdu = (0.5*np.abs(tdu1+tdu2) *
                       PSC2[:, ii].reshape((Mu_tangent.size, 1)))
                Sd = PopuSource_AB(Ite_pop[ii, 0, low_tag, 0],
                                   Ite_pop[ii, 0, up_tag, 0],
                                   Aul[up_tag, low_tag],
                                   Bul[up_tag, low_tag],
                                   Blu[up_tag, low_tag])  # * F_vl_i[xx][ii]
                Sd1 = PopuSource_AB(Ite_pop[ii+1, 0, low_tag, 0],
                                    Ite_pop[ii+1, 0, up_tag, 0],
                                    Aul[up_tag, low_tag],
                                    Bul[up_tag, low_tag],
                                    Blu[up_tag, low_tag])  # * F_vl_i[xx][ii]
                Idu = ji_in_all[xx, ii+1, :, :]*1.  # !!!CHECK!!!
                ji_in_all[xx, ii, :, :],lambda_approx_in[xx, ii, :, :]\
                         = FOSC(tdu, Sd1, Sd, Idu)
                ji_in_all[xx, ii, ((Mu_tangent > Alt_ref[ii])), :] = 0
    # test = trapz_inte_edge(PSC_sin[:, -1], deg[:, -1])
    # PSC_cos = np.arccos(PSC/(Alt_ref*1.e3+Radius))
    # PSC_cos1 = PSC/(Alt_ref*1.e3+Radius)
    # PSC_sin = (1-PSC_cos**2)**0.5
    # PSC_cos = np.cos( np.arccos(PSC_cos1)*1.)  # degree (pi)
                lambda_approx_in[xx, ii, ((Mu_tangent > Alt_ref[ii])), :] = 0
                # mu; 0->i,
            elif ii == 0:  # SOSC is not available (FOSC elif ii>=0)
                tdu2 = Abs_coeff[ii+1] * F_vl_i[xx][ii+1]
                tdu1 = Abs_coeff[ii] * F_vl_i[xx][ii]
                tdu2 = Abs_coeff[ii+1] * \
                DopplerWind(Temp[ii+1], Fre_range_i[xx], Para, wind_v[:,ii+1], shift_direction='Red')#F_vl_i[xx][ii+1]
                tdu1 = Abs_coeff[ii] * \
                DopplerWind(Temp[ii], Fre_range_i[xx], Para, wind_v[:,ii], shift_direction='Red')#F_vl_i[xx][ii]
#                tdu = (0.5*np.abs(tdu1+tdu2).reshape((1, Fre_range.size)) *
#                       PSC2[:, ii].reshape((Mu_tangent.size, 1)))
                tdu = (0.5*np.abs(tdu1+tdu2) *
                       PSC2[:, ii].reshape((Mu_tangent.size, 1)))
                Sd = PopuSource_AB(Ite_pop[ii, 0, low_tag, 0],
                                   Ite_pop[ii, 0, up_tag, 0],
                                   Aul[up_tag, low_tag],
                                   Bul[up_tag, low_tag],
                                   Blu[up_tag, low_tag])  # * F_vl_i[xx][ii]
                Sd1 = PopuSource_AB(Ite_pop[ii+1, 0, low_tag, 0],
                                    Ite_pop[ii+1, 0, up_tag, 0],
                                    Aul[up_tag, low_tag],
                                    Bul[up_tag, low_tag],
                                    Blu[up_tag, low_tag])  # * F_vl_i[xx][ii]
                Sd3 = PopuSource_AB(Ite_pop[ii-1, 0, low_tag, 0],
                                    Ite_pop[ii-1, 0, up_tag, 0],
                                    Aul[up_tag, low_tag],
                                    Bul[up_tag, low_tag],
                                    Blu[up_tag, low_tag])  # * F_vl_i[xx][ii]
#                tdu3 = Abs_coeff[ii-1] * F_vl_i[xx][ii-1]
                tdu3 = Abs_coeff[ii-1] * \
                DopplerWind(Temp[ii-1], Fre_range_i[xx], Para, wind_v[:,ii-1], shift_direction='Red')#F_vl_i[xx][ii]
#                tdb = (0.5*np.abs(tdu1+tdu3).reshape((1, Fre_range.size)) *
#                       PSC2[:, ii-1].reshape((Mu_tangent.size, 1)))
                tdb = (0.5*np.abs(tdu1+tdu3) *
                       PSC2[:, ii-1].reshape((Mu_tangent.size, 1)))
                Idu = ji_in_all[xx, ii+1, :, :]*1.  # !!!CHECK!!!
                ji_in_all[xx, ii, :, :],lambda_approx_in[xx, ii, :, :]\
                = FOSC(tdu, Sd1, Sd, Idu)
                ji_in_all[xx, ii, ((Mu_tangent > Alt_ref[ii])), :] = 0
                lambda_approx_in[xx, ii, ((Mu_tangent > Alt_ref[ii])), :] = 0
                # mu; 0->i,
            else:  # SOSC
                tdu2 = Abs_coeff[ii+1] * F_vl_i[xx][ii+1]
                tdu1 = Abs_coeff[ii] * F_vl_i[xx][ii]
                tdu2 = Abs_coeff[ii+1] * \
                DopplerWind(Temp[ii+1], Fre_range_i[xx], Para, wind_v[:,ii+1], shift_direction='Red')#F_vl_i[xx][ii+1]
                tdu1 = Abs_coeff[ii] * \
                DopplerWind(Temp[ii], Fre_range_i[xx], Para, wind_v[:,ii], shift_direction='Red')#F_vl_i[xx][ii]
#                tdu = (0.5*np.abs(tdu1+tdu2).reshape((1, Fre_range.size)) *
#                       PSC2[:, ii].reshape((Mu_tangent.size, 1)))
                tdu = (0.5*np.abs(tdu1+tdu2) *
                       PSC2[:, ii].reshape((Mu_tangent.size, 1)))
                Sd = PopuSource_AB(Ite_pop[ii, 0, low_tag, 0],
                                   Ite_pop[ii, 0, up_tag, 0],
                                   Aul[up_tag, low_tag],
                                   Bul[up_tag, low_tag],
                                   Blu[up_tag, low_tag])  # * F_vl_i[xx][ii]
                Sd1 = PopuSource_AB(Ite_pop[ii+1, 0, low_tag, 0],
                                    Ite_pop[ii+1, 0, up_tag, 0],
                                    Aul[up_tag, low_tag],
                                    Bul[up_tag, low_tag],
                                    Blu[up_tag, low_tag])  # * F_vl_i[xx][ii]
                Sd3 = PopuSource_AB(Ite_pop[ii-1, 0, low_tag, 0],
                                    Ite_pop[ii-1, 0, up_tag, 0],
                                    Aul[up_tag, low_tag],
                                    Bul[up_tag, low_tag],
                                    Blu[up_tag, low_tag])  # * F_vl_i[xx][ii]
                tdu3 = Abs_coeff[ii-1] * F_vl_i[xx][ii-1]
                tdu3 = Abs_coeff[ii-1] * \
                DopplerWind(Temp[ii-1], Fre_range_i[xx], Para, wind_v[:,ii-1], shift_direction='Red')#F_vl_i[xx][ii]
#                tdb = (0.5*np.abs(tdu1+tdu3).reshape((1, Fre_range.size)) *
#                       PSC2[:, ii-1].reshape((Mu_tangent.size, 1)))
                tdb = (0.5*np.abs(tdu1+tdu3) *
                       PSC2[:, ii-1].reshape((Mu_tangent.size, 1)))
                Idu = ji_in_all[xx, ii+1, :, :]*1.  # !!!CHECK!!!
                ji_in_all[xx, ii, :, :], lambda_approx_in[xx, ii, :, :]\
                         = SOSC(tdu, tdb,
                                Sd1, Sd, Sd3,
                                Idu)
                ji_in_all[xx, ii, (Mu_tangent == Alt_ref[ii]), :],\
                lambda_approx_in[xx, ii, (Mu_tangent == Alt_ref[ii]), :]\
                          = FOSC(tdu[Mu_tangent == Alt_ref[ii]],
                                    Sd1,
                                    Sd,
                                    Idu[Mu_tangent == Alt_ref[ii]])
                ji_in_all[xx, ii, ((Mu_tangent > Alt_ref[ii])), :] = 0
                lambda_approx_in[xx, ii, ((Mu_tangent > Alt_ref[ii])), :] = 0
                # mu; 0->i,
    for i in range(Alt_ref.size):  # spatial points
        #  print(i)
        B_int = B_place*0.  # intensity for each transitoin
        B_int_lamda = B_place*0.  # intensity for each transitoin for lambda ope
        A_int_lamda = B_place*0.  # approximated lambda operate
        SF = np.ones(Nt)
        for xx in range(Nt):  # transitions
            Para=[Freq_array[xx]*1.e9, 18.0153]
            Fre_range = Fre_range_i[xx]
            # Line_Shape = F_vl_i[xx]
            Abs_coeff = Abs_ite[xx]
            Fre_range = Fre_range_i[xx]
            up_tag = int(Tran_tag[xx][1])
            low_tag = int(Tran_tag[xx][2])
            # Up_pop = Ite_pop[i, 0, up_tag, 0]
            # Low_pop = Ite_pop[i, 0, low_tag, 0]
            if i == 0:  # Lower Boundary(u>0) outgoing
                if continuum_surface_temperature_unit == 'Planck':
#                    ji_out_all[xx, i, :, :] = planck(Fre_range, Temp[i])
                    ji_out_all[xx, i, :, :] = Bv_T(Fre_range, Temp[i])
                elif continuum_surface_temperature_unit == 'RJ':
                    ji_out_all[xx, i, :, :] = rayleighjeans(Fre_range, Temp[i])
                elif continuum_surface_temperature_unit == 'RJ_obs':
                    _t_phys = h * Freq_array[xx] * 1.e9 / \
                              (k * np.log(h * Freq_array[xx] * 1.e9
                                          /k/Temp[i] + 1))
                    ji_out_all[xx, i, :, :] = planck(Fre_range, _t_phys)
                else:
                    ji_out_all[xx, i, :, :] = Bv_T(Fre_range, Temp[i])
                ji_out_all[:, i, (Mu_tangent > Alt_ref[i]), :] = 0
            elif i < Alt_ref.size-1:  # SOSC
                tl_1 = Abs_coeff[i] * F_vl_i[xx][i]
                # GS-iteration
                if iteration == 'MUGA':
                    tl_2 = basic(new_pop[i-1, 0, low_tag, 0],
                                 new_pop[i-1, 0, up_tag, 0],
                                 Blu[up_tag, low_tag],
                                 Bul[up_tag, low_tag],
                                 Freq_array[xx]*1.e9)*F_vl_i[xx][i-1]
                elif iteration == 'LI' or iteration == 'MALI':
                    tl_2 = basic(Ite_pop[i-1, 0, low_tag, 0],
                                 Ite_pop[i-1, 0, up_tag, 0],
                                 Blu[up_tag, low_tag],
                                 Bul[up_tag, low_tag],
                                 Freq_array[xx]*1.e9)*F_vl_i[xx][i-1]
                # tdl2 = calc_abscoeff(xx, IteNum+1, i-1) * F_vl_i[xx][i-1]
                tl1 = (0.5*np.abs(tl_1+tl_2).reshape((1, Fre_range.size)) *
                       PSC2[:, i-1].reshape((Mu_tangent.size, 1)))  # (Mu, Fre)
                tl_3 = Abs_coeff[i+1] * F_vl_i[xx][i+1]
                tl3 = (0.5*np.abs(tl_1+tl_3).reshape((1, Fre_range.size)) *
                       PSC2[:, i].reshape((Mu_tangent.size, 1)))  # (Mu, Fre)
                """===Doppler=========="""
                tl_1 = Abs_coeff[i] * \
                DopplerWind(Temp[i], Fre_range_i[xx], Para, wind_v[:,i], shift_direction='Blue')
                if iteration == 'MUGA':
                    tl_2 = basic(new_pop[i-1, 0, low_tag, 0],
                                 new_pop[i-1, 0, up_tag, 0],
                                 Blu[up_tag, low_tag],
                                 Bul[up_tag, low_tag],
                                 Freq_array[xx]*1.e9)*\
                           DopplerWind(Temp[i-1], Fre_range_i[xx], Para, wind_v[:,i-1], 
                                       shift_direction='Blue')  # F_vl_i[xx][i-1]
                elif iteration == 'LI' or iteration == 'MALI':
                    tl_2 = basic(Ite_pop[i-1, 0, low_tag, 0],
                                 Ite_pop[i-1, 0, up_tag, 0],
                                 Blu[up_tag, low_tag],
                                 Bul[up_tag, low_tag],
                                 Freq_array[xx]*1.e9)*\
                           DopplerWind(Temp[i-1], Fre_range_i[xx], Para, wind_v[:,i-1], 
                                       shift_direction='Blue')  # F_vl_i[xx][i-1]
                tl_3 = Abs_coeff[i+1] * \
                DopplerWind(Temp[i+1], Fre_range_i[xx], Para, wind_v[:,i+1], shift_direction='Red')
                tl1 = (0.5*np.abs(tl_1+tl_2) *
                       PSC2[:, i-1].reshape((Mu_tangent.size, 1)))  # (Mu, Fre)
                tl3 = (0.5*np.abs(tl_1+tl_3) *
                       PSC2[:, i].reshape((Mu_tangent.size, 1)))  # (Mu, Fre)
                """======================"""
                Sl2 = PopuSource_AB(Ite_pop[i, 0, low_tag, 0],
                                    Ite_pop[i, 0, up_tag, 0],
                                    Aul[up_tag, low_tag],
                                    Bul[up_tag, low_tag],
                                    Blu[up_tag, low_tag])  # * F_vl_i[xx][i]
                SF[xx] = Sl2
                if iteration == 'MUGA':
                    Sl1 = PopuSource_AB(new_pop[i-1, 0, low_tag, 0],
                                        new_pop[i-1, 0, up_tag, 0],
                                        Aul[up_tag, low_tag],
                                        Bul[up_tag, low_tag],
                                        Blu[up_tag, low_tag])  # * F_vl_i[xx][i]
                elif iteration == 'LI' or iteration == 'MALI':
                    Sl1 = PopuSource_AB(Ite_pop[i-1, 0, low_tag, 0],
                                        Ite_pop[i-1, 0, up_tag, 0],
                                        Aul[up_tag, low_tag],
                                        Bul[up_tag, low_tag],
                                        Blu[up_tag, low_tag])  # * F_vl_i[xx][i]
                Sl3 = PopuSource_AB(Ite_pop[i+1, 0, low_tag, 0],
                                    Ite_pop[i+1, 0, up_tag, 0],
                                    Aul[up_tag, low_tag],
                                    Bul[up_tag, low_tag],
                                    Blu[up_tag, low_tag])
                Il = ji_out_all[xx, i-1, :, :]*1.  #
                # ji_out_all[xx, i, :, :] = FOSC(tl1, Sl1, Sl2, Il)  # (12.113)
                ji_out_all[xx, i, :, :],lambda_approx_out[xx, i, :, :]\
                          = SOSC(tl1, tl3,
                                 Sl1, Sl2, Sl3,
                                 Il)
                ji_out_all[xx, i, Mu_tangent > Alt_ref[i], :]=0
                lambda_approx_out[xx, i, Mu_tangent > Alt_ref[i], :] = 0
                ji_out_all[xx, i, Mu_tangent == Alt_ref[i], :]\
                = ji_in_all[xx, i, Mu_tangent == Alt_ref[i], :]
                lambda_approx_out[xx, i, Mu_tangent == Alt_ref[i], :]\
                = lambda_approx_in[xx, i, Mu_tangent == Alt_ref[i], :]
                I3 = ji_in_all[xx, i+1, :, :]*1.  # !!!CHECK!!!
                ji_in_all[xx, i, :, :],lambda_approx_out[xx, i, :, :]\
                         = SOSC(tl3, tl1,
                                Sl3, Sl2, Sl1,
                                I3)
                ji_in_all[xx, i, (Mu_tangent == Alt_ref[i]), :],\
                lambda_approx_out[xx, i, (Mu_tangent == Alt_ref[i]), :]\
                             = FOSC(tl3[Mu_tangent == Alt_ref[i]],
                                    Sl3,
                                    Sl2,
                                    I3[Mu_tangent == Alt_ref[i]])
                ji_in_all[xx, i, ((Mu_tangent > Alt_ref[i])), :] = 0
                lambda_approx_out[xx, i, ((Mu_tangent > Alt_ref[i])), :] = 0
#                ji_in_all[xx, i, ((Mu_tangent > Alt_ref[i-1])), :] = 0
                if np.argwhere(Mu_tangent == Alt_ref[i]).size!=1:
                    logger.error(' '.join((i, xx, "error!")))

            elif i == Alt_ref.size-1:  # FOSC
                tl_1 = Abs_coeff[i] * F_vl_i[xx][i]
                # GS-iteration
                if iteration == 'MUGA':
                    tl_2 = basic(new_pop[i-1, 0, low_tag, 0],
                                 new_pop[i-1, 0, up_tag, 0],
                                 Blu[up_tag, low_tag],
                                 Bul[up_tag, low_tag],
                                 Freq_array[xx]*1.e9)*F_vl_i[xx][i-1]
                elif iteration == 'MALI' or iteration == 'LI':
                    tl_2 = basic(new_pop[i-1, 0, low_tag, 0],
                                 new_pop[i-1, 0, up_tag, 0],
                                 Blu[up_tag, low_tag],
                                 Bul[up_tag, low_tag],
                                 Freq_array[xx]*1.e9)*F_vl_i[xx][i-1]
                else:
                    raise ValueError(f'Invalid iteration method {iteration}')
                # tdl2 = calc_abscoeff(xx, IteNum+1, i-1) * F_vl_i[xx][i-1]
                tl1 = (0.5*np.abs(tl_1+tl_2).reshape((1, Fre_range.size)) *
                       PSC2[:, i-1].reshape((Mu_tangent.size, 1)))  # (Mu, Fre)
                """===Doppler=========="""
                tl_1 = (Abs_coeff[i] *
                        DopplerWind(Temp[i], Fre_range_i[xx], Para,
                                    wind_v[:, i], shift_direction='Blue'))
                if iteration == 'MUGA':
                    tl_2 = basic(new_pop[i-1, 0, low_tag, 0],
                                 new_pop[i-1, 0, up_tag, 0],
                                 Blu[up_tag, low_tag],
                                 Bul[up_tag, low_tag],
                                 Freq_array[xx]*1.e9)*\
                           DopplerWind(Temp[i-1], Fre_range_i[xx], Para, wind_v[:,i-1], 
                                       shift_direction='Blue')  # F_vl_i[xx][i-1]
                elif iteration == 'LI' or iteration == 'MALI':
                    tl_2 = basic(Ite_pop[i-1, 0, low_tag, 0],
                                 Ite_pop[i-1, 0, up_tag, 0],
                                 Blu[up_tag, low_tag],
                                 Bul[up_tag, low_tag],
                                 Freq_array[xx]*1.e9)*\
                           DopplerWind(Temp[i-1], Fre_range_i[xx], Para, wind_v[:,i-1], 
                                       shift_direction='Blue')  # F_vl_i[xx][i-1]
                tl1 = (0.5*np.abs(tl_1+tl_2) *
                       PSC2[:, i-1].reshape((Mu_tangent.size, 1)))  # (Mu, Fre)
                """======================"""

                Sl = PopuSource_AB(Ite_pop[i, 0, low_tag, 0],
                                   Ite_pop[i, 0, up_tag, 0],
                                   Aul[up_tag, low_tag],
                                   Bul[up_tag, low_tag],
                                   Blu[up_tag, low_tag])  # * F_vl_i[xx][i]
                if iteration == 'MUGA':
                    Sl1 = PopuSource_AB(new_pop[i-1, 0, low_tag, 0],
                                        new_pop[i-1, 0, up_tag, 0],
                                        Aul[up_tag, low_tag],
                                        Bul[up_tag, low_tag],
                                        Blu[up_tag, low_tag])  # * F_vl_i[xx][i]
                elif iteration == 'LI' or iteration == 'MALI':
                    Sl1 = PopuSource_AB(Ite_pop[i-1, 0, low_tag, 0],
                                        Ite_pop[i-1, 0, up_tag, 0],
                                        Aul[up_tag, low_tag],
                                        Bul[up_tag, low_tag],
                                        Blu[up_tag, low_tag])  # * F_vl_i[xx][i]
                Il = ji_out_all[xx, i-1, :, :]*1.  #
                ji_out_all[xx, i, :, :],lambda_approx_out[xx, i, :, :]\
                          = FOSC(tl1, Sl1, Sl, Il)  # (12.113)
                ji_out_all[xx, i, Mu_tangent > Alt_ref[i], :]=0
                lambda_approx_out[xx, i, Mu_tangent > Alt_ref[i], :] = 0
                ji_out_all[xx, i,
                           Mu_tangent == Alt_ref[i],
                           :] = ji_in_all[xx, i, Mu_tangent == Alt_ref[i], :]
                lambda_approx_out[xx, i,
                           Mu_tangent == Alt_ref[i],
                           :] = lambda_approx_in[xx, i, Mu_tangent == Alt_ref[i], :]
        weighttemp = (mu_weight[:, i].reshape(mu_weight[:, i].size, 1) *
                      np.ones((mu_weight[:, i].size, Fre_range_i[xx].size)))
        weighttemp = weighttemp*1./weighttemp[:, 0].sum()
        for xx in range(Nt):  # transitions
            Para=[Freq_array[xx]*1.e9, 18.0153]
            if wind_v is not None:
                incoming_doppler = DopplerWind(Temp[i],
                                               Fre_range_i[xx],
                                               Para, wind_v[:,i],
                                               shift_direction='Red')
                outgoing_doppler = DopplerWind(Temp[i],
                                               Fre_range_i[xx],
                                               Para, wind_v[:,i],
                                               shift_direction='Blue')
                Fre_weight_i = trapz_inte_edge(incoming_doppler, Fre_range_i[xx])
                Fre_weight_o = trapz_inte_edge(outgoing_doppler, Fre_range_i[xx])
                J_mean = np.sum((ji_out_all[xx, i, :, :] * Fre_weight_o +
                                 ji_in_all[xx, i, :, :] * Fre_weight_i) *
                                weighttemp)*0.5
                l_ap= np.sum((lambda_approx_in[xx, i, :, :] * Fre_weight_i +
                              lambda_approx_out[xx, i, :, :] * Fre_weight_o) *
                             weighttemp)*0.5
#                if (l_ap>1) or (l_ap<0):
#                    print(l_ap, i, xx,
#                          lambda_approx_in[xx, i, :, :].min(),
#                          lambda_approx_out[xx, i, :, :].min(),
#                          lambda_approx_in[xx, i, :, :].max(),
#                          lambda_approx_out[xx, i, :, :].max())
            else: 
                Fre_weight = trapz_inte_edge(F_vl_i[xx][i], Fre_range_i[xx])
                j_mu = np.sum((ji_out_all[xx, i, :, :] +
                              ji_in_all[xx, i, :, :]) *
                              weighttemp,
                              axis=0)*0.5
                J_mean = (j_mu * Fre_weight).sum()  # * FreDelta
                """Lambda operator """
                L_ap= np.sum((lambda_approx_in[xx, i, :, :] +
                              lambda_approx_out[xx, i, :, :]) *
                             weighttemp,
                             axis=0)*0.5
                if (L_ap>1) or (L_ap<0):
                    logger.info(
                        L_ap, i,
                        lambda_approx_in[xx, i, :, :].min(),
                        lambda_approx_out[xx, i, :, :].min(),
                        lambda_approx_in[xx, i, :, :].max(),
                        lambda_approx_out[xx, i, :, :].mmx()
                    )
                    L_ap[L_ap<0]=0
                    L_ap[L_ap>1]=1
                #j_mean_corr[xx,i] = J_mean
                l_ap = (L_ap * Fre_weight).sum()  # * FreDelta
#            l_ap[l_ap<0]=0
#            l_ap[l_ap>1]=1
            if (l_ap>1) or (l_ap<0):
                logger.info(l_ap, i, lambda_approx_in[xx, i, :, :].min(),
                  lambda_approx_out[xx, i, :, :].min())
            B_int[B_place == xx] = J_mean*1.
            if Alt_ref[i] == 0:
                Fre_weight = trapz_inte_edge(F_vl_i[xx][i], Fre_range_i[xx])
                J_mean = (Bv_T(Fre_range_i[xx], Temp[i]) * Fre_weight).sum()
                B_int_lamda[B_place == xx] = J_mean*1.
            elif (i>0) & (i<Alt_ref.size-1) & (iteration == 'MUGA' or iteration == 'MALI'):
                B_int_lamda[B_place == xx] = J_mean*1. - l_ap*SF[xx]
                #print(l_ap*SF[xx])
                #B_int_lamda[B_place == xx] = SF[xx]+(J_mean*1. - SF[xx])/(1.-l_ap)
                A_int_lamda[(B_place == xx)&(RaRaA>0)] = l_ap
                #j_eff_corr[xx,i] = J_mean*1. - l_ap*SF[xx]
            elif (i>0) & (i<Alt_ref.size-1) & (iteration == 'LI'):
                B_int_lamda[B_place == xx] = J_mean*1.
                #print(l_ap*SF[xx])
                #B_int_lamda[B_place == xx] = SF[xx]+(J_mean*1. - SF[xx])/(1.-l_ap)
                A_int_lamda[(B_place == xx)&(RaRaA>0)] = 0.
                #j_eff_corr[xx,i] = J_mean*1. - l_ap*SF[xx]
        RaRij = (RaRaB_absorption+RaRaB_induced) * B_int
        RaRii = np.eye(Ni)*(RaRij.sum(axis=0))*-1.
        A_m = (RaRaA+RaRaAd+RaRij+RaRii+CoRa_block[i])*-1.
        if (i>0) & (i<Alt_ref.size-1) & (iteration == 'MUGA' or iteration == 'MALI'):  # preconditioning part
            RaRij_lambda = (RaRaB_absorption+RaRaB_induced) * B_int_lamda
            RaRii_lambda = np.eye(Ni)*(RaRij_lambda.sum(axis=0))*-1.
            P_m = (RaRaA*(1-A_int_lamda)+(-RaRaA*(1-A_int_lamda)).sum(axis=0)*np.eye(Ni)+RaRij_lambda+RaRii_lambda+CoRa_block[i])*-1.
            #P_m = ((-RaRaA*(1-A_int_lamda)).sum(axis=0)+RaRii_lambda+CoRa_block[i])*-1.
            #P_m = P_m.diagonal()*np.eye(Ni)
            P_m[-1, :] = 1.
        elif (i>0) & (i<Alt_ref.size-1) & (iteration == 'LI'):  # preconditioning part
            RaRij_lambda = (RaRaB_absorption+RaRaB_induced) * B_int_lamda
            RaRii_lambda = np.eye(Ni)*(RaRij_lambda.sum(axis=0))*-1.
            P_m = (RaRaA+(-RaRaA).sum(axis=0)*np.eye(Ni)+RaRij_lambda+RaRii_lambda+CoRa_block[i])*-1.
            #P_m = ((-RaRaA*(1-A_int_lamda)).sum(axis=0)+RaRii_lambda+CoRa_block[i])*-1.
            #P_m = P_m.diagonal()*np.eye(Ni)
            P_m[-1, :] = 1.
        A_m[-1, :] = 1.
        b = np.zeros((Ni, 1))*1.
        b[-1] = Ite_pop[i][0].sum()
        n_old = Ite_pop[i][0]*1.
        #"""correction method (normal?)"""Don't repeat yourself
        #n_delta = b - np.dot(A_m, n_old)
#        if (i>0)&(i<Alt_ref.size-1):  # preconditioning part
#            ##"""method1"""
#            n_new = (np.eye(Ni)-np.linalg.inv(P_m).dot(A_m)).dot(n_old) + np.linalg.inv(P_m).dot(b)
#            ##"""method 2"""
#            #residual_r = b - np.dot(P_m, n_old)
#            #n_delta = np.linalg.inv(P_m).dot(residual_r)
#            #omega = 2./(1+(1-(A_int_lamda**2).max())**0.5)
#            #n_new = n_old + n_delta#*omega  # plus or minus?
#        else:
#            residual_r = b - np.dot(A_m, n_old)
#            n_new = np.linalg.inv(A_m).dot(b)
#            n_delta = n_new-n_old
        # """Inversion method"""
        if (i > 0) & (i < Alt_ref.size-1):  # preconditioning part
            n_new = np.linalg.inv(P_m).dot(b)
        elif i == 0:
            n_new = n_old
        elif i == Alt_ref.size-1:
            n_new = np.linalg.inv(A_m).dot(b)
        else:
            logger.info('check alt. grid', i, Alt_ref[i])
        n_delta = n_new-n_old
        if update_population is True:
            new_pop[i, 0, :] = n_new
        else:
            n_new = n_old
            new_pop[i, 0, :] = n_new #this is for no update
#        print(n_new)
        max_true_error[i] = np.abs((n_delta/n_new)*100).max()
        if (i > 0) and (i < Alt_ref.size-1) and (iteration == 'MUGA'):
            for xx in range(Nt):  # transitions
                Para=[Freq_array[xx]*1.e9, 18.0153]
                Fre_range = Fre_range_i[xx]
                # Line_Shape = F_vl_i[xx]
                Abs_coeff = Abs_ite[xx]
                Fre_range = Fre_range_i[xx]
                up_tag = int(Tran_tag[xx][1])
                low_tag = int(Tran_tag[xx][2])
                # tl_1 = Abs_coeff[i] * F_vl_i[xx][i]
                tl_1 = basic(new_pop[i, 0, low_tag, 0],
                             new_pop[i, 0, up_tag, 0],
                             Blu[up_tag, low_tag],
                             Bul[up_tag, low_tag],
                             Freq_array[xx]*1.e9)*F_vl_i[xx][i-1]
                tl_2 = basic(new_pop[i-1, 0, low_tag, 0],
                             new_pop[i-1, 0, up_tag, 0],
                             Blu[up_tag, low_tag],
                             Bul[up_tag, low_tag],
                             Freq_array[xx]*1.e9)*F_vl_i[xx][i-1]
                # tdl2 = calc_abscoeff(xx, IteNum+1, i-1) * F_vl_i[xx][i-1]
                tl1 = (0.5*np.abs(tl_1+tl_2).reshape((1, Fre_range.size)) *
                       PSC2[:, i-1].reshape((Mu_tangent.size, 1)))  # (Mu, Fre)
                tl_3 = Abs_coeff[i+1] * F_vl_i[xx][i+1]
                tl3 = (0.5*np.abs(tl_1+tl_3).reshape((1, Fre_range.size)) *
                       PSC2[:, i].reshape((Mu_tangent.size, 1)))  # (Mu, Fre)
                """===Doppler=========="""
                tl_1 = basic(new_pop[i, 0, low_tag, 0],
                             new_pop[i, 0, up_tag, 0],
                             Blu[up_tag, low_tag],
                             Bul[up_tag, low_tag],
                             Freq_array[xx]*1.e9)*\
                       DopplerWind(Temp[i], Fre_range_i[xx], Para, wind_v[:,i], shift_direction='Blue')
                tl_2 = basic(new_pop[i-1, 0, low_tag, 0],
                             new_pop[i-1, 0, up_tag, 0],
                             Blu[up_tag, low_tag],
                             Bul[up_tag, low_tag],
                             Freq_array[xx]*1.e9)*\
                       DopplerWind(Temp[i-1], Fre_range_i[xx], Para, wind_v[:,i-1], 
                                   shift_direction='Blue')  # F_vl_i[xx][i-1]
                tl_3 = Abs_coeff[i+1] * \
                DopplerWind(Temp[i+1], Fre_range_i[xx], Para, wind_v[:,i+1], shift_direction='Red')
                tl1 = (0.5*np.abs(tl_1+tl_2) *
                       PSC2[:, i-1].reshape((Mu_tangent.size, 1)))  # (Mu, Fre)
                tl3 = (0.5*np.abs(tl_1+tl_3) *
                       PSC2[:, i].reshape((Mu_tangent.size, 1)))  # (Mu, Fre)
                """======================"""
                Sl2 = PopuSource_AB(new_pop[i, 0, low_tag, 0],
                                    new_pop[i, 0, up_tag, 0],
                                    Aul[up_tag, low_tag],
                                    Bul[up_tag, low_tag],
                                    Blu[up_tag, low_tag])  # * F_vl_i[xx][i]
                Sl1 = PopuSource_AB(new_pop[i-1, 0, low_tag, 0],
                                    new_pop[i-1, 0, up_tag, 0],
                                    Aul[up_tag, low_tag],
                                    Bul[up_tag, low_tag],
                                    Blu[up_tag, low_tag])  # * F_vl_i[xx][i]
                Sl3 = PopuSource_AB(Ite_pop[i+1, 0, low_tag, 0],
                                    Ite_pop[i+1, 0, up_tag, 0],
                                    Aul[up_tag, low_tag],
                                    Bul[up_tag, low_tag],
                                    Blu[up_tag, low_tag])
                Il = ji_out_all[xx, i-1, :, :]*1.  # !!!CHECK!!!
                # ji_out_all[xx, i, :, :] = FOSC(tl1, Sl1, Sl2, Il)  # (12.113)
                ji_out_all[xx, i, :, :],lambda_approx_out[xx, i, :, :]\
                          = SOSC(tl1, tl3,
                                 Sl1, Sl2, Sl3,
                                 Il)
                ji_out_all[xx, i, Mu_tangent > Alt_ref[i], :]=0
                lambda_approx_out[xx, i, Mu_tangent > Alt_ref[i], :] = 0
                ji_out_all[xx, i, Mu_tangent == Alt_ref[i], :]\
                = ji_in_all[xx, i, Mu_tangent == Alt_ref[i], :]
                lambda_approx_out[xx, i, Mu_tangent == Alt_ref[i], :]\
                = lambda_approx_in[xx, i, Mu_tangent == Alt_ref[i], :]
    if out_put_spectra is True:
        return new_pop,ji_out_all
    else:
        return new_pop




def jmean_calc(freq_w, mu_w, j_fre_mu):
    j_fre = np.sum(j_fre_mu*mu_w, axis=0)
    j_mean = np.sum(j_fre * freq_w).sum()
    return j_mean


def trans_rate_matrix(trans_place, radiation_coeff, jmean,
                      acoeff_rate, coll_rate):
    j_matrix = trans_place*0.
    for hoge in range(jmean.size):
        j_matrix[trans_place == hoge] = jmean[hoge]
    radiation_rate = radiation_coeff*j_matrix
    radiation_rate_diag = (np.eye(trans_place.shape[0]) *
                           (radiation_rate.sum(axis=0))*-1.)
    trans_matrix = (acoeff_rate +
                    radiation_rate +
                    radiation_rate_diag +
                    coll_rate)*-1.
    return trans_matrix


def pop_calc(trans_rm, old_pop):
    b = np.zeros((trans_rm.shape[0], 1))*1.
    b[-1] = old_pop.sum()
    n_new = np.linalg.inv(trans_rm).dot(b)
    return n_new




def MALI(Ite_pop, Abs_ite,
         PSC2, Mu_tangent, mu_weight,
         Alt_ref, Temp,
         Fre_range_i, Freq_array, F_vl_i,
         B_place,
         Nt, Ni, Aul, Bul, Blu,
         RaRaB_absorption, RaRaB_induced, RaRaA, RaRaAd, CoRa_block,
         Tran_tag):
    ji_in_all = np.zeros((Nt,
                          Alt_ref.size,
                          Mu_tangent.size,
                          Fre_range_i[0].size))
    ji_out_all = ji_in_all*0.
    lambda_approx_in = np.zeros((Nt,
                                 Alt_ref.size,
                                 Mu_tangent.size,
                                 Fre_range_i[0].size))
    lambda_approx_out = np.zeros((Nt,
                                  Alt_ref.size,
                                  Mu_tangent.size,
                                  Fre_range_i[0].size))
    # for IteNum in range(0, 100):
    max_true_error = np.ones(Alt_ref.size)*0.
    new_pop = np.array(Ite_pop)*0.+0.
    for ii in range(Alt_ref.size)[::-1]:  # spatial points
        # print(ii)
        for xx in range(Nt):  # transitions
            # Line_Shape = F_vl_i[xx]
            Abs_coeff = Abs_ite[xx]
            Fre_range = Fre_range_i[xx]
            up_tag = int(Tran_tag[xx][1])
            low_tag = int(Tran_tag[xx][2])
            # Up_pop = Ite_pop[ii, 0, up_tag, 0]
            # Low_pop = Ite_pop[ii, 0, low_tag, 0]
            if ii == Alt_ref.size-1:  # upper Boundary(u>0) incoming
                B_v_cosmic = Bv_T(Fre_range, 2.375)
                ji_in_all[xx, ii, :, :] = 0.*B_v_cosmic  # Ladi comparison
            elif ii == Alt_ref.size-1-1:
                tdu2 = Abs_coeff[ii+1] * F_vl_i[xx][ii+1]
                tdu1 = Abs_coeff[ii] * F_vl_i[xx][ii]
                tdu = (0.5*np.abs(tdu1+tdu2).reshape((1, Fre_range.size)) *
                       PSC2[:, ii].reshape((Mu_tangent.size, 1)))
                Sd = PopuSource_AB(Ite_pop[ii, 0, low_tag, 0],
                                   Ite_pop[ii, 0, up_tag, 0],
                                   Aul[up_tag, low_tag],
                                   Bul[up_tag, low_tag],
                                   Blu[up_tag, low_tag])  # * F_vl_i[xx][ii]
                Sd1 = PopuSource_AB(Ite_pop[ii+1, 0, low_tag, 0],
                                    Ite_pop[ii+1, 0, up_tag, 0],
                                    Aul[up_tag, low_tag],
                                    Bul[up_tag, low_tag],
                                    Blu[up_tag, low_tag])  # * F_vl_i[xx][ii]
                Idu = ji_in_all[xx, ii+1, :, :]*1.  # !!!CHECK!!!
                ji_in_all[xx, ii, :, :],lambda_approx_in[xx, ii, :, :]\
                         = FOSC(tdu, Sd1, Sd, Idu)
                ji_in_all[xx, ii, ((Mu_tangent > Alt_ref[ii])), :] = 0
                lambda_approx_in[xx, ii, ((Mu_tangent > Alt_ref[ii])), :] = 0
                # mu; 0->i,
            elif ii == 0:  # SOSC is not available (FOSC elif ii>=0)
                tdu2 = Abs_coeff[ii+1] * F_vl_i[xx][ii+1]
                tdu1 = Abs_coeff[ii] * F_vl_i[xx][ii]
                tdu = (0.5*np.abs(tdu1+tdu2).reshape((1, Fre_range.size)) *
                       PSC2[:, ii].reshape((Mu_tangent.size, 1)))
                Sd = PopuSource_AB(Ite_pop[ii, 0, low_tag, 0],
                                   Ite_pop[ii, 0, up_tag, 0],
                                   Aul[up_tag, low_tag],
                                   Bul[up_tag, low_tag],
                                   Blu[up_tag, low_tag])  # * F_vl_i[xx][ii]
                Sd1 = PopuSource_AB(Ite_pop[ii+1, 0, low_tag, 0],
                                    Ite_pop[ii+1, 0, up_tag, 0],
                                    Aul[up_tag, low_tag],
                                    Bul[up_tag, low_tag],
                                    Blu[up_tag, low_tag])  # * F_vl_i[xx][ii]
                Sd3 = PopuSource_AB(Ite_pop[ii-1, 0, low_tag, 0],
                                    Ite_pop[ii-1, 0, up_tag, 0],
                                    Aul[up_tag, low_tag],
                                    Bul[up_tag, low_tag],
                                    Blu[up_tag, low_tag])  # * F_vl_i[xx][ii]
                tdu3 = Abs_coeff[ii-1] * F_vl_i[xx][ii-1]
                tdb = (0.5*np.abs(tdu1+tdu3).reshape((1, Fre_range.size)) *
                       PSC2[:, ii-1].reshape((Mu_tangent.size, 1)))
                Idu = ji_in_all[xx, ii+1, :, :]*1.  # !!!CHECK!!!
                ji_in_all[xx, ii, :, :],lambda_approx_in[xx, ii, :, :]\
                = FOSC(tdu, Sd1, Sd, Idu)
                ji_in_all[xx, ii, ((Mu_tangent > Alt_ref[ii])), :] = 0
                lambda_approx_in[xx, ii, ((Mu_tangent > Alt_ref[ii])), :] = 0
                # mu; 0->i,
            else:  # SOSC
                tdu2 = Abs_coeff[ii+1] * F_vl_i[xx][ii+1]
                tdu1 = Abs_coeff[ii] * F_vl_i[xx][ii]
                tdu = (0.5*np.abs(tdu1+tdu2).reshape((1, Fre_range.size)) *
                       PSC2[:, ii].reshape((Mu_tangent.size, 1)))
                Sd = PopuSource_AB(Ite_pop[ii, 0, low_tag, 0],
                                   Ite_pop[ii, 0, up_tag, 0],
                                   Aul[up_tag, low_tag],
                                   Bul[up_tag, low_tag],
                                   Blu[up_tag, low_tag])  # * F_vl_i[xx][ii]
                Sd1 = PopuSource_AB(Ite_pop[ii+1, 0, low_tag, 0],
                                    Ite_pop[ii+1, 0, up_tag, 0],
                                    Aul[up_tag, low_tag],
                                    Bul[up_tag, low_tag],
                                    Blu[up_tag, low_tag])  # * F_vl_i[xx][ii]
                Sd3 = PopuSource_AB(Ite_pop[ii-1, 0, low_tag, 0],
                                    Ite_pop[ii-1, 0, up_tag, 0],
                                    Aul[up_tag, low_tag],
                                    Bul[up_tag, low_tag],
                                    Blu[up_tag, low_tag])  # * F_vl_i[xx][ii]
                tdu3 = Abs_coeff[ii-1] * F_vl_i[xx][ii-1]
                tdb = (0.5*np.abs(tdu1+tdu3).reshape((1, Fre_range.size)) *
                       PSC2[:, ii-1].reshape((Mu_tangent.size, 1)))
                Idu = ji_in_all[xx, ii+1, :, :]*1.  # !!!CHECK!!!
                ji_in_all[xx, ii, :, :], lambda_approx_in[xx, ii, :, :]\
                         = SOSC(tdu, tdb,
                                Sd1, Sd, Sd3,
                                Idu)
                ji_in_all[xx, ii, (Mu_tangent == Alt_ref[ii]), :],\
                lambda_approx_in[xx, ii, (Mu_tangent == Alt_ref[ii]), :]\
                          = FOSC(tdu[Mu_tangent == Alt_ref[ii]],
                                    Sd1,
                                    Sd,
                                    Idu[Mu_tangent == Alt_ref[ii]])
                ji_in_all[xx, ii, ((Mu_tangent > Alt_ref[ii])), :] = 0
                lambda_approx_in[xx, ii, ((Mu_tangent > Alt_ref[ii])), :] = 0
                # mu; 0->i,
    for i in range(Alt_ref.size):  # spatial points
        #  print(i)
        B_int = B_place*0.  # intensity for each transitoin
        B_int_lamda = B_place*0.  # intensity for each transitoin for lambda ope
        A_int_lamda = B_place*0.  # approximated lambda operate
        SF = np.ones(Nt)
        for xx in range(Nt):  # transitions
            Fre_range = Fre_range_i[xx]
            # Line_Shape = F_vl_i[xx]
            Abs_coeff = Abs_ite[xx]
            Fre_range = Fre_range_i[xx]
            up_tag = int(Tran_tag[xx][1])
            low_tag = int(Tran_tag[xx][2])
            # Up_pop = Ite_pop[i, 0, up_tag, 0]
            # Low_pop = Ite_pop[i, 0, low_tag, 0]
            if i == 0:  # Lower Boundary(u>0) outgoing
                ji_out_all[xx, i, :, :] = Bv_T(Fre_range, Temp[i])
                ji_out_all[:, i, (Mu_tangent > Alt_ref[i]), :] = 0
            elif i < Alt_ref.size-1:  # SOSC
                tl_1 = Abs_coeff[i] * F_vl_i[xx][i]
                # GS-iteration
                tl_2 = basic(Ite_pop[i-1, 0, low_tag, 0],
                             Ite_pop[i-1, 0, up_tag, 0],
                             Blu[up_tag, low_tag],
                             Bul[up_tag, low_tag],
                             Freq_array[xx]*1.e9)*F_vl_i[xx][i-1]
                # tdl2 = calc_abscoeff(xx, IteNum+1, i-1) * F_vl_i[xx][i-1]
                tl1 = (0.5*np.abs(tl_1+tl_2).reshape((1, Fre_range.size)) *
                       PSC2[:, i-1].reshape((Mu_tangent.size, 1)))  # (Mu, Fre)
                tl_3 = Abs_coeff[i+1] * F_vl_i[xx][i+1]
                tl3 = (0.5*np.abs(tl_1+tl_3).reshape((1, Fre_range.size)) *
                       PSC2[:, i].reshape((Mu_tangent.size, 1)))  # (Mu, Fre)
                Sl2 = PopuSource_AB(Ite_pop[i, 0, low_tag, 0],
                                    Ite_pop[i, 0, up_tag, 0],
                                    Aul[up_tag, low_tag],
                                    Bul[up_tag, low_tag],
                                    Blu[up_tag, low_tag])  # * F_vl_i[xx][i]
                SF[xx] = Sl2
                Sl1 = PopuSource_AB(Ite_pop[i-1, 0, low_tag, 0],
                                    Ite_pop[i-1, 0, up_tag, 0],
                                    Aul[up_tag, low_tag],
                                    Bul[up_tag, low_tag],
                                    Blu[up_tag, low_tag])  # * F_vl_i[xx][i]
                Sl3 = PopuSource_AB(Ite_pop[i+1, 0, low_tag, 0],
                                    Ite_pop[i+1, 0, up_tag, 0],
                                    Aul[up_tag, low_tag],
                                    Bul[up_tag, low_tag],
                                    Blu[up_tag, low_tag])
                Il = ji_out_all[xx, i-1, :, :]*1.  #
                # ji_out_all[xx, i, :, :] = FOSC(tl1, Sl1, Sl2, Il)  # (12.113)
                ji_out_all[xx, i, :, :],lambda_approx_out[xx, i, :, :]\
                          = SOSC(tl1, tl3,
                                 Sl1, Sl2, Sl3,
                                 Il)
                ji_out_all[xx, i, Mu_tangent > Alt_ref[i], :]=0
                lambda_approx_out[xx, i, Mu_tangent > Alt_ref[i], :] = 0
                ji_out_all[xx, i, Mu_tangent == Alt_ref[i], :]\
                = ji_in_all[xx, i, Mu_tangent == Alt_ref[i], :]
                lambda_approx_out[xx, i, Mu_tangent == Alt_ref[i], :]\
                = lambda_approx_in[xx, i, Mu_tangent == Alt_ref[i], :]
                I3 = ji_in_all[xx, i+1, :, :]*1.  # !!!CHECK!!!
                ji_in_all[xx, i, :, :],lambda_approx_out[xx, i, :, :]\
                         = SOSC(tl3, tl1,
                                Sl3, Sl2, Sl1,
                                I3)
                ji_in_all[xx, i, (Mu_tangent == Alt_ref[i]), :],\
                lambda_approx_out[xx, i, (Mu_tangent == Alt_ref[i]), :]\
                             = FOSC(tl3[Mu_tangent == Alt_ref[i]],
                                    Sl3,
                                    Sl2,
                                    I3[Mu_tangent == Alt_ref[i]])
                ji_in_all[xx, i, ((Mu_tangent > Alt_ref[i])), :] = 0
                lambda_approx_out[xx, i, ((Mu_tangent > Alt_ref[i])), :] = 0
#                ji_in_all[xx, i, ((Mu_tangent > Alt_ref[i-1])), :] = 0
            elif i == Alt_ref.size-1:  # FOSC
                tl_1 = Abs_coeff[i] * F_vl_i[xx][i]
                # GS-iteration
                tl_2 = basic(Ite_pop[i-1, 0, low_tag, 0],
                             Ite_pop[i-1, 0, up_tag, 0],
                             Blu[up_tag, low_tag],
                             Bul[up_tag, low_tag],
                             Freq_array[xx]*1.e9)*F_vl_i[xx][i-1]
                # tdl2 = calc_abscoeff(xx, IteNum+1, i-1) * F_vl_i[xx][i-1]
                tl1 = (0.5*np.abs(tl_1+tl_2).reshape((1, Fre_range.size)) *
                       PSC2[:, i-1].reshape((Mu_tangent.size, 1)))  # (Mu, Fre)
                Sl = PopuSource_AB(Ite_pop[i, 0, low_tag, 0],
                                   Ite_pop[i, 0, up_tag, 0],
                                   Aul[up_tag, low_tag],
                                   Bul[up_tag, low_tag],
                                   Blu[up_tag, low_tag])  # * F_vl_i[xx][i]
                Sl1 = PopuSource_AB(Ite_pop[i-1, 0, low_tag, 0],
                                    Ite_pop[i-1, 0, up_tag, 0],
                                    Aul[up_tag, low_tag],
                                    Bul[up_tag, low_tag],
                                    Blu[up_tag, low_tag])  # * F_vl_i[xx][i]
                Il = ji_out_all[xx, i-1, :, :]*1.  #
                ji_out_all[xx, i, :, :],lambda_approx_out[xx, i, :, :]\
                          = FOSC(tl1, Sl1, Sl, Il)  # (12.113)
                ji_out_all[xx, i, Mu_tangent > Alt_ref[i], :]=0
                lambda_approx_out[xx, i, Mu_tangent > Alt_ref[i], :] = 0
                ji_out_all[xx, i,
                           Mu_tangent == Alt_ref[i],
                           :] = ji_in_all[xx, i, Mu_tangent == Alt_ref[i], :]
                lambda_approx_out[xx, i,
                           Mu_tangent == Alt_ref[i],
                           :] = lambda_approx_in[xx, i, Mu_tangent == Alt_ref[i], :]
        for xx in range(Nt):  # transitions
            Fre_weight = trapz_inte_edge(F_vl_i[xx][i], Fre_range_i[xx])
            weighttemp = (mu_weight[:, i].reshape(mu_weight[:, i].size, 1) *
                          np.ones((mu_weight[:, i].size, Fre_weight.size)))
            weighttemp = weighttemp*1./weighttemp[:, 0].sum()
            j_mu = np.sum((ji_out_all[xx, i, :, :] +
                          ji_in_all[xx, i, :, :]) *
                          weighttemp,
                          axis=0)*0.5
            """Lambda operator """
            MALI_LI = 'MALI'
            L_ap= np.sum((lambda_approx_in[xx, i, :, :] +
                          lambda_approx_out[xx, i, :, :]) *
                          weighttemp,
                          axis=0)*0.5
            L_ap[L_ap<0]=0
            J_mean = (j_mu * Fre_weight).sum()  # * FreDelta
            l_ap = (L_ap * Fre_weight).sum()  # * FreDelta
            if Alt_ref[i] == 0:
                J_mean = (Bv_T(Fre_range_i[xx], Temp[i]) * Fre_weight).sum()
            B_int[B_place == xx] = J_mean*1.
            if (i>0) & (i<Alt_ref.size-1):
                B_int_lamda[B_place == xx] = J_mean*1. - l_ap*SF[xx]
                A_int_lamda[(B_place == xx)&(RaRaA>0)] = l_ap
                if MALI_LI=='LI':
                    B_int_lamda[B_place == xx] = J_mean*1.
                    A_int_lamda[(B_place == xx)&(RaRaA>0)] = 0
                #j_eff_corr[xx,i] = J_mean*1. - l_ap*SF[xx]
        RaRij = (RaRaB_absorption+RaRaB_induced) * B_int
        RaRii = np.eye(Ni)*(RaRij.sum(axis=0))*-1.
        A_m = (RaRaA+RaRaAd+RaRij+RaRii+CoRa_block[i])*-1.
        if (i>0) & (i<Alt_ref.size-1):  # preconditioning part
            RaRij_lambda = (RaRaB_absorption+RaRaB_induced) * B_int_lamda
            RaRii_lambda = np.eye(Ni)*(RaRij_lambda.sum(axis=0))*-1.
            P_m = (RaRaA*(1.-A_int_lamda)+(-RaRaA*(1.-A_int_lamda)).sum(axis=0)*np.eye(Ni)+RaRij_lambda+RaRii_lambda+CoRa_block[i])*-1.
            #P_m = ((-RaRaA*(1-A_int_lamda)).sum(axis=0)+RaRii_lambda+CoRa_block[i])*-1.
            #P_m = P_m.diagonal()*np.eye(Ni)
            P_m[-1, :] = 1.
        A_m[-1, :] = 1.
        b = np.zeros((Ni, 1))*1.
        b[-1] = Ite_pop[i][0].sum()
        n_old = Ite_pop[i][0]*1.
        #"""correction method (normal?)"""
        #n_delta = b - np.dot(A_m, n_old)
#        if (i>0)&(i<Alt_ref.size-1):  # preconditioning part
#            ##"""method1"""
#            n_new = (np.eye(Ni)-np.linalg.inv(P_m).dot(A_m)).dot(n_old) + np.linalg.inv(P_m).dot(b)
#            ##"""method 2"""
#            #residual_r = b - np.dot(P_m, n_old)
#            #n_delta = np.linalg.inv(P_m).dot(residual_r)
#            #omega = 2./(1+(1-(A_int_lamda**2).max())**0.5)
#            #n_new = n_old + n_delta#*omega  # plus or minus?
#        else:
#            residual_r = b - np.dot(A_m, n_old)
#            n_new = np.linalg.inv(A_m).dot(b)
#            n_delta = n_new-n_old
        # """Inversion method"""
        if (i>0) & (i<Alt_ref.size-1):  # preconditioning part
            n_new = np.linalg.inv(P_m).dot(b)
        else:
            n_new = np.linalg.inv(A_m).dot(b)
        n_delta = n_new-n_old
        # """Population input """
        new_pop[i, 0, :] = n_new
        max_true_error[i] = np.abs((n_delta/n_new)*100).max()
    return new_pop


