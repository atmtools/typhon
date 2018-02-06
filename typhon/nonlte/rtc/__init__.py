
import numba

import numpy as np
from ..spectra.source_function import Bv_T, PopuSource_AB
from ..spectra.abscoeff import basic

@numba.jit
def FOSC(tau, Sb, Sm, Ib):
    """ First Order Short Characteristics 

    See "Theory of stellar atmospheres: An introduction to astrophysical 
    non-equilibrium quantitative spectroscopic analysis" by Ivan Hubeny
    and Dimitri Mihalas, ISBN 978-0-691-16328-4

    Parameters:
        tau: optical depth between two layers 
        Sb: Source function at adjacent grid points 
        Sm: Source function at target point 
        Ib: Intensity adjacent grid points

    Returns: 
        Inwards-directed intensity, lambda used in equation
    """
    yd = tau - 1. + np.exp(-tau)  # (12.120)
    
    dev_cond = tau * 1.
    dev_cond[tau == 0] = np.nan

    lambda_m = yd/dev_cond  # (12.117)
    lambda_b = - (yd / dev_cond) + 1. - np.exp(-tau)  # (12.118, 116)
    Im = Ib * np.exp(-tau) + lambda_m * Sm + lambda_b * Sb  # (12.114)
    Im[tau == 0] = Ib[tau == 0]
    return Im, lambda_m


@numba.jit
def SOSC(tau1, tau3, S1, S2, S3, I1):
    """ Second Order Short Characteristics

    See "Theory of stellar atmospheres: An introduction to astrophysical 
    non-equilibrium quantitative spectroscopic analysis" by Ivan Hubeny
    and Dimitri Mihalas, ISBN 978-0-691-16328-4

    and

    See "Fast multilevel radiative transfer" by Frederic Paletou and 
    Ludovick Leger, arXiv:astro-ph/0507021v3 4 Jul 2006

    Parameters:
        tau1,3: optical depth at both adjacent grid
        S1,3: Source function at adjacent grid points 
        S2: Source function at target point 
        I1: Intensity entering/leaving grids
    
    Returns:
        Outgoing/Entering intensity, lambda used in equation

    """
    x1 = 1. - np.exp(-tau1)
    y1 = tau1 - 1. + np.exp(-1.*tau1)
    z1 = tau1**2 - 2*y1

    la1_dev = tau1*(tau3+tau1)
    la1_dev[la1_dev == 0] = np.nan
    la2_dev = tau1*tau3
    la2_dev[la2_dev == 0] = np.nan
    la3_dev = tau3*(tau1*tau3)
    la3_dev[la3_dev == 0] = np.nan

    lambda_3 = (z1-tau1*y1)/(tau3*(tau3+tau1))  # (19c)
    lambda_2 = ((tau3+tau1)*y1-z1)/(la2_dev)  # (19b)
    lambda_1 = x1+(z1-(tau3+2*tau1)*y1)/la1_dev  # (19a)

    discrimination = lambda_3 * S3 + lambda_2 * S2 + lambda_1 * S1  # (12.124)

    taumask = tau1*1.
    taumask[tau1 != tau1] = np.inf

    discrimination[taumask <= 1.e-6] = 0  # remove calc. error

    I2 = I1 * np.exp(-tau1) + discrimination
    return I2, lambda_2


""" Takayoshi Yamada:  Work in progress below.  Change sparingly.
"""
def CalcSpectra(Ite_pop, Abs_ite,
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
    return ji_out_all[:,-1,:,:], lambda_approx_out[:,-1,:,:]
