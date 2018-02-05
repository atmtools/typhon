# -*- coding: utf-8 -*-
import numpy as np
from scipy.constants import c, k, h
from ..mathmatics import Trapz_inte_edge


def read_swi_molecules(molecules_state_consts, Temp, Mole):
    fn = open(molecules_state_consts+'oH2O16.lev_7levels', 'rb')
    # fn = open(RateConstantPath+'orthoH2O-7levels.txt', 'r')
    fr = fn.readlines()
    fn.close()
    "NUMBER OF ENERGY LEVELS"
    Ni = int(fr[3])
    E_cm1 = np.array([float(fr[xx].split()[1]) for xx in range(5, 12)])
    Weighti = np.array([float(fr[xx].split()[2]) for xx in range(5, 12)])
    # ====two level
    # Ni = int(2)
    # E_cm1 = np.array([float(fr[xx].split()[1]) for xx in range(5, 7)])
    # Weighti = np.array([float(fr[xx].split()[2]) for xx in range(5, 7)])
    # ======
    Ai = np.zeros((Ni, Ni))
    Freqi = np.zeros((Ni, Ni))
    E_Ki = np.zeros((Ni, Ni))
    B_place = np.ones((Ni, Ni))*-1.
    Tran_tag = []
    Ai_tag = []
    for xx in range(15, 24):  # Nt Number of transition ,24 , two level ->16):
        """up: temp[1]-1, low: temp[2]-1"""
        temp = np.array(fr[xx].split()).astype(np.float64)
        Tran_tag.append(temp[:3]-1)  # for 1... -> 0....
        Ai[int(temp[1]-1), int(temp[2])-1] = temp[3]
        Ai_tag.append(temp[3])
        # Freqi[int(temp[1]-1), int(temp[2])-1] = temp[4]
        _deltaEcm1 = E_cm1[int(temp[1])-1] - E_cm1[int(temp[2])-1]
        Freqi[int(temp[1]-1), int(temp[2])-1] = _deltaEcm1*c*100.*1.e-9  # GHz
        E_Ki[int(temp[1]-1), int(temp[2])-1] = temp[5]
        B_place[int(temp[1]-1), int(temp[2])-1] = int(temp[0]-1)  # induced
        B_place[int(temp[2])-1, int(temp[1]-1)] = int(temp[0]-1)  # absorption

    Tran_tag = np.array(Tran_tag)*1
    Ai_tag = np.array(Ai_tag)*1.
    Nt = len(Tran_tag)  # int(9)
    Freq_array = np.array([])
    for xx in range(Nt):
        _up = int(Tran_tag[xx][1])
        _low = int(Tran_tag[xx][2])
        Freq_array = np.hstack((Freq_array, Freqi[_up, _low]))

    E_ji = E_cm1*100*c*h
    # Weighti*np.exp(-E_ji/k/150.)  # Qr
    """  l
      |     |  |n1|
    u |     |  |n2|
      |     |  |n3|
    """
    """
    Radiation rate matrix
    """
    Bul = Ai*c**2/(2*h*(Freqi*1.e9)**3)
    Blu = Bul*Weighti.reshape((Ni, 1))/Weighti
    Bul[Bul != Bul] = 0  # [i,j]: i->j
    Blu[Blu != Blu] = 0  # [i,j]: j->i
    # RaRaB_induced = -Bul + Bul.T
    # RaRaB_absorption = Blu-Blu.T
    RaRaB_induced = Bul.T  # [i,j] i<-j
    RaRaB_absorption = Blu  # [i,j] i<-j

    RaRaA = Ai.T
    # Diagonal
    RaRaAd = np.zeros((Ni, Ni))  #
    RaRaBd = np.zeros((Ni, Ni))  #
    for xx in range(Ni):
        cond = Tran_tag[:, 1] == xx
        RaRaAd[xx, xx] += Ai_tag[cond].sum()*-1.
        RaRaBd[xx, xx] += (Bul[xx]+Blu.T[xx]).sum()*-1.
#    return RaRaA, RaRaAd, RaRaBd, RaRaB_induced, RaRaB_absorption, Tran_tag, Nt
#
#
#def read_swi_molecules_state_collision(molecules_state_consts, Temp, Mole,
#                                       Tran_tag, Nt):
    RateConst = np.loadtxt(molecules_state_consts+'H2O'+'_col.txt')*1.e-6
    "Temperature of Raw data"
    colt = np.array([100,  200,  300,  400,  500,  600,  700,  800])*1.
    coltt = np.around(np.arange(0., 400., 0.01), 2)  # We calculate 0.1 K
    col_tk = []
    for xx in range(Nt):
        col_tk.append(np.interp(coltt, colt, RateConst[xx]))
    col_tk = np.array(col_tk)
    "set col rate for each atmospheric Temp."
    col_ul, col_lu = [], []
    for xx in range(Nt):
        _up = int(Tran_tag[xx][1])
        _low = int(Tran_tag[xx][2])
        col_ul.append(np.array([col_tk[xx][np.argwhere(coltt == round(xxx, 2))][0][0]
                               for xxx in np.around(Temp, 2)]))
        col_lu.append((col_ul[xx]*Weighti[_up]/Weighti[_low]) *
                      np.exp(-h*Freqi[_up, _low]*1.e9/k/Temp))

    rate_ul, rate_lu = Mole*col_ul, Mole*col_lu  # molecules/s
    J_ETS, J_LTE = [], []
    for xx in range(Nt):
        _up = int(Tran_tag[xx][1])
        _low = int(Tran_tag[xx][2])
        J_ETS.append(2.*h*(Freqi[_up, _low]*1.e9)**3/c**2 /
                     (((rate_lu[xx])/(rate_ul[xx])*(rate_ul[xx]/Ai[_up, _low]) /
                       (1.+rate_ul[xx]/Ai[_up, _low]))**-1 *
                      Weighti[_up]/Weighti[_low] - 1.)
                     )  # See (5,2)(3.35)
        J_LTE.append(2.*h*(Freqi[_up, _low]*1.e9)**3 /
                     c**2 /
                     ((rate_ul[xx])/(rate_lu[xx]) *
                      Weighti[_up]/Weighti[_low] - 1.)
                     )  # See (3.35)
        # P.plot(h*(Freqi[_up, _low]*1.e9/(k*np.log(2.*h*
        #             (Freqi[_up, _low]*1.e9)**3/c**2/J_LTE[xx]+1))))

        # P.plot(h*(Freqi[_up, _low]*1.e9/(k*np.log(2.*h*
        #            (Freqi[_up, _low]*1.e9)**3/c**2/J_ETS[xx] +1)))
        #       , label = str(Freqi[_up, _low]))


    J_ETS = np.array(J_ETS)
    J_ETS_plot = h*(Freq_array.reshape(Nt, 1)*1.e9 /
                      (k*np.log(2.*h*(Freq_array.reshape(Nt, 1)*1.e9)**3 /
                                 c**2/J_ETS + 1.)
                       ))

    #==============================================================================
    # P.figure()
    # [P.plot(J_ETS_plot[xx], Alt_ref, label=str(round(Freq_array[xx], 3))) for xx in
    #  range(Nt)]
    # P.plot(Temp, Alt_ref, 'k', label='T$_k$')
    # P.legend(loc=0)
    # P.grid()
    #==============================================================================

    CoRa_block = []  # Collisional Rate matrix
    for hoge in range(Temp.size):
        Cul, Clu = rate_ul.T[hoge], rate_lu.T[hoge]
        CoRa = np.zeros((Ni, Ni))
        for xx in range(Nt):
            _up = int(Tran_tag[xx][1])
            _low = int(Tran_tag[xx][2])
            # CoRa[_up, _low] += -Cul[xx]+Clu[xx]
            # CoRa[_low, _up] += Cul[xx]-Clu[xx]
            CoRa[_up, _low] += Clu[xx]
            CoRa[_low, _up] += Cul[xx]
        # Diagonal
        for xx in range(Ni):
            cond = Tran_tag[:, 1] == xx
            CoRa[xx, xx] -= Cul[cond].sum()
            cond = Tran_tag[:, 2] == xx
            CoRa[xx, xx] -= Clu[cond].sum()
        CoRa_block.append(CoRa*1.)

    CoRa_block = np.array(CoRa_block)*1.
    return RaRaA, RaRaAd, RaRaBd, RaRaB_induced, RaRaB_absorption, Tran_tag, CoRa_block

'''
not Now,
But I have to make a method that
calculate population from JPL catalogue



ATRASU_dir = '/home/yamada/workspace/'
LineParaPath = ATRASU_dir+'ATRASU/spectroscopy/HITRAN/'  # read HITRAN,
LineParaPathJPL = ATRASU_dir+'ATRASU/spectroscopy/JPL/'  # read JPL

FunctionPosition = ATRASU_dir+'ATRASU/function/'
import sys
sys.path.append(FunctionPosition)
from Func_all_test import read_line_all, read_molpara, LineShape, Bv_T


def Bv_T(Freq, T, Freq_c):
    # Bv_out = 2.*C.h*Freq**3/C.c**2/(N.exp(C.h*Freq_c/C.k/T)-1.)
    Bv_out = 2.*C.h*Freq**3/C.c**2/(N.exp(C.h*Freq/C.k/T)-1.)
    return Bv_out


frequency, flag, lgint, GUP, QN1, QN2, ELO, DR = \
 read_line_all('H2O', LineParaPathJPL)
M, flag, Qrs_T, Qrs = read_molpara('H2O', LineParaPathJPL)

"""
Read absorption coeff.
for Temp.
"""
intensities = []
for xx in range(Nt):
    _up = int(Tran_tag[xx][1])
    _low = int(Tran_tag[xx][2])
    freq_cond = N.argmin(N.abs(frequency-Freqi[_up, _low]))
    ju, jkau, jkcu = (QN1[freq_cond][:3]).astype(int)
    jl, jkal, jkcl = (QN2[freq_cond][:3]).astype(int)
    # print (ju*2+1)*(2*((jkau+jkcu)%2) + 1)
    nn = DR[freq_cond]/2.
    qrs2 = [x for x in Qrs]
    qrs_t = [x for x in Qrs_T]
    qrs2.reverse()
    qrs_t.reverse()
    qrt_ticks = N.array([round(xx, 1) for xx in N.arange(50, 300+0.1, 0.1)])
    qrs = N.interp(qrt_ticks, qrs_t, qrs2)
    lower = (ELO[freq_cond])*C.c*100.*C.h
    upper = lower + C.h*frequency[freq_cond]*1.e9
    "Temperature dependance"
    T = N.around(Temp, 1)*1.
    QT = N.array([qrs[N.argwhere(qrt_ticks == T[xxx])[0][0]]
                  for xxx in range(T.size)])
    Svt2 = lgint[freq_cond]*(300./T)**(nn+1.)*N.exp(-(1./T-1./300.)*lower/C.k)
    intensities.append(lgint[freq_cond]*(Qrs[0]/QT) *
                       (N.exp(-(lower)/C.k/T)-N.exp(-(upper)/C.k/T)) /
                       (N.exp(-(lower)/C.k/300.)-N.exp(-(upper)/C.k/300.)))


intensities = N.array(intensities)


"""
H2O Partiion function from all JPL levels(636levels)
"""
qnl = [int(QN2[Jj][0])*1.e6+int(QN2[Jj][1])*1.e4 +
       int(QN2[Jj][2])*1.e2+int(QN2[Jj][3]) for Jj in range(frequency.size)]
qnh = [int(QN1[Jj][0])*1.e6+int(QN1[Jj][1])*1.e4 +
       int(QN1[Jj][2])*1.e2+int(QN1[Jj][3]) for Jj in range(frequency.size)]
# from upper states!!
# higher
u, indicesu, indexu = N.unique(qnh, return_index=True, return_inverse=True)
# lower
qnll, indices, index = N.unique(qnl, return_index=True, return_inverse=True)
mix = N.hstack((qnll, u))
# all_quantum_state
mixu, mixin, mixindex = N.unique(mix, return_index=True, return_inverse=True)
qnll2 = mixu
eloo = N.ones(qnll2.size)  # make_all_energy
eloo[mixin <= qnll.size] = ELO[indices]*C.h*C.c*100  # added_lower state
# added_from_upper
eloo[mixin >= qnll.size] = ELO[mixin[mixin >= qnll.size]-qnll.size] *\
                           C.h*C.c*100+C.h *\
                           frequency[mixin[mixin >= qnll.size]-qnll.size]*1.e9
Jnum = [int(qnll2[x]*1.e-6*2+1.) for x in range(qnll2.size)]
Kanum = [int(("{0:08d}".format(int(qnll2[x])))[2:4])
         for x in range(qnll2.size)]
kasurplus = [x % 2 for x in Kanum]
Kcnum = [int(("{0:08d}".format(int(qnll2[x])))[4:6])
         for x in range(qnll2.size)]
kcsurplus = [x % 2 for x in Kcnum]
even_odd = N.array([(kasurplus[x] + kcsurplus[x]) % 2
                    for x in range(len(Kanum))])
Sjk = 2*even_odd+1.

"""
Calculate
non-LTE Rotational probability (ni/Ntotal)
"""
RoTemp = N.reshape(Temp*1., (Temp.size, 1))  # shoud be Shape(Nt,Temp)
Ntotal = N.array([Jnum[x]*Sjk[x]*N.exp(-(eloo[x])/C.k/RoTemp)
                  for x in range(qnll2.size)]).cumsum(axis=0)[-1]

# Jnum*Sjk*N.exp(-(eloo)/C.k/RoTemp)
# partion func for give level(N)
Qr = Weighti*N.exp(-(E_cm1*100*C.c*C.h)/(C.k*RoTemp))
RoPr = Qr/Ntotal  # This is for all transitions
RoPr = Qr/(Qr.sum(axis=1).reshape(RoTemp.size, 1))  # this is given transition
"""make line strength for each transition"""
linet = []
for xx in range(Nt):
    gdu, gdl = Weighti[Tran_tag[xx][1:].astype(int)]
    _up = int(Tran_tag[xx][1])
    _low = int(Tran_tag[xx][2])
    Aei = Ai[_up, _low]
    line_const = (C.c*10**2)**2*Aei /\
                 (8.*N.pi*(Freq_array[xx]*1.e9)**2) *\
                 (gdu/gdl)*1.e-6*1.e14  # Hz->MHz,cm^2 ->nm^2
    # W = C.h*C.c*E_cm1[_low]*100.  # energy level above ground state
    "This is the function of calculating H2O intensity"
    line = (1.-N.exp(-C.h*(Freq_array[xx]*1.e9)/C.k/RoTemp))*line_const
    linet.append(line[:, 0]*RoPr[:, _low])  # line intensity non-LTE

"""plot
xx=2  # Nt = 8 shows the big difference!
P.clf()
P.plot(linet[xx], label='Calc Qr')
P.plot(intensities[xx], label='JPL approx.')
P.legend(loc=0)
"""
Ni_LTE = Mole.reshape((Mole.size, 1))*RoPr*0.75
Ite_pop = [[Ni_LTE[i].reshape((Ni, 1))] for i in range(Mole.size)]
'''
