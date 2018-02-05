# -*- coding: utf-8 -*-
from scipy.constants import h, c, k
import numpy as np


def PopuSource(LowerPop, UpperPop, LowerDeg, UpperDeg, Freq_c):
    sij = (2.*h*Freq_c**3)/(c**2*(LowerPop*UpperDeg/UpperPop/LowerDeg-1.))
    return sij


def PopuSource_AB(LowerPop, UpperPop, Aul, Bul, Blu):
    sij = (UpperPop*Aul)/(LowerPop*Blu - UpperPop*Bul)
    return sij


"""
PopuSource(Ite_pop[0][0][0],
           Ite_pop[0][0][1],
           MolPara.weighti[0],
           MolPara.weighti[1],
           MolPara.freqi[1][0]*1.e9)
PopuSource_AB(Ite_pop[0][0][0],
              Ite_pop[0][0][1],
              MolPara.ai[1,0],
              MolPara.bul[1,0],
              MolPara.blu[1,0])
MolPara.blu[1,0]
MolPara.bul[1,0]
MolPara.ai[1,0]
Ite_pop[0][0][0]
Ite_pop[0][0][1]
MolPara.weighti[1]
MolPara.weighti[0]
MolPara.freqi[1][0]*1.e9
"""


def Bv_T(Freq, T):
    # brbr = 1  # .e7*1.e-4
    Bv_out = 2.*h*Freq**3/c**2/(np.exp(h*Freq/k/T)-1.)
    return Bv_out
