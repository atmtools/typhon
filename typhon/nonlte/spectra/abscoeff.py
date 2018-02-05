# -*- coding: utf-8 -*-
import numpy as np
from scipy.constants import h


def basic(LowerPop, UpperPop, Blu, Bul, Freq):
    u"""
    calculate absorption coefficient.\n
    Freq should be Hz
    """
    _const = h*Freq/4./np.pi
    return (LowerPop*Blu - UpperPop*Bul)*_const


def calc_abscoeff(tran, itera):
    _up, _low = Tran_tag[tran][1:].astype(int)
    _v0 = Freqi[_up, _low]*1.e9
    _const = C.h*_v0/4./N.pi
    lowerPop = N.array(Ite_pop)[:, itera, _low, 0]
    upperPop = N.array(Ite_pop)[:, itera, _up, 0]
    BLU = Blu[_up, _low]
    BUL = Bul[_up, _low]
    # AUL = Ai[_up, _low]
    return (lowerPop*BLU - upperPop*BUL)*_const


def calc_abscoeff2(alt, tran, itera):
    _up, _low = Tran_tag[tran][1:].astype(int)
    _v0 = Freqi[_up, _low]*1.e9
    _const = C.h*_v0/4./N.pi
    lowerPop = N.array(Ite_pop)[alt][itera][_low][0]
    upperPop = N.array(Ite_pop)[alt][itera][_up][0]
    BLU = Blu[_up, _low]
    BUL = Bul[_up, _low]
    # AUL = Ai[_up, _low]
    return (lowerPop*BLU - upperPop*BUL)*_const




