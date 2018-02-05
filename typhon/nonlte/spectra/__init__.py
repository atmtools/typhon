# -*- coding: utf-8 -*-


import numpy as np
from scipy import constants as const
from . import *
from .lineshape import DLV

def calc_abscoeff(Tran_tag, Freqi, Ite_pop, Blu, Bul, tran, itera, alt=None):
    """Function for calculation the absorption coefficient(s)
    
    .. math::
        x = \\sigma^3
        
    More info
    
    Parameters:
        Tran_tag: lol
        
        Freqi: lal
        
    Returns:
        Absorption coefficient(s)
    """
    _up, _low = Tran_tag[tran][1:].astype(int)
    _v0 = Freqi[_up, _low]*1.e9
    _const = const.h*_v0/4./np.pi
    if alt is None:
        lowerPop = np.array(Ite_pop)[:, itera, _low, 0]
        upperPop = np.array(Ite_pop)[:, itera, _up, 0]
    else:
        lowerPop = np.array(Ite_pop)[alt][itera][_low][0]
        upperPop = np.array(Ite_pop)[alt][itera][_up][0]
    BLU = Blu[_up, _low]
    BUL = Bul[_up, _low]
    # AUL = Ai[_up, _low]
    return (lowerPop*BLU - upperPop*BUL)*_const
