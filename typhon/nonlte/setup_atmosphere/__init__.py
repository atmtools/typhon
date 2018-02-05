# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from ..const import *


def swifile(atmosphere):
    """Read Files from SWI mission
    only for H2O molecules
    """
    alt = np.loadtxt(atmosphere+'Alt.out', delimiter=',')  # should be km!!
    pre = np.loadtxt(atmosphere+'Pre.out', delimiter=',')  # should be Pa
    temp = np.loadtxt(atmosphere+'Temp.out', delimiter=',')  # K
    mole = np.loadtxt(atmosphere+'H2O'+'.out', delimiter=',')  # m-3
    alt = np.r_[0, alt]
    pre = np.r_[pre[0], pre]
    temp = np.r_[temp[0], temp]
    mole = np.r_[mole[0], mole]*H*H*O  # isotope in the Earth
    # inter polate
    alt_ref = np.r_[0, alt[1:-1]+4.5/2.]
#==============================================================================
# Templ = np.interp(Alt, Alt_ref, Temp)  # Templ grid same with layer boundary
# Molel = np.interp(Alt, Alt_ref, Mole)  # Temp grid change!!!
# Prel = np.interp(Alt, Alt_ref, Pre)  # Temp grid change!!!
# LayerWidth = 1.
# Alt_ref = np.r_[0, Alt[1:-1]+4.5/2.]
# Alt_new = np.arange(0, Alt[-1], LayerWidth)
# Alt_new = np.arange(0, Alt[-1], LayerWidth)
# Alt =  np.r_[0, Alt_new+LayerWidth/2.]
# Pre = np.interp(Alt_new, Alt_ref, Pre)
# Temp = np.interp(Alt_new, Alt_ref, Temp)
# Mole = np.interp(Alt_new, Alt_ref, Mole)
#==============================================================================
    return alt, pre, temp, mole, alt_ref


def plot_alt_temp_mole(atmosphere):
    alt, pre, temp, mole, alt_ref = swifile(atmosphere)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(mole*1.e-6,alt_ref,'b-')  # ,label='Number density(SSL=60)')
    plt.xlabel('Number density [cm$^{-3}$]',fontsize=24,weight='bold')
    plt.xscale('log')
    plt.ylabel('Altitude [km]',fontsize=24,weight='bold')
    ax2=ax.twiny()
    ax2.plot(temp,alt_ref,'k-', label='Temperature')
    ax2.set_xlabel("Temperature [K]",fontsize=24,weight='bold')
    ax2.plot([],[],'b-', label='H$_{2}$O Number density')
    plt.legend()
    return fig
