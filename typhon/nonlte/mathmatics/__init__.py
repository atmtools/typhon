import numpy as np

def Trapz_inte_edge(i_dy, i_dx):
    dx_pre = np.abs((i_dx[1:]-i_dx[:-1])/2.)
    dx_center = dx_pre[1:] + dx_pre[:-1]
    dy_center = np.abs(i_dy[1:] + i_dy[:-1])/2.
    trapz_center = (dy_center[1:] + dy_center[:-1])*dx_center/2.
    First_dydx = (i_dy[0] + dy_center[0])*dx_center[0]/2.
    End_dydx = (i_dy[-1] + dy_center[-1])*dx_center[-1]/2.
    InteTrapz = np.r_[First_dydx, trapz_center, End_dydx]
    return InteTrapz




