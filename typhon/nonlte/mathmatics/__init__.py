import numpy as np

def trapz_inte_edge(i_dy, i_dx):
    """ Trapezoidal integration including edge grids

    Parameters:
        i_dy: Array of indices representing y-axis
        i_dx: Array of indices representing x-axis
    
    Returns:
        Concatenated step array of the axis from first to end
    """
    dx_pre = 0.5 * np.abs(i_dx[1:] - i_dx[:-1])
    dx_center = dx_pre[1:] + dx_pre[:-1]
    dy_center = 0.5 * np.abs(i_dy[1:] + i_dy[:-1])
    trapz_center = 0.5 * (dy_center[1:] + dy_center[:-1]) * dx_center
    first_dydx = 0.5 * (i_dy[0] + dy_center[0]) * dx_center[0]
    last_dydx = 0.5 * (i_dy[-1] + dy_center[-1]) * dx_center[-1]
    inte_trapz = np.r_[first_dydx, trapz_center, last_dydx]
    return inte_trapz
