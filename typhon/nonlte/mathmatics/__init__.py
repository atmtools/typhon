import numpy as np

def trapz_inte_edge(y, x):
    """ Trapezoidal integration including edge grids

    Parameters:
        y: Array of y-axis value
        x: Array of x-axis value

        Returns:
        Area corresponded to each y (or x) value
            For Example,
             Area corresponded to at y_n is
            ..math:
                0.5*y_n ((x_{n} - x_{n-1}) + (x_{n+1} - x_{n}))
             Area corresponded to at y_0 (start point(edge)) is
            ..math:
                0.5*y_0(x_{1} - x_{0})
    """
    weight_x_0 = 0.5 * (x[1] - x[0])
    weight_x_f = 0.5 * (x[-1] - x[-2])
    weight_x_n = 0.5 * (x[1:-1] - x[:-2]) + 0.5 * (x[2:] - x[1:-1])
    weight_x = np.r_[weight_x_0, weight_x_n, weight_x_f]
    return weight_x*y
