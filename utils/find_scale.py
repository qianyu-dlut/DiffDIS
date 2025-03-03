import numpy as np
from scipy.optimize import minimize, least_squares

def mse_scaled_shifted(scale_shift, A, B):
    """ 
    Calculate the mean squared error between the scaled and shifted A and B.
    scale_shift[0] is the scale factor, scale_shift[1] is the shift factor.
    """
    scale, shift = scale_shift
    A_transformed = scale * A + shift
    # target = np.mean((A_transformed - B) ** 2)
    # target = np.sqrt(np.mean((A_transformed - B) ** 2))
    # target = np.mean(np.abs(A_transformed - B))
    target = np.mean(np.abs(A_transformed - B) / B)
    
    return target

def find_scale_shift(A, B):
    """
    Find the scale and shift factors to minimize the difference between A and B.
    """
    initial_guess = [80, 0]  # Initial guess for scale and shift
    result = minimize(mse_scaled_shifted, 
                      initial_guess, 
                      args=(A, B),
                      method='Nelder-Mead',
                      options={'maxiter': 5000, 'xatol': 1e-8, 'yatol': 1e-8, 'fatol': 1e-8})
    # result = least_squares(mse_scaled_shifted,
    #                        x0=initial_guess,
    #                        args=(A, B),
    #                        verbose=1, x_scale='jac',
    #                        ftol=1e-8, xtol=1e-8, gtol=1e-8,
    #                        method='dogbox', max_nfev=5000)

    scale, shift = result.x
    return scale, shift

if __name__=='__main__':
    # Example usage
    A = np.array([3, 4, 1, 8, 4, 0, 4, 5])  # Your depth map A
    B = np.array([9, 5, 4, 3, 6, 8, 9, 2])*80  # Your depth map B

    scale, shift = find_scale_shift(A, B)
    print(f"Scale: {scale}, Shift: {shift}")
