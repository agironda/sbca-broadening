'''
calculate the broadening mechanisms for the following cases:

dE / E = dtheta / tan (B)

Johann:
dtheta_j =  [d_sbca / 2 R sin ( B + a )] ^ 2 / 8 tan (B + a) 

Sagittal:
dtheta_s = 1/2 * (d_sbca / 2 R) ^ 2 / tan^3(B+a) 

In-plane dispersive, along X
dtheta_x = Sx / R sin (B + a)

In-plane off-circle, along Y
dtheta_y = Sy * d_sbca / 2 * R^2

Out-of-plane perpendicular, along Z
dtheta_z = [ Sz / R sin (B + a) ] ^ 2 * tan (B)

'''

import numpy as np

def _calc_resolution(d_theta, bragg):
    # dE / E = dtheta / tan(B)
    resolution = d_theta / np.tan(np.radians(bragg)) 
    return resolution

def _dtheta_x(source_x, curvature, bragg, alpha):
    # dtheta_x = Sx / R sin (B + a)
    dtheta_x = source_x / (curvature * np.sin(np.radians(bragg + alpha)))
    return _dtheta_x

def _dtheta_y(source_y, d_sbca, curvature):
    # dtheta_y = Sy * d_sbca / 2 * R^2
    dtheta_y = source_y * d_sbca / (2 * curvature^2)
    return dtheta_y

def _dtheta_z(source_z, curvature, bragg, alpha):
    # dtheta_z = [ Sz / R sin (B + a) ] ^ 2 * tan (B)
    quant = source_z / (curvature * np.sin(np.radians(bragg+alpha)))
    dtheta_z = quant^2 * np.tan(np.radians(bragg))
    return dtheta_z

def _dtheta_j(d_sbca, curvature, bragg, alpha):
    # dtheta_j =  [d_sbca / 2 R sin ( B + a )] ^ 2 / 8 tan (B + a) 
    quant = d_sbca / (2 * curvature * np.sin(np.radians(bragg + alpha)))
    dtheta_j = quant^2 / (8 * np.tan(np.radians(bragg + alpha)))
    return dtheta_j

def _dtheta_s(d_sbca, curvature, bragg, alpha):
    # dtheta_s = 1/2 * (d_sbca / 2 R) ^ 2 / tan^3(B+a) 
    quant = (d_sbca / (2 * curvature))^2
    dtheta_s =  quant / (2 * np.tan(np.radians(bragg + alpha))^3) 
    return dtheta_s

