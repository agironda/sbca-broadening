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

def surface_broadening(
        *,
        diam_sbca,
        bending_radius,
        bragg,
        alpha=0.0,
        energy=None,
        output="rad",
        ):
    params = dict(
        diam_sbca=diam_sbca,
        bending_radius=bending_radius,
        bragg=bragg,
        alpha=alpha,
        energy=energy,
        output=output,
    )

    johann = johann_broadening(**params)
    sagittal = sagittal_broadening(**params)

    return {
        "johann": johann,
        "sagittal": sagittal,
        "total": _quadrature(johann, sagittal),
    }

def source_broadening_xyz(
        *,
        source_x,
        source_y,
        source_z,
        diam_sbca,
        bending_radius,
        bragg,
        alpha=0.0,
        energy=None,
        output="rad",
        ):
    params = dict(
        bending_radius=bending_radius,
        bragg=bragg,
        alpha=alpha,
        energy=energy,
        output=output,
    )

    x = source_broadening_x(source_x=source_x, **params)
    y = source_broadening_y(
        source_y=source_y,
        diam_sbca=diam_sbca,
        bending_radius=bending_radius,
        bragg=bragg,
        energy=energy,
        output=output,
    )
    z = source_broadening_z(source_z=source_z, **params)

    return {
        "x": x,
        "y": y,
        "z": z,
        "total": _quadrature(x, y, z),
    }

def geometric_broadening(
        *,
        source_x,
        source_y,
        source_z,
        diam_sbca,
        bending_radius,
        bragg,
        alpha=0.0,
        energy=None,
        output="rad",
        ):
    params = dict(
        diam_sbca=diam_sbca,
        bending_radius=bending_radius,
        bragg=bragg,
        alpha=alpha,
        energy=energy,
        output=output,
    )

    source = source_broadening_xyz(
        source_x=source_x,
        source_y=source_y,
        source_z=source_z,
        **params,
    )

    surface = surface_broadening(**params)

    return {
        "source": source,
        "surface": surface,
        "total": _quadrature(source["total"], surface["total"]),
    }

def johann_broadening(
        *, 
        diam_sbca, 
        bending_radius, 
        bragg, 
        alpha=0.0, 
        energy=None, 
        output='rad'): 
    dtheta = _dtheta_j(diam_sbca, bending_radius, bragg, alpha)
    return _convert_broadening(dtheta, bragg, energy, output)

def sagittal_broadening(
        *, 
        diam_sbca, 
        bending_radius, 
        bragg, 
        alpha=0.0, 
        energy=None, 
        output='rad'
        ): 
    dtheta = _dtheta_s(diam_sbca, bending_radius, bragg, alpha)
    return _convert_broadening(dtheta, bragg, energy, output)

def source_broadening_x(
        *, 
        source_x, 
        bending_radius, 
        bragg, 
        alpha=0.0, 
        energy=None, 
        output='rad'
        ): 
    dtheta = _dtheta_x(source_x, bending_radius, bragg, alpha)
    return _convert_broadening(dtheta, bragg, energy, output)

def source_broadening_y(
        *, 
        source_y,
        diam_sbca, 
        bending_radius, 
        bragg, 
        energy=None, 
        output='rad'
        ): 
    dtheta = _dtheta_y(source_y, diam_sbca, bending_radius)
    return _convert_broadening(dtheta, bragg, energy, output)

def source_broadening_z(
        *, 
        source_z, 
        bending_radius, 
        bragg, 
        alpha=0.0, 
        energy=None, 
        output='rad'
        ): 
    dtheta = _dtheta_z(source_z, bending_radius, bragg, alpha)
    return _convert_broadening(dtheta, bragg, energy, output)

def _convert_broadening(dtheta, bragg, energy=None, output='rad'):
    '''Convert angular broadening to the requested units.'''
    dtheta = np.abs(dtheta)

    if output == 'rad':
        return dtheta

    if output == 'deg':
        return np.degrees(dtheta)

    relative = dtheta / np.tan(np.radians(bragg))

    if output == 'relative':
        return relative

    if output in ('eV', 'meV'):
        if energy is None:
            raise ValueError('energy must be provided for eV or meV output')

        dE = relative * energy

        if output == 'meV':
            dE *= 1000

        return dE

    raise ValueError(f'Invalid output: {output}')


def _dtheta_x(source_x, curvature, bragg, alpha):
    '''Angular broadening from the in-plane dispersive source dimension.'''
    # dtheta_x = Sx / R sin (B + a)
    dtheta_x = source_x / (curvature * np.sin(np.radians(bragg + alpha)))
    return dtheta_x

def _dtheta_y(source_y, d_sbca, curvature):
    '''Angular broadening from the in-plane transverse source dimension.'''
    # dtheta_y = Sy * d_sbca / 2 * R^2
    dtheta_y = source_y * d_sbca / (2 * curvature**2)
    return dtheta_y

def _dtheta_z(source_z, curvature, bragg, alpha):
    '''Angular broadening from the out-of-plane source dimension.'''
    # dtheta_z = [ Sz / R sin (B + a) ] ^ 2 * tan (B)
    quant = source_z / (curvature * np.sin(np.radians(bragg+alpha)))
    dtheta_z = quant**2 * np.tan(np.radians(bragg))
    return dtheta_z

def _dtheta_j(d_sbca, curvature, bragg, alpha):
    '''Angular broadening from meridional Johann error along SBCA surface.'''
    # dtheta_j =  [d_sbca / 2 R sin ( B + a )] ^ 2 / 8 tan (B + a) 
    quant = d_sbca / (2 * curvature * np.sin(np.radians(bragg + alpha)))
    dtheta_j = quant**2 / (8 * np.tan(np.radians(bragg + alpha)))
    return dtheta_j

def _dtheta_s(d_sbca, curvature, bragg, alpha):
    '''Angular broadening from sagittal error along SBCA surface.'''
    # dtheta_s = 1/2 * (d_sbca / 2 R) ^ 2 / tan^3(B+a) 
    quant = (d_sbca / (2 * curvature))**2
    dtheta_s =  quant / (2 * np.tan(np.radians(bragg + alpha))**3) 
    return dtheta_s

def _quadrature(*values):
    '''Calculate the root-sum-square of broadening contributions.'''
    return np.sqrt(sum(np.asarray(v)**2 for v in values))

