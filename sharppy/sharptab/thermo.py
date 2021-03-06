''' Thermodynamic Library '''
from __future__ import division
import numpy as np
import numpy.ma as ma
import scipy as scipy
import scipy.optimize as opt
from sharppy.sharptab.utils import *
from sharppy.sharptab.constants import *

__all__ = ['drylift', 'thalvl', 'lcltemp', 'theta', 'wobf']
__all__ += ['satlift', 'wetlift', 'lifted', 'lifted_mod', 'vappres', 'mixratio']
__all__ += ['temp_at_mixrat', 'wetbulb', 'thetaw', 'thetaws', 'thetae', 'thetaes']
__all__ += ['thetad', 'thetads', 'thetav', 'virtemp', 'sat_temp', 'relh']
__all__ += ['ftoc', 'ctof', 'ctok', 'ktoc', 'ftok', 'ktof']


# Constants Used
c1 = 0.0498646455 ; c2 = 2.4082965 ; c3 = 7.07475
c4 = 38.9114 ; c5 = 0.0915 ; c6 = 1.2035
eps = 0.62197

def drylift(p, t, td):
    '''
    Lifts a parcel to the LCL and returns its new level and temperature.

    Parameters
    ----------
    p : number, numpy array
        Pressure of initial parcel in hPa
    t : number, numpy array
        Temperature of inital parcel in C
    td : number, numpy array
        Dew Point of initial parcel in C

    Returns
    -------
    p2 : number, numpy array
        LCL pressure in hPa
    t2 : number, numpy array
        LCL Temperature in C

    '''
    t2 = lcltemp(t, td)
    p2 = thalvl(theta(p, t, 1000.), t2)
    return p2, t2


def lcltemp(t, td):
    '''
    Returns the temperature (C) of a parcel when raised to its LCL.

    Parameters
    ----------
    t : number, numpy array
        Temperature of the parcel (C)
    td : number, numpy array
        Dewpoint temperature of the parcel (C)

    Returns
    -------
    Temperature (C) of the parcel at it's LCL.

    '''
    s = t - td
    dlt = s * (1.2185 + 0.001278 * t + s * (-0.00219 + 1.173e-5 * s -
        0.0000052 * t))
    return t - dlt


def thalvl(theta, t):
    '''
    Returns the level (hPa) of a parcel.

    Parameters
    ----------
    theta : number, numpy array
        Potential temperature of the parcel (C)
    t : number, numpy array
        Temperature of the parcel (C)

    Returns
    -------
    Pressure Level (hPa [float]) of the parcel
    '''

    t = t + ZEROCNK
    theta = theta + ZEROCNK
    return 1000. / (np.power((theta / t),(1./ROCP)))


def theta(p, t, p2=1000.):
    '''
    Returns the potential temperature (C) of a parcel.

    Parameters
    ----------
    p : number, numpy array
        The pressure of the parcel (hPa)
    t : number, numpy array
        Temperature of the parcel (C)
    p2 : number, numpy array (default 1000.)
        Reference pressure level (hPa)
    
    Returns
    -------
    Potential temperature (C)

    '''
    return ((t + ZEROCNK) * np.power((p2 / p),ROCP)) - ZEROCNK


def thetaw(p, t, td):
    '''
    Returns the wetbulb potential temperature (C) of a parcel.

    Parameters
    ----------
    p : number
        The pressure of the parcel (hPa)
    t : number
        Temperature of the parcel (C)
    td : number
        Dew point of parcel (C)

    Returns
    -------
    Wetbulb potential temperature (C)

    '''
    p2, t2 = drylift(p, t, td)
    return wetlift(p2, t2, 1000.)


def thetaws(p, t):
    '''
    Returns the saturation wetbulb potential temperature (C) of
    a parcel. Unlike wetbulb potential temperature (Theta-w),
    which is a function of pressure, temperature, and dewpoint,
    Theta-ws is a function of pressure and temperature only and
    assumes the parcel is completely saturated at the start.

    Parameters
    ----------
    p : number
        The pressure of the parcel (hPa)
    t : number
        Temperature of the parcel (C)

    Returns
    -------
    Saturation wetbulb porential temperature (C)

    '''
    return wetlift(p, t, 1000)


def thetae(p, t, td):
    '''
    Returns the equivalent potential temperature (C) of a parcel.

    Parameters
    ----------
    p : number
        The pressure of the parcel (hPa)
    t : number
        Temperature of the parcel (C)
    td : number
        Dew point of parcel (C)

    Returns
    -------
    Equivalent potential temperature (C)

    '''
    p2, t2 = drylift(p, t, td)
    return theta(100., wetlift(p2, t2, 100.), 1000.)


def thetaes(p, t):
    '''
    Returns the saturation equivalent potential temperature (C)
    of a parcel.  Unlike equivalent potential temperature, which
    is a function of pressure, temperature, and dewpoint, Theta-es
    is a function of pressure and temperature only and assumes the
    parcel is completely saturated at the start.

    Parameters
    ----------
    p : number
        The pressure of the parcel (hPa)
    t : number
        Temperature of the parcel (C)

    Returns
    -------
    Saturation equivalent porential temperature (C)

    '''
    return theta(100., wetlift(p, t, 100.), 1000.)


def thetad(p, td):
    '''
    Returns the dewpoint potential temperature (C) of a parcel.

    Parameters
    ----------
    p : number
        The pressure of the parcel (hPa)
    td : number
        Dew point of parcel (C)

    Returns
    -------
    Dewpoint potential temperature (C)

    '''
    mxr = mixratio(p, td)
    return temp_at_mixrat(mxr, 1000)


def thetads(p, t):
    '''
    Returns the saturation dewpoint potential temperature (C)
    of a parcel.  This is a function of pressure and ambient
    temperature and assumes that the parcel is completely
    saturated at the start.

    Parameters
    ----------
    p : number
        The pressure of the parcel (hPa)
    t : number
        Temperature of the parcel (C)

    Returns
    -------
    Saturation dewpoint potential temperature (C)

    '''
    mxr = mixratio(p, t)
    return temp_at_mixrat(mxr, 1000)


def thetav(p, t, td):
    '''
    Returns the virtual potential temperature (C) of a parcel.
    If td is masked, then it returns the temperature passed to
    the function.

    Parameters
    ----------
    p : number
        The pressure of the parcel (hPa)
    t : number
        Temperature of the parcel (C)
    td : number
        Dew point of parcel (C)

    Returns
    -------
    Virtual potential Temperature (C)

    '''
    return theta(p, virtemp(p, t, td), 1000)


def virtemp(p, t, td):
    '''
    Returns the virtual temperature (C) of a parcel.  If 
    td is masked, then it returns the temperature passed to the 
    function.

    Parameters
    ----------
    p : number
        The pressure of the parcel (hPa)
    t : number
        Temperature of the parcel (C)
    td : number
        Dew point of parcel (C)

    Returns
    -------
    Virtual temperature (C)

    '''
    
    tk = t + ZEROCNK
    w = 0.001 * mixratio(p, td)
    vt = (tk * (1. + w / eps) / (1. + w)) - ZEROCNK
    if not QC(vt):
        return t
    else:
        return vt

def sat_temp(p, vt):
    '''
    For a given virtual temperature (C), returns the saturated
    ambient temperature (C) of a parcel (i.e. the ambient temperature
    at which the parcel is saturated).

    Parameters
    ----------
    p : number
        The pressure of the parcel (hPa)
    vt : number
        Virtual temperature of the parcel (C)

    Returns
    -------
    Saturated ambient temperature (C)

    '''

    def sat_f(t, p, vt):
        return virtemp(p, t, t) - vt
    sat_temp = opt.newton(sat_f, vt, args=(p, vt), tol=1e-12, maxiter=50)
    return sat_temp

def relh(p, t, td):
    '''
    Returns the relative humidity (%) of a parcel.

    Parameters
    ----------
    p : number
        The pressure of the parcel (hPa)
    t : number
        Temperature of the parcel (C)
    td : number
        Dew point of parcel (C)

    Returns
    -------
    Relative humidity (%) of a parcel

    '''
    return 100. * mixratio(p, td) / mixratio(p, t)

def wobf(t):
    '''
    Implementation of the Wobus Function for computing the moist adiabats.

    Parameters
    ----------
    t : number, numpy array
        Temperature (C)

    Returns
    -------
    Correction to theta (C) for calculation of saturated potential temperature.

    '''
    t = t - 20
    if type(t) == type(np.array([])) or type(t) == type(np.ma.array([])):
        npol = 1. + t * (-8.841660499999999e-3 + t * ( 1.4714143e-4 + t * (-9.671989000000001e-7 + t * (-3.2607217e-8 + t * (-3.8598073e-10)))))
        npol = 15.13 / (np.power(npol,4))
        ppol = t * (4.9618922e-07 + t * (-6.1059365e-09 + t * (3.9401551e-11 + t * (-1.2588129e-13 + t * (1.6688280e-16)))))
        ppol = 1 + t * (3.6182989e-03 + t * (-1.3603273e-05 + ppol))
        ppol = (29.93 / np.power(ppol,4)) + (0.96 * t) - 14.8
        correction = np.zeros(t.shape, dtype=np.float64)
        correction[t <= 0] = npol[t <= 0]
        correction[t > 0] = ppol[t > 0]
        return correction
    else:
        if t is np.ma.masked:
            return t
        if t <= 0:
            npol = 1. + t * (-8.841660499999999e-3 + t * ( 1.4714143e-4 + t * (-9.671989000000001e-7 + t * (-3.2607217e-8 + t * (-3.8598073e-10)))))
            npol = 15.13 / (np.power(npol,4))
            return npol
        else:
            ppol = t * (4.9618922e-07 + t * (-6.1059365e-09 + t * (3.9401551e-11 + t * (-1.2588129e-13 + t * (1.6688280e-16)))))
            ppol = 1 + t * (3.6182989e-03 + t * (-1.3603273e-05 + ppol))
            ppol = (29.93 / np.power(ppol,4)) + (0.96 * t) - 14.8
            return ppol



def satlift(p, thetam):
    '''
    Returns the temperature (C) of a saturated parcel (thm) when lifted to a
    new pressure level (hPa)

    Parameters
    ----------
    p : number
        Pressure to which parcel is raised (hPa)
    thetam : number
        Saturated Potential Temperature of parcel (C)

    Returns
    -------
    Temperature (C) of saturated parcel at new level

    '''
    #if type(p) == type(np.array([p])) or type(thetam) == type(np.array([thetam])):
    if np.fabs(p - 1000.) - 0.001 <= 0: return thetam
    eor = 999
    while np.fabs(eor) - 0.1 > 0:
        if eor == 999:                  # First Pass
            pwrp = np.power((p / 1000.),ROCP)
            t1 = (thetam + ZEROCNK) * pwrp - ZEROCNK
            e1 = wobf(t1) - wobf(thetam)
            rate = 1
        else:                           # Successive Passes
            rate = (t2 - t1) / (e2 - e1)
            t1 = t2
            e1 = e2
        t2 = t1 - (e1 * rate)
        e2 = (t2 + ZEROCNK) / pwrp - ZEROCNK
        e2 += wobf(t2) - wobf(e2) - thetam
        eor = e2 * rate
    return t2 - eor


def wetlift(p, t, p2):
    '''
    Lifts a parcel moist adiabatically to its new level.

    Parameters
    -----------
    p : number
        Pressure of initial parcel (hPa)
    t : number
        Temperature of initial parcel (C)
    p2 : number
        Pressure of final level (hPa)

    Returns
    -------
    Temperature (C)

    '''
    thta = theta(p, t, 1000.)
    if thta is np.ma.masked or p2 is np.ma.masked:
        return np.ma.masked
    thetam = thta - wobf(thta) + wobf(t)
    return satlift(p2, thetam)



def lifted(p, t, td, lev):
    '''
    Calculate temperature (C) of parcel (defined by p, t, td) lifted
    to the specified pressure level.

    Parameters
    ----------
    p : number
        Pressure of initial parcel in hPa
    t : number
        Temperature of initial parcel in C
    td : number
        Dew Point of initial parcel in C
    lev : number
        Pressure to which parcel is lifted in hPa

    Returns
    -------
    Temperature (C) of lifted parcel

    '''
    p2, t2 = drylift(p, t, td)
    return wetlift(p2, t2, lev)


def lifted_mod(p, t, td, lev):
    '''
    A modification of the "lifted" function (q.v.) that calculates the
    ambient and dewpoint temperatures (C) of a parcel lifted to the
    specified level.  First, the level of the Lifted Condensation Level
    (LCL) is determined.  If the level to be lifted to lies below the
    LCL, then the parcel is only lifted dry-adiabatically to its new
    level.  If the level to be lifted lies at or above the LCL, then
    the parcel is lifted first dry-adiabatically to the LCL, then
    lifted moist-adiabatically to its new level.

    Parameters
    ----------
    p : number, numpy array
        Pressure of initial parcel in hPa
    t : number, numpy array
        Temperature of inital parcel in C
    td : number, numpy array
        Dew Point of initial parcel in C

    Returns
    -------
    t2 : number, numpy array
        Temperature in C
    td2 : number, numpy array
        Dew Point temperature in C
    '''
    plcl = drylift(p, t, td)[0]
    if lev > plcl:
        t2 = theta(p, t, lev)
        mxr_p = mixratio(p, td)
        td2 = temp_at_mixrat(mxr_p, lev)
        return t2, td2
    else:
        t2 = lifted(p, t, td, lev)
        td2 = t2
        return t2, td2


def vappres(t):
    '''
    Returns the vapor pressure of dry air at given temperature

    Parameters
    ------
    t : number, numpy array
        Temperature of the parcel (C)

    Returns
    -------
    Vapor Pressure of dry air

    '''
    pol = t * (1.1112018e-17 + (t * -3.0994571e-20))
    pol = t * (2.1874425e-13 + (t * (-1.789232e-15 + pol)))
    pol = t * (4.3884180e-09 + (t * (-2.988388e-11 + pol)))
    pol = t * (7.8736169e-05 + (t * (-6.111796e-07 + pol)))
    pol = 0.99999683 + (t * (-9.082695e-03 + pol))
    return 6.1078 / pol**8


def mixratio(p, t):
    '''
    Returns the mixing ratio (g/kg) of a parcel

    Parameters
    ----------
    p : number, numpy array
        Pressure of the parcel (hPa)
    t : number, numpy array
        Temperature of the parcel (hPa)

    Returns
    -------
    Mixing Ratio (g/kg) of the given parcel

    '''
    x = 0.02 * (t - 12.5 + (7500. / p))
    wfw = 1. + (0.0000045 * p) + (0.0014 * x * x)
    fwesw = wfw * vappres(t)
    return 621.97 * (fwesw / (p - fwesw))


def temp_at_mixrat(w, p):
    '''
    Returns the temperature (C) of air at the given mixing ratio (g/kg) and
    pressure (hPa)

    Parameters
    ----------
    w : number, numpy array
        Mixing Ratio (g/kg)
    p : number, numpy array
        Pressure (hPa)

    Returns
    -------
    Temperature (C) of air at given mixing ratio and pressure
    '''
    x = np.log10(w * p / (622. + w))
    x = (np.power(10.,((c1 * x) + c2)) - c3 + (c4 * np.power((np.power(10,(c5 * x)) - c6),2))) - ZEROCNK
    return x


def wetbulb(p, t, td):
    '''
    Calculates the wetbulb temperature (C) for the given parcel

    Parameters
    ----------
    p : number
        Pressure of parcel (hPa)
    t : number
        Temperature of parcel (C)
    td : number
        Dew Point of parcel (C)

    Returns
    -------
    Wetbulb temperature (C)
    '''
    p2, t2 = drylift(p, t, td)
    return wetlift(p2, t2, p)


def ctof(t):
    '''
    Convert temperature from Celsius to Fahrenheit

    Parameters
    ----------
    t : number, numpy array
        The temperature in Celsius

    Returns
    -------
    Temperature in Fahrenheit (number or numpy array)

    '''
    return (1.8 * t) + 32.


def ftoc(t):
    '''
    Convert temperature from Fahrenheit to Celsius

    Parameters
    ----------
    t : number, numpy array
        The temperature in Fahrenheit

    Returns
    -------
    Temperature in Celsius (number or numpy array)

    '''
    return (t - 32.) * (5. / 9.)


def ktoc(t):
    '''
    Convert temperature from Kelvin to Celsius

    Parameters
    ----------
    t : number, numpy array
        The temperature in Kelvin

    Returns
    -------
    Temperature in Celsius (number or numpy array)

    '''
    return t - ZEROCNK


def ctok(t):
    '''
    Convert temperature from Celsius to Kelvin

    Parameters
    ----------
    t : number, numpy array
        The temperature in Celsius

    Returns
    -------
    Temperature in Kelvin (number or numpy array)

    '''
    return t + ZEROCNK


def ktof(t):
    '''
    Convert temperature from Kelvin to Fahrenheit

    Parameters
    ----------
    t : number, numpy array
        The temperature in Kelvin

    Returns
    -------
    Temperature in Fahrenheit (number or numpy array)

    '''
    return ctof(ktoc(t))


def ftok(t):
    '''
    Convert temperature from Fahrenheit to Kelvin

    Parameters
    ----------
    t : number, numpy array
        The temperature in Fahrenheit

    Returns
    -------
    Temperature in Kelvin (number or numpy array)

    '''
    return ctok(ftoc(t))
