'''Fire Parameter Routines'''
from __future__ import division
from sharppy.sharptab import thermo, utils, winds, interp
import numpy as np
import numpy.ma as ma
from sharppy.sharptab.constants import *


__all__ = ['fosberg', 'haines_la', 'haines_ma', 'haines_ha', 'cbi']

## Routines implemented in Python by Greg Blumberg - CIMMS and Kelton Halbert (OU SoM)
## wblumberg@ou.edu, greg.blumberg@noaa.gov, kelton.halbert@noaa.gov, keltonhalbert@ou.edu

def fosberg(prof):
    '''
        The Fosberg Fire Weather Index
        Adapted from code donated by Rich Thompson - NOAA Storm Prediction Center

        Description:
        The FWI (Fire Weather Index) is defined by a quantitative model that provides
        a nonlinear filter of meteorological data which results in a linear relationship
        between the combined meteorological variables of relative humidity and wind speed,
        and the behavior of wildfires. Thus the index deals with only the weather conditions,
        not the fuels. Several sets of conditions have been defined by Fosberg (Fosberg, 1978)
        to apply this to fire weather management. The upper limits have been set to give an
        index value of 100 if the moisture content is zero and the wind is 30 mph. 

        Thus, the numbers range from 0 to 100 and if any number is larger than 100, it is set back to 100. 
        The index can be used to measure changes in fire weather conditions. Over several years of use, 
        Fosberg index values of 50 or greater generally appear significant on a national scale.
        The SPC fire weather verification scheme uses the Fosberg Index, but with a check for
        both temperature (60F) and adjective fire danger rating (3-High, 4-Very High, 5-Extreme).

        Description Source - http://www.spc.noaa.gov/exper/firecomp/INFO/fosbinfo.html

        WARNING: This function has not been fully tested.

        Parameters
        ----------
        prof - Profile object

        Returns
        -------
        fosberg - the Fosberg Fire Weather Index

    '''
    tmpf = thermo.ctof(prof.tmpc[prof.get_sfc()])
    fmph = utils.KTS2MPH(prof.wspd[prof.get_sfc()])

    rh = thermo.relh(prof.pres[prof.sfc], prof.tmpc[prof.sfc], prof.dwpc[prof.sfc])
    if (rh <= 10):
        em = 0.03229 + 0.281073*rh - 0.000578*rh*tmpf
    elif (10 < rh <= 50):
        em = 2.22749 + 0.160107*rh - 0.014784*tmpf
    else:
        em = 21.0606 + 0.005565*rh*rh - 0.00035*rh*tmpf - 0.483199*rh

    em30 = em/30
    u_sq = fmph*fmph
    fmdc = 1 - 2*em30 + 1.5*em30*em30 - 0.5*em30*em30*em30

    fosberg = (fmdc*np.sqrt(1+u_sq))/0.3002

    return fosberg

def haines_la(prof):
    '''
        Haines Index, Low Altitude

        The Haines Index (or Lower Atmospheric Severity Index) was developed in order to help forecast fire "blow up"
        potential.  It is a function of lower atmospheric stability and moisture content.  Three versions were
        developed, to be used depending on the altitude of the surface.  Values of 2 or 3 indicate very low "blow up"
        potential; a value of 4 indicates low potential; a value of 5 indicates moderate potential; and a value of
        6 indicates high potential.

        The low altitude version was developed primarily for use when the surface level pressure is 950 mb or higher.

        Parameters
        ----------
        prof - Profile object

        Returns
        -------
        haines_la : number
            Haines Index, Low Altitude (number)
    '''

    tmp950 = interp.temp(prof, 950)
    tmp850 = interp.temp(prof, 850)
    dpt850 = interp.dwpt(prof, 850)

    lr98 = tmp950 - tmp850
    tdd850 = tmp850 - dpt850

    if lr98 <= 3:
        stab_tm = 1
    elif 3 < lr98 and lr98 < 8:
        stab_tm = 2
    else:
        stab_tm = 3
    
    if tdd850 <= 5:
        mois_tm = 1
    elif 5 < tdd850 and tdd850 < 10:
        mois_tm = 2
    else:
        mois_tm = 3
    
    haines_la = stab_tm + mois_tm

    return haines_la

def haines_ma(prof):
    '''
        Haines Index, Middle Altitude

        The Haines Index (or Lower Atmospheric Severity Index) was developed in order to help forecast fire "blow up"
        potential.  It is a function of lower atmospheric stability and moisture content.  Three versions were
        developed, to be used depending on the altitude of the surface.  Values of 2 or 3 indicate very low "blow up"
        potential; a value of 4 indicates low potential; a value of 5 indicates moderate potential; and a value of
        6 indicates high potential.

        The middle altitude version was developed primarily for use when the surface level pressure is between 950
        and 850 mb.

        Parameters
        ----------
        prof - Profile object

        Returns
        -------
        haines_ma : number
            Haines Index, Middle Altitude (number)
    '''

    tmp850 = interp.temp(prof, 850)
    tmp700 = interp.temp(prof, 700)
    dpt850 = interp.dwpt(prof, 850)

    lr87 = tmp850 - tmp700
    tdd850 = tmp850 - dpt850

    if lr87 <= 5:
        stab_tm = 1
    elif 5 < lr87 and lr87 < 11:
        stab_tm = 2
    else:
        stab_tm = 3
    
    if tdd850 <= 5:
        mois_tm = 1
    elif 5 < tdd850 and tdd850 < 13:
        mois_tm = 2
    else:
        mois_tm = 3
    
    haines_ma = stab_tm + mois_tm

    return haines_ma

def haines_ha(prof):
    '''
        Haines Index, High Altitude

        The Haines Index (or Lower Atmospheric Severity Index) was developed in order to help forecast fire "blow up"
        potential.  It is a function of lower atmospheric stability and moisture content.  Three versions were
        developed, to be used depending on the altitude of the surface.  Values of 2 or 3 indicate very low "blow up"
        potential; a value of 4 indicates low potential; a value of 5 indicates moderate potential; and a value of
        6 indicates high potential.

        The low altitude version was developed primarily for use when the surface level pressure is between 850 and
        700 mb.

        Parameters
        ----------
        prof - Profile object

        Returns
        -------
        haines_ha : number
            Haines Index, High Altitude (number)
    '''

    tmp700 = interp.temp(prof, 700)
    tmp500 = interp.temp(prof, 500)
    dpt700 = interp.dwpt(prof, 700)

    lr75 = tmp700 - tmp500
    tdd700 = tmp700 - dpt700

    if lr75 <= 17:
        stab_tm = 1
    elif 17 < lr75 and lr75 < 22:
        stab_tm = 2
    else:
        stab_tm = 3
    
    if tdd700 <= 14:
        mois_tm = 1
    elif 14 < tdd700 and tdd700 < 21:
        mois_tm = 2
    else:
        mois_tm = 3
    
    haines_ha = stab_tm + mois_tm

    return haines_ha

def cbi(prof):
    '''
        Chandler Burning Index

        This index uses air temperature and relative humidity to create a numerical index of fire danger.
        It is based solely on weather conditions, with no adjustment for fuel moisture.  Values of less
        than 50 indicate low fire danger; Values between 50 and 75 indicate moderate fire danger; values
        between 75 and 90 indicate high fire danger; values between 90 and 97.5 indicate very high fire
        danger; and values above 97.5 indicate extreme fire danger.

        Parameters
        ----------
        prof - Profile object

        Returns
        -------
        cbi : number
            Chandler Burning Index (number)
    '''

    sfc_tmp = prof.tmpc[prof.sfc]
    sfc_rh = thermo.relh(prof.pres[prof.sfc], prof.tmpc[prof.sfc], prof.dwpc[prof.sfc])

    cbi = ((( 110 - (1.373 * sfc_rh)) - (0.54 * (10.2 - sfc_tmp))) * (124 * (10 ** (-0.0142 * sfc_rh)))) / 60

    return cbi
