''' Thermodynamic Parameter Routines '''
from __future__ import division
import numpy as np
import numpy.ma as ma
from sharppy.sharptab import interp, utils, thermo, winds
from sharppy.sharptab.constants import *

'''
    This file contains various functions to perform the calculation of various convection indices.
    Because of this, parcel lifting routines are also found in this file.
    Functions denoted with a (*) in the docstring refer to functions that were added to the SHARPpy package that 
    were not ported from the Storm Prediction Center.  They have been included as they have been used by the 
    community in an effort to expand SHARPpy to support the many parameters used in atmospheric science. 
    
    While the logic for these functions are based in the scientific literature, validation
    of the output from these functions is occasionally difficult to perform.  Although we have made an effort
    to resolve code issues when they arise, values from these functions may be erronious and may require additional 
    inspection by the user.  We appreciate any contributions by the meteorological community that can
    help better validate these SHARPpy functions!
    
'''

__all__ = ['DefineParcel', 'Parcel', 'inferred_temp_adv']
__all__ += ['k_index', 't_totals', 'c_totals', 'v_totals', 'precip_water']
__all__ += ['inversion', 'temp_lvl', 'max_temp', 'mean_mixratio', 'mean_theta', 'mean_thetae', 'mean_thetaes', 'mean_thetaw', 'mean_thetaws', 'mean_relh']
__all__ += ['lapse_rate', 'most_unstable_level', 'parcelx', 'bulk_rich']
__all__ += ['bunkers_storm_motion', 'effective_inflow_layer']
__all__ += ['convective_temp', 'esp', 'pbl_top', 'precip_eff', 'dcape', 'sig_severe']
__all__ += ['dgz', 'ship', 'stp_cin', 'stp_fixed', 'scp', 'mmp', 'wndg', 'sherb', 'tei', 'cape']
__all__ += ['mburst', 'dcp', 'ehi', 'sweat', 'hgz', 'lhp']
__all__ += ['spot', 'wbz', 'thomp', 'tq', 's_index', 'boyden', 'dci', 'pii', 'ko', 'brad', 'rack', 'jeff', 'sc_totals']
__all__ += ['esi', 'vgp', 'aded1', 'aded2', 'ei', 'eehi', 'vtp']
__all__ += ['snsq', 'snow']
__all__ += ['windex', 'wmsi', 'dmpi1', 'dmpi2', 'hmi', 'mwpi']
__all__ += ['hi', 'ulii', 'ssi', 'csv', 'z_index', 'k_high1', 'k_high2', 'swiss00', 'swiss12', 'fin', 'yon1', 'yon2']
__all__ += ['fsi', 'fog_point', 'fog_threat']
__all__ += ['mvv', 'tsi', 'jli', 'ncape', 'ncinh', 'lsi', 'mcsi1', 'mcsi2', 'cii1', 'cii2']
__all__ += ['cpst1', 'cpst2', 'cpst3']
__all__ += ['tie']
__all__ += ['t1_gust']

class DefineParcel(object):
    '''
        Create a parcel from a supplied profile object.
        
        Parameters
        ----------
        prof : profile object
        Profile object
        
        Optional Keywords
        flag : int (default = 1)
        Parcel Selection
        1: Observed Surface Parcel
        2: Forecast Surface Parcel
        3: Most Unstable Parcel
        4: Mean Mixed Layer Parcel
        5: User Defined Parcel
        6: Mean Effective Layer Parcel
        7: Convective Temperature Parcel
        
        Optional Keywords (Depending on Parcel Selected)
        Parcel (flag) == 1: Observed Surface Parcel
        None
        Parcel (flag) == 2: Forecast Surface Parcel
        pres : number (default = 100 hPa)
        Depth over which to mix the boundary layer; only changes
        temperature; does not affect moisture
        Parcel (flag) == 3: Most Unstable Parcel
        pres : number (default = 400 hPa)
        Depth over which to look for the the most unstable parcel
        starting from the surface pressure
        Parcel (flag) == 4: Mixed Layer Parcel
        pres : number (default = 100 hPa)
        Depth over which to mix the surface parcel
        Parcel (flag) == 5: User Defined Parcel
        pres : number (default = SFC - 100 hPa)
        Pressure of the parcel to lift
        tmpc : number (default = Temperature at the provided pressure)
        Temperature of the parcel to lift
        dwpc : number (default = Dew Point at the provided pressure)
        Dew Point of the parcel to lift
        Parcel (flag) == 6: Effective Inflow Layer
        ecape : number (default = 100)
        The minimum amount of CAPE a parcel needs to be considered
        part of the inflow layer
        ecinh : number (default = -250)
        The maximum amount of CINH allowed for a parcel to be
        considered as part of the inflow layer
        Parcel (flag) == 7: Convective Temperature Parcel
        pres : number (default = 100 hPa)
        Depth over which to mix the boundary layer; only changes
        temperature; does not affect moisture
        
        '''
    def __init__(self, prof, flag, **kwargs):
        self.flag = flag
        if flag == 1:
            self.presval = prof.pres[prof.sfc]
            self.__sfc(prof)
        elif flag == 2:
            self.presval = kwargs.get('pres', 100)
            self.__fcst(prof, **kwargs)
        elif flag == 3:
            self.presval = kwargs.get('pres', 300)
            self.__mu(prof, **kwargs)
        elif flag == 4:
            self.presval = kwargs.get('pres', 100)
            self.__ml(prof, **kwargs)
        elif flag == 5:
            self.presval = kwargs.get('pres', prof.pres[prof.sfc])
            self.__user(prof, **kwargs)
        elif flag == 6:
            self.presval = kwargs.get('pres', 100)
            self.__effective(prof, **kwargs)
        elif flag == 7:
            self.presval = kwargs.get('pres', 100)
            self.__convective(prof, **kwargs)
        else:
            #print 'Defaulting to Surface Parcel'
            self.presval = kwargs.get('pres', prof.gSndg[prof.sfc])
            self.__sfc(prof)
    
    
    def __sfc(self, prof):
        '''
            Create a parcel using surface conditions
            
            '''
        self.desc = 'Surface Parcel'
        self.pres = prof.pres[prof.sfc]
        self.tmpc = prof.tmpc[prof.sfc]
        self.dwpc = prof.dwpc[prof.sfc]
    
    
    def __fcst(self, prof, **kwargs):
        '''
            Create a parcel using forecast conditions.
            
            '''
        self.desc = 'Forecast Surface Parcel'
        self.tmpc = max_temp(prof)
        self.pres = prof.pres[prof.sfc]
        pbot = self.pres; ptop = self.pres - 100.
        self.dwpc = thermo.temp_at_mixrat(mean_mixratio(prof, pbot, ptop, exact=True), self.pres)
    
    
    def __mu(self, prof, **kwargs):
        '''
            Create the most unstable parcel within the lowest XXX hPa, where
            XXX is supplied. Default XXX is 400 hPa.
            
            '''
        self.desc = 'Most Unstable Parcel in Lowest %.2f hPa' % self.presval
        pbot = prof.pres[prof.sfc]
        ptop = pbot - self.presval
        self.pres = most_unstable_level(prof, pbot=pbot, ptop=ptop)
        self.tmpc = interp.temp(prof, self.pres)
        self.dwpc = interp.dwpt(prof, self.pres)
    
    
    def __ml(self, prof, **kwargs):
        '''
            Create a mixed-layer parcel with mixing within the lowest XXX hPa,
            where XXX is supplied. Default is 100 hPa.

            If
            
            '''
        self.desc = '%.2f hPa Mixed Layer Parcel' % self.presval
        pbot = kwargs.get('pbot', prof.pres[prof.sfc])
        ptop = pbot - self.presval
        self.pres = pbot
        mtheta = mean_theta(prof, pbot, ptop, exact=True)
        self.tmpc = thermo.theta(1000., mtheta, self.pres)
        mmr = mean_mixratio(prof, pbot, ptop, exact=True)
        self.dwpc = thermo.temp_at_mixrat(mmr, self.pres)
    
    
    def __user(self, prof, **kwargs):
        '''
            Create a user defined parcel.
            
            '''
        self.desc = '%.2f hPa Parcel' % self.presval
        self.pres = self.presval
        self.tmpc = kwargs.get('tmpc', interp.temp(prof, self.pres))
        self.dwpc = kwargs.get('dwpc', interp.dwpt(prof, self.pres))
    
    
    def __effective(self, prof, **kwargs):
        '''
            Create the mean-effective layer parcel.
            
            '''
        ecape = kwargs.get('ecape', 100)
        ecinh = kwargs.get('ecinh', -250)
        pbot, ptop = effective_inflow_layer(prof, ecape, ecinh)
        if utils.QC(pbot) and pbot > 0:
            self.desc = '%.2f hPa Mean Effective Layer Centered at %.2f' % ( pbot-ptop, (pbot+ptop)/2.)
            mtha = mean_theta(prof, pbot, ptop)
            mmr = mean_mixratio(prof, pbot, ptop)
            self.pres = (pbot+ptop)/2.
            self.tmpc = thermo.theta(1000., mtha, self.pres)
            self.dwpc = thermo.temp_at_mixrat(mmr, self.pres)
        else:
            self.desc = 'Defaulting to Surface Layer'
            self.pres = prof.pres[prof.sfc]
            self.tmpc = prof.tmpc[prof.sfc]
            self.dwpc = prof.dwpc[prof.sfc]
        if utils.QC(pbot): self.pbot = pbot
        else: self.pbot = ma.masked
        if utils.QC(ptop): self.ptop = ptop
        else: self.pbot = ma.masked
    

    def __convective(self, prof, **kwargs):
        '''
            Create the convective temperature parcel.

            '''
        self.desc = 'Convective Temperature Parcel'
        self.tmpc = convective_temp(prof, **kwargs)
        self.pres = prof.pres[prof.sfc]
        pbot = self.pres; ptop = self.pres - 100.
        self.dwpc = thermo.temp_at_mixrat(mean_mixratio(prof, pbot, ptop, exact=True), self.pres)


class Parcel(object):
    '''
        Initialize the parcel variables
        
        Parameters
        ----------
        pbot : number
        Lower-bound (pressure; hPa) that the parcel is lifted
        ptop : number
        Upper-bound (pressure; hPa) that the parcel is lifted
        pres : number
        Pressure of the parcel to lift (hPa)
        tmpc : number
        Temperature of the parcel to lift (C)
        dwpc : number
        Dew Point of the parcel to lift (C)
        
        '''
    def __init__(self, **kwargs):
        self.pres = ma.masked # Parcel beginning pressure (mb)
        self.tmpc = ma.masked # Parcel beginning temperature (C)
        self.dwpc = ma.masked # Parcel beginning dewpoint (C)
        self.ptrace = ma.masked # Parcel trace pressure (mb)
        self.ttrace = ma.masked # Parcel trace temperature (C)
        self.blayer = ma.masked # Pressure of the bottom of the layer the parcel is lifted (mb)
        self.tlayer = ma.masked # Pressure of the top of the layer the parcel is lifted (mb)
        self.entrain = 0. # A parcel entrainment setting (not yet implemented)
        self.lclpres = ma.masked # Parcel LCL (lifted condensation level) pressure (mb)
        self.lclhght = ma.masked # Parcel LCL height (m AGL)
        self.lfcpres = ma.masked # Parcel LFC (level of free convection) pressure (mb)
        self.lfchght = ma.masked # Parcel LFC height (m AGL)
        self.elpres = ma.masked # Parcel EL (equilibrium level) pressure (mb)
        self.elhght = ma.masked # Parcel EL height (m AGL)
        self.mplpres = ma.masked # Maximum Parcel Level (mb)
        self.mplhght = ma.masked # Maximum Parcel Level (m AGL)
        self.bplus = ma.masked # Parcel CAPE (J/kg)
        self.bminus = ma.masked # Parcel CIN (J/kg)
        self.bfzl = ma.masked # Parcel CAPE up to freezing level (J/kg)
        self.b3km = ma.masked # Parcel CAPE up to 3 km (J/kg)
        self.b4km = ma.masked # Parcel CAPE up to 4 km (J/kg)
        self.b6km = ma.masked # Parcel CAPE up to 6 km (J/kg)
        self.p0c = ma.masked # Pressure value at 0 C  (mb)
        self.pm10c = ma.masked # Pressure value at -10 C (mb)
        self.pm20c = ma.masked # Pressure value at -20 C (mb)
        self.pm30c = ma.masked # Pressure value at -30 C (mb)
        self.hght0c = ma.masked # Height value at 0 C (m AGL)
        self.hghtm10c = ma.masked # Height value at -10 C (m AGL)
        self.hghtm20c = ma.masked # Height value at -20 C (m AGL)
        self.hghtm30c = ma.masked # Height value at -30 C (m AGL)
        self.wm10c = ma.masked # w velocity at -10 C ?
        self.wm20c = ma.masked # w velocity at -20 C ?
        self.wm30c = ma.masked # Wet bulb at -30 C ? 
        self.li5 = ma.masked # Lifted Index at 500 mb (C)
        self.li3 = ma.masked # Lifted Index at 300 mb (C)
        self.brnshear = ma.masked # Bulk Richardson Number Shear
        self.brnu = ma.masked # Bulk Richardson Number U (kts)
        self.brnv = ma.masked # Bulk Richardson Number V (kts)
        self.brn = ma.masked # Bulk Richardson Number (unitless)
        self.limax = ma.masked # Maximum Lifted Index (C)
        self.limaxpres = ma.masked # Pressure at Maximum Lifted Index (mb)
        self.cap = ma.masked # Cap Strength (C)
        self.cappres = ma.masked # Cap strength pressure (mb)
        self.bmin = ma.masked # Buoyancy minimum in profile (C)
        self.bminpres = ma.masked # Buoyancy minimum pressure (mb)
        for kw in kwargs: setattr(self, kw, kwargs.get(kw))

def hgz(prof):
    '''
        Hail Growth Zone Levels
    
        This function finds the pressure levels for the dendritic 
        growth zone (from -10 C to -30 C).  If either temperature cannot be found,
        it is set to be the surface pressure.

        Parameters
        ----------
        prof : profile object
        Profile Object
        
        Returns
        -------
        pbot : number
        Pressure of the bottom level (mb)
        ptop : number 
        Pressure of the top level (mb)
    '''

    pbot = temp_lvl(prof, -10)
    ptop = temp_lvl(prof, -30)

    if not utils.QC(pbot):
        pbot = prof.pres[prof.sfc]
    if not utils.QC(ptop):
        ptop = prof.pres[prof.sfc]

    return pbot, ptop


def dgz(prof):
    '''
        Dendritic Growth Zone Levels
    
        This function finds the pressure levels for the dendritic 
        growth zone (from -12 C to -17 C).  If either temperature cannot be found,
        it is set to be the surface pressure.

        Parameters
        ----------
        prof : profile object
        Profile Object
        
        Returns
        -------
        pbot : number
        Pressure of the bottom level (mb)
        ptop : number 
        Pressure of the top level (mb)
    '''

    pbot = temp_lvl(prof, -12)
    ptop = temp_lvl(prof, -17)

    if not utils.QC(pbot):
        pbot = prof.pres[prof.sfc]
    if not utils.QC(ptop):
        ptop = prof.pres[prof.sfc]

    return pbot, ptop

def lhp(prof):
    '''
        Large Hail Parameter (*)

        From Johnson and Sugden (2014), EJSSM

        Parameters
        ----------
        prof : profile object
            ConvectiveProfile object

        Returns
        -------
        lhp : number
            large hail parameter (unitless)
    '''

    mag06_shr = utils.KTS2MS(utils.mag(*prof.sfc_6km_shear))

    if prof.mupcl.bplus >= 400 and mag06_shr >= 14:
        lr75 = prof.lapserate_700_500
        zbot, ztop = interp.hght(prof, hgz(prof))
        thk_hgz = ztop - zbot

        term_a = (((prof.mupcl.bplus - 2000.)/1000.) +\
                 ((3200 - thk_hgz)/500.) +\
                 ((lr75 - 6.5)/2.))

        if term_a < 0:
            term_a = 0

        p_1km, p_3km, p_6km = interp.pres(prof, interp.to_msl(prof, [1000, 3000, 6000]))
        shear_el = utils.KTS2MS(utils.mag(*winds.wind_shear(prof, pbot=prof.pres[prof.sfc], ptop=prof.mupcl.elpres)))
        grw_el_dir = interp.vec(prof, prof.mupcl.elpres)[0]
        grw_36_dir = utils.comp2vec(*winds.mean_wind(prof, pbot=p_3km, ptop=p_6km))[0]
        grw_alpha_el = grw_el_dir - grw_36_dir

        if grw_alpha_el > 180:
            grw_alpha_el = -10

        srw_01_dir = utils.comp2vec(*winds.sr_wind(prof, pbot=prof.pres[prof.sfc], ptop=p_1km, stu=prof.srwind[0], stv=prof.srwind[1]))[0]
        srw_36_dir = utils.comp2vec(*winds.sr_wind(prof, pbot=p_3km, ptop=p_6km, stu=prof.srwind[0], stv=prof.srwind[1]))[0]
        srw_alpha_mid = srw_36_dir - srw_01_dir

        term_b = (((shear_el - 25.)/5.) +\
                  ((grw_alpha_el + 5.)/20.) +\
                  ((srw_alpha_mid - 80.)/10.))
        if term_b < 0:
            term_b = 0

        lhp = term_a * term_b + 5

    else:
        lhp = 0

    return lhp


def ship(prof, **kwargs):
    '''
        Calculate the Sig Hail Parameter (SHIP)

        Parameters
        ----------
        prof : Profile object
        mupcl : (optional) Most-Unstable Parcel
        lr75 : (optional) 700 - 500 mb lapse rate (C/km)
        h5_temp : (optional) 500 mb temperature (C)
        shr06 : (optional) 0-6 km shear (m/s)
        frz_lvl : (optional) freezing level (m)

        Returns
        -------
        ship : number
            significant hail parameter (unitless)

        Ryan Jewell (SPC) helped in correcting this equation as the SPC
        sounding help page version did not have the correct information
        of how SHIP was calculated.

        The significant hail parameter (SHIP; SPC 2014) is
        an index developed in-house at the SPC. (Johnson and Sugden 2014)
    '''
      
    mupcl = kwargs.get('mupcl', None)
    sfc6shr = kwargs.get('sfc6shr', None)
    frz_lvl = kwargs.get('frz_lvl', None)
    h5_temp = kwargs.get('h5_temp', None)
    lr75 = kwargs.get('lr75', None)

    if not mupcl:
        try:
            mupcl = prof.mupcl
        except:
            mulplvals = DefineParcel(prof, flag=3, pres=300)
            mupcl = cape(prof, lplvals=mulplvals)
    mucape = mupcl.bplus
    mumr = thermo.mixratio(mupcl.pres, mupcl.dwpc)

    if not frz_lvl:
        frz_lvl = interp.hght(prof, temp_lvl(prof, 0))

    if not h5_temp:
        h5_temp = interp.temp(prof, 500.)

    if not lr75:
        lr75 = lapse_rate(prof, 700., 500., pres=True)

    if not sfc6shr:
        try:
            sfc_6km_shear = prof.sfc_6km_shear
        except:
            sfc = prof.pres[prof.sfc]
            p6km = interp.pres(prof, interp.to_msl(prof, 6000.))
            sfc_6km_shear = winds.wind_shear(prof, pbot=sfc, ptop=p6km)
    
    sfc_6km_shear = utils.mag(sfc_6km_shear[0], sfc_6km_shear[1])
    shr06 = utils.KTS2MS(sfc_6km_shear)
    
    if shr06 > 27:
        shr06 = 27.
    elif shr06 < 7:
        shr06 = 7.

    if mumr > 13.6:
        mumr = 13.6
    elif mumr < 11.:
        mumr = 11.

    if h5_temp > -5.5:
        h5_temp = -5.5

    ship = -1. * (mucape * mumr * lr75 * h5_temp * shr06) / 42000000.
    
    if mucape < 1300:
        ship = ship*(mucape/1300.)
    
    if lr75 < 5.8:
        ship = ship*(lr75/5.8)

    if frz_lvl < 2400:
        ship = ship * (frz_lvl/2400.)
    
    return ship

def stp_cin(mlcape, esrh, ebwd, mllcl, mlcinh):
    '''
        Significant Tornado Parameter (w/CIN)

        From Thompson et al. 2012 WAF, page 1139

        Parameters
        ----------
        mlcape : Mixed-layer CAPE from the parcel class (J/kg)
        esrh : effective storm relative helicity (m2/s2)
        ebwd : effective bulk wind difference (m/s)
        mllcl : mixed-layer lifted condensation level (m)
        mlcinh : mixed-layer convective inhibition (J/kg)

        Returns
        -------
        stp_cin : number
            significant tornado parameter (unitless)

    '''
    cape_term = mlcape / 1500.
    eshr_term = esrh / 150.
    
    if ebwd < 12.5:
        ebwd_term = 0.
    elif ebwd > 30.:
        ebwd_term = 1.5
    else:
        ebwd_term  = ebwd / 20.

    if mllcl < 1000.:
        lcl_term = 1.0
    elif mllcl > 2000.:
        lcl_term = 0.0
    else:
        lcl_term = ((2000. - mllcl) / 1000.)

    if mlcinh > -50:
        cinh_term = 1.0
    elif mlcinh < -200:
        cinh_term = 0
    else:
        cinh_term = ((mlcinh + 200.) / 150.)

    stp_cin = np.maximum(cape_term * eshr_term * ebwd_term * lcl_term * cinh_term, 0)
    return stp_cin

def stp_fixed(sbcape, sblcl, srh01, bwd6):
    '''
        Significant Tornado Parameter (fixed layer)
   
        From Thompson et al. 2003

        Parameters
        ----------
        sbcape : Surface based CAPE from the parcel class (J/kg)
        sblcl : Surface based lifted condensation level (LCL) (m)
        srh01 : Surface to 1 km storm relative helicity (m2/s2)
        bwd6 : Bulk wind difference between 0 to 6 km (m/s)

        Returns
        -------
        stp_fixed : number
            signifcant tornado parameter (fixed-layer)
    '''
    
    # Calculate SBLCL term
    if sblcl < 1000.: # less than 1000. meters
        lcl_term = 1.0
    elif sblcl > 2000.: # greater than 2000. meters
        lcl_term = 0.0
    else:
        lcl_term = ((2000.-sblcl)/1000.)

    # Calculate 6BWD term
    if bwd6 > 30.: # greater than 30 m/s
        bwd6 = 30
    elif bwd6 < 12.5:
        bwd6 = 0.0
    
    bwd6_term = bwd6 / 20.

    cape_term = sbcape / 1500.
    srh_term = srh01 / 150.

    stp_fixed = cape_term * lcl_term * srh_term * bwd6_term
   
    return stp_fixed

def scp(mucape, srh, ebwd):
    '''
        Supercell Composite Parameter

        From Thompson et al. 2004

        Parameters
        ----------
        prof : Profile object
        mucape : Most Unstable CAPE from the parcel class (J/kg) (optional)
        srh : the effective SRH from the winds.helicity function (m2/s2)
        ebwd : effective bulk wind difference (m/s)

        Returns
        -------
        scp : number
            supercell composite parameter
    
    '''

    if ebwd > 20:
        ebwd = 20.
    elif ebwd < 10:
        ebwd = 0.
     
    muCAPE_term = mucape / 1000.
    esrh_term = srh / 50.
    ebwd_term = ebwd / 20.

    scp = muCAPE_term * esrh_term * ebwd_term
    return scp

def k_index(prof):
    '''
        Calculates the K-Index from a profile object
        
        Parameters
        ----------
        prof : profile object
        Profile Object
        
        Returns
        -------
        kind : number
        K-Index
        
        '''
    t8 = interp.temp(prof, 850.)
    t7 = interp.temp(prof, 700.)
    t5 = interp.temp(prof, 500.)
    td7 = interp.dwpt(prof, 700.)
    td8 = interp.dwpt(prof, 850.)
    return t8 - t5 + td8 - (t7 - td7)


def t_totals(prof):
    '''
        Calculates the Total Totals Index from a profile object
        
        Parameters
        ----------
        prof : profile object
        Profile Object
        
        Returns
        -------
        t_totals : number
        Total Totals Index
        
        '''
    return c_totals(prof) + v_totals(prof)


def c_totals(prof):
    '''
        Calculates the Cross Totals Index from a profile object
        
        Parameters
        ----------
        prof : profile object
        Profile Object
        
        Returns
        -------
        c_totals : number
        Cross Totals Index
        
        '''
    return interp.dwpt(prof, 850.) - interp.temp(prof, 500.)


def v_totals(prof):
    '''
        Calculates the Vertical Totals Index from a profile object
        
        Parameters
        ----------
        prof : profile object
        Profile Object
        
        Returns
        -------
        v_totals : number
        Vertical Totals Index
        
        '''
    return interp.temp(prof, 850.) - interp.temp(prof, 500.)


def precip_water(prof, pbot=None, ptop=400, dp=-1, exact=False):
    '''
        Calculates the precipitable water from a profile object within the
        specified layer. The default layer (lower=-1 & upper=-1) is defined to
        be surface to 400 hPa.
        
        Parameters
        ----------
        prof : profile object
        Profile Object
        pbot : number (optional; default surface)
        Pressure of the bottom level (hPa)
        ptop : number (optional; default 400 hPa)
        Pressure of the top level (hPa).
        dp : negative integer (optional; default = -1)
        The pressure increment for the interpolated sounding
        exact : bool (optional; default = False)
        Switch to choose between using the exact data (slower) or using
        interpolated sounding at 'dp' pressure levels (faster)
        
        Returns
        -------
        pwat : number,
        Precipitable Water (in)
        '''
    if not pbot: pbot = prof.pres[prof.sfc]

    if prof.pres[-1] > ptop:
        ptop = prof.pres[-1]

    if exact:
        ind1 = np.where(pbot > prof.pres)[0].min()
        ind2 = np.where(ptop < prof.pres)[0].max()
        dwpt1 = interp.dwpt(prof, pbot)
        dwpt2 = interp.dwpt(prof, ptop)
        mask = ~prof.dwpc.mask[ind1:ind2+1] * ~prof.pres.mask[ind1:ind2+1]
        dwpt = np.concatenate([[dwpt1], prof.dwpc[ind1:ind2+1][mask], [dwpt2]])
        p = np.concatenate([[pbot], prof.pres[ind1:ind2+1][mask], [ptop]])
    else:
        dp = -1
        p = np.arange(pbot, ptop+dp, dp, dtype=type(pbot))
        dwpt = interp.dwpt(prof, p)
    w = thermo.mixratio(p, dwpt)
    return (((w[:-1]+w[1:])/2 * (p[:-1]-p[1:])) * 0.00040173).sum()


def inferred_temp_adv(prof, lat=35):
    '''
        Inferred Temperature Advection

        SHARP code deduced by Greg Blumberg.  Not based on actual SPC code.

        Calculates the inferred temperature advection from the surface pressure
        and up every 100 mb assuming all winds are geostrophic.  The units returned are
        in C/hr.  If no latitude is specified the function defaults to 35 degrees North.

        This function doesn't compare well to SPC in terms of magnitude of the results.  The direction
        and relative magnitude I think I've got right. My calculations seem to be consistently less
        than those seen on the SPC website.  Although, this function may be right as the SPC values seem
        a little high for typical synoptic scale geostrophic temperature advection (10 Kelvin/day is typical).

        This code uses Equation 4.1.139 from Bluestein's "Synoptic-Dynamic Meteorology in Midlatitudes (Volume I)"

        Parameters
        ----------
        prof : Profile object
        lat : latitude in decimal degrees (optional)

        Returns
        -------
        temp_adv : an array of temperature advection values in C/hr
        pressure_bounds: a 2D array indicating the top and bottom bounds of the temperature advection layers.
    '''

    omega = (2. * np.pi) / (86164.)
       
    dp = -100
    pres_idx = np.where(prof.pres >= 100.)[0]
    pressures = np.arange(prof.pres[prof.get_sfc()], prof.pres[pres_idx][-1], dp, dtype=type(prof.pres[prof.get_sfc()])) # Units: mb
    temps = thermo.ctok(interp.temp(prof, pressures))
    heights = interp.hght(prof, pressures)
    temp_adv = np.empty(len(pressures) - 1)
    dirs = interp.vec(prof, pressures)[0]
    pressure_bounds = np.empty((len(pressures) - 1, 2))

    if utils.QC(lat):
        f = 2. * omega * np.sin(np.radians(lat)) # Units: (s**-1)
    else:
        temp_adv[:] = np.nan
        return temp_adv, pressure_bounds

    multiplier = (f / 9.81) * (np.pi / 180.) # Units: (s**-1 / (m/s**2)) * (radians/degrees)

    for i in xrange(1, len(pressures)):
        bottom_pres = pressures[i-1]
        top_pres = pressures[i]
        # Get the temperatures from both levels (in Kelvin)
        btemp = temps[i-1]
        ttemp = temps[i]
        # Get the two heights of the top and bottom layer
        bhght = heights[i-1] # Units: meters
        thght = heights[i] # Units: meters
        bottom_wdir = dirs[i-1] # Meteorological degrees (degrees from north)
        top_wdir = dirs[i] # same units as top_wdir
        
        # Calculate the average temperature
        avg_temp = (ttemp + btemp) * 2.
        
        # Calculate the mean wind between the two levels (this is assumed to be geostrophic)
        mean_u, mean_v = winds.mean_wind(prof, pbot=bottom_pres, ptop=top_pres)
        mean_wdir, mean_wspd = utils.comp2vec(mean_u, mean_v) # Wind speed is in knots here
        mean_wspd = utils.KTS2MS(mean_wspd) # Convert this geostrophic wind speed to m/s
        
        # Here we calculate the change in wind direction with height (thanks to Andrew Mackenzie for help with this)
        # The sign of d_theta will dictate whether or not it is warm or cold advection
        mod = 180 - bottom_wdir
        top_wdir = top_wdir + mod
        
        if top_wdir < 0:
            top_wdir = top_wdir + 360
        elif top_wdir >= 360:
            top_wdir = top_wdir - 360
        d_theta = top_wdir - 180.
        
        # Here we calculate t_adv (which is -V_g * del(T) or the local change in temperature term)
        # K/s  s * rad/m * deg   m^2/s^2          K        degrees / m
        t_adv = multiplier * np.power(mean_wspd,2) * avg_temp * (d_theta / (thght - bhght)) # Units: Kelvin / seconds
        
        # Append the pressure bounds so the person knows the pressure
        pressure_bounds[i-1, :] = bottom_pres, top_pres
        temp_adv[i-1] = t_adv*60.*60. # Converts Kelvin/seconds to Kelvin/hour (or Celsius/hour)

    return temp_adv, pressure_bounds


def inversion(prof, pbot=None, ptop=None, dp=-1, exact=False):
    '''
        Finds the layers where temperature inversions are occurring.

        Parameters
        ----------
        prof : profile object
        Profile Object
        pbot : number (optional; default surface)
        Pressure of the bottom level (hPa)
        ptop : number (optional; default top of sounding)
        Pressure of the top level (hPa).
        dp : negative integer (optional; default = -1)
        The pressure increment for the interpolated sounding
        exact : bool (optional; default = False)
        Switch to choose between using the exact data (slower) or using
        interpolated sounding at 'dp' pressure levels (faster)
        
        Returns
        -------
        inv_bot : An array of bases of inversion layers
        inv_top : An array of tops of inversion layers
    '''

    if not pbot: pbot = prof.pres[prof.sfc]
    if not ptop: ptop = prof.pres[prof.top]
    
    if not utils.QC(interp.vtmp(prof, pbot)): pbot = prof.pres[prof.sfc]
    if not utils.QC(interp.vtmp(prof, ptop)): return ma.masked
    
    if exact:
        ind1 = np.where(pbot > prof.pres)[0].min()
        ind2 = np.where(ptop < prof.pres)[0].max()
        vtmp1 = interp.vtmp(prof, pbot)
        vtmp2 = interp.vtmp(prof, ptop)
        hght1 = interp.hght(prof, pbot)
        hght2 = interp.hght(prof, ptop)
        mask = ~prof.vtmp.mask[ind1:ind2+1] * ~prof.hght.mask[ind1:ind2+1]
        vtmp = np.concatenate([[vtmp1], prof.vtmp[ind1:ind2+1][mask], [vtmp2]])
        hght = np.concatenate([[hght1], prof.hght[ind1:ind2+1][mask], [hght2]])
    else:
        ps = np.arange(pbot, ptop+dp, dp)
        vtmp = interp.vtmp(prof, ps)
        hght = interp.hght(prof, ps)
    
    lr = ((vtmp[1:] - vtmp[:-1]) / (hght[1:] - hght[:-1])) * -1000
    ind3 = ma.where(lr < 0)[0]

    if exact:
        ind4bot = ind3 + ind1 - 1
        ind4top = ind3 + ind1
        inv_bot = prof.pres[ind4bot]
        inv_top = prof.pres[ind4top]
    else:
        ind4 = ma.where(ind3[1:] - ind3[:-1] > 1)[0]
        ind4bot = ind3[ind4+1]
        ind4bot = ma.append(ind3[0], ind4bot)
        ind4top = ind3[ind4]
        ind4top = ma.append(ind4top, ind3.max())
        inv_bot = ps[ind4bot]
        inv_top = ps[ind4top]
    
    return inv_bot, inv_top


def temp_lvl(prof, temp):
    '''
        Calculates the level (hPa) of the first occurrence of the specified
        temperature.
        
        Parameters
        ----------
        prof : profile object
        Profile Object
        temp : number
        Temperature being searched (C)
        
        Returns
        -------
        First Level of the temperature (hPa)
        
        '''
    difft = prof.tmpc - temp
    ind1 = ma.where(difft >= 0)[0]
    ind2 = ma.where(difft <= 0)[0]
    if len(ind1) == 0 or len(ind2) == 0:
        return ma.masked
    inds = np.intersect1d(ind1, ind2)
    if len(inds) > 0:
        return prof.pres[inds][0]
    diff1 = ind1[1:] - ind1[:-1]
    ind = np.where(diff1 > 1)[0] + 1
    try:
        ind = ind.min()
    except:
        ind = ind1[-1]

    return np.power(10, np.interp(temp, [prof.tmpc[ind+1], prof.tmpc[ind]],
                            [prof.logp[ind+1], prof.logp[ind]]))


def max_temp(prof, mixlayer=100):
    '''
        Calculates a maximum temperature forecast based on the depth of the mixing
        layer and low-level temperatures
        
        Parameters
        ----------
        prof : profile object
        Profile Object
        mixlayer : number (optional; default = 100)
        Top of layer over which to "mix" (hPa)
        
        Returns
        -------
        mtemp : number
        Forecast Maximum Temperature
        
        '''
    mixlayer = prof.pres[prof.sfc] - mixlayer
    temp = thermo.ctok(interp.temp(prof, mixlayer)) + 2
    return thermo.ktoc(temp * (prof.pres[prof.sfc] / mixlayer)**ROCP)


def mean_relh(prof, pbot=None, ptop=None, dp=-1, exact=False):
    '''
        Calculates the mean relative humidity from a profile object within the
        specified layer.
        
        Parameters
        ----------
        prof : profile object
        Profile Object
        pbot : number (optional; default surface)
        Pressure of the bottom level (hPa)
        ptop : number (optional; default 400 hPa)
        Pressure of the top level (hPa)
        dp : negative integer (optional; default = -1)
        The pressure increment for the interpolated sounding
        exact : bool (optional; default = False)
        Switch to choose between using the exact data (slower) or using
        interpolated sounding at 'dp' pressure levels (faster)
        
        Returns
        -------
        Mean Relative Humidity
        
        '''
    if not pbot: pbot = prof.pres[prof.sfc]
    if not ptop: ptop = prof.pres[prof.sfc] - 100.
    if not utils.QC(interp.temp(prof, pbot)): pbot = prof.pres[prof.sfc]
    if not utils.QC(interp.temp(prof, ptop)): return ma.masked
    if exact:
        ind1 = np.where(pbot > prof.pres)[0].min()
        ind2 = np.where(ptop < prof.pres)[0].max()
        dwpt1 = interp.dwpt(prof, pbot)
        dwpt2 = interp.dwpt(prof, ptop)
        mask = ~prof.dwpc.mask[ind1:ind2+1] * ~prof.pres.mask[ind1:ind2+1]
        dwpt = np.concatenate([[dwpt1], prof.dwpc[ind1:ind2+1][mask],
                               [dwpt2]])
        p = np.concatenate([[pbot], prof.pres[ind1:ind2+1][mask], [ptop]])
    else:
        dp = -1
        p = np.arange(pbot, ptop+dp, dp, dtype=type(pbot))
        tmp = interp.temp(prof, p)
        dwpt = interp.dwpt(prof, p)
    rh = thermo.relh(p, tmp, dwpt)
    return ma.average(rh, weights=p)

def mean_omega(prof, pbot=None, ptop=None, dp=-1, exact=False):
    '''
        Calculates the mean omega from a profile object within the
        specified layer.
        
        Parameters
        ----------
        prof : profile object
        Profile Object
        pbot : number (optional; default surface)
        Pressure of the bottom level (hPa)
        ptop : number (optional; default 400 hPa)
        Pressure of the top level (hPa)
        dp : negative integer (optional; default = -1)
        The pressure increment for the interpolated sounding
        exact : bool (optional; default = False)
        Switch to choose between using the exact data (slower) or using
        interpolated sounding at 'dp' pressure levels (faster)
        
        Returns
        -------
        Mean Omega
        
        '''
    if hasattr(prof, 'omeg'): 
        if prof.omeg.all() is np.ma.masked:
            return prof.missing
    else:
        return prof.missing
    if not pbot: pbot = prof.pres[prof.sfc]
    if not ptop: ptop = prof.pres[prof.sfc] - 100.
    if not utils.QC(interp.omeg(prof, pbot)): pbot = prof.pres[prof.sfc]
    if not utils.QC(interp.omeg(prof, ptop)): return ma.masked
    if exact:
        # This condition of the if statement is not tested
        omeg = prof.omeg
        ind1 = np.where(pbot > prof.pres)[0].min()
        ind2 = np.where(ptop < prof.pres)[0].max()
        omeg1 = interp.omeg(prof, pbot)
        omeg2 = interp.omeg(prof, ptop)
        omeg = omeg[ind1:ind2+1]
        mask = ~omeg.mask
        omeg = np.concatenate([[omeg1], omeg[mask], omeg[mask], [omeg2]])
        tott = omeg.sum() / 2.
        num = float(len(omeg)) / 2.
        thta = tott / num
    else:
        dp = -1
        p = np.arange(pbot, ptop+dp, dp, dtype=type(pbot))
        omeg = interp.omeg(prof, p)
        omeg = ma.average(omeg, weights=p)
    return omeg

def mean_mixratio(prof, pbot=None, ptop=None, dp=-1, exact=False):
    '''
        Calculates the mean mixing ratio from a profile object within the
        specified layer.
        
        Parameters
        ----------
        prof : profile object
        Profile Object
        pbot : number (optional; default surface)
        Pressure of the bottom level (hPa)
        ptop : number (optional; default 400 hPa)
        Pressure of the top level (hPa)
        dp : negative integer (optional; default = -1)
        The pressure increment for the interpolated sounding
        exact : bool (optional; default = False)
        Switch to choose between using the exact data (slower) or using
        interpolated sounding at 'dp' pressure levels (faster)
        
        Returns
        -------
        Mean Mixing Ratio
        
        '''
    if not pbot: pbot = prof.pres[prof.sfc]
    if not ptop: ptop = prof.pres[prof.sfc] - 100.
    if not utils.QC(interp.temp(prof, pbot)): pbot = prof.pres[prof.sfc]
    if not utils.QC(interp.temp(prof, ptop)): return ma.masked
    if exact:
        ind1 = np.where(pbot > prof.pres)[0].min()
        ind2 = np.where(ptop < prof.pres)[0].max()
        dwpt1 = interp.dwpt(prof, pbot)
        dwpt2 = interp.dwpt(prof, ptop)
        mask = ~prof.dwpc.mask[ind1:ind2+1] * ~prof.pres.mask[ind1:ind2+1]
        dwpt = np.concatenate([[dwpt1], prof.dwpc[ind1:ind2+1][mask], prof.dwpc[ind1:ind2+1][mask], [dwpt2]])
        p = np.concatenate([[pbot], prof.pres[ind1:ind2+1][mask],prof.pres[ind1:ind2+1][mask], [ptop]])
        totd = dwpt.sum() / 2.
        totp = p.sum() / 2.
        num = float(len(dwpt)) / 2.
        w = thermo.mixratio(totp/num, totd/num)
    
    else:
        dp = -1
        p = np.arange(pbot, ptop+dp, dp, dtype=type(pbot))
        dwpt = interp.dwpt(prof, p)
        w = ma.average(thermo.mixratio(p, dwpt))
    return w

def mean_thetae(prof, pbot=None, ptop=None, dp=-1, exact=False):
    '''
        Calculates the mean theta-e from a profile object within the
        specified layer.
        
        Parameters
        ----------
        prof : profile object
        Profile Object
        pbot : number (optional; default surface)
        Pressure of the bottom level (hPa)
        ptop : number (optional; default 400 hPa)
        Pressure of the top level (hPa)
        dp : negative integer (optional; default = -1)
        The pressure increment for the interpolated sounding
        exact : bool (optional; default = False)
        Switch to choose between using the exact data (slower) or using
        interpolated sounding at 'dp' pressure levels (faster)
        
        Returns
        -------
        Mean Theta-E
        
        '''
    if not pbot: pbot = prof.pres[prof.sfc]
    if not ptop: ptop = prof.pres[prof.sfc] - 100.
    if not utils.QC(interp.temp(prof, pbot)): pbot = prof.pres[prof.sfc]
    if not utils.QC(interp.temp(prof, ptop)): return ma.masked
    if exact:
        ind1 = np.where(pbot > prof.pres)[0].min()
        ind2 = np.where(ptop < prof.pres)[0].max()
        thetae1 = thermo.thetae(pbot, interp.temp(prof, pbot), interp.dwpt(prof, pbot))
        thetae2 = thermo.thetae(ptop, interp.temp(prof, ptop), interp.dwpt(prof, pbot))
        thetae = np.ma.empty(prof.pres[ind1:ind2+1].shape)
        for i in np.arange(0, len(thetae), 1):
            thetae[i] = thermo.thetae(prof.pres[ind1:ind2+1][i],  prof.tmpc[ind1:ind2+1][i], prof.dwpc[ind1:ind2+1][i])
        mask = ~thetae.mask
        thetae = np.concatenate([[thetae1], thetae[mask], thetae[mask], [thetae2]])
        tott = thetae.sum() / 2.
        num = float(len(thetae)) / 2.
        thtae = tott / num
    else:
        dp = -1
        p = np.arange(pbot, ptop+dp, dp, dtype=type(pbot))
        #temp = interp.temp(prof, p)
        #dwpt = interp.dwpt(prof, p)
        #thetae = np.empty(p.shape)
        #for i in np.arange(0, len(thetae), 1):
        #   thetae[i] = thermo.thetae(p[i], temp[i], dwpt[i])
        thetae = interp.thetae(prof, p)
        thtae = ma.average(thetae, weights=p)
    return thtae

def mean_thetaes(prof, pbot=None, ptop=None, dp=-1, exact=False):
    '''
        Calculates the mean theta-es from a profile object within the
        specified layer.
        
        Parameters
        ----------
        prof : profile object
        Profile Object
        pbot : number (optional; default surface)
        Pressure of the bottom level (hPa)
        ptop : number (optional; default 400 hPa)
        Pressure of the top level (hPa)
        dp : negative integer (optional; default = -1)
        The pressure increment for the interpolated sounding
        exact : bool (optional; default = False)
        Switch to choose between using the exact data (slower) or using
        interpolated sounding at 'dp' pressure levels (faster)
        
        Returns
        -------
        Mean Theta-ES
        
        '''
    if not pbot: pbot = prof.pres[prof.sfc]
    if not ptop: ptop = prof.pres[prof.sfc] - 100.
    if not utils.QC(interp.temp(prof, pbot)): pbot = prof.pres[prof.sfc]
    if not utils.QC(interp.temp(prof, ptop)): return ma.masked
    if exact:
        ind1 = np.where(pbot > prof.pres)[0].min()
        ind2 = np.where(ptop < prof.pres)[0].max()
        thetaes1 = thermo.thetaes(pbot, interp.temp(prof, pbot))
        thetaes2 = thermo.thetaes(ptop, interp.temp(prof, ptop))
        thetaes = np.ma.empty(prof.pres[ind1:ind2+1].shape)
        for i in np.arange(0, len(thetaes), 1):
            thetaes[i] = thermo.thetaes(prof.pres[ind1:ind2+1][i],  prof.tmpc[ind1:ind2+1][i])
        mask = ~thetaes.mask
        thetaes = np.concatenate([[thetaes1], thetaes[mask], thetaes[mask], [thetaes2]])
        tott = thetaes.sum() / 2.
        num = float(len(thetaes)) / 2.
        thtaes = tott / num
    else:
        dp = -1
        p = np.arange(pbot, ptop+dp, dp, dtype=type(pbot))
        temp = interp.temp(prof, p)
        thetaes = np.empty(p.shape)
        for i in np.arange(0, len(thetaes), 1):
           thetaes[i] = thermo.thetaes(p[i], temp[i])
        thtaes = ma.average(thetaes, weights=p)
    return thtaes

def mean_theta(prof, pbot=None, ptop=None, dp=-1, exact=False):
    '''
        Calculates the mean theta from a profile object within the
        specified layer.
        
        Parameters
        ----------
        prof : profile object
        Profile Object
        pbot : number (optional; default surface)
        Pressure of the bottom level (hPa)
        ptop : number (optional; default 400 hPa)
        Pressure of the top level (hPa)
        dp : negative integer (optional; default = -1)
        The pressure increment for the interpolated sounding
        exact : bool (optional; default = False)
        Switch to choose between using the exact data (slower) or using
        interpolated sounding at 'dp' pressure levels (faster)
        
        Returns
        -------
        Mean Theta
        
        '''
    if not pbot: pbot = prof.pres[prof.sfc]
    if not ptop: ptop = prof.pres[prof.sfc] - 100.
    if not utils.QC(interp.temp(prof, pbot)): pbot = prof.pres[prof.sfc]
    if not utils.QC(interp.temp(prof, ptop)): return ma.masked
    if exact:
        ind1 = np.where(pbot > prof.pres)[0].min()
        ind2 = np.where(ptop < prof.pres)[0].max()
        theta1 = thermo.theta(pbot, interp.temp(prof, pbot))
        theta2 = thermo.theta(ptop, interp.temp(prof, ptop))
        theta = thermo.theta(prof.pres[ind1:ind2+1],  prof.tmpc[ind1:ind2+1])
        mask = ~theta.mask
        theta = np.concatenate([[theta1], theta[mask], theta[mask], [theta2]])
        tott = theta.sum() / 2.
        num = float(len(theta)) / 2.
        thta = tott / num
    else:
        dp = -1
        p = np.arange(pbot, ptop+dp, dp, dtype=type(pbot))
        temp = interp.temp(prof, p)
        theta = thermo.theta(p, temp)
        thta = ma.average(theta, weights=p)
    return thta

def mean_thetaw(prof, pbot=None, ptop=None, dp=-1, exact=False):
    '''
        Calculates the mean theta-w from a profile object within the
        specified layer.
        
        Parameters
        ----------
        prof : profile object
        Profile Object
        pbot : number (optional; default surface)
        Pressure of the bottom level (hPa)
        ptop : number (optional; default 400 hPa)
        Pressure of the top level (hPa)
        dp : negative integer (optional; default = -1)
        The pressure increment for the interpolated sounding
        exact : bool (optional; default = False)
        Switch to choose between using the exact data (slower) or using
        interpolated sounding at 'dp' pressure levels (faster)
        
        Returns
        -------
        Mean Theta-W
        
        '''
    if not pbot: pbot = prof.pres[prof.sfc]
    if not ptop: ptop = prof.pres[prof.sfc] - 100.
    if not utils.QC(interp.temp(prof, pbot)): pbot = prof.pres[prof.sfc]
    if not utils.QC(interp.temp(prof, ptop)): return ma.masked
    if exact:
        ind1 = np.where(pbot > prof.pres)[0].min()
        ind2 = np.where(ptop < prof.pres)[0].max()
        thetaw1 = thermo.thetaw(pbot, interp.temp(prof, pbot), interp.dwpt(prof, pbot))
        thetaw2 = thermo.thetaw(ptop, interp.temp(prof, ptop), interp.dwpt(prof, pbot))
        thetaw = np.ma.empty(prof.pres[ind1:ind2+1].shape)
        for i in np.arange(0, len(thetaw), 1):
            thetaw[i] = thermo.thetaw(prof.pres[ind1:ind2+1][i],  prof.tmpc[ind1:ind2+1][i], prof.dwpc[ind1:ind2+1][i])
        mask = ~thetaw.mask
        thetaw = np.concatenate([[thetaw1], thetaw[mask], thetaw[mask], [thetaw2]])
        tott = thetaw.sum() / 2.
        num = float(len(thetaw)) / 2.
        thtaw = tott / num
    else:
        dp = -1
        p = np.arange(pbot, ptop+dp, dp, dtype=type(pbot))
        temp = interp.temp(prof, p)
        dwpt = interp.dwpt(prof, p)
        thetaw = thermo.thetaw(p, temp, dwpt)
        thtaw = ma.average(thetaw, weights=p)
    return thtaw

def mean_thetaws(prof, pbot=None, ptop=None, dp=-1, exact=False):
    '''
        Calculates the mean theta-ws from a profile object within the
        specified layer.
        
        Parameters
        ----------
        prof : profile object
        Profile Object
        pbot : number (optional; default surface)
        Pressure of the bottom level (hPa)
        ptop : number (optional; default 400 hPa)
        Pressure of the top level (hPa)
        dp : negative integer (optional; default = -1)
        The pressure increment for the interpolated sounding
        exact : bool (optional; default = False)
        Switch to choose between using the exact data (slower) or using
        interpolated sounding at 'dp' pressure levels (faster)
        
        Returns
        -------
        Mean Theta-WS
        
        '''
    if not pbot: pbot = prof.pres[prof.sfc]
    if not ptop: ptop = prof.pres[prof.sfc] - 100.
    if not utils.QC(interp.temp(prof, pbot)): pbot = prof.pres[prof.sfc]
    if not utils.QC(interp.temp(prof, ptop)): return ma.masked
    if exact:
        ind1 = np.where(pbot > prof.pres)[0].min()
        ind2 = np.where(ptop < prof.pres)[0].max()
        thetaws1 = thermo.thetaws(pbot, interp.temp(prof, pbot))
        thetaws2 = thermo.thetaws(ptop, interp.temp(prof, ptop))
        thetaws = np.ma.empty(prof.pres[ind1:ind2+1].shape)
        for i in np.arange(0, len(thetaws), 1):
            thetaws[i] = thermo.thetaws(prof.pres[ind1:ind2+1][i],  prof.tmpc[ind1:ind2+1][i])
        mask = ~thetaws.mask
        thetaws = np.concatenate([[thetaws1], thetaws[mask], thetaws[mask], [thetaws2]])
        tott = thetaws.sum() / 2.
        num = float(len(thetaws)) / 2.
        thtaws = tott / num
    else:
        dp = -1
        p = np.arange(pbot, ptop+dp, dp, dtype=type(pbot))
        temp = interp.temp(prof, p)
        thetaws = np.empty(p.shape)
        for i in np.arange(0, len(thetaws), 1):
           thetaws[i] = thermo.thetaws(p[i], temp[i])
        thtaws = ma.average(thetaws, weights=p)
    return thtaws

def lapse_rate(prof, lower, upper, pres=True):
    '''
        Calculates the lapse rate (C/km) from a profile object
        
        Parameters
        ----------
        prof : profile object
        Profile Object
        lower : number
        Lower Bound of lapse rate
        upper : number
        Upper Bound of lapse rate
        pres : bool (optional; default = True)
        Flag to determine if lower/upper are pressure [True]
        or height [False]
        
        Returns
        -------
        lapse rate  (float [C/km])
        '''
    if pres:
        if (prof.pres[-1] > upper): return ma.masked 
        p1 = lower
        p2 = upper
        z1 = interp.hght(prof, lower)
        z2 = interp.hght(prof, upper)
    else:
        z1 = interp.to_msl(prof, lower)
        z2 = interp.to_msl(prof, upper)
        p1 = interp.pres(prof, z1)
        p2 = interp.pres(prof, z2)
    tv1 = interp.vtmp(prof, p1)
    tv2 = interp.vtmp(prof, p2)
    return (tv2 - tv1) / (z2 - z1) * -1000.


def most_unstable_level(prof, pbot=None, ptop=None, dp=-1, exact=False):
    '''
        Finds the most unstable level between the lower and upper levels.
        
        Parameters
        ----------
        prof : profile object
        Profile Object
        pbot : number (optional; default surface)
        Pressure of the bottom level (hPa)
        ptop : number (optional; default 400 hPa)
        Pressure of the top level (hPa)
        dp : negative integer (optional; default = -1)
        The pressure increment for the interpolated sounding
        exact : bool (optional; default = False)
        Switch to choose between using the exact data (slower) or using
        interpolated sounding at 'dp' pressure levels (faster)
        
        Returns
        -------
        Pressure level of most unstable level (hPa)
        
        '''
    if not pbot: pbot = prof.pres[prof.sfc]
    if not ptop: ptop = prof.pres[prof.sfc] - 400
    if not utils.QC(interp.temp(prof, pbot)): pbot = prof.pres[prof.sfc]
    if not utils.QC(interp.temp(prof, ptop)): return ma.masked
    if exact:
        ind1 = np.where(pbot > prof.pres)[0].min()
        ind2 = np.where(ptop < prof.pres)[0].max()
        t1 = interp.temp(prof, pbot)
        t2 = interp.temp(prof, ptop)
        d1 = interp.dwpt(prof, pbot)
        d2 = interp.dwpt(prof, ptop)
        t = prof.tmpc[ind1:ind2+1]
        d = prof.dwpc[ind1:ind2+1]
        p = prof.pres[ind1:ind2+1]
        mask = ~t.mask * ~d.mask * ~p.mask
        t = np.concatenate([[t1], t[mask], [t2]])
        d = np.concatenate([[d1], d[mask], [d2]])
        p = np.concatenate([[pbot], p[mask], [ptop]])
    else:
        dp = -1
        p = np.arange(pbot, ptop+dp, dp, dtype=type(pbot))
        t = interp.temp(prof, p)
        d = interp.dwpt(prof, p)
    p2, t2 = thermo.drylift(p, t, d)
    mt = thermo.wetlift(p2, t2, 1000.) # Essentially this is making the Theta-E profile, which we are already doing in the Profile object!
    ind = np.where(np.fabs(mt - np.nanmax(mt)) < TOL)[0]
    return p[ind[0]]

def parcelTraj(prof, parcel, smu=None, smv=None):
    '''
        Parcel Trajectory Routine (Storm Slinky)
        Coded by Greg Blumberg

        This routine is a simple 3D thermodynamic parcel trajectory model that
        takes a thermodynamic profile and a parcel trace and computes the
        trajectory of a parcel that is lifted to its LFC, then given a 5 m/s
        nudge upwards, and then left to accelerate up to the EL.  (Based on a description
        in the AWIPS 2 Online Training.)
        
        This parcel is assumed to be moving horizontally via the storm motion vector, which
        if not supplied is taken to be the Bunkers Right Mover storm motion vector.
        As the parcel accelerates upwards, it is advected by the storm relative winds.
        The environmental winds are assumed to be steady-state.
        
        This simulates the path a parcel in a storm updraft would take using pure parcel theory.
        
        Parameters
        ----------
        prof : Profile object
        parcel : parcel object
        smu: optional (storm motion vector u)
        smv: optional (storm motion vector v)
        
        Returns
        -------
        pos_vector : a list of tuples, where each element of the list is a location of the parcel in time
        theta : the tilt of the updraft measured by the angle of the updraft with respect to the horizon
        '''
    
    t_parcel = parcel.ttrace # temperature
    p_parcel = parcel.ptrace # mb
    elhght = parcel.elhght # meter
    
    y_0 = 0 # meter
    x_0 = 0 # meter
    z_0 = parcel.lfchght # meter
    p_0 = parcel.lfcpres # meter
    
    g = 9.8 # m/s**2
    t_0 = 0 # seconds
    w_0 = 5 # m/s (the initial parcel nudge)
    u_0 = 0 # m/s
    v_0 = 0 # m/s (initial parcel location, which is storm motion relative)
    
    delta_t = 25 # the trajectory increment
    pos_vector = [(x_0,y_0,z_0)]
    speed_vector = [(u_0, v_0, w_0)]
    
    if smu==None or smv==None:
        smu = prof.srwind[0] # Expected to be in knots
        smv = prof.srwind[1] # Is expected to be in knots

    if parcel.bplus < 1e-3:
        # The parcel doesn't have any positively buoyant areas.
        return np.ma.masked, np.nan

    if not utils.QC(elhght):
        elhght = prof.hght[-1]

    while z_0 < elhght:
        t_1 = delta_t + t_0 # the time step increment
        
        # Compute the vertical acceleration
        env_tempv = interp.vtmp(prof, p_0) + 273.15
        pcl_tempv = interp.generic_interp_pres(np.log10(p_0), np.log10(p_parcel.copy())[::-1], t_parcel[::-1]) + 273.15
        accel = g * ((pcl_tempv - env_tempv) / env_tempv)
        
        # Compute the vertical displacement
        z_1 = (.5 * accel * np.power(t_1 - t_0, 2)) + (w_0 * (t_1 - t_0)) + z_0
        w_1 = accel * (t_1 - t_0) + w_0
        
        # Compute the parcel-relative winds
        u, v = interp.components(prof, p_0)
        u_0 = utils.KTS2MS(u - smu)
        v_0 = utils.KTS2MS(v - smv)
        
        # Compute the horizontal displacements
        x_1 = u_0 * (t_1 - t_0) + x_0
        y_1 = v_0 * (t_1 - t_0) + y_0
        
        pos_vector.append((x_1, y_1, z_1))
        speed_vector.append((u_0, v_0, w_1))

        # Update parcel position
        z_0 = z_1
        y_0 = y_1
        x_0 = x_1
        t_0 = t_1
        p_0 = interp.pres(prof, interp.to_msl(prof, z_1))
        if ma.is_masked(p_0):
            print "p_0 is masked. Can't continue slinky"
            break
        
        # Update parcel vertical velocity
        w_0 = w_1
    
    # Compute the angle tilt of the updraft
    r = np.sqrt(np.power(pos_vector[-1][0], 2) + np.power(pos_vector[-1][1], 2))
    theta = np.degrees(np.arctan2(pos_vector[-1][2],r))
    return pos_vector, theta

def cape(prof, pbot=None, ptop=None, dp=-1, **kwargs):
    '''        
        Lifts the specified parcel, calculates various levels and parameters from
        the profile object. Only B+/B- are calculated based on the specified layer. 
        
        This is a convenience function for effective_inflow_layer and convective_temp, 
        as well as any function that needs to lift a parcel in an iterative process.
        This function is a stripped back version of the parcelx function, that only
        handles bplus and bminus. The intention is to reduce the computation time in
        the iterative functions by reducing the calculations needed.

        This method of creating a stripped down parcelx function for CAPE/CIN calculations
        was developed by Greg Blumberg and Kelton Halbert and later implemented in
        SPC's version of SHARP to speed up their program.
        
        For full parcel objects, use the parcelx function.
        
        !! All calculations use the virtual temperature correction unless noted. !!
        
        Parameters
        ----------
        prof : profile object
        Profile Object
        pbot : number (optional; default surface)
        Pressure of the bottom level (hPa)
        ptop : number (optional; default 400 hPa)
        Pressure of the top level (hPa)
        pres : number (optional)
        Pressure of parcel to lift (hPa)
        tmpc : number (optional)
        Temperature of parcel to lift (C)
        dwpc : number (optional)
        Dew Point of parcel to lift (C)
        dp : negative integer (optional; default = -1)
        The pressure increment for the interpolated sounding
        exact : bool (optional; default = False)
        Switch to choose between using the exact data (slower) or using
        interpolated sounding at 'dp' pressure levels (faster)
        flag : number (optional; default = 5)
        Flag to determine what kind of parcel to create; See DefineParcel for
        flag values
        lplvals : lifting parcel layer object (optional)
        Contains the necessary parameters to describe a lifting parcel
        
        Returns
        -------
        pcl : parcel object
        Parcel Object
    
    '''
    flag = kwargs.get('flag', 5)
    pcl = Parcel(pbot=pbot, ptop=ptop)
    pcl.lplvals = kwargs.get('lplvals', DefineParcel(prof, flag))
    if prof.pres.compressed().shape[0] < 1: return pcl
    
    # Variables
    pres = kwargs.get('pres', pcl.lplvals.pres)
    tmpc = kwargs.get('tmpc', pcl.lplvals.tmpc)
    dwpc = kwargs.get('dwpc', pcl.lplvals.dwpc)
    pcl.pres = pres
    pcl.tmpc = tmpc
    pcl.dwpc = dwpc
    totp = 0.
    totn = 0.
    tote = 0.
    cinh_old = 0.
    
    # See if default layer is specified
    if not pbot:
        pbot = prof.pres[prof.sfc]
        pcl.blayer = pbot
        pcl.pbot = pbot
    if not ptop:
        ptop = prof.pres[prof.pres.shape[0]-1]
        pcl.tlayer = ptop
        pcl.ptop = ptop
    
    # Make sure this is a valid layer
    if pbot > pres:
        pbot = pres
        pcl.blayer = pbot
    if type(interp.vtmp(prof, pbot)) == type(ma.masked): return ma.masked
    if type(interp.vtmp(prof, ptop)) == type(ma.masked): return ma.masked
    
    # Begin with the Mixing Layer
    pe1 = pbot
    h1 = interp.hght(prof, pe1)
    tp1 = thermo.virtemp(pres, tmpc, dwpc)
    
    # Lift parcel and return LCL pres (hPa) and LCL temp (C)
    pe2, tp2 = thermo.drylift(pres, tmpc, dwpc)
    blupper = pe2
    h2 = interp.hght(prof, pe2)
    te2 = interp.vtmp(prof, pe2)
    
    # Calculate lifted parcel theta for use in iterative CINH loop below
    # RECALL: lifted parcel theta is CONSTANT from LPL to LCL
    theta_parcel = thermo.theta(pe2, tp2, 1000.)
    
    # Environmental theta and mixing ratio at LPL
    bltheta = thermo.theta(pres, interp.temp(prof, pres), 1000.)
    blmr = thermo.mixratio(pres, dwpc)
    
    # ACCUMULATED CINH IN THE MIXING LAYER BELOW THE LCL
    # This will be done in 'dp' increments and will use the virtual
    # temperature correction where possible
    pp = np.arange(pbot, blupper+dp, dp, dtype=type(pbot))
    hh = interp.hght(prof, pp)
    tmp_env_theta = thermo.theta(pp, interp.temp(prof, pp), 1000.)
    tmp_env_dwpt = interp.dwpt(prof, pp)
    tv_env = thermo.virtemp(pp, tmp_env_theta, tmp_env_dwpt)
    tmp1 = thermo.virtemp(pp, theta_parcel, thermo.temp_at_mixrat(blmr, pp))
    tdef = (tmp1 - tv_env) / thermo.ctok(tv_env)

    tidx1 = np.arange(0, len(tdef)-1, 1)
    tidx2 = np.arange(1, len(tdef), 1)
    lyre = G * (tdef[tidx1]+tdef[tidx2]) / 2 * (hh[tidx2]-hh[tidx1])
    totn = lyre[lyre < 0].sum()
    if not totn: totn = 0.
    
    # Move the bottom layer to the top of the boundary layer
    if pbot > pe2:
        pbot = pe2
        pcl.blayer = pbot

    if pbot < prof.pres[-1]:
        # Check for the case where the LCL is above the 
        # upper boundary of the data (e.g. a dropsonde)
        return pcl
    
    # Find lowest observation in layer
    lptr = ma.where(pbot > prof.pres)[0].min()
    uptr = ma.where(ptop < prof.pres)[0].max()
    
    # START WITH INTERPOLATED BOTTOM LAYER
    # Begin moist ascent from lifted parcel LCL (pe2, tp2)
    pe1 = pbot
    h1 = interp.hght(prof, pe1)
    te1 = interp.vtmp(prof, pe1)
    tp1 = thermo.wetlift(pe2, tp2, pe1)
    lyre = 0
    lyrlast = 0
    for i in xrange(lptr, prof.pres.shape[0]):
        if not utils.QC(prof.tmpc[i]): continue
        pe2 = prof.pres[i]
        h2 = prof.hght[i]
        te2 = prof.vtmp[i]
        tp2 = thermo.wetlift(pe1, tp1, pe2)
        tdef1 = (thermo.virtemp(pe1, tp1, tp1) - te1) / thermo.ctok(te1)
        tdef2 = (thermo.virtemp(pe2, tp2, tp2) - te2) / thermo.ctok(te2)
        lyrlast = lyre
        lyre = G * (tdef1 + tdef2) / 2. * (h2 - h1)
        
        # Add layer energy to total positive if lyre > 0
        if lyre > 0: totp += lyre
        # Add layer energy to total negative if lyre < 0, only up to EL
        else:
            if pe2 > 500.: totn += lyre

        tote += lyre
        pelast = pe1
        pe1 = pe2
        h1 = h2
        te1 = te2
        tp1 = tp2
        # Is this the top of the specified layer
        if i >= uptr and not utils.QC(pcl.bplus):
            pe3 = pe1
            h3 = h1
            te3 = te1
            tp3 = tp1
            lyrf = lyre
            if lyrf > 0:
                pcl.bplus = totp - lyrf
                pcl.bminus = totn
            else:
                pcl.bplus = totp
                if pe2 > 500.: pcl.bminus = totn + lyrf
                else: pcl.bminus = totn
            pe2 = ptop
            h2 = interp.hght(prof, pe2)
            te2 = interp.vtmp(prof, pe2)
            tp2 = thermo.wetlift(pe3, tp3, pe2)
            tdef3 = (thermo.virtemp(pe3, tp3, tp3) - te3) / thermo.ctok(te3)
            tdef2 = (thermo.virtemp(pe2, tp2, tp2) - te2) / thermo.ctok(te2)
            lyrf = G * (tdef3 + tdef2) / 2. * (h2 - h3)
            if lyrf > 0: pcl.bplus += lyrf
            else:
                if pe2 > 500.: pcl.bminus += lyrf
            if pcl.bplus == 0: pcl.bminus = 0.
    return pcl

def parcelx(prof, pbot=None, ptop=None, dp=-1, **kwargs):
    '''
        Lifts the specified parcel, calculated various levels and parameters from
        the profile object. B+/B- are calculated based on the specified layer.
        
        !! All calculations use the virtual temperature correction unless noted. !!
        
        Parameters
        ----------
        prof : profile object
        Profile Object
        pbot : number (optional; default surface)
        Pressure of the bottom level (hPa)
        ptop : number (optional; default 400 hPa)
        Pressure of the top level (hPa)
        pres : number (optional)
        Pressure of parcel to lift (hPa)
        tmpc : number (optional)
        Temperature of parcel to lift (C)
        dwpc : number (optional)
        Dew Point of parcel to lift (C)
        dp : negative integer (optional; default = -1)
        The pressure increment for the interpolated sounding
        exact : bool (optional; default = False)
        Switch to choose between using the exact data (slower) or using
        interpolated sounding at 'dp' pressure levels (faster)
        flag : number (optional; default = 5)
        Flag to determine what kind of parcel to create; See DefineParcel for
        flag values
        lplvals : lifting parcel layer object (optional)
        Contains the necessary parameters to describe a lifting parcel
        
        Returns
        -------
        pcl : parcel object
        Parcel Object
        
        '''
    flag = kwargs.get('flag', 5)
    pcl = Parcel(pbot=pbot, ptop=ptop)
    pcl.lplvals = kwargs.get('lplvals', DefineParcel(prof, flag))
    if prof.pres.compressed().shape[0] < 1: return pcl
    
    # Variables
    pres = kwargs.get('pres', pcl.lplvals.pres)
    tmpc = kwargs.get('tmpc', pcl.lplvals.tmpc)
    dwpc = kwargs.get('dwpc', pcl.lplvals.dwpc)
    pcl.pres = pres
    pcl.tmpc = tmpc
    pcl.dwpc = dwpc
    cap_strength = -9999.
    cap_strengthpres = -9999.
    li_max = -9999.
    li_maxpres = -9999.
    totp = 0.
    totn = 0.
    tote = 0.
    cinh_old = 0.
    
    # See if default layer is specified
    if not pbot:
        pbot = prof.pres[prof.sfc]
        pcl.blayer = pbot
        pcl.pbot = pbot
    if not ptop:
        ptop = prof.pres[prof.pres.shape[0]-1]
        pcl.tlayer = ptop
        pcl.ptop = ptop
    # Make sure this is a valid layer
    if pbot > pres:
        pbot = pres
        pcl.blayer = pbot
    if type(interp.vtmp(prof, pbot)) == type(ma.masked): return ma.masked
    if type(interp.vtmp(prof, ptop)) == type(ma.masked): return ma.masked
    
    # Begin with the Mixing Layer
    pe1 = pbot
    h1 = interp.hght(prof, pe1)
    tp1 = thermo.virtemp(pres, tmpc, dwpc)
    ttrace = [tp1]
    ptrace = [pe1]
    
    # Lift parcel and return LCL pres (hPa) and LCL temp (C)
    pe2, tp2 = thermo.drylift(pres, tmpc, dwpc)
    blupper = pe2
    h2 = interp.hght(prof, pe2)
    te2 = interp.vtmp(prof, pe2)
    pcl.lclpres = min(pe2, prof.pres[prof.sfc]) # Make sure the LCL pressure is
                                                # never below the surface
    pcl.lclhght = interp.to_agl(prof, h2)
    ptrace.append(pe2)
    ttrace.append(thermo.virtemp(pe2, tp2, tp2))
    
    # Calculate lifted parcel theta for use in iterative CINH loop below
    # RECALL: lifted parcel theta is CONSTANT from LPL to LCL
    theta_parcel = thermo.theta(pe2, tp2, 1000.)
    
    # Environmental theta and mixing ratio at LPL
    bltheta = thermo.theta(pres, interp.temp(prof, pres), 1000.)
    blmr = thermo.mixratio(pres, dwpc)
    
    # ACCUMULATED CINH IN THE MIXING LAYER BELOW THE LCL
    # This will be done in 'dp' increments and will use the virtual
    # temperature correction where possible
    pp = np.arange(pbot, blupper+dp, dp, dtype=type(pbot))
    hh = interp.hght(prof, pp)
    tmp_env_theta = thermo.theta(pp, interp.temp(prof, pp), 1000.)
    tmp_env_dwpt = interp.dwpt(prof, pp)
    tv_env = thermo.virtemp(pp, tmp_env_theta, tmp_env_dwpt)
    tmp1 = thermo.virtemp(pp, theta_parcel, thermo.temp_at_mixrat(blmr, pp))
    tdef = (tmp1 - tv_env) / thermo.ctok(tv_env)


    tidx1 = np.arange(0, len(tdef)-1, 1)
    tidx2 = np.arange(1, len(tdef), 1)
    lyre = G * (tdef[tidx1]+tdef[tidx2]) / 2 * (hh[tidx2]-hh[tidx1])
    totn = lyre[lyre < 0].sum()
    if not totn: totn = 0.
    
    # Move the bottom layer to the top of the boundary layer
    if pbot > pe2:
        pbot = pe2
        pcl.blayer = pbot
    
    # Calculate height of various temperature levels
    p0c = temp_lvl(prof, 0.)
    pm10c = temp_lvl(prof, -10.)
    pm20c = temp_lvl(prof, -20.)
    pm30c = temp_lvl(prof, -30.)
    hgt0c = interp.hght(prof, p0c)
    hgtm10c = interp.hght(prof, pm10c)
    hgtm20c = interp.hght(prof, pm20c)
    hgtm30c = interp.hght(prof, pm30c)
    pcl.p0c = p0c
    pcl.pm10c = pm10c
    pcl.pm20c = pm20c
    pcl.pm30c = pm30c
    pcl.hght0c = hgt0c
    pcl.hghtm10c = hgtm10c
    pcl.hghtm20c = hgtm20c
    pcl.hghtm30c = hgtm30c

    if pbot < prof.pres[-1]:
        # Check for the case where the LCL is above the 
        # upper boundary of the data (e.g. a dropsonde)
        return pcl

    # Find lowest observation in layer
    lptr = ma.where(pbot >= prof.pres)[0].min()
    uptr = ma.where(ptop <= prof.pres)[0].max()
    
    # START WITH INTERPOLATED BOTTOM LAYER
    # Begin moist ascent from lifted parcel LCL (pe2, tp2)
    pe1 = pbot
    h1 = interp.hght(prof, pe1)
    te1 = interp.vtmp(prof, pe1)
    tp1 = thermo.wetlift(pe2, tp2, pe1)
    lyre = 0
    lyrlast = 0

    iter_ranges = np.arange(lptr, prof.pres.shape[0])
    ttraces = ma.zeros(len(iter_ranges))
    ptraces = ma.zeros(len(iter_ranges))
    ttraces[:] = ptraces[:] = ma.masked

    for i in iter_ranges:
        if not utils.QC(prof.tmpc[i]): continue
        pe2 = prof.pres[i]
        h2 = prof.hght[i]
        te2 = prof.vtmp[i]
        #te2 = thermo.virtemp(prof.pres[i], prof.tmpc[i], prof.dwpc[i])
        tp2 = thermo.wetlift(pe1, tp1, pe2)
        tdef1 = (thermo.virtemp(pe1, tp1, tp1) - te1) / thermo.ctok(te1)
        tdef2 = (thermo.virtemp(pe2, tp2, tp2) - te2) / thermo.ctok(te2)

        ptraces[i-iter_ranges[0]] = pe2
        ttraces[i-iter_ranges[0]] = thermo.virtemp(pe2, tp2, tp2)
        lyrlast = lyre
        lyre = G * (tdef1 + tdef2) / 2. * (h2 - h1)

        # Add layer energy to total positive if lyre > 0
        if lyre > 0: totp += lyre
        # Add layer energy to total negative if lyre < 0, only up to EL
        else:
            if pe2 > 500.: totn += lyre
        
        # Check for Max LI
        mli = thermo.virtemp(pe2, tp2, tp2) - te2
        if  mli > li_max:
            li_max = mli
            li_maxpres = pe2
        
        # Check for Max Cap Strength
        mcap = te2 - mli
        if mcap > cap_strength:
            cap_strength = mcap
            cap_strengthpres = pe2
        
        tote += lyre
        pelast = pe1
        pe1 = pe2
        te1 = te2
        tp1 = tp2
        
        # Is this the top of the specified layer
        if i >= uptr and not utils.QC(pcl.bplus):
            pe3 = pe1
            h3 = h2
            te3 = te1
            tp3 = tp1
            lyrf = lyre
            if lyrf > 0:
                pcl.bplus = totp - lyrf
                pcl.bminus = totn
            else:
                pcl.bplus = totp
                if pe2 > 500.: pcl.bminus = totn + lyrf
                else: pcl.bminus = totn
            pe2 = ptop
            h2 = interp.hght(prof, pe2)
            te2 = interp.vtmp(prof, pe2)
            tp2 = thermo.wetlift(pe3, tp3, pe2)
            tdef3 = (thermo.virtemp(pe3, tp3, tp3) - te3) / thermo.ctok(te3)
            tdef2 = (thermo.virtemp(pe2, tp2, tp2) - te2) / thermo.ctok(te2)
            lyrf = G * (tdef3 + tdef2) / 2. * (h2 - h3)
            if lyrf > 0: pcl.bplus += lyrf
            else:
                if pe2 > 500.: pcl.bminus += lyrf
            if pcl.bplus == 0: pcl.bminus = 0.
        
        # Is this the freezing level
        if te2 < 0. and not utils.QC(pcl.bfzl):
            pe3 = pelast
            h3 = interp.hght(prof, pe3)
            te3 = interp.vtmp(prof, pe3)
            tp3 = thermo.wetlift(pe1, tp1, pe3)
            lyrf = lyre
            if lyrf > 0.: pcl.bfzl = totp - lyrf
            else: pcl.bfzl = totp
            if not utils.QC(p0c) or p0c > pe3:
                pcl.bfzl = 0
            elif utils.QC(pe2):
                te2 = interp.vtmp(prof, pe2)
                tp2 = thermo.wetlift(pe3, tp3, pe2)
                tdef3 = (thermo.virtemp(pe3, tp3, tp3) - te3) / \
                    thermo.ctok(te3)
                tdef2 = (thermo.virtemp(pe2, tp2, tp2) - te2) / \
                    thermo.ctok(te2)
                lyrf = G * (tdef3 + tdef2) / 2. * (hgt0c - h3)
                if lyrf > 0: pcl.bfzl += lyrf
        
        # Is this the -10C level
        if te2 < -10. and not utils.QC(pcl.wm10c):
            pe3 = pelast
            h3 = interp.hght(prof, pe3)
            te3 = interp.vtmp(prof, pe3)
            tp3 = thermo.wetlift(pe1, tp1, pe3)
            lyrf = lyre
            if lyrf > 0.: pcl.wm10c = totp - lyrf
            else: pcl.wm10c = totp
            if not utils.QC(pm10c) or pm10c > pcl.lclpres:
                pcl.wm10c = 0
            elif utils.QC(pe2):
                te2 = interp.vtmp(prof, pe2)
                tp2 = thermo.wetlift(pe3, tp3, pe2)
                tdef3 = (thermo.virtemp(pe3, tp3, tp3) - te3) / \
                    thermo.ctok(te3)
                tdef2 = (thermo.virtemp(pe2, tp2, tp2) - te2) / \
                    thermo.ctok(te2)
                lyrf = G * (tdef3 + tdef2) / 2. * (hgtm10c - h3)
                if lyrf > 0: pcl.wm10c += lyrf
        
        # Is this the -20C level
        if te2 < -20. and not utils.QC(pcl.wm20c):
            pe3 = pelast
            h3 = interp.hght(prof, pe3)
            te3 = interp.vtmp(prof, pe3)
            tp3 = thermo.wetlift(pe1, tp1, pe3)
            lyrf = lyre
            if lyrf > 0.: pcl.wm20c = totp - lyrf
            else: pcl.wm20c = totp
            if not utils.QC(pm20c) or pm20c > pcl.lclpres:
                pcl.wm20c = 0
            elif utils.QC(pe2):
                te2 = interp.vtmp(prof, pe2)
                tp2 = thermo.wetlift(pe3, tp3, pe2)
                tdef3 = (thermo.virtemp(pe3, tp3, tp3) - te3) / \
                    thermo.ctok(te3)
                tdef2 = (thermo.virtemp(pe2, tp2, tp2) - te2) / \
                    thermo.ctok(te2)
                lyrf = G * (tdef3 + tdef2) / 2. * (hgtm20c - h3)
                if lyrf > 0: pcl.wm20c += lyrf
        
        # Is this the -30C level
        if te2 < -30. and not utils.QC(pcl.wm30c):
            pe3 = pelast
            h3 = interp.hght(prof, pe3)
            te3 = interp.vtmp(prof, pe3)
            tp3 = thermo.wetlift(pe1, tp1, pe3)
            lyrf = lyre
            if lyrf > 0.: pcl.wm30c = totp - lyrf
            else: pcl.wm30c = totp
            if not utils.QC(pm30c) or pm30c > pcl.lclpres:
                pcl.wm30c = 0
            elif utils.QC(pe2):
                te2 = interp.vtmp(prof, pe2)
                tp2 = thermo.wetlift(pe3, tp3, pe2)
                tdef3 = (thermo.virtemp(pe3, tp3, tp3) - te3) / \
                    thermo.ctok(te3)
                tdef2 = (thermo.virtemp(pe2, tp2, tp2) - te2) / \
                    thermo.ctok(te2)
                lyrf = G * (tdef3 + tdef2) / 2. * (hgtm30c - h3)
                if lyrf > 0: pcl.wm30c += lyrf
        
        # Is this the 3km level
        if pcl.lclhght < 3000.:
            if interp.to_agl(prof, h1) <= 3000. and interp.to_agl(prof, h2) >= 3000. and not utils.QC(pcl.b3km):
                pe3 = pelast
                h3 = interp.hght(prof, pe3)
                te3 = interp.vtmp(prof, pe3)
                tp3 = thermo.wetlift(pe1, tp1, pe3)
                lyrf = lyre
                if lyrf > 0: pcl.b3km = totp - lyrf
                else: pcl.b3km = totp
                h4 = interp.to_msl(prof, 3000.)
                pe4 = interp.pres(prof, h4)
                if utils.QC(pe2):
                    te2 = interp.vtmp(prof, pe4)
                    tp2 = thermo.wetlift(pe3, tp3, pe4)
                    tdef3 = (thermo.virtemp(pe3, tp3, tp3) - te3) / \
                        thermo.ctok(te3)
                    tdef2 = (thermo.virtemp(pe4, tp2, tp2) - te2) / \
                        thermo.ctok(te2)
                    lyrf = G * (tdef3 + tdef2) / 2. * (h4 - h3)
                    if lyrf > 0: pcl.b3km += lyrf
        else: pcl.b3km = 0.
        
        # Is this the 4km level
        if pcl.lclhght < 4000.:
            if interp.to_agl(prof, h1) <= 4000. and interp.to_agl(prof, h2) >= 4000. and not utils.QC(pcl.b4km):
                pe3 = pelast
                h3 = interp.hght(prof, pe3)
                te3 = interp.vtmp(prof, pe3)
                tp3 = thermo.wetlift(pe1, tp1, pe3)
                lyrf = lyre
                if lyrf > 0: pcl.b4km = totp - lyrf
                else: pcl.b4km = totp
                h4 = interp.to_msl(prof, 4000.)
                pe4 = interp.pres(prof, h4)
                if utils.QC(pe2):
                    te2 = interp.vtmp(prof, pe4)
                    tp2 = thermo.wetlift(pe3, tp3, pe4)
                    tdef3 = (thermo.virtemp(pe3, tp3, tp3) - te3) / \
                        thermo.ctok(te3)
                    tdef2 = (thermo.virtemp(pe4, tp2, tp2) - te2) / \
                        thermo.ctok(te2)
                    lyrf = G * (tdef3 + tdef2) / 2. * (h4 - h3)
                    if lyrf > 0: pcl.b4km += lyrf
        else: pcl.b4km = 0.

        # Is this the 6km level
        if pcl.lclhght < 6000.:
            if interp.to_agl(prof, h1) <= 6000. and interp.to_agl(prof, h2) >= 6000. and not utils.QC(pcl.b6km):
                pe3 = pelast
                h3 = interp.hght(prof, pe3)
                te3 = interp.vtmp(prof, pe3)
                tp3 = thermo.wetlift(pe1, tp1, pe3)
                lyrf = lyre
                if lyrf > 0: pcl.b6km = totp - lyrf
                else: pcl.b6km = totp
                h4 = interp.to_msl(prof, 6000.)
                pe4 = interp.pres(prof, h4)
                if utils.QC(pe2):
                    te2 = interp.vtmp(prof, pe4)
                    tp2 = thermo.wetlift(pe3, tp3, pe4)
                    tdef3 = (thermo.virtemp(pe3, tp3, tp3) - te3) / \
                        thermo.ctok(te3)
                    tdef2 = (thermo.virtemp(pe4, tp2, tp2) - te2) / \
                        thermo.ctok(te2)
                    lyrf = G * (tdef3 + tdef2) / 2. * (h4 - h3)
                    if lyrf > 0: pcl.b6km += lyrf
        else: pcl.b6km = 0.
        
        h1 = h2

        # LFC Possibility
        if lyre >= 0. and lyrlast <= 0.:
            tp3 = tp1
            #te3 = te1
            pe2 = pe1
            pe3 = pelast
            if interp.vtmp(prof, pe3) < thermo.virtemp(pe3, thermo.wetlift(pe2, tp3, pe3), thermo.wetlift(pe2, tp3, pe3)):
                # Found an LFC, store height/pres and reset EL/MPL
                pcl.lfcpres = pe3
                pcl.lfchght = interp.to_agl(prof, interp.hght(prof, pe3))
                pcl.elpres = ma.masked
                pcl.elhght = ma.masked
                pcl.mplpres = ma.masked
            else:
                while interp.vtmp(prof, pe3) > thermo.virtemp(pe3, thermo.wetlift(pe2, tp3, pe3), thermo.wetlift(pe2, tp3, pe3)) and pe3 > 0:
                    pe3 -= 5
                if pe3 > 0:
                    # Found a LFC, store height/pres and reset EL/MPL
                    pcl.lfcpres = pe3
                    pcl.lfchght = interp.to_agl(prof, interp.hght(prof, pe3))
                    cinh_old = totn
                    tote = 0.
                    li_max = -9999.
                    if cap_strength < 0.: cap_strength = 0.
                    pcl.cap = cap_strength
                    pcl.cappres = cap_strengthpres

                    pcl.elpres = ma.masked
                    pcl.elhght = ma.masked
                    pcl.mplpres = ma.masked

            # Hack to force LFC to be at least at the LCL
            if pcl.lfcpres >= pcl.lclpres:
                pcl.lfcpres = pcl.lclpres
                pcl.lfchght = pcl.lclhght

        # EL Possibility
        if lyre <= 0. and lyrlast >= 0.:
            tp3 = tp1
            #te3 = te1
            pe2 = pe1
            pe3 = pelast
            while interp.vtmp(prof, pe3) < thermo.virtemp(pe3, thermo.wetlift(pe2, tp3, pe3), thermo.wetlift(pe2, tp3, pe3)):
                pe3 -= 5
            pcl.elpres = pe3
            pcl.elhght = interp.to_agl(prof, interp.hght(prof, pcl.elpres))
            pcl.mplpres = ma.masked
            pcl.limax = -li_max
            pcl.limaxpres = li_maxpres
        
        # MPL Possibility
        if tote < 0. and not utils.QC(pcl.mplpres) and utils.QC(pcl.elpres):
            pe3 = pelast
            h3 = interp.hght(prof, pe3)
            te3 = interp.vtmp(prof, pe3)
            tp3 = thermo.wetlift(pe1, tp1, pe3)
            totx = tote - lyre
            pe2 = pelast
            while totx > 0:
                pe2 -= 1
                te2 = interp.vtmp(prof, pe2)
                tp2 = thermo.wetlift(pe3, tp3, pe2)
                h2 = interp.hght(prof, pe2)
                tdef3 = (thermo.virtemp(pe3, tp3, tp3) - te3) / \
                    thermo.ctok(te3)
                tdef2 = (thermo.virtemp(pe2, tp2, tp2) - te2) / \
                    thermo.ctok(te2)
                lyrf = G * (tdef3 + tdef2) / 2. * (h2 - h3)
                totx += lyrf
                tp3 = tp2
                te3 = te2
                pe3 = pe2
            pcl.mplpres = pe2
            pcl.mplhght = interp.to_agl(prof, interp.hght(prof, pe2))
        
        # 500 hPa Lifted Index
        if prof.pres[i] <= 500. and not utils.QC(pcl.li5):
            a = interp.vtmp(prof, 500.)
            b = thermo.wetlift(pe1, tp1, 500.)
            pcl.li5 = a - thermo.virtemp(500, b, b)
        
        # 300 hPa Lifted Index
        if prof.pres[i] <= 300. and not utils.QC(pcl.li3):
            a = interp.vtmp(prof, 300.)
            b = thermo.wetlift(pe1, tp1, 300.)
            pcl.li3 = a - thermo.virtemp(300, b, b)
    
#    pcl.bminus = cinh_old

    if not utils.QC(pcl.bplus): pcl.bplus = totp
    
    # Calculate BRN if available
    bulk_rich(prof, pcl)
    
    # Save params
    if np.floor(pcl.bplus) == 0: pcl.bminus = 0.
    pcl.ptrace = ma.concatenate((ptrace, ptraces))
    pcl.ttrace = ma.concatenate((ttrace, ttraces))

    # Find minimum buoyancy from Trier et al. 2014, Part 1
    idx = np.ma.where(pcl.ptrace >= 500.)[0]
    if len(idx) != 0:
        b = pcl.ttrace[idx] - interp.vtmp(prof, pcl.ptrace[idx])
        idx2 = np.ma.argmin(b)
        pcl.bmin = b[idx2]
        pcl.bminpres = pcl.ptrace[idx][idx2]

    return pcl


def bulk_rich(prof, pcl):
    '''
        Calculates the Bulk Richardson Number for a given parcel.
        
        Parameters
        ----------
        prof : profile object
        Profile object
        pcl : parcel object
        Parcel object
        
        Returns
        -------
        Bulk Richardson Number
        
        '''
    # Make sure parcel is initialized
    if not utils.QC(pcl.lplvals):
        pbot = ma.masked
    elif pcl.lplvals.flag > 0 and pcl.lplvals.flag < 4:
        ptop = interp.pres(prof, interp.to_msl(prof, 6000.))
        pbot = prof.pres[prof.sfc]
    else:
        h0 = interp.hght(prof, pcl.pres)
        try:
            pbot = interp.pres(prof, h0-500.)
        except:
            pbot = ma.masked
        if utils.QC(pbot): pbot = prof.pres[prof.sfc]
        h1 = interp.hght(prof, pbot)
        ptop = interp.pres(prof, h1+6000.)
    
    if not utils.QC(pbot) or not utils.QC(ptop):
        pcl.brnshear = ma.masked
        pcl.brn = ma.masked
        pcl.brnu = ma.masked
        pcl.brnv = ma.masked
        pcl.brnshear = ma.masked
        pcl.brnu = ma.masked
        pcl.brnv = ma.masked
        pcl.brn = ma.masked

        return pcl
    
    # Calculate the lowest 500m mean wind
    p = interp.pres(prof, interp.hght(prof, pbot)+500.)
    mnlu, mnlv = winds.mean_wind(prof, pbot, p)
    
    # Calculate the 6000m mean wind
    mnuu, mnuv = winds.mean_wind(prof, pbot, ptop)
    
    # Make sure CAPE and Shear are available
    if not utils.QC(pcl.bplus) or not utils.QC(mnlu) or not utils.QC(mnuu):
        pcl.brnshear = ma.masked
        pcl.brnu = ma.masked
        pcl.brnv = ma.masked
        pcl.brn = ma.masked
        return pcl
    
    # Calculate shear between levels
    dx = mnuu - mnlu
    dy = mnuv - mnlv
    pcl.brnu = dx
    pcl.brnv = dy
    pcl.brnshear = utils.KTS2MS(utils.mag(dx, dy))
    pcl.brnshear = pcl.brnshear**2 / 2.
    pcl.brn = pcl.bplus / pcl.brnshear
    return pcl


def effective_inflow_layer(prof, ecape=100, ecinh=-250, **kwargs):
    '''
        Calculates the top and bottom of the effective inflow layer based on
        research by Thompson et al. (2004).

        Parameters
        ----------
        prof : profile object
        Profile object
        ecape : number (optional; default=100)
        Minimum amount of CAPE in the layer to be considered part of the
        effective inflow layer.
        echine : number (optional; default=250)
        Maximum amount of CINH in the layer to be considered part of the
        effective inflow layer
        mupcl : parcel object
        Most Unstable Layer parcel

        Returns
        -------
        pbot : number
        Pressure at the bottom of the layer (hPa)
        ptop : number
        Pressure at the top of the layer (hPa)

    '''
    mupcl = kwargs.get('mupcl', None)
    if not mupcl:
        try:
            mupcl = prof.mupcl
        except:
            mulplvals = DefineParcel(prof, flag=3, pres=300)
            mupcl = cape(prof, lplvals=mulplvals)
    mucape   = mupcl.bplus
    mucinh = mupcl.bminus
    pbot = ma.masked
    ptop = ma.masked
    if mucape != 0:
        if mucape >= ecape and mucinh > ecinh:
            # Begin at surface and search upward for effective surface
            for i in xrange(prof.sfc, prof.top):
                pcl = cape(prof, pres=prof.pres[i], tmpc=prof.tmpc[i], dwpc=prof.dwpc[i])
                if pcl.bplus >= ecape and pcl.bminus > ecinh:
                    pbot = prof.pres[i]
                    break

            if not utils.QC(pbot): 
                return ma.masked, ma.masked

            bptr = i
            # Keep searching upward for the effective top
            for i in xrange(bptr+1, prof.top):
                if not prof.dwpc[i] or not prof.tmpc[i]:
                    continue
                pcl = cape(prof, pres=prof.pres[i], tmpc=prof.tmpc[i], dwpc=prof.dwpc[i])
                if pcl.bplus < ecape or pcl.bminus <= ecinh: #Is this a potential "top"?
                    j = 1
                    while not utils.QC(prof.dwpc[i-j]) and not utils.QC(prof.tmpc[i-j]):
                        j += 1
                    ptop = prof.pres[i-j]
                    if ptop > pbot: ptop = pbot
                    break

    return pbot, ptop

def bunkers_storm_motion(prof, **kwargs):
    '''
        Compute the Bunkers Storm Motion for a right moving supercell using a
        parcel based approach. This code is consistent with the findings in
        Bunkers et. al 2014, using the Effective Inflow Base as the base, and
        65% of the most unstable parcel equilibrium level height using the
        pressure weighted mean wind.

        Parameters
        ----------
        prof : profile object
        Profile Object
        pbot : float (optional)
        Base of effective-inflow layer (hPa)
        mupcl : parcel object (optional)
        Most Unstable Layer parcel

        Returns
        -------
        rstu : number
        Right Storm Motion U-component
        rstv : number
        Right Storm Motion V-component
        lstu : number
        Left Storm Motion U-component
        lstv : number
        Left Storm Motion V-component

    '''
    d = utils.MS2KTS(7.5)   # Deviation value emperically derived at 7.5 m/s
    mupcl = kwargs.get('mupcl', None)
    pbot = kwargs.get('pbot', None)
    if not mupcl:
        try:
            mupcl = prof.mupcl
        except:
            mulplvals = DefineParcel(prof, flag=3, pres=400)
            mupcl = parcelx(prof, lplvals=mulplvals)
    mucape = mupcl.bplus
    mucinh = mupcl.bminus
    muel = mupcl.elhght
    if not pbot:
        pbot, ptop = effective_inflow_layer(prof, 100, -250, mupcl=mupcl)
    if pbot and mucape > 100. and utils.QC(muel):
        base = interp.to_agl(prof, interp.hght(prof, pbot))
        depth = muel - base
        htop = base + ( depth * (65./100.) )
        ptop = interp.pres(prof, interp.to_msl(prof, htop))
        mnu, mnv = winds.mean_wind(prof, pbot, ptop)
        sru, srv = winds.wind_shear(prof, pbot, ptop)
        srmag = utils.mag(sru, srv)
        uchg = d / srmag * srv
        vchg = d / srmag * sru
        rstu = mnu + uchg
        rstv = mnv - vchg
        lstu = mnu - uchg
        lstv = mnv + vchg
    else:
        rstu, rstv, lstu, lstv = winds.non_parcel_bunkers_motion(prof)
    
    return rstu, rstv, lstu, lstv


def convective_temp(prof, **kwargs):
    '''
        Computes the convective temperature, assuming no change in the moisture
        profile. Parcels are iteratively lifted until only mincinh is left as a
        cap. The first guess is the observed surface temperature.
        
        Parameters
        ----------
        prof : profile object
        Profile Object
        mincinh : parcel object (optional; default -1)
        Amount of CINH left at CI
        pres : number (optional)
        Pressure of parcel to lift (hPa)
        tmpc : number (optional)
        Temperature of parcel to lift (C)
        dwpc : number (optional)
        Dew Point of parcel to lift (C)
        
        Returns
        -------
        Convective Temperature (float) in degrees C
        
        '''
    mincinh = kwargs.get('mincinh', 0.)
    mmr = mean_mixratio(prof)
    pres = kwargs.get('pres', prof.pres[prof.sfc])
    tmpc = kwargs.get('tmpc', prof.tmpc[prof.sfc])
    dwpc = kwargs.get('dwpc', thermo.temp_at_mixrat(mmr, pres))
    
    # Do a quick search to fine whether to continue. If you need to heat
    # up more than 25C, don't compute.
    pcl = cape(prof, flag=5, pres=pres, tmpc=tmpc+25., dwpc=dwpc)
    if pcl.bplus == 0. or pcl.bminus < mincinh: return ma.masked
    excess = dwpc - tmpc
    if excess > 0: tmpc = tmpc + excess + 4.
    pcl = cape(prof, flag=5, pres=pres, tmpc=tmpc, dwpc=dwpc)
    if pcl.bplus == 0.: pcl.bminus = ma.masked
    while pcl.bminus < mincinh:
        if pcl.bminus < -100: tmpc += 2.
        else: tmpc += 0.5
        pcl = cape(prof, flag=5, pres=pres, tmpc=tmpc, dwpc=dwpc)
        if pcl.bplus == 0.: pcl.bminus = ma.masked
    return tmpc

def tei(prof):
    '''
        Theta-E Index (TEI)
        TEI is the difference between the surface theta-e and the minimum theta-e value
        in the lowest 400 mb AGL
       
        Note: This is the definition of TEI on the SPC help page,
        but these calculations do not match up with the TEI values on the SPC Online Soundings.
        The TEI values online are more consistent with the max Theta-E
        minus the minimum Theta-E found in the lowest 400 mb AGL.

        This is what our TEI calculation shall be for the time being.

        Parameters
        ----------
        prof : Profile object
        
        Returns
        -------
        tei : theta-e index
        '''
    
    sfc_theta = prof.thetae[prof.sfc]
    sfc_pres = prof.pres[prof.sfc]
    top_pres = sfc_pres - 400.
    
    layer_idxs = ma.where(prof.pres >= top_pres)[0]
    min_thetae = ma.min(prof.thetae[layer_idxs])
    max_thetae = ma.max(prof.thetae[layer_idxs])

    #tei = sfc_theta - min_thetae
    tei = max_thetae - min_thetae
    return tei

def esp(prof, **kwargs):
    
    '''
        Enhanced Stretching Potential (ESP)
        This composite parameter identifies areas where low-level buoyancy
        and steep low-level lapse rates are co-located, which may
        favor low-level vortex stretching and tornado potential.
       
        REQUIRES: 0-3 km MLCAPE (from MLPCL)

        Parameters
        ----------
        prof : Profile object
        mlpcl : Mixed-Layer Parcel object (optional)

        Returns
        -------
        esp : ESP index
        '''
     
    mlpcl = kwargs.get('mlpcl', None)
    if not mlpcl:
        try:
            mlpcl = prof.mlpcl
        except:
            mlpcl = parcelx(prof, flag=4)
    mlcape = mlpcl.b3km
    
    lr03 = prof.lapserate_3km # C/km
    if lr03 < 7. or mlpcl.bplus < 250.:
        return 0
    esp = (mlcape / 50.) * ((lr03 - 7.0) / (1.0))
    
    return esp

def sherb(prof, **kwargs):
    '''
        Severe Hazards In Environments with Reduced Buoyancy (SHERB) Parameter (*)

        A composite parameter designed to assist forecasters in the High-Shear
        Low CAPE (HSLC) environment.  This allows better discrimination 
        between significant severe and non-severe convection in HSLC enviroments.

        It can detect significant tornadoes and significant winds.  Values above
        1 are more likely associated with significant severe.

        See Sherburn et. al. 2014 WAF for more information

        REQUIRES (if effective==True): The effective inflow layer be defined

        Parameters
        ----------
        prof : Profile object
        effective : True or False...use the effective layer computation or not
                    the effective bulk wind difference (prof.ebwd) must exist first
                    if not specified it will default to False (optional)
        ebottom : bottom of the effective inflow layer (mb) (optional) 
        etop :top of the effective inflow layer (mb) (optional) 
        mupcl : Most-Unstable Parcel (optional)

        Returns
        -------
        sherb : an integer for the SHERB parameter
                if effective==True and an effective inflow layer cannot be found,
                this function returns prof.missing

        '''

    effective = kwargs.get('effective', False)
    ebottom = kwargs.get('ebottom', None)
    etop = kwargs.get('etop', None)

    lr03 = lapse_rate(prof, 0, 3000, pres=False)
    lr75 = lapse_rate(prof, 700, 500, pres=True)

    if effective == False:
        p3km = interp.pres(prof, interp.to_msl(prof, 3000))
        sfc_pres = prof.pres[prof.sfc]
        shear = utils.KTS2MS(utils.mag(*winds.wind_shear(prof, pbot=sfc_pres, ptop=p3km)))
        sherb = ( shear / 26. ) * ( lr03 / 5.2 ) * ( lr75 / 5.6 )
    else:
        if hasattr(prof, 'ebwd'):
            # If the Profile object has the attribute "ebwd"
            shear = utils.KTS2MS(utils.mag( prof.ebwd[0], prof.ebwd[1] ))

        elif ((not ebottom) or (not etop)) or \
             ((not hasattr(prof, 'ebottom') or not hasattr(prof, 'etop'))):
            # if the effective inflow layer hasn't been specified via the function arguments
            # or doesn't exist in the Profile object we need to calculate it, but we need mupcl
            if ebottom is None or etop is None:
                # only calculate ebottom and etop if they're not supplied by the kwargs
                if not hasattr(prof, 'mupcl') or not kwargs.get('mupcl', None):
                    # If the mupcl attribute doesn't exist in the Profile
                    # or the mupcl hasn't been passed as an argument
                    # compute the mupcl
                    mulplvals = DefineParcel(prof, flag=3, pres=300)
                    mupcl = cape(prof, lplvals=mulplvals)
                else:
                    mupcl = prof.mupcl
           
                # Calculate the effective inflow layer
                ebottom, etop = effective_inflow_layer( prof, mupcl=mupcl )
            
            if ebottom is np.masked or etop is np.masked:
                # If the inflow layer doesn't exist, return missing
                return prof.missing
            else:
                # Calculate the Effective Bulk Wind Difference
                ebotm = interp.to_agl(prof, interp.hght(prof, ebottom))
                depth = ( mupcl.elhght - ebotm ) / 2
                elh = interp.pres(prof, interp.to_msl(prof, ebotm + depth))
                ebwd = winds.wind_shear(prof, pbot=ebottom, ptop=elh)
        else:
            # If there's no way to compute the effective SHERB
            # because there's no information about how to get the
            # inflow layer, return missing.
            return prof.missing
        
        shear = utils.KTS2MS(utils.mag( prof.ebwd[0], prof.ebwd[1] ))
        sherb = ( shear / 27. ) * ( lr03 / 5.2 ) * ( lr75 / 5.6 )

    return sherb

def mmp(prof, **kwargs):

    """
        MCS Maintenance Probability (MMP)
        The probability that a mature MCS will maintain peak intensity
        for the next hour.
        
        This equation was developed using proximity soundings and a regression equation
        Uses MUCAPE, 3-8 km lapse rate, maximum bulk shear, 3-12 km mean wind speed
        From Coniglio et. al. 2006 WAF
        
        REQUIRES: MUCAPE (J/kg) 

        Parameters
        ----------
        prof : Profile object
        mupcl : Most-Unstable Parcel object (optional)
        
        Returns
        -------
        mmp : MMP index (%)
        
        Note:
        Per Mike Coniglio (personal comm.), the maximum deep shear value is computed by
        computing the shear vector between all the wind vectors
        in the lowest 1 km and all the wind vectors in the 6-10 km layer.
        The maximum speed shear from this is the max_bulk_shear value (m/s).
        """
    
    mupcl = kwargs.get('mupcl', None)
    if not mupcl:
        try:
            mupcl = prof.mupcl
        except:
            mulplvals = DefineParcel(prof, flag=3, pres=300)
            mupcl = cape(prof, lplvals=mulplvals)
    mucape = mupcl.bplus

    if mucape < 100.:
        return 0.

    agl_hght = interp.to_agl(prof, prof.hght)
    lowest_idx = np.where(agl_hght <= 1000)[0]
    highest_idx = np.where((agl_hght >= 6000) & (agl_hght < 10000))[0]
    if len(lowest_idx) == 0 or len(highest_idx) == 0:
        return ma.masked
    possible_shears = np.empty((len(lowest_idx),len(highest_idx)))
    pbots = interp.pres(prof, prof.hght[lowest_idx])
    ptops = interp.pres(prof, prof.hght[highest_idx])

    for b in xrange(len(pbots)):
        for t in xrange(len(ptops)):
            if b < t: continue
            u_shear, v_shear = winds.wind_shear(prof, pbot=pbots[b], ptop=ptops[t])
            possible_shears[b,t] = utils.mag(u_shear, v_shear)
    max_bulk_shear = utils.KTS2MS(np.nanmax(possible_shears.ravel()))
    lr38 = lapse_rate(prof, 3000., 8000., pres=False)
    plower = interp.pres(prof, interp.to_msl(prof, 3000.))
    pupper = interp.pres(prof, interp.to_msl(prof, 12000.))
    mean_wind_3t12 = winds.mean_wind( prof, pbot=plower, ptop=pupper)
    mean_wind_3t12 = utils.KTS2MS(utils.mag(mean_wind_3t12[0], mean_wind_3t12[1]))

    a_0 = 13.0 # unitless
    a_1 = -4.59*10**-2 # m**-1 * s
    a_2 = -1.16 # K**-1 * km
    a_3 = -6.17*10**-4 # J**-1 * kg
    a_4 = -0.17 # m**-1 * s
    
    mmp = 1. / (1. + np.exp(a_0 + (a_1 * max_bulk_shear) + (a_2 * lr38) + (a_3 * mucape) + (a_4 * mean_wind_3t12)))
    
    return mmp

def wndg(prof, **kwargs):
    '''
        Wind Damage Parameter (WNDG)

        A non-dimensional composite parameter that identifies areas
        where large CAPE, steep low-level lapse rates,
        enhanced flow in the low-mid levels, and minimal convective
        inhibition are co-located.
        
        WNDG values > 1 favor an enhanced risk for scattered damaging
        outflow gusts with multicell thunderstorm clusters, primarily
        during the afternoon in the summer.
        
        REQUIRES: MLCAPE (J/kg), MLCIN (J/kg)

        Parameters
        ----------
        prof : Profile object
        mlpcl : Mixed-Layer Parcel object (optional) 

        Returns
        -------
        wndg : WNDG index
        '''
    
    mlpcl = kwargs.get('mlpcl', None)
    if not mlpcl:
        try:
            mlpcl = prof.mlpcl
        except:
            mllplvals = DefineParcel(prof, flag=4)
            mlpcl = cape(prof, lplvals=mllplvals)
    mlcape = mlpcl.bplus

    lr03 = lapse_rate( prof, 0, 3000., pres=False ) # C/km
    bot = interp.pres( prof, interp.to_msl( prof, 1000. ) )
    top = interp.pres( prof, interp.to_msl( prof, 3500. ) )
    mean_wind = winds.mean_wind(prof, pbot=bot, ptop=top) # needs to be in m/s
    mean_wind = utils.KTS2MS(utils.mag(mean_wind[0], mean_wind[1]))
    mlcin = mlpcl.bminus # J/kg
    
    if lr03 < 7:
        lr03 = 0.
    
    if mlcin < -50:
        mlcin = -50.
    wndg = (mlcape / 2000.) * (lr03 / 9.) * (mean_wind / 15.) * ((50. + mlcin)/40.)
    
    return wndg


def sig_severe(prof, **kwargs):
    '''
        Significant Severe (SigSevere)
        Craven and Brooks, 2004
        
        REQUIRES: MLCAPE (J/kg), 0-6km Shear (kts)

        Parameters
        ----------
        prof : Profile object
        mlpcl : Mixed-Layer Parcel object (optional) 

        Returns
        -------
        sigsevere : significant severe parameter (m3/s3)
    '''
     
    mlpcl = kwargs.get('mlpcl', None)
    sfc6shr = kwargs.get('sfc6shr', None)
    if not mlpcl:
        try:
            mlpcl = prof.mlpcl
        except:
            mllplvals = DefineParcel(prof, flag=4)
            mlpcl = cape(prof, lplvals=mllplvals)
    mlcape = mlpcl.bplus

    if not sfc6shr:
        try:
            sfc_6km_shear = prof.sfc_6km_shear
        except:
            sfc = prof.pres[prof.sfc]
            p6km = interp.pres(prof, interp.to_msl(prof, 6000.))
            sfc_6km_shear = winds.wind_shear(prof, pbot=sfc, ptop=p6km)

    sfc_6km_shear = utils.mag(sfc_6km_shear[0], sfc_6km_shear[1])
    shr06 = utils.KTS2MS(sfc_6km_shear)
    
    sigsevere = mlcape * shr06
    return sigsevere

def dcape(prof):
    '''
        Downdraft CAPE (DCAPE)
        
        Adapted from John Hart's (SPC) DCAPE code in NSHARP donated by Rich Thompson (SPC)

        Calculates the downdraft CAPE value using the downdraft parcel source found in the lowest
        400 mb of the sounding.  This downdraft parcel is found by identifying the minimum 100 mb layer 
        averaged Theta-E.

        Afterwards, this parcel is lowered to the surface moist adiabatically (w/o virtual temperature
        correction) and the energy accumulated is called the DCAPE.

		Future adaptations of this function may utilize the Parcel/DefineParcel object.

        Parameters
        ----------
        prof : Profile object
        
        Returns
        -------
        dcape : downdraft CAPE (J/kg)
        ttrace : downdraft parcel trace temperature (C)
        ptrace : downdraft parcel trace pressure (mb)
        '''
    
    sfc_pres = prof.pres[prof.sfc]
    prof_thetae = prof.thetae
    prof_wetbulb = prof.wetbulb
    mask1 = prof_thetae.mask
    mask2 = prof.pres.mask
    mask = np.maximum( mask1, mask2 )
    prof_thetae = prof_thetae[~mask]
    prof_wetbulb = prof_wetbulb[~mask]
    pres = prof.pres[~mask]
    hght = prof.hght[~mask]
    dwpc = prof.dwpc[~mask]
    tmpc = prof.tmpc[~mask]
    idx = np.where(pres >= sfc_pres - 400.)[0]

    # Find the minimum average theta-e in a 100 mb layer
    mine = 1000.0
    minp = -999.0
    for i in idx:
        thta_e_mean = mean_thetae(prof, pbot=pres[i], ptop=pres[i]-100.)
        if utils.QC(thta_e_mean) and thta_e_mean < mine:
            minp = pres[i] - 50.
            mine = thta_e_mean

    upper = minp
    uptr = np.where(pres >= upper)[0]
    uptr = uptr[-1]
    
    # Define parcel starting point
    tp1 = thermo.wetbulb(upper, interp.temp(prof, upper), interp.dwpt(prof, upper))
    pe1 = upper
    te1 = interp.temp(prof, pe1)
    h1 = interp.hght(prof, pe1)
    tote = 0
    lyre = 0

    # To keep track of the parcel trace from the downdraft
    ttrace = [tp1] 
    ptrace = [upper]

    # Lower the parcel to the surface moist adiabatically and compute
    # total energy (DCAPE)
    iter_ranges = xrange(uptr, -1, -1)
    ttraces = ma.zeros(len(iter_ranges))
    ptraces = ma.zeros(len(iter_ranges))
    ttraces[:] = ptraces[:] = ma.masked
    for i in iter_ranges:
        pe2 = pres[i]
        te2 = tmpc[i]
        h2 = hght[i]
        tp2 = thermo.wetlift(pe1, tp1, pe2)

        if utils.QC(te1) and utils.QC(te2):
            tdef1 = (tp1 - te1) / (thermo.ctok(te1))
            tdef2 = (tp2 - te2) / (thermo.ctok(te2))
            lyrlast = lyre
            lyre = 9.8 * (tdef1 + tdef2) / 2.0 * (h2 - h1)
            tote += lyre

        ttraces[i] = tp2
        ptraces[i] = pe2

        pe1 = pe2
        te1 = te2
        h1 = h2
        tp1 = tp2
    drtemp = tp2 # Downrush temp in Celsius

    return tote, ma.concatenate((ttrace, ttraces[::-1])), ma.concatenate((ptrace, ptraces[::-1]))

def precip_eff(prof, **kwargs):
    '''
        Precipitation Efficiency (*)

        This calculation comes from Noel and Dobur 2002, published
        in NWA Digest Vol 26, No 34.

        The calculation multiplies the PW from the whole atmosphere
        by the 1000 - 700 mb mean relative humidity (in decimal form)

        Values on the SPC Mesoanalysis range from 0 to 2.6.

        Larger values means that the precipitation is more efficient.

        Parameters
        ----------
        prof : Profile object
               if the Profile object does not have a pwat attribute
               this function will perform the calculation.
        pwat : (optional) precomputed precipitable water vapor (inch)
        pbot : (optional) the bottom pressure of the RH layer (mb)
        ptop : (optional) the top pressure of the RH layer (mb)

        Returns
        -------
        precip_efficency : the PE value (units inches)

    '''
    
    pw = kwargs.get('pwat', None)
    pbot = kwargs.get('pbot', 1000)
    ptop = kwargs.get('ptop', 700)

    if pw is None or not hasattr(prof, 'pwat'):
        pw = precip_water(prof)
    else:
        pw = prof.pwat

    mean_rh = mean_relh(prof, pbot=pbot, ptop=ptop) / 100.

    return pw*mean_rh

def pbl_top(prof):
    '''
        Planetary Boundary Layer Depth
        Adapted from NSHARP code donated by Rich Thompson (SPC)

        Calculates the planetary boundary layer depth by calculating the 
        virtual potential temperature of the surface parcel + .5 K, and then searching
        for the location above the surface where the virtual potential temperature of the profile
        is greater than the surface virtual potential temperature.

        While this routine suggests a parcel lift, this Python adaptation does not use loop
        like parcelx().

        Parameters
        ----------
        prof : Profile object

        Returns
        -------
        ppbl_top : the pressure that corresponds to the top of the PBL
    '''

    thetav = thermo.theta(prof.pres, thermo.virtemp(prof.pres, prof.tmpc, prof.dwpc))
    try:
        level = np.where(thetav[prof.sfc]+.5 < thetav)[0][0]
    except IndexError:
        print "Warning: PBL top could not be found."
        level = thetav.shape[0] - 1

    return prof.pres[level]

def dcp(prof):
    '''
        Derecho Composite Parameter (*)

        This parameter is based on a data set of 113 derecho events compiled by Evans and Doswell (2001).
        The DCP was developed to identify environments considered favorable for cold pool "driven" wind
        events through four primary mechanisms:

        1) Cold pool production [DCAPE]
        2) Ability to sustain strong storms along the leading edge of a gust front [MUCAPE]
        3) Organization potential for any ensuing convection [0-6 km shear]
        4) Sufficient flow within the ambient environment to favor development along downstream portion of the
            gust front [0-6 km mean wind].

        This index is fomulated as follows:
        DCP = (DCAPE/980)*(MUCAPE/2000)*(0-6 km shear/20 kt)*(0-6 km mean wind/16 kt)

        Reference:
        Evans, J.S., and C.A. Doswell, 2001: Examination of derecho environments using proximity soundings. Wea. Forecasting, 16, 329-342.

        Parameters
        ----------
        prof : Profile object

        Returns
        -------
        dcp : number
            Derecho Composite Parameter (unitless)

    '''
    sfc = prof.pres[prof.sfc]
    p6km = interp.pres(prof, interp.to_msl(prof, 6000.))
    dcape_val = getattr(prof, 'dcape', dcape( prof )[0])
    mupcl = getattr(prof, 'mupcl', parcelx(prof, flag=3))
    sfc_6km_shear = getattr(prof, 'sfc_6km_shear', winds.wind_shear(prof, pbot=sfc, ptop=p6km))
    mean_6km = getattr(prof, 'mean_6km', utils.comp2vec(*winds.mean_wind(prof, pbot=sfc, ptop=p6km)))
    mag_shear = utils.mag(sfc_6km_shear[0], sfc_6km_shear[1])
    mag_mean_wind = mean_6km[1]

    dcp = (dcape_val/980.) * (mupcl.bplus/2000.) * (mag_shear / 20. ) * (mag_mean_wind / 16.)

    return dcp


def mburst(prof):
    '''
        Microburst Composite Index

        Formulated by Chad Entremont NWS JAN 12/7/2014
        Code donated by Rich Thompson (SPC)

        Below is taken from the SPC Mesoanalysis:
        The Microburst Composite is a weighted sum of the following individual parameters: SBCAPE, SBLI,
        lapse rates, vertical totals (850-500 mb temperature difference), DCAPE, and precipitable water.

        All of the terms are summed to arrive at the final microburst composite value.
        The values can be interpreted in the following manner: 3-4 infers a "slight chance" of a microburst;
        5-8 infers a "chance" of a microburst; >= 9 infers that microbursts are "likely".
        These values can also be viewed as conditional upon the existence of a storm.
	
	    This code was updated on 9/11/2018 - TT was being used in the function instead of VT.
    	The original SPC code was checked to confirm this was the problem.
	    This error was not identified during the testing phase for some reason.

        Parameters
        ----------
        prof : Profile object

        Returns
        -------
        mburst : number
            Microburst Composite (unitless)
    '''

    sbpcl = getattr(prof, 'sfcpcl', parcelx(prof, flag=1))
    lr03 = getattr(prof, 'lapserate_3km', lapse_rate( prof, 0., 3000., pres=False ))
    vt = getattr(prof, 'vertical_totals', v_totals(prof))
    dcape_val = getattr(prof, 'dcape', dcape( prof )[0])
    pwat = getattr(prof, 'pwat', precip_water( prof ))
    tei_val = thetae_diff(prof)

    sfc_thetae = thermo.thetae(sbpcl.lplvals.pres, sbpcl.lplvals.tmpc, sbpcl.lplvals.dwpc)

    # SFC Theta-E term
    if thermo.ctok(sfc_thetae) >= 355:
        te = 1
    else:
        te = 0

    # Surface-based CAPE Term
    if not utils.QC(sbpcl.bplus):
        sbcape_term = np.nan
    else:
        if sbpcl.bplus < 2000:
            sbcape_term = -5
        if sbpcl.bplus >= 2000:
            sbcape_term = 0
        if sbpcl.bplus >= 3300:
            sbcape_term = 1
        if sbpcl.bplus >= 3700:
            sbcape_term = 2
        if sbpcl.bplus >= 4300:
            sbcape_term = 4

    # Surface based LI term
    if not utils.QC(sbpcl.li5):
        sbli_term = np.nan
    else:
        if sbpcl.li5 > -7.5:
            sbli_term = 0
        if sbpcl.li5 <= -7.5:
            sbli_term = 1
        if sbpcl.li5 <= -9.0:
            sbli_term = 2
        if sbpcl.li5 <= -10.0:
            sbli_term = 3

    # PWAT Term
    if not utils.QC(pwat):
        pwat_term = np.nan
    else:
        if pwat < 1.5:
            pwat_term = -3
        else:
            pwat_term = 0

    # DCAPE Term
    if not utils.QC(dcape_val):
        dcape_term = np.nan
    else:
        if pwat > 1.70:
            if dcape_val > 900:
                dcape_term = 1
            else:
                dcape_term = 0
        else:
            dcape_term = 0

    # Lapse Rate Term
    if not utils.QC(lr03):
        lr03_term = np.nan
    else:
        if lr03 <= 8.4:
            lr03_term = 0
        else:
            lr03_term = 1

    # Vertical Totals term
    if not utils.QC(vt):
        vt_term = np.nan
    else:
        if vt < 27:
            vt_term = 0
        elif vt >= 27 and vt < 28:
            vt_term = 1
        elif vt >= 28 and vt < 29:
            vt_term = 2
        else:
            vt_term = 3

    # TEI term?
    if not utils.QC(tei_val):
        ted = np.nan
    else:
        if tei_val >= 35:
            ted = 1
        else:
            ted = 0

    mburst = te + sbcape_term + sbli_term + pwat_term + dcape_term + lr03_term + vt_term + ted

    if mburst < 0:
        mburst = 0
    if np.isnan(mburst):
        mburst = np.ma.masked

    return mburst

def ehi(prof, pcl, hbot, htop, stu=0, stv=0):
    '''
        Energy-Helicity Index

        Computes the energy helicity index (EHI) using a parcel
        object and a profile object.

        The equation is EHI = (CAPE * HELICITY) / 160000.

        Parameters
        ----------
        prof : Profile object
        pcl : Parcel object
        hbot : number
            Height of the bottom of the helicity layer [m]
        htop : number
            Height of the top of the helicity layer [m]
        stu : number
            Storm-relative wind U component [kts]
            (optional; default=0)
        stv : number
            Storm-relative wind V component [kts]
            (optional; default=0)

        Returns
        -------
        ehi : number
            Energy Helicity Index (unitless)
    '''

    helicity = winds.helicity(prof, hbot, htop, stu = stu, stv = stv)[0]
    ehi = (helicity * pcl.bplus) / 160000.

    return ehi

def sweat(prof):
    '''
        SWEAT Index (*)

        Computes the SWEAT (Severe Weather Threat Index) using the following numbers:

        1.) 850 Dewpoint
        2.) Total Totals Index
        3.) 850 mb wind speed
        4.) 500 mb wind speed
        5.) Direction of wind at 500
        6.) Direction of wind at 850
	
	    Formulation taken from 
	    Notes on Analysis and Severe-Storm Forecasting Procedures of the Air Force Global Weather Central, 1972
	    by RC Miller.

        Parameters
        ----------
        prof : Profile object

        Returns
        -------
        sweat : number
            SWEAT Index (number)
    '''

    td850 = interp.dwpt(prof, 850)
    vec850 = interp.vec(prof, 850)
    vec500 = interp.vec(prof, 500)
    tt = getattr(prof, 'totals_totals', t_totals( prof ))

    if td850 > 0:
        term1 = 12. * td850
    else:
        term1 = 0

    if tt < 49:
        term2 = 0
    else:
        term2 = 20. * (tt - 49)

    term3 = 2 * vec850[1]
    term4 = vec500[1]
    if 130 <= vec850[0] and 250 >= vec850[0] and 210 <= vec500[0] and 310 >= vec500[0] and vec500[0] - vec850[0] > 0 and vec850[1] >= 15 and vec500[1] >= 15:
        term5 = 125 * (np.sin( np.radians(vec500[0] - vec850[0])) + 0.2)
    else:
        term5 = 0

    sweat = term1 + term2 + term3 + term4 + term5

    return sweat

def thetae_diff(prof):
    '''
        thetae_diff()

        Adapted from code for thetae_diff2() provided by Rich Thompson (SPC)

        Find the maximum and minimum Theta-E values in the lowest 3000 m of
        the sounding and returns the difference.  Only positive difference values
        (where the minimum Theta-E is above the maximum) are returned.

        Parameters
        ----------
        prof : Profile object

        Returns
        -------
        thetae_diff : the Theta-E difference between the max and min values (K)
    '''

    thetae = getattr(prof, 'thetae', prof.get_thetae_profile())
    idx = np.where(interp.to_agl(prof, prof.hght) <= 3000)[0]
    maxe_idx = np.ma.argmax(thetae[idx])
    mine_idx = np.ma.argmin(thetae[idx])

    maxe_pres = prof.pres[idx][maxe_idx]
    mine_pres = prof.pres[idx][mine_idx]

    thetae_diff = thetae[idx][maxe_idx] - thetae[idx][mine_idx]

    if maxe_pres < mine_pres:
        return 0
    else:
        return thetae_diff

def spot(prof):
    '''
        SPOT Index (*)

        The Surface Potential (SPOT) Index, unlike most other forecasting indices,
        uses only data collected from the surface level.  As such, it has the
        advantage of being able to use surface plots, which usually update hourly
        instead of every 12 to 24 hours as with upper-air observations.

        Using the the SWEAT and SPOT values together tends to offer more skill
        than using either index by itself.

        The SPOT Index is computed using the following numbers:

        1.) Surface ambient temperature (in degrees Fahrenheit)
        2.) Surface dewpoint temperature (in degrees Fahrenheit)
        3.) Altimeter setting (in inches of mercury (inHg))
        4.) Wind direction
        5.) Wind speed (in knots)

        Parameters
        ----------
        prof : Profile object

        Returns
        -------
        spot : number
            SPOT Index (number)
    '''

    tmpc_sfc = prof.tmpc[prof.sfc]
    dwpc_sfc = prof.dwpc[prof.sfc]
    sfc_pres = prof.pres[prof.sfc]
    sfc_hght = prof.hght[prof.sfc]
    wdirsfc = prof.wdir[prof.sfc]
    wspdsfc = prof.wspd[prof.sfc]

    # Translate temperatures from Celsius to Fahrenheit
    tmpf = thermo.ctof(tmpc_sfc)
    dwpf = thermo.ctof(dwpc_sfc)

    # Calculate altimeter setting
    asm = sfc_pres * (( 1 + ((( 1013.25 / sfc_pres ) ** (501800000/2637400451) ) * (( 0.0065 * sfc_hght ) / 288.15))) ** (2637400451/501800000)) # Calculate altimeter setting in mb
    asi = asm * 15200/514731 # Translate altimeter setting from mb to inHg

    # Ambient temperature factor
    taf = tmpf - 60

    # Dewpoint temperature factor
    tdf = dwpf - 55

    # Altimeter setting factor
    if tmpf < 50 and asi < 29.50:
        asf = 50 * ( 30 - asi )
    else:
        asf = 100 * ( 30 - asi )
    
    # Wind vector factor
    if 0 <= wdirsfc and wdirsfc < 40:
        if dwpf < 55:
            wvf = -2 * wspdsfc
        else:
            wvf = -1 * wspdsfc
    elif 40 <= wdirsfc and wdirsfc < 70:
        wvf = 0
    elif 70 <= wdirsfc and wdirsfc < 130:
        if dwpf < 55:
            wvf = wspdsfc / 2
        else:
            wvf = wspdsfc
    elif 130 <= wdirsfc and wdirsfc <= 210:
        if dwpf < 55:
            wvf = wspdsfc
        else:
            wvf = 2 * wspdsfc
    elif 210 < wdirsfc and wdirsfc <= 230:
        if dwpf < 55:
            wvf = 0
        elif 55 <= dwpf and dwpf <= 60:
            wvf = wspdsfc / 2
        else:
            wvf = wspdsfc
    elif 230 < wdirsfc and wdirsfc <= 250:
        if dwpf < 55:
            wvf = -2 * wspdsfc
        elif 55 <= dwpf and dwpf <= 60:
            wvf = -1 * wspdsfc
        else:
            wvf = wspdsfc
    else:
        wvf = -2 * wspdsfc
    
    spot = taf + tdf + asf + wvf

    return spot

def wbz(prof):
    '''
        Wetbulb Zero height

        The wetbulb zero (WBZ) height identifies the height in feet  AGL at which the
        wetbulb temperature equals 0 degrees C.  It is assumed that hailstones above 
        this level do not melt, as even if the ambient temperature is above freezing,
        evaporation would absorb latent heat, chilling the hailstones.  However, if
        the wetbulb temperature is above 0, hailstones will melt as they fall.

        A WBZ height of less than 6,000 feet is usually associated with a relatively
        cool airmass with low CAPE, making it unlikely for large hail to form
        (although there have been occasional instances of large hail falling in areas
        of low WBZ).  A WBZ height of over 12,000 feet suggests that a hailstone will
        fall through a very deep column of warm air over an extended period of time,
        and will likely melt substantially or even completely by the time it reaches
        the ground.

        WBZ heights of between 6,000 and 12,000 feet are usually considered to be a 
        "sweet spot" of sorts for hail to form, with heights in the 7,000-9,000 foot
        range being most associated with large hail.

        Parameters
        ----------
        prof : Profile object

        Returns
        -------
        wbp : mb
            Wetbulb Zero (mb)
        wbz : feet
            Wetbulb Zero (feet AGL)
    '''

    wetbulb = getattr(prof, 'wetbulb', prof.get_wetbulb_profile())
    ind1 = ma.where(wetbulb >= 0)[0]
    ind2 = ma.where(wetbulb <= 0)[0]
    if len(ind1) == 0 or len(ind2) == 0:
        wbp = ma.masked
    inds = np.intersect1d(ind1, ind2)
    if len(inds) > 0:
        wbp = prof.pres[inds][0]
    diff1 = ind1[1:] - ind1[:-1]
    ind = np.where(diff1 > 1)[0] + 1
    try:
        ind = ind.min()
    except:
        ind = ind1[-1]
    
    wbp = np.power(10, np.interp(0, [wetbulb[ind+1], wetbulb[ind]],
            [prof.logp[ind+1], prof.logp[ind]]))
    
    wbz = utils.M2FT(interp.to_agl(prof, interp.hght(prof, wbp)))
    
    return wbp, wbz

def thomp(prof, pcl):
    '''
        Thompson Index (*)

        The Thompson Index is a combination of the K Index and Lifted Index.
        It attempts to integrate elevated moisture into the index, using the
        850 mb dewpoint and 700 mb humidity.  Accordingly, it works best in
        tropical and mountainous locations.

        Parameters
        ----------
        prof : Profile object

        Returns
        -------
        thomp : number
            Thompson Index (number)
    '''

    ki = getattr(prof, 'k_index', k_index(prof))

    thomp = ki - pcl.li5

    return thomp

def tq(prof):
    '''
        TQ Index (*)

        The TQ index is used for assessing the probability of low-topped
        convection.  Values of more than 12 indicate an unstable lower
        troposphere with thunderstorms possible outside of stratiform clouds.
        Values of of more than 17 indicate an unstable lower troposphere
        with thunderstorms possible when stratiform clouds are present.

        Parameters
        ----------
        prof : Profile object

        Returns
        -------
        tq : number
            TQ Index (number)
    '''

    tmp850 = interp.temp(prof, 850)
    dpt850 = interp.dwpt(prof, 850)
    tmp700 = interp.temp(prof, 700)
    
    tq = tmp850 + dpt850 - ( 1.7 * tmp700 )

    return tq

def s_index(prof):
    '''
        S-Index (*)

        This European index is a mix of the K Index and Vertical Totals Index.
        The S-Index was developed by the German Military Geophysical Office.

        Parameters
        ----------
        prof : Profile object

        Returns
        -------
        s_index : number
            S-Index (number)
    '''

    ki = getattr(prof, 'k_index', k_index(prof))
    vt = getattr(prof, 'vertical_totals', v_totals(prof))
    tmp5 = interp.temp(prof, 500)

    if vt < 22:
        af = 6
    elif 22 <= vt and vt <= 25:
        af = 2
    else:
        af = 0
    
    s_index = ki - ( tmp5 + af )

    return s_index

def boyden(prof):
    '''
        Boyden Index (*)

        This index, used in Europe, does not factor in moisture.
        It evaluates thickness and mid-level warmth.  It was defined
        in 1963 by C. J. Boyden.

        Parameters
        ----------
        prof : Profile object

        Returns
        -------
        boyden : number
            Boyden Index (number)
    '''

    # Height in decameters (dam)
    h700 = interp.hght(prof, 700) / 10
    h1000 = interp.hght(prof, 1000) / 10
    tmp700 = interp.temp(prof, 700)
    
    boyden = h700 - h1000 - tmp700 - 200

    return boyden

def dci(prof, pcl):
    '''
        Deep Convective Index (*)

        This index is a combination of parcel theta-e at 850 mb and
        Lifted Index.  This attempts to further improve the Lifted Index.
        It was defined by W. R. Barlow in 1993.

        Parameters
        ----------
        prof : Profile object

        Returns
        -------
        dci : number
            Deep Convective Index (number)
    '''

    tmp850 = interp.temp(prof, 850)
    dpt850 = interp.dwpt(prof, 850)

    dci = tmp850 + dpt850 - pcl.li5

    return dci

def pii(prof):
    '''
        Potential Instability Index (*)

        This index relates potential instability in the middle atmosphere with
        thickness.  It was proposed by A. J. Van Delden in 2001.  Positive values
        indicate increased potential for convective weather.
        
        The units in the original formulation are in degrees Kelvin per meter (K/m);
        however, this formulation will use degrees Kelvin per kilometer (K/km) so as
        to make the values easier to read.

        Parameters
        ----------
        prof : Profile object

        Returns
        -------
        pii : number
            Potential Instability Index (number)
    '''

    te925 = thermo.thetae(925, interp.temp(prof, 925), interp.dwpt(prof, 925))
    te500 = thermo.thetae(500, interp.temp(prof, 500), interp.dwpt(prof, 500))
    z500 = interp.hght(prof, 500)
    z925 = interp.hght(prof, 925)

    th95 = ( z500 - z925 ) / 1000

    pii = ( te925 - te500 ) / th95

    return pii

def ko(prof):
    '''
        KO Index (*)

        This index was developed by Swedish meteorologists and used heavily by the
        Deutsche Wetterdienst.  It compares values of equivalent potential
        temperature at different levels.  It was developed by T. Andersson, M.
        Andersson, C. Jacobsson, and S. Nilsson.

        Parameters
        ----------
        prof : Profile object

        Returns
        -------
        ko : number
            KO Index (number)
    '''

    te500 = thermo.thetae(500, interp.temp(prof, 500), interp.dwpt(prof, 500))
    te700 = thermo.thetae(700, interp.temp(prof, 700), interp.dwpt(prof, 700))
    te850 = thermo.thetae(850, interp.temp(prof, 850), interp.dwpt(prof, 850))
    sfc_pres = prof.pres[prof.sfc]

    if sfc_pres < 1000:
        pr1s = sfc_pres
    else:
        pr1s = 1000
    
    te1s = thermo.thetae(pr1s, interp.temp(prof, pr1s), interp.dwpt(prof, pr1s))

    ko = ( 0.5 * ( te500 + te700 ) ) - ( 0.5 * ( te850 + te1s ) )

    return ko

def brad(prof):
    '''
        Bradbury Index (*)

        Also known as the Potential Wet-Bulb Index, this index is used in Europe.
        It is a measure of the potential instability between 850 and 500 mb.  It 
        was defined in 1977 by T. A. M. Bradbury.

        Parameters
        ----------
        prof : Profile object

        Returns
        -------
        brad : number
            Bradbury Index (number)
    '''

    qw500 = thermo.thetaw(500, interp.temp(prof, 500), interp.dwpt(prof, 500))
    qw850 = thermo.thetaw(850, interp.temp(prof, 850), interp.dwpt(prof, 850))

    brad = qw500 - qw850

    return brad

def rack(prof):
    '''
        Rackliff Index (*)

        This index, used primarily in Europe during the 1950s, is a simple comparison
        of the 900 mb wet bulb temperature with the 500 mb dry bulb temperature.
        It is believed to have been developed by Peter Rackliff during the 1940s.

        Parameters
        ----------
        prof : Profile object

        Returns
        -------
        rack : number
            Rackliff Index (number)
    '''

    qw900 = thermo.thetaw(900, interp.temp(prof, 900), interp.dwpt(prof, 900))
    tmp500 = interp.temp(prof, 500)

    rack = qw900 - tmp500

    return rack

def jeff(prof):
    '''
        Jefferson Index (*)

        A European stability index, the Jefferson Index was intended to be an improvement of the
        Rackliff Index. The change would make it less dependent on temperature. The version used
        since the 1960s is a slight modification of G. J. Jefferson's 1963 definition.

        Parameters
        ----------
        prof : Profile object

        Returns
        -------
        jeff : number
            Jefferson Index (number)
    '''

    qw850 = thermo.thetaw(850, interp.temp(prof, 850), interp.dwpt(prof, 850))
    tmp500 = interp.temp(prof, 500)
    tdd700 = interp.tdd(prof, 700)

    jeff = ( 1.6 * qw850 ) - tmp500 - ( 0.5 * tdd700 ) - 8

    return jeff

def sc_totals(prof):
    '''
        Surface-based Cross Totals (*)

        This index, developed by J. Davies in 1988, is a modification of the Cross Totals index that
        replaces the 850 mb dewpoint with the surface dewpoint.  As such, this index will usually
        give a higher value than the original Cross Totals.

        Parameters
        ----------
        prof : Profile object

        Returns
        -------
        sc_totals : number
            Surface-based Cross Totals (number)
    '''

    sc_totals = prof.dwpc[prof.sfc] - interp.temp(prof, 500)

    return sc_totals

def esi(prof, sbcape):
    '''
        Energy Shear Index (*)

        This index, proposed as a way of parameterizing updraft duration, multiplies SBCAPE by
        850 mb-6 km AGL mean vertical shear magnitude in m/s.  A 2002 study by Brimelow and Reuter
        indicated considerable success with using this index to forecast large hail.  An ESI
        value approaching 5 is considered favorable for large hail, with values above 5 not
        having much further significance.

        Parameters
        ----------
        prof : Profile object
        sbcape : Surface-based Convective Available Potential Energy (J/kg)

        Returns
        -------
        esi : number
            Energy Shear Index (unitless)
    '''

    p6km = interp.pres(prof, interp.to_msl(prof, 6000))
    shr_850mb_6km = winds.wind_shear(prof, pbot=850, ptop=p6km)
    shr_850mb_6km = utils.KTS2MS(utils.mag(*shr_850mb_6km)) / ( 6000 - interp.to_agl(prof, interp.hght(prof, 850)) )

    esi = shr_850mb_6km * sbcape

    return esi

def vgp(prof, pcl, **kwargs):
    '''
        Vorticity Generation Potential (*)

        The Vorticity Generation Potential index was developed by Erik Rasmussen and
        David Blanchard in 1998.  It assesses the possibility for vorticity being
        tilted into the vertical to create rotating updrafts.

        The formula is:

        VGP = sqrt(CAPE) * U03

        where U03 is the mean shear between the surface and 3 km AGL.
        
        Parameters
        ----------
        prof : Profile object
        pcl : Parcel object

        Returns
        -------
        vgp : number
            Vorticity Generation Potential (number)
    '''

    sfc3shr = kwargs.get('sfc3shr', None)

    if not sfc3shr:
        try:
            sfc_3km_shear = prof.sfc_3km_shear
        except:
            sfc_pres = prof.pres[prof.sfc]
            p3km = interp.pres(prof, interp.to_msl(prof, 3000.))
            sfc_3km_shear = winds.wind_shear(prof, pbot=sfc_pres, ptop=p3km)
    
    shr03 = utils.KTS2MS(utils.mag(*sfc_3km_shear))
    mag03_shr = shr03 / 3000

    vgp = mag03_shr * ( pcl.bplus ** 0.5 )

    return vgp

def aded1(prof):
    '''
        Adedokun Index, version 1 (*)

        The Adedokun Index (created by J. A. Adedokun in 1981 and 1982) was developed
        in two versions for forecasting precipitation in west Africa.  The Index 
        lowers a 500 mb parcel moist adiabatically to 1000 mb, then compares it to
        the wet bulb potential temperature (theta-w) of a specified level.

        Version 1 subtracts the parcel's temperature from the theta-w of the 850 mb
        level.  This version has been found to be better for forecasting non-
        occurrence of precipitation.

        For both versions, values >= -1 were defined to be indicative of precipitation
        occurrence while values < -1 were defined to be indicative of precipitation
        non-occurrence.

        Parameters
        ----------
        prof : Profile object

        Returns
        -------
        aded1 : number
            Adedokun Index, version 1 (number)
    '''

    pclm500 = thermo.thetaws(500, interp.temp(prof, 500))
    thtw850 = thermo.thetaw(850, interp.temp(prof, 850), interp.dwpt(prof, 850))

    aded1 = thtw850 - pclm500

    return aded1

def aded2(prof):
    '''
        Adedokun Index, version 2 (*)

        The Adedokun Index (created by J. A. Adedokun in 1981 and 1982) was developed
        in two versions for forecasting precipitation in west Africa.  The Index 
        lowers a 500 mb parcel moist adiabatically to 1000 mb, then compares it to
        the wet bulb potential temperature (theta-w) of a specified level.

        Version 2 subtracts the parcel's temperature from the theta-w of the surface
        level.  This version has been found to be better for forecasting occurrence
        of precipitation.

        For both versions, values >= -1 were defined to be indicative of precipitation
        occurrence while values < -1 were defined to be indicative of precipitation
        non-occurrence.

        Parameters
        ----------
        prof : Profile object

        Returns
        -------
        aded1 : number
            Adedokun Index, version 2 (number)
    '''

    pclm500 = thermo.thetaws(500, interp.temp(prof, 500))
    thtw_sfc = thermo.thetaw(prof.pres[prof.sfc], prof.tmpc[prof.sfc], prof.dwpc[prof.sfc])

    aded2 = thtw_sfc - pclm500

    return aded2

def ei(prof):
    '''
        Energy Index (*)

        The Energy Index (also known as the Total Energy Index) was developed by G. L.
        Darkow in 1968.  It calculates the moist static energy at the 500 and 850 mb levels
        and then subtracts the latter from the former.  The energy is calculated in units
        of cal/gm.  Negative values indicate instability.

        Parameters
        ----------
        prof : Profile object

        Returns
        -------
        ei : number
            Energy Index (number)
    '''

    tmp500 = thermo.ctok(interp.temp(prof, 500)) # Temperature in degrees Kelvin
    hght500 = interp.hght(prof, 500)
    mxr500 = thermo.mixratio(500, interp.dwpt(prof,500))
    tmp850 = thermo.ctok(interp.temp(prof, 850)) # Temperature in degrees Kelvin
    hght850 = interp.hght(prof, 850)
    mxr850 = thermo.mixratio(850, interp.dwpt(prof, 850))

    # Calculate moist static energy in joules/kilogram
    mse5_j = ( 1004.6851 * tmp500 ) + ( G * hght500 ) + ( 2500 * mxr500 )
    mse8_j = ( 1004.6851 * tmp850 ) + ( G * hght850 ) + ( 2500 * mxr850 )

    # Convert moist static energy to calories/gram
    mse5_c = mse5_j / 4186.8
    mse8_c = mse8_j / 4186.8

    ei = mse5_c - mse8_c

    return ei

def eehi(prof, sbcape, mlcape, sblcl, mllcl, srh01, bwd6, **kwargs):
    '''
        Enhanced Energy Helicity Index (*)

        The original 0-1 km EHI presented a normalized product of 0-1 km storm-relative helicity
        (SRH) and 100 mb mean parcel (ML) CAPE. This modified version more closely mimics the
        fixed-layer significant tornado parameter with its inclusion of the same fixed-layer (0-6 km)
        bulk wind difference term (SHR6), and the addition of a 4 km AGL max vertical velocity term (WMAX4).

        If surface-based (SB) CAPE exceeds the MLCAPE, the ML lifting condensation level (LCL) in less than
        1000 m AGL, and the surface temperature - dewpoint depression is no more than 10 F, then the SB
        parcel is used in the EEHI calculation. Otherwise, the calculation defaults to the ML parcel.

        The index is formulated as follows:

        EEHI = ((CAPE * 0-1 km SRH)/ 160000) * SRH6 * WMAX4

        The 0-6 km bulk wind difference term is capped at a value of 1.5 for SRH6 greater than 30 m/s,
        (SHR6 / 20 m/s) for values from 12.5-30 m/s, and set to 0.0 when SHR6 is less than 12.5 m/s.
        The WMAX4 term is capped at 1.5 for WMAX4 greater than 30 m/s, (WMAX4 / 20 m/s) for values
        from 10-30 m/s, and set to 0.0 when WMAX4 is less than 10 m/s. Lastly, the entire index is
        set to 0.0 if the average of the SBLCL and MLLCL is greater than 2000 m AGL.

        This enhanced EHI is meant to highlight tornadic supercell potential into a lower range of buoyancy,
        compared to the fixed-layer significant tornado parameter, with decreased false alarms compared to
        the original 0-1 km EHI. The WMAX4 term reflects the thermodynamic potential for low-level vortex
        stretching, while the SB parcel is used for CAPE calculations in relatively moist environments more
        typical of the cool season or tropical cyclones. Values greater than 1 are associated with greater
        probabilities of tornadic supercells.

        Parameters
        ----------
        prof : Profile object
        pcl : Parcel object
        mlcape : Mixed-layer CAPE from the parcel class (J/kg)
        sbcape : Surface based CAPE from the parcel class (J/kg)
        sblcl : Surface based lifted condensation level (m)
        mllcl : Mixed-layer lifted condensation level (m)
        srh01 : 0-1 km storm-relative helicity (m2/s2)
        bwd6 : Bulk wind difference between 0 to 6 km (m/s)

        Returns
        -------
        eehi : number
            Enhanced Energy Helicity Index (unitless)
    '''

    tddsfc = thermo.ctof(prof.tmpc[prof.sfc]) - thermo.ctof(prof.dwpc[prof.sfc])
    sbpcl = getattr(prof, 'sfcpcl', parcelx(prof, flag=1))
    mupcl = getattr(prof, 'mupcl', parcelx(prof, flag=3))

    mlpcl = kwargs.get('mlpcl', None)
    if not mlpcl:
        try:
            mlpcl = prof.mlpcl
        except:
            mlpcl = parcelx(prof, flag=4)
    
    if sbcape > mlcape and mllcl < 1000 and tddsfc <= 10:
        capef = sbcape
    else:
        capef = mlcape
    
    cape4 = mupcl.b4km
    wmax4 = ( 2 * cape4 ) ** 0.5

    if bwd6 > 30:
        srh6f = 1.5
    elif bwd6 < 12.5:
        srh6f = 0
    else:
        srh6f = bwd6 / 20
    
    if wmax4 > 30:
        wmax4f = 1.5
    elif wmax4 < 10:
        wmax4f = 0
    else:
        wmax4f = wmax4 / 20
    
    if ( sblcl + mllcl ) / 2 > 2000:
        eehi = 0
    else:
        eehi = (( capef * srh01 ) / 160000 ) * srh6f * wmax4f
    
    return eehi

def vtp(prof, mlcape, esrh, ebwd, mllcl, mlcinh, **kwargs):
    '''
        Violent Tornado Parameter (*)

        From Hampshire et. al. 2017, JOM page 8.

        Research using observed soundings found that 0-3 km CAPE and 0-3 km lapse rate were notable
        discriminators of violent tornado environments (versus weak and/or significant tornado environments).
        These parameters were combined into the effective layer version of the Significant Tornado Parameter
        (STP) to create the Violent Tornado Parameter (VTP).

        Parameters
        ----------
        prof : Profile object
        mlcape : Mixed-layer CAPE from the parcel class (J/kg)
        esrh : effective storm relative helicity (m2/s2)
        ebwd : effective bulk wind difference (m/s)
        mllcl : mixed-layer lifted condensation level (m)
        mlcinh : mixed-layer convective inhibition (J/kg)

        Returns
        -------
        vtp : number
            Violent Tornado Parameter (unitless)
    '''

    cape_term = mlcape / 1500.
    eshr_term = esrh / 150.
    lr03_term = lapse_rate(prof, 0, 3000, pres=False) / 6.5

    mlpcl = kwargs.get('mlpcl', None)
    if not mlpcl:
        try:
            mlpcl = prof.mlpcl
        except:
            mlpcl = parcelx(prof, flag=4)
    
    if ebwd < 12.5:
        ebwd_term = 0.
    elif ebwd > 30.:
        ebwd_term = 1.5
    else:
        ebwd_term  = ebwd / 20.

    if mllcl < 1000.:
        lcl_term = 1.0
    elif mllcl > 2000.:
        lcl_term = 0.0
    else:
        lcl_term = ((2000. - mllcl) / 1000.)

    if mlcinh > -50:
        cinh_term = 1.0
    elif mlcinh < -200:
        cinh_term = 0
    else:
        cinh_term = ((mlcinh + 200.) / 150.)
    
    if mlpcl.b3km > 100:
        cape3_term = 2
    else:
        cape3_term = mlpcl.b3km / 50

    vtp = np.maximum(cape_term * eshr_term * ebwd_term * lcl_term * cinh_term * cape3_term * lr03_term, 0)

    return vtp

def snsq(prof):
    '''
        Snow Squall Parameter (*)

        From Banacos et. al. 2014, JOM page 142.

        A non-dimensional composite parameter that combines 0-2 km AGL relative humidity, 0-2 km AGL
        potential instability (theta-e decreases with height), and 0-2 km AGL mean wind speed (m/s).
        The intent of the parameter is to identify areas with low-level potential instability, sufficient
        moisture, and strong winds to support snow squall development. Surface potential temperatures
        (theta) and MSL pressure are also plotted to identify strong baroclinic zones which often
        provide the focused low-level ascent in cases of narrow snow bands.

        Parameters
        ----------
        prof : Profile object

        Returns
        -------
        snsq : number
            Snow Squall Parameter (unitless)
    '''

    sfc_pres = prof.pres[prof.sfc]
    pres_2km = interp.pres(prof, interp.to_msl(prof, 2000))
    relh02 = mean_relh(prof, pbot=sfc_pres, ptop=pres_2km)
    sfc_thetae = prof.thetae[prof.sfc]
    thetae_2km = interp.thetae(prof, pres_2km)
    thetae_d02 = thetae_2km - sfc_thetae
    mw02 = utils.KTS2MS(utils.mag(*winds.mean_wind_npw(prof, pbot=sfc_pres, ptop=pres_2km)))
    sfc_wtb = prof.wetbulb[prof.sfc]

    if relh02 < 60:
        relhf = 0
    else:
        relhf = ( relh02 - 60 ) / 15
    
    if thetae_d02 > 4:
        thetaef = 0
    else:
        thetaef = ( 4 - thetae_d02 ) / 4
    
    mwf = mw02 / 9

    if sfc_wtb >= 1:
        snsq = 0
    else:
        snsq = relhf * thetaef * mwf
    
    return snsq

def snow(prof):
    '''
        Snow Index (*)

        This index uses two thickness layers: the 850-700 mb thickness layer and the 1000-850 mb thickness
        layer.  A value of greater than 4179 indicates liquid precipitation; a value of 4179 indicates
        mixed precipitation; and a value of less than 4179 indicates solid precipitation.

        Parameters
        ----------
        prof : Profile object

        Returns
        -------
        snow : number
            Snow Index (number)
    '''

    hght1000 = interp.hght(prof, 1000)
    hght850 = interp.hght(prof, 850)
    hght700 = interp.hght(prof, 700)

    thick78 = hght700 - hght850
    thick18 = hght1000 - hght850

    snow = thick78 + ( 2 * thick18 )

    return snow

def windex(prof, **kwargs):
    '''
        Wind Index (*)

        This index, a measure of microburst potential and downdraft instability, estimates maximum
        convective wind gust speeds.  Created by Donald McCann in 1994, the index is displayed in knots.

        Parameters
        ----------
        prof : Profile object

        Returns
        -------
        windex : knots
            WINDEX (knots)
    '''

    frz_lvl = kwargs.get('frz_lvl', None)
    sfc_pres = prof.pres[prof.sfc]
    pres_1km = interp.pres(prof, interp.to_msl(prof, 1000))
    mxr01 = mean_mixratio(prof, pbot=sfc_pres, ptop=pres_1km)
    
    if not frz_lvl:
        frz_lvl = interp.hght(prof, temp_lvl(prof, 0))
    
    frz_pres = interp.pres(prof, frz_lvl)
    frz_dwpt = interp.dwpt(prof, frz_pres)
    mxr_frz = thermo.mixratio(frz_pres, frz_dwpt)
    hm_m = interp.to_agl(prof, frz_lvl)
    hm_km = hm_m / 1000
    lr_frz = lapse_rate(prof, 0, hm_m, pres=False)

    if mxr01 > 12:
        rq = 1
    else:
        rq = mxr01 / 12

    windex = 5 * ( ( hm_km * rq * ((lr_frz ** 2 ) - 30 + mxr01 - ( 2 * mxr_frz )) ) ** 0.5 )

    return windex

def wmsi(prof, **kwargs):
    '''
        Wet Microburst Severity Index (*)

        This index, developed by K. L. Pryor and G. P. Ellrod in 2003, was developed to better
        assess the potential deverity of wet microbursts.  WSMI is a product of CAPE
        (specifically from the most unstable parcel) and delta-Theta-E (see the Theta-E
        Index parameter).

        Parameters
        ----------
        prof : Profile object

        Returns
        -------
        wmsi : number
            Wet Microburst Severity Index (number)
    '''

    mupcl = kwargs.get('mupcl', None)
    sfc_theta = prof.thetae[prof.sfc]
    sfc_pres = prof.pres[prof.sfc]
    top_pres = sfc_pres - 400.

    if not mupcl:
        try:
            mupcl = prof.mupcl
        except:
            mulplvals = DefineParcel(prof, flag=3, pres=300)
            mupcl = cape(prof, lplvals=mulplvals)
    mucape = mupcl.bplus

    layer_idxs = ma.where(prof.pres >= top_pres)[0]
    min_thetae = ma.min(prof.thetae[layer_idxs])

    dthetae = sfc_theta - min_thetae

    wmsi = ( mucape * dthetae ) / 1000

    return wmsi

def dmpi1(prof):
    '''
        Dry Microburst Potential Index, version 1 (*)

        This index was primarily derived by R. Wakimoto in 1985 to forecast potential for
        dry microbursts.
        
        The original index, calculated using soundings in the region of Denver, CO, used
        the 700 and 500 mb layers for its calculations.  However, the RAOB Program manual
        recommends the use of the 5,000 and 13,000-ft AGL layers so that the results can
        be consistently used for any worldwide sounding, regardless of station elevation.
        The decision was made to split the index into two versions, the original (version
        1) and the RAOB (version 2).

        Parameters
        ----------
        prof : Profile object

        Returns
        -------
        dmpi1 : number
            Dry Microburst Potential Index, version 1 (number)
    '''

    tdd500 = interp.tdd(prof, 500)
    tdd700 = interp.tdd(prof, 700)
    lr75 = lapse_rate(prof, 700, 500, pres=True)

    dmpi1 = lr75 + tdd700 - tdd500

    return dmpi1

def dmpi2(prof):
    '''
        Dry Microburst Potential Index, version 2 (*)

        This index was primarily derived by R. Wakimoto in 1985 to forecast potential for
        dry microbursts.
        
        The original index, calculated using soundings in the region of Denver, CO, used
        the 700 and 500 mb layers for its calculations.  However, the RAOB Program manual
        recommends the use of the 5,000 and 13,000-ft AGL layers so that the results can
        be consistently used for any worldwide sounding, regardless of station elevation.
        The decision was made to split the index into two versions, the original (version
        1) and the RAOB (version 2).

        Parameters
        ----------
        prof : Profile object

        Returns
        -------
        dmpi2 : number
            Dry Microburst Potential Index, version 2 (number)
    '''

    lvl5 = interp.to_msl(prof, utils.FT2M(5000))
    lvl13 = interp.to_msl(prof, utils.FT2M(13000))
    pres5 = interp.pres(prof, lvl5)
    pres13 = interp.pres(prof, lvl13)
    tdd5 = interp.tdd(prof, pres5)
    tdd13 = interp.tdd(prof, pres13)
    lr_513 = lapse_rate(prof, lvl5, lvl13, pres=False)

    dmpi2 = lr_513 + tdd5 - tdd13

    return dmpi2

def hmi(prof):
    '''
        Hybrid Microburst Index (*)

        This index, created by K. L. Pryor in 2006, is designed to detect conditions
        favorable for both wet and dry microbursts.

        Parameters
        ----------
        prof : Profile object

        Returns
        -------
        hmi : number
            Hybrid Microburst Index
    '''
    
    tdd850 = interp.tdd(prof, 850)
    tdd670 = interp.tdd(prof, 670)
    lr_86 = lapse_rate(prof, 850, 670, pres=True)

    hmi = lr_86 + tdd850 - tdd670

    return hmi

def mwpi(prof, pcl):
    '''
        Microburst Windspeed Potential Index (*)

        This index is designed to improve the Hybrid Microburst Index by adding a
        term related to CAPE values.

        Parameters
        ----------
        prof : Profile object
        pcl : Parcel object

        Returns
        -------
        mwpi : number
            Microburst Windspeed Potential Index (number)
    '''

    hmi_t = getattr(prof, 'hmi', hmi(prof))

    mwpi = ( pcl.bplus / 100 ) + hmi_t

    return mwpi

def mdpi(prof):
    '''
        Microburst Day Potential Index (*)

        This index, developed jointly by the USAFs 45th Weather Squadronm and NASA's Applied
        Meteorology Uint (AMU) in 1995, calculates the risk of a microburst based on the maximum
        Theta-E temperature difference at two levels: the lowest 150 mb and the 650-500 mb levels.
        If the MDPI value is at least 1, microbursts are likely.

        Parameters
        ----------
        prof : Profile object

        Returns
        -------
        mdpi : number
            Microburst Day Potential Index (number)
    '''
    
    thetae = getattr(prof, 'thetae', prof.get_thetae_profile())
    sfc_pres = prof.pres[prof.sfc]
    upr_pres = sfc_pres - 150.
    
    layer_idxs_low = ma.where(prof.pres >= upr_pres)[0]
    layer_idxs_high = np.logical_and(650 >= prof.pres, prof.pres >= 500)
    min_thetae = ma.min(prof.thetae[layer_idxs_high])
    max_thetae = ma.max(prof.thetae[layer_idxs_low])

    mdpi = ( max_thetae - min_thetae ) / 30

    return mdpi

def hi(prof):
    '''
        Humidity Index (*)

        This index, derived by Z. Litynska in 1976, calculates moisture and instability using the dewpoint
        depressions of several levels.  It has proven to be fairly reliable, especially in the
        Mediterranean regions of the world.  Lower values indicate higher moisture content and greater
        instability potential.

        Parameters
        ----------
        prof : Profile object

        Returns
        -------
        hi :number
            Humidity Index (number)
    '''

    tdd850 = interp.tdd(prof, 850)
    tdd700 = interp.tdd(prof, 700)
    tdd500 = interp.tdd(prof, 500)

    hi = tdd850 + tdd700 + tdd500

    return hi

def ulii(prof):
    '''
        Upper Level Instability Index (*)

        This index was developed as part of a method for computing wind gusts produced by
        high-based thunderstorms, typically in the Rocky Mountains region.  It makes use of
        the 400-mb ambient temperature, the 300-mb ambient temperature, and a parcel lifted
        from the 500 mb level to both the 400 and 300-mb levels.

        Parameters
        ----------
        prof : Profile object

        Returns
        -------
        ulii : number
            Upper Level Instability Index (number)
    '''

    tmp500 = interp.temp(prof, 500)
    dpt500 = interp.dwpt(prof, 500)
    vtp400 = interp.vtmp(prof, 400)
    vtp300 = interp.vtmp(prof, 300)
    t_pcl54 = thermo.lifted(500, tmp500, dpt500, 400)
    t_pcl53 = thermo.lifted(500, tmp500, dpt500, 300)
    vt_pcl54 = thermo.virtemp(400, t_pcl54, t_pcl54)
    vt_pcl53 = thermo.virtemp(300, t_pcl53, t_pcl53)

    ulii = ( vtp400 - vt_pcl54 ) + ( vtp300 - vt_pcl53 )

    return ulii

def ssi(prof):
    '''
        Showalter Stability Index (*)

        This index, one of the first forecasting indices ever constructed, lifts a parcel
        from 850 mb to 500 mb, then compares it with the ambient temperature (similar to
        the lifted index).  It does not work well in mountainous areas, and cannot be used
        when the 850 mb level is below ground.  The SSI was devised by Albert Showalter in
        1947.

        The version used here makes use of the virtual temperature correction, much like the
        lifted indices in this program.

        Parameters
        ----------
        prof : Profile object

        Returns
        -------
        ssi : number
            Showalter Stability Index (number)
    '''

    tmp850 = interp.temp(prof, 850)
    dpt850 = interp.dwpt(prof, 850)
    vtp500 = interp.vtmp(prof, 500)

    t_pcl85 = thermo.lifted(850, tmp850, dpt850, 500)
    vt_pcl85 = thermo.virtemp(500, t_pcl85, t_pcl85)

    ssi = vtp500 - vt_pcl85

    return ssi

def csv(prof):
    '''
        "C" Stability Value (*)

        Formulation taken from Cox 1961, BAMS v.42 pg. 770.

        This index was originally derived in an effort to forecast thunderstorm potential.  It is found by
        raising the potential temperature at the 850 mb level moist adiabatically up to 600 mb, then subtracting
        the 600 mb ambient temperature from it.

        Parameters
        ----------
        prof : Profile object

        Returns
        -------
        csv : number
            "C" Stability Value (number)
    '''

    theta850 = thermo.theta(850, interp.temp(prof, 850))
    wtlft86 = thermo.wetlift(1000, theta850, 600)
    tmp600 = interp.temp(prof, 600)

    csv = wtlft86 - tmp600

    return csv

def z_index(prof):
    '''
        Z-Index (*)

        Formulation taken from Randerson 1977, MWR v.105 pg. 711.

        This index was developed by D. Randerson in 1977 in an effort to forecast thunderstorms over Nevada.
        It makes use of a regression equation that uses multiple variables: surface pressure (mb), surface
        temperature (degrees C), surface dewpoint depression (degrees C), 850 mb ambient temperature (degrees C),
        850 mb dewpoint depression (degrees C), 700 mb height (m), 500 mb ambient temperature (degrees C), the
        U-component of the 500 mb wind (kts), and 500 mb dewpoint temperature (degrees C).

        If the Z value is 0, then the probability of a thunderstorm is about 50%.  If the Z value is positive,
        then the probability is less than 50%.  If the Z value is negative, then the probability is over 50%.

        Parameters
        ----------
        prof : Profile object

        Returns
        -------
        z_index : number
            Z-Index (number)
    '''

    pres_sfc = prof.pres[prof.sfc]
    tmp_sfc = prof.tmpc[prof.sfc]
    tdd_sfc = tmp_sfc - prof.dwpc[prof.sfc]
    tmp850 = interp.temp(prof, 850)
    tdd850 = interp.tdd(prof, 850)
    hght700 = interp.hght(prof, 700)
    tmp500 = interp.temp(prof, 500)
    dpt500 = interp.dwpt(prof, 500)
    vec500 = interp.vec(prof, 500)
    u500 = utils.vec2comp(vec500[0], vec500[1])[0]

    z_d = ( 165.19 * pres_sfc ) - ( 14.63 * tmp_sfc ) + ( 11.73 * tdd_sfc ) + ( 31.52 * tmp850 ) + ( 38.22 * tdd850 ) - ( 17.30 * hght700 ) + ( 85.89 * tmp500 ) + ( 12.69 * u500 ) - ( 12.85 * dpt500 )
    z_index = 0.01 * ( z_d - 93200 )

    return z_index

def k_high1(prof):
    '''
        K-Index, high altitude version 1 (*)

        Formulation taken from Modahl 1979, JAM v.18 pg. 675.

        This index was derived by A. Modahl as a variant of the K-Index to be used in high-altitude areas.
        However, testing of the initial modified version (version 1) suggested that omitting the
        temperature lapse rate term and leaving just the 850 mb dewpoint temperature and 500 mb dewpoint
        depression terms (version 2) would give results similar to the initial version.

        Parameters
        ----------
        prof : Profile object

        Returns
        -------
        k_high1 : number
            K-Index, high altitude version 1
    '''

    tmp700 = interp.temp(prof, 700)
    tmp300 = interp.temp(prof, 300)
    dpt850 = interp.dwpt(prof, 850)
    tdd500 = interp.tdd(prof, 500)

    k_high1 = ( tmp700 - tmp300 ) + dpt850 - tdd500

    return k_high1

def k_high2(prof):
    '''
        K-Index, high altitude version 2 (*)

        Formulation taken from Modahl 1979, JAM v.18 pg. 675.

        This index was derived by A. Modahl as a variant of the K-Index to be used in high-altitude areas.
        However, testing of the initial modified version (version 1) suggested that omitting the
        temperature lapse rate term and leaving just the 850 mb dewpoint temperature and 500 mb dewpoint
        depression terms (version 2) would give results similar to the initial version.

        Parameters
        ----------
        prof : Profile object

        Returns
        -------
        k_high2 : number
            K-Index, high altitude version 2
    '''

    dpt850 = interp.dwpt(prof, 850)
    tdd500 = interp.tdd(prof, 500)

    k_high2 = dpt850 - tdd500

    return k_high2

def swiss00(prof):
    '''
        Stability and Wind Shear index for thunderstorms in Switzerland, 00z version (SWISS00) (*)

        This index is one of two versions of a forecasting index that was developed for use in forecasting
        thunderstorms in Switzerland (see Huntrieser et. al., WAF v.12 pgs. 108-125).  This version was
        developed for forecasting nocturnal thunderstorms using soundings taken around 00z.  It makes use
        of the Showalter Index, the 3-6 km AGL wind shear, and the dewpoint depression at the 600 mb level.

        Parameters
        ----------
        prof : Profile object

        Returns
        -------
        swiss00 : number
            Stability and Wind Shear index for thunderstorms in Switzerland, 00z version (number)
    '''

    si850 = getattr(prof, 'ssi', ssi(prof))
    p3km, p6km = interp.pres(prof, interp.to_msl(prof, np.array([3000., 6000.])))
    ws36 = utils.KTS2MS(utils.mag(*winds.wind_shear(prof, pbot=p3km, ptop=p6km)))
    tdd600 = interp.tdd(prof, 600)

    swiss00 = si850 + ( 0.4 * ws36 ) + ( tdd600 / 10 )

    return swiss00

def swiss12(prof):
    '''
        Stability and Wind Shear index for thunderstorms in Switzerland, 12z version (SWISS00) (*)

        This index is one of two versions of a forecasting index that was developed for use in forecasting
        thunderstorms in Switzerland (see Huntrieser et. al., WAF v.12 pgs. 108-125).  This version was
        developed for forecasting nocturnal thunderstorms using soundings taken around 12z.  It makes use
        of the Surface-based Lifted Index, the 0-3 km AGL wind shear, and the dewpoint depression at the
        650 mb level.

        Parameters
        ----------
        prof : Profile object

        Returns
        -------
        swiss12 : number
            Stability and Wind Shear index for thunderstorms in Switzerland, 12z version (number)
    '''

    sbpcl = getattr(prof, 'sfcpcl', parcelx(prof, flag=1))
    sli = sbpcl.li5
    p_sfc = prof.pres[prof.sfc]
    p3km = interp.pres(prof, interp.to_msl(prof, 3000))
    ws03 = utils.KTS2MS(utils.mag(*winds.wind_shear(prof, pbot=p_sfc, ptop=p3km)))
    tdd650 = interp.tdd(prof, 650)

    swiss12 = sli - ( 0.3 * ws03 ) + ( 0.3 * tdd650 )

    return swiss12

def fin(prof):
    '''
        FIN Index (*)

        Formulation taken from Ukkonen et. al. 2017, JAMC 56 pg. 2349

        This index is a modified version of the SWISS12 Index (q.v.) that makes use of the Most Unstable
        Lifted Index, the 700 mb dewpoint depression, and the wind shear between the surface and 750 mb.
        Negative values indicate favorable instability and shear for thunderstorm development.

        Parameters
        ----------
        prof : Profile object

        Returns
        -------
        fin : number
            FIN Index (number)        
    '''
    
    # Calculate MULI
    mupcl = getattr(prof, 'mupcl', parcelx(prof, flag=3))
    muli = mupcl.li5

    # Calculate 700 mb dewpoint depression
    tdd700 = interp.tdd(prof, 700)

    # Calculate surface-750 mb shear
    sfc_pres = prof.pres[prof.sfc]
    ws_sfc_750mb = utils.KTS2MS(utils.mag(*winds.wind_shear(prof, pbot=sfc_pres, ptop=750)))

    fin = muli + ( tdd700 / 10 ) - ( ws_sfc_750mb / 10 )

    return fin

def yon1(prof):
    '''
        Yonetani Index, version 1 (*)

        This index, derived by T. Yonetani in 1979, was developed to help forecast thunderstorms
        over the Kanto Plains region of Japan.  It makes use of the environmental lapse rates at
        the 900-850 mb and 850-500 mb levels, the average relative humidity of the 900-850 mb
        level, and the moist adiabatic lapse rate of the ambient air temperature at 850 mb.
        Positive values indicate a likely chance for thunderstorms.

        Parameters
        ----------
        prof : Profile object

        Returns
        -------
        yon1 : number
            Yonetani Index, version 1 (number)
    '''

    lr98 = lapse_rate(prof, 900, 850, pres=True)
    lr85 = lapse_rate(prof, 850, 500, pres=True)
    rh98 = mean_relh(prof, 900, 850) / 100

    # Calculate moist adiabatic lapse rate at the ambient temperature at 850 mb
    tmp850c = interp.temp(prof, 850)
    tmp850k = thermo.ctok(tmp850c)
    mxr850 = thermo.mixratio(850, tmp850c) / 1000
    cp_m = 1.0046851 * ( 1 + ( 1.84 * mxr850 ) )
    gocp = G / cp_m
    lvocp = 2.5e6 / cp_m
    lvord = 1680875 / 193
    eps = 0.62197
    num = 1 + ( lvord * ( mxr850 / tmp850k ) )
    denom = 1 + ( lvocp * lvord * ( ( mxr850 * eps ) / ( tmp850k ** 2 ) ) )
    malr850 = gocp * ( num / denom )

    if rh98 > 0.57:
        final_term = 15
    else:
        final_term = 16.5
    
    yon1 = ( 0.966 * lr98 ) + ( 2.41 * ( lr85 - malr850 ) ) + ( 9.66 * rh98 ) - final_term

    return yon1

def yon2(prof):
    '''
        Yonetani Index, version 2 (*)

        This index is a modification of the original Yonetani Index that was developed in an effort
        to better predict thunderstorms over the island of Cyprus (see Jacovides and Yonetani 1990,
        WAF v.5 pgs. 559-569).  It makes use of the same variables as the original index, but the
        weighing factors are rearrainged.  Positive values indicate a likely chance for
        thunderstorms.

        Parameters
        ----------
        prof : Profile object

        Returns
        -------
        yon2 : number
            Yonetani Index, version 2 (number)
    '''

    lr98 = lapse_rate(prof, 900, 850, pres=True)
    lr85 = lapse_rate(prof, 850, 500, pres=True)
    rh98 = mean_relh(prof, 900, 850) / 100

    # Calculate moist adiabatic lapse rate at the ambient temperature at 850 mb
    tmp850c = interp.temp(prof, 850)
    tmp850k = thermo.ctok(tmp850c)
    mxr850 = thermo.mixratio(850, tmp850c) / 1000
    cp_m = 1.0046851 * ( 1 + ( 1.84 * mxr850 ) )
    gocp = G / cp_m
    lvocp = 2.5e6 / cp_m
    lvord = 1680875 / 193
    eps = 0.62197
    num = 1 + ( lvord * ( mxr850 / tmp850k ) )
    denom = 1 + ( lvocp * lvord * ( ( mxr850 * eps ) / ( tmp850k ** 2 ) ) )
    malr850 = gocp * ( num / denom )

    if rh98 > 0.50:
        final_term = 13
    else:
        final_term = 14.5
    
    yon2 = ( 0.964 * lr98 ) + ( 2.46 * ( lr85 - malr850 ) ) + ( 9.64 * rh98 ) - final_term

    return yon2

def fsi(prof):
    '''
        Fog Stability Index
        
        Although this index was developed by USAF meteorologists for use in Germany, it can
        also be applied to similar climates.  The index is designed to indicate the
        potential for radiation fog.  Lower values indicate higher chances of radiation fog.

        Parameters
        ----------
        prof : Profile object

        Returns
        -------
        fsi : number
            Fog Stability Index (number)
    '''

    tmp_sfc = prof.tmpc[prof.sfc]
    dpt_sfc = prof.dwpc[prof.sfc]
    tmp850 = interp.temp(prof, 850)
    vec850 = interp.vec(prof, 850)

    fsi = ( 4 * tmp_sfc ) - ( 2 * ( tmp850 - dpt_sfc ) ) + vec850[1]

    return fsi

def fog_point(prof, pcl):
    '''
        Fog Point (*)

        This value indicates the temperature at which radiation fog will form.  It is
        determined by following the saturation mixing ratio line from the dew point curve
        at the LCL pressure level to the surface.

        Parameters
        ----------
        prof : Profile object
        pcl : Parcel object

        Returns
        -------
        fog_point : (float [C])
            Fog Point (Celsuis)
    '''

    dpt_lcl = interp.dwpt(prof, pcl.lclpres)
    mxr_lcl = thermo.mixratio(pcl.lclpres, dpt_lcl)
    sfc_pres = prof.pres[prof.sfc]

    fog_point = thermo.temp_at_mixrat(mxr_lcl, sfc_pres)

    return fog_point

def fog_threat(prof, pcl):
    '''
        Fog Threat (*)

        This value indicates the potential for radiation fog.  It is calculated by
        subtracting the fog point from the 850 mb wet-bulb potential temperature.
        Lower values indicate a higher likelihood for radiation fog.

        Parameters
        ----------
        prof : Profile object
        pcl : Parcel object

        Returns
        -------
        fog_threat : number
            Fog Threat (number)
    '''

    fp = getattr(prof, 'fog_point', fog_point(prof, pcl))
    thtw850 = thermo.thetaw(850, interp.temp(prof, 850), interp.dwpt(prof, 850))

    fog_threat = thtw850 - fp

    return fog_threat

def mvv(prof, pcl):
    '''
        Maximum Vertical Velocity (*)

        This is the maximum vertical velocity of the potential convective updraft.
        MVV is a function of CAPE.

        Parameters
        ----------
        prof : Profile object
        pcl : Parcel object

        Returns
        -------
        mvv : (float [m/s])
            Maximum Vertical Velocity (meters/second)
    '''

    mvv = ( 2 * pcl.bplus ) ** 0.5

    return mvv

def tsi(prof, pcl, hbot, htop, stu=0, stv=0):
    '''
        Thunderstorm Severity Index (*)

        This index is used to help measure and predict the severity of
        thunderstorm events, using a regression equation based around instability,
        shear, helicity, and storm motion.  Lower values indicate higher
        potential severity of a thunderstorm event; however, this is all
        contigent on whether or not thunderstorms actually do occur.

        Parameters
        ----------
        prof : Profile object
        pcl : Parcel object
        hbot : number
            Height of the bottom of the helicity layer [m]
        htop : number
            Height of the top of the helicity layer [m]
        stu : number
            Storm-relative wind U component [kts]
            (optional; default=0)
        stv : number
            Storm-relative wind V component [kts]
            (optional; default=0)

        Returns
        -------
        tsi : number
            Thunderstorm Severity Index (number)
    '''

    hght_t = interp.to_agl(prof, prof.hght[prof.top])
    wmax_c = winds.max_wind(prof, 0, hght_t, all=False)
    wmax = utils.mag(wmax_c[0], wmax_c[1], missing=MISSING)
    srh = winds.helicity(prof, hbot, htop, stu = stu, stv = stv)[0]
    ehis = getattr(prof, 'ehi', ehi(prof, pcl, hbot, htop, stu = stu, stv = stv))
    sspd = utils.mag(stu, stv, missing=MISSING)

    tsi = 4.943709 - ( 0.000777 * pcl.bplus ) - ( 0.004005 * wmax ) + ( 0.181217 * ehis ) - ( 0.026867 * sspd ) - (0.006479 * srh )

    return tsi

def jli(prof):
    '''
        Johnson Lag Index (*)

        Developed by D. L. Johnson in 1982, this index was based on a series of soundings
        made during experiments in the 1970s.  It is a parametric index that takes into
        account temperature and moisture differences at several layers, and is used to
        predict the likelihood of convective weather within several hours of the original
        sounding.  Negative values indicate increasing chances for convective weather.

        Parameters
        ----------
        prof : Profile object

        Returns
        -------
        jli : number
            Johnson Lag Index (number)
    '''

    tmp800 = interp.temp(prof, 800)
    tmp650 = interp.temp(prof, 650)
    tmp500 = interp.temp(prof, 500)
    thetae900 = thermo.thetae(900, interp.temp(prof, 900), interp.dwpt(prof, 900))
    thetae800 = thermo.thetae(800, interp.temp(prof, 800), interp.dwpt(prof, 800))
    thetae750 = thermo.thetae(750, interp.temp(prof, 750), interp.dwpt(prof, 750))
    thetae700 = thermo.thetae(700, interp.temp(prof, 700), interp.dwpt(prof, 700))

    dt68 = tmp650 - tmp800
    dt56 = tmp500 - tmp650
    dte89 = thetae800 - thetae900
    dte75 = thetae700 - thetae750

    jli = ( -11.5 - dt68 ) + ( 2 * ( dt56 + 14.9 ) ) + (2 * ( dte89 + 3.5 ) ) - ( ( 3.0 + dte75 ) / 3 )

    return jli

def ncape(prof, pcl):
    '''
        Normalized CAPE (*)

        NCAPE is CAPE that is divided by the depth of the positive-buoyancy layer.  Values
        around or less than 0.1 suggest a relatively "skinny" CAPE profile with relatively
        weak parcel acceleration.  Values around 0.3 or above suggest a relatively "fat"
        CAPE profile with large parcel accelerations possible.  Larger parcel accelerations
        can likely lead to stronger, more sustained updrafts.

        Parameters
        ----------
        prof : Profile object
        pcl : Parcel object

        Returns
        -------
        ncape : number
            NCAPE (number)
    '''

    p_buoy_depth = pcl.elhght - pcl.lfchght

    ncape = pcl.bplus / p_buoy_depth

    return ncape

def ncinh(prof, pcl):
    '''
        Normalized CINH (*)

        NCINH is CINH that is divided by the depth of the negative-buoyancy layer.  Values
        around or greater than -0.01 suggest a relatively "skinny" CINH profile that only
        requires relatively weak parcel acceleration to overcome the cap.  Values around 
        -0.03 or below suggest a relatively "fat" CINH profile with large parcel accelerations
        required to overcome the cap.

        Parameters
        ----------
        prof : Profile object
        pcl : Parcel object

        Returns
        -------
        ncinh : number
            NCINH (number)
    '''

    n_buoy_depth = pcl.lfchght

    ncinh = pcl.bminus / n_buoy_depth

    return ncinh

def lsi(prof):
    '''
        Lid Strength Index (*)

        Formulation taken from Carson et. al. 1980, BAMS v.61 pg. 1022.

        The Lid Strength Index was originally derived as an analogue for the Lifted Index,
        but as a way to measure the strength of the cap rather than stability.  It uses the
        mean theta-w of the lowest 50 mb, the maximum theta-ws in the atmosphere below 500
        mb, and the average theta-ws between the maximum theta-ws layer and 500 mb.
        
        Values below 1 indicate a very weak cap that would be easy to break; values between
        1 and 2 indicate a cap that is just strong enough to suppress convection while still
        being eventually breakable; and values above 2 indicate a very strong cap that is
        unlikely to be broken.

        Parameters
        ----------
        prof : Profile object

        Returns
        -------
        lsi : number
            Lid Strength Index (number)
    '''

    sfc_pres = prof.pres[prof.sfc]
    pres_50 = sfc_pres - 50
    thetaws = getattr(prof, 'thetaws', prof.get_thetaws_profile())

    thtw_lw = mean_thetaw(prof, sfc_pres, pres_50)

    idx = ma.where(prof.pres >= 500)[0]
    max_idx = np.ma.argmax(thetaws[idx])
    max_thetaws = thetaws[idx][max_idx]
    max_pres = prof.pres[idx][max_idx]

    thtw_up = mean_thetaw(prof, max_pres, 500)

    lsi = ( thtw_lw - thtw_up ) - ( max_thetaws - thtw_lw )

    return lsi

def mcsi1(prof, lat=35):
    '''
        MCS Index, version 1 (*)

        Formulation taken from Jirak and Cotton 2007, WAF v.22 pg. 825.

        The MCS Index was originally derived by I. Jirak and W. Cotton in 2007 as an attempt
        to determine the likelihood that convection will develop into a mesoscale convective
        system (MCS).  It makes use of the most-unstable Lifted Index, 0-3 km AGL bulk shear,
        and temperature advection at the 700 mb level.

        In WAF 24 pages 351-355, Bunkers warned that the results produced by the original
        equation (version 1) could be strongly biased in gridded datasets by the temperature
        advection term.  In response, in WAF v.24 pgs. 356-360, Jirak and Cotton created a
        second version (version 2) that rebalanced the equation so as to reduce the biasing.

        MCSI values on below -1.5 are considered unfavorable for MCS development; between -1.5
        and 0 are considered marginal; between 0 and 3 are considered favorable; and values
        exceeding 3 are considered very favorable.

        Parameters
        ----------
        prof : Profile object

        Returns
        -------
        mcsi1 : number
            MCS Index, version 1 (number)
    '''

    # Calculate LI
    mupcl = getattr(prof, 'mupcl', parcelx(prof, flag=3))
    muli = mupcl.li5

    # Calculate shear
    p3km = interp.pres(prof, interp.to_msl(prof, 3000))
    sfc_pres = prof.pres[prof.sfc]
    mag03_shr = utils.KTS2MS(utils.mag(*winds.wind_shear(prof, pbot=sfc_pres, ptop=p3km)))

    # Calculate 700 mb temperature advection
    omega = (2. * np.pi) / (86164.)
    b_pr = 750 # Pressure of bottom of layer
    m_pr = 700 # Pressure of middle of layer
    t_pr = 650 # Pressure of top of layer
    m_tmp = thermo.ctok(interp.temp(prof, m_pr)) # Temperature of middle of layer (Kelvin)
    b_ht = interp.hght(prof, b_pr) # Height ASL of bottom of layer (meters)
    t_ht = interp.hght(prof, t_pr) # Height ASL of top of layer (meters)
    b_wdir = interp.vec(prof, b_pr)[0] # Wind direction at bottom of layer (degrees from north)
    t_wdir = interp.vec(prof, t_pr)[0] # Wind direction at top of layer (degrees from north)
    m_wspd = utils.KTS2MS(interp.vec(prof, m_pr)[1]) # Wind speed at middle of layer (meters/second)

    if utils.QC(lat):
        f = 2. * omega * np.sin(np.radians(lat)) # Units: (s**-1)
    else:
        t7_adv = np.nan
        return mcsi1
    
    multiplier = (f / 9.81) * (np.pi / 180.) # Units: (s**-1 / (m/s**2)) * (radians/degrees)
    
    # Calculate change in wind direction with height; this will help determine whether advection is warm or cold
    mod = 180 - b_wdir
    t_wdir = t_wdir + mod
        
    if t_wdir < 0:
        t_wdir = t_wdir + 360
    elif t_wdir >= 360:
        t_wdir = t_wdir - 360
    d_theta = t_wdir - 180.

    # Here we calculate t_adv (which is -V_g * del(T) or the local change in temperature term)
    # K/s  s * rad/m * deg   m^2/s^2          K        degrees / m
    t7_adv = multiplier * np.power(m_wspd,2) * m_tmp * (d_theta / (t_ht - b_ht)) # Units: Kelvin / seconds 

    # Calculate LI term
    li_term = -( muli + 4.4 ) / 3.3

    # Calculate shear term
    shr_term = ( mag03_shr - 11.5 ) / 5

    # Calculate advection term
    adv_term = ( t7_adv - 4.5e-5 ) / 7.3e-5

    # Calculate equation
    mcsi1 = li_term + shr_term + adv_term

    return mcsi1

def mcsi2(prof, lat=35):
    '''
        MCS Index, version 2 (*)

        Formulation taken from Jirak and Cotton 2009, WAF v.24 pg. 359.

        The MCS Index was originally derived by I. Jirak and W. Cotton in 2007 as an attempt
        to determine the likelihood that convection will develop into a mesoscale convective
        system (MCS).  It makes use of the most-unstable Lifted Index, 0-3 km AGL bulk shear,
        and temperature advection at the 700 mb level.

        In WAF 24 pages 351-355, Bunkers warned that the results produced by the original
        equation (version 1) could be strongly biased in gridded datasets by the temperature
        advection term.  In response, in WAF v.24 pgs. 356-360, Jirak and Cotton created a
        second version (version 2) that rebalanced the equation so as to reduce the biasing.

        MCSI values on below -1.5 are considered unfavorable for MCS development; between -1.5
        and 0 are considered marginal; between 0 and 3 are considered favorable; and values
        exceeding 3 are considered very favorable.

        Parameters
        ----------
        prof : Profile object

        Returns
        -------
        mcsi2 : number
            MCS Index, version 2 (number)
    '''

    # Calculate LI
    mupcl = getattr(prof, 'mupcl', parcelx(prof, flag=3))
    muli = mupcl.li5

    # Calculate shear
    p3km = interp.pres(prof, interp.to_msl(prof, 3000))
    sfc_pres = prof.pres[prof.sfc]
    mag03_shr = utils.KTS2MS(utils.mag(*winds.wind_shear(prof, pbot=sfc_pres, ptop=p3km)))

    # Calculate 700 mb temperature advection
    omega = (2. * np.pi) / (86164.)
    b_pr = 750 # Pressure of bottom of layer
    m_pr = 700 # Pressure of middle of layer
    t_pr = 650 # Pressure of top of layer
    m_tmp = thermo.ctok(interp.temp(prof, m_pr)) # Temperature of middle of layer (Kelvin)
    b_ht = interp.hght(prof, b_pr) # Height ASL of bottom of layer (meters)
    t_ht = interp.hght(prof, t_pr) # Height ASL of top of layer (meters)
    b_wdir = interp.vec(prof, b_pr)[0] # Wind direction at bottom of layer (degrees from north)
    t_wdir = interp.vec(prof, t_pr)[0] # Wind direction at top of layer (degrees from north)
    m_wspd = utils.KTS2MS(interp.vec(prof, m_pr)[1]) # Wind speed at middle of layer (meters/second)

    if utils.QC(lat):
        f = 2. * omega * np.sin(np.radians(lat)) # Units: (s**-1)
    else:
        t7_adv = np.nan
        return mcsi1
    
    multiplier = (f / 9.81) * (np.pi / 180.) # Units: (s**-1 / (m/s**2)) * (radians/degrees)
    
    # Calculate change in wind direction with height; this will help determine whether advection is warm or cold
    mod = 180 - b_wdir
    t_wdir = t_wdir + mod
        
    if t_wdir < 0:
        t_wdir = t_wdir + 360
    elif t_wdir >= 360:
        t_wdir = t_wdir - 360
    d_theta = t_wdir - 180.

    # Here we calculate t_adv (which is -V_g * del(T) or the local change in temperature term)
    # K/s  s * rad/m * deg   m^2/s^2          K        degrees / m
    t7_adv = multiplier * np.power(m_wspd,2) * m_tmp * (d_theta / (t_ht - b_ht)) # Units: Kelvin / seconds 

    # Calculate LI term
    li_term = -( muli + 4.4 ) / 3.3

    # Calculate shear term
    shr_term = ( mag03_shr - 11.5 ) / 4.1

    # Calculate advection term
    adv_term = ( t7_adv - 4.5e-5 ) / 1.6e-4

    # Calculate equation
    mcsi2 = li_term + shr_term + adv_term

    return mcsi2

def cii1(prof):
    '''
        Convective Instability Index, version 1 (*)

        This index was developed by W. D. Bonner, R. M. Reap, and J. E. Kemper in 1971.  It uses
        the equivalent potential temperature (Theta-e) at the surface, 850 mb, and 700 mb levels.
        Values <= 0 are indicative of convective instability and perhaps potential storm development.

        Parameters
        ----------
        prof : Profile object

        Returns
        -------
        cii1 : number
            Convective Instability Index, version 1 (number)
    '''

    te_sfc = prof.thetae[prof.sfc]
    te850 = interp.thetae(prof, 850)
    te700 = interp.thetae(prof, 700)

    te_s8 = ( te_sfc + te850 ) / 2

    cii1 = te700 - te_s8

    return cii1

def cii2(prof):
    '''
        Convective Instability Index, version 2 (*)

        This index was derived by D. A. Barber in 1975.  It subtracts the average Theta-e value in
        the 600-500 mb layer from the average Theta-e value in the lowest 100 mb.  Values >= 0
        indicate likely convective instability.

        Parameters
        ----------
        prof : Profile object

        Returns
        -------
        cii2 : number
            Convective Instability Index, version 2 (number)
    '''

    sfc_pres = prof.pres[prof.sfc]
    top_pres = sfc_pres - 100

    te_low100 = mean_thetae(prof, pbot=sfc_pres, ptop=top_pres)
    te65 = mean_thetae(prof, pbot=600, ptop=500)

    cii2 = te_low100 - te65

    return cii2

def cpst1(mlcape, bwd6, srh03, mlcinh):
    '''
        Conditional Probability of a Significant Tornado, version 1 (*)

        This equation is one of three that were derived in Togstead et. al., Weather and
        Forecasting 2011 p. 729-743, as part of an effort to develop logistic regression
        equations that could help assess the probability of the occurrence of significant
        tornadoes (i.e. tornadoes rated EF2 or higher on the Enhanced Fujita (EF) scale).

        This equation makes use of mixed-layer CAPE, 0-6 km bulk shear, 0-3 km storm-relative
        helicity, and mixed-layer CIN.
        
        Parameters
        ----------
        mlcape : Mixed-layer CAPE from the parcel class (J/kg)
        bwd6 : 0-6 km bulk shear (m/s)
        srh03 : 0-3 km storm-relative helicity (m2/s2)
        mlcinh : mixed-layer convective inhibition (J/kg)

        Returns
        -------
        cpst1 : percent
            Conditional Probability of a Significant Tornado, version 1 (percent)
    '''

    # Normalization values taken from the original paper.
    mlcape_n = 40.7
    bwd6_n = 23.4
    srh03_n = 164.8
    mlcinh_n = 58.1

    # f(x) in the original paper.
    reg = -4.69 + ( 2.98 * ( ( ( mlcape ** 0.5 ) / mlcape_n ) * ( bwd6 / bwd6_n ) ) ) + ( 1.67  * ( srh03 / srh03_n ) ) + ( 1.82 * ( mlcinh / mlcinh_n ) )

    # P in the original paper.
    cpst1 = 100 / ( 1 + np.exp(-reg) )

    return cpst1

def cpst2(mlcape, bwd6, bwd1, mlcinh):
    '''
        Conditional Probability of a Significant Tornado, version 2 (*)

        This equation is one of three that were derived in Togstead et. al., Weather and
        Forecasting 2011 p. 729-743, as part of an effort to develop logistic regression
        equations that could help assess the probability of the occurrence of significant
        tornadoes (i.e. tornadoes rated EF2 or higher on the Enhanced Fujita (EF) scale).

        This equation makes use of mixed-layer CAPE, 0-6 km bulk shear, 0-1 km bulk shear,
        and mixed-layer CIN.

        Parameters
        ----------
        mlcape : Mixed-layer CAPE from the parcel class (J/kg)
        bwd6 : 0-6 km bulk wind difference (m/s)
        bwd1 : 0-1 km bulk wind difference (m/s)
        mlcinh : mixed-layer convective inhibition (J/kg)

        Returns
        -------
        cpst2 : percent
            Conditional Probability of a Significant Tornado, version 2 (percent)
    '''

    # Normalization values taken from the original paper.
    mlcape_n = 40.7
    bwd6_n = 23.4
    bwd1_n = 11.0
    mlcinh_n = 58.1

    # f(x) in the original paper.
    reg = -5.67 + ( 3.11 * ( ( ( mlcape ** 0.5 ) / mlcape_n ) * ( bwd6 / bwd6_n ) ) ) + ( 2.23  * ( bwd1 / bwd1_n ) ) + ( 1.38  * ( mlcinh / mlcinh_n ) )

    # P in the original paper.
    cpst2 = 100 / ( 1 + np.exp(-reg) )

    return cpst2

def cpst3(mlcape, bwd6, bwd1, mllcl, mlcinh):
    '''
        Conditional Probability of a Significant Tornado, version 3 (*)

        This equation is one of three that were derived in Togstead et. al., Weather and
        Forecasting 2011 p. 729-743, as part of an effort to develop logistic regression
        equations that could help assess the probability of the occurrence of significant
        tornadoes (i.e. tornadoes rated EF2 or higher on the Enhanced Fujita (EF) scale).

        This equation makes use of mixed-layer CAPE, 0-6 km bulk shear, 0-1 km bulk shear,
        mixed-layer LCL, and mixed-layer CIN.

        The original paper states that, out the three conditional probability equations
        derived, this one has the lowest chi square score.  This is due to the fact that,
        outside of very high LCL heights (i.e. near or above 2000 m AGL), LCL is less
        discriminatory of tornadic vs. nontornadic environments than the other components
        that were used in this and the other equations.

        Parameters
        ----------
        mlcape : Mixed-layer CAPE from the parcel class (J/kg)
        bwd6 : 0-6 km bulk wind difference (m/s)
        bwd1 : 0-1 km bulk wind difference (m/s)
        mllcl : mixed-layer lifted condensation level (m)
        mlcinh : mixed-layer convective inhibition (J/kg)

        Returns
        -------
        cpst3 : percent
            Conditional Probability of a Significant Tornado, version 3 (percent)
    '''

    # Normalization values taken from the original paper.
    mlcape_n = 40.7
    bwd6_n = 23.4
    bwd1_n = 11.0
    mllcl_n = 1170.0
    mlcinh_n = 58.1

    # f(x) in the original paper.
    reg = -4.73 + ( 3.21 * ( ( ( mlcape ** 0.5 ) / mlcape_n ) * ( bwd6 / bwd6_n ) ) ) + ( 0.78 * ( ( bwd1 / bwd1_n ) / ( mllcl / mllcl_n ) ) ) + ( 1.06 * ( mlcinh / mlcinh_n ) )

    # P in the original paper.
    cpst3 = 100 / ( 1 + np.exp(-reg) )

    return cpst3

def tie(prof):
    '''
        Tornado Intensity Equation (*)

        Formulation taken from Colquhoun and Riley 1996, WAF v.11 pg. 367.

        This equation is a regression equation designed to help predict the likely intensity
        of a tornado forming within the proximity of a sounding station, given surface-based
        Lifted Index and surface to 500 mb wind shear.

        Parameters
        ----------
        prof : Profile object

        Returns
        -------
        tie : number
            Tornado Intensity Equation (number)
    '''

    sbpcl = getattr(prof, 'sfcpcl', parcelx(prof, flag=1))
    sli = sbpcl.li5
    p_sfc = prof.pres[prof.sfc]
    sfc_600mb_shr = utils.KTS2MS(utils.mag(*winds.wind_shear(prof, pbot=p_sfc, ptop=600)))

    tie = ( -0.145 * sli ) + ( 0.136 * sfc_600mb_shr ) - 1.5

    return tie

def t1_gust(prof):
    '''
        T1 Gust (*)

        Formulation taken from 
	    Notes on Analysis and Severe-Storm Forecasting Procedures of the Air Force Global Weather Central, 1972
	    by RC Miller.

        This parameter estimates the maximum average wind gusts.  If the sounding has an inversion
        layer with a top less than 200 mb above the ground, then the maximum temperature in the
        inversion is moist adiabatically lifted to 600 mb; if no inversion is present or if the top
        of the inversion is above 200 mb above the ground, then the maximum forecast surface
        temperature is moist adiabatically lifted to 600 mb.  In either case, the lifted temperature
        is subrtacted from the 600 mb ambient temperature; the square root of the difference is then
        multiplied by 13 to get the likely T1 Average Gust.

        The maximum peak gust is calculated by adding one third (1/3) of the mean wind in the lower
        5,000 feet AGL to the T1 Average Gust value.

        For the direction of the gusts, the mean wind direction in the level from 10,000 to 14,000
        feet AGL is used.

        Parameters
        ----------
        prof : Profile object

        Returns
        -------
        t1_avg : knots
            T1 Average Gust (knots)
        t1_peak : knots
            T1 Peak Gust (knots)
        t1_dir
            T1 Gust Direction (degrees)
    '''

    inv_top = getattr(prof, 'inversion', inversion(prof, pbot=None, ptop=600)[1][0])
    sfc_pres = prof.pres[prof.sfc]

    if not utils.QC(inv_top) or inv_top < sfc_pres - 200:
        max_tmp = getattr(prof, 'max_temp', max_temp(prof))
        max_tmp_pcl = thermo.wetlift(sfc_pres, max_tmp, 600)
    else:
        idx = np.logical_and(inv_top >= prof.pres, prof.pres >= 600)
        max_idx = np.ma.argmax(prof.tmpc[idx])
        max_pres = prof.pres[idx][max_idx]
        max_tmp = prof.tmpc[idx][max_idx]
        max_tmp_pcl = thermo.wetlift(max_pres, max_tmp, 600)
    
    tmp600 = interp.temp(prof, 600)
    t1_diff = max_tmp_pcl - tmp600

    t1_avg = 13 * (t1_diff ** 0.5)
    
    pres5k = interp.pres(prof, interp.to_msl(prof, utils.FT2M(5000)))
    mn_wd_5k = utils.mag(*winds.mean_wind(prof, pbot=sfc_pres, ptop=pres5k))
    
    t1_peak = t1_avg + ( mn_wd_5k / 3 )

    pres10k = interp.pres(prof, interp.to_msl(prof, utils.FT2M(10000)))
    pres14k = interp.pres(prof, interp.to_msl(prof, utils.FT2M(14000)))

    mn_wd_10_14 = winds.mean_wind(prof, pbot=pres10k, ptop=pres14k)
    t1_dir = utils.comp2vec(mn_wd_10_14[0], mn_wd_10_14[1])[0]

    return t1_avg, t1_peak, t1_dir
