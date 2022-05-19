import numpy as np
from scipy import interpolate

def MagsToFluxes(filename, Teff=5527., PrintFluxes=False):

    # see mag2fluxconv for important notes and refs!

    band = np.loadtxt(filename, unpack=True, usecols=[0], dtype='object')
    mag, merr = np.loadtxt(filename, unpack=True, usecols=[1,2])
    
    theta = Teff / 5040.
    nband = band.shape[0]
    lameff = np.zeros(nband)
    weff = np.zeros(nband)
    flux = np.zeros(nband)

    for i in range(nband):
        if band[i] == 'U':
            lameff[i] = np.polyval([3476.,162.,86.,-63.][::-1], theta) / 1.e4
            weff[i] = np.polyval([612.,346.,-741.,269.][::-1], theta) / 1.e4
            lamflamzp = 1.51e-5
            flux[i] = lamflamzp * 10**(-0.4 * mag[i])
        elif band[i] == 'B':
            lameff[i] = np.polyval([4336., 201., 235., -115.][::-1], theta) / 1.e4
            weff[i] = np.polyval([863., 494., -833., 192.][::-1], theta) / 1.e4
            lamflamzp = 2.90e-5
            flux[i] = lamflamzp * 10**(-0.4 * mag[i])
        elif band[i] == 'J2M':
            lameff[i] = np.polyval([1.23,0.08,0.0,-0.01][::-1], theta)
            weff[i] = np.polyval([0.16,0.08,-0.23,0.11][::-1], theta)
            lamflamzp = 1594. * 3e-9 / 1.235**2 * 1.235
            flux[i] = lamflamzp * 10**(-0.4 * mag[i])
        elif band[i] == 'H2M':
            lameff[i] = np.polyval([1.64,0.05,0.02,-0.02][::-1], theta)
            weff[i] = np.polyval([0.24,0.09,-0.21,0.07][::-1], theta)
            lamflamzp = 1024. * 3e-9 / 1.662**2 * 1.662
            flux[i] = lamflamzp * 10**(-0.4 * mag[i])
        elif band[i] == 'K2M':
            lameff[i] = np.polyval([2.15,0.03,0.02,-0.01][::-1], theta)
            weff[i] = np.polyval([0.25,0.04,-0.06,0.01][::-1], theta)
            lamflamzp = 666.7 * 3.e-9 / 2.159**2 * 2.159
            flux[i] = lamflamzp * 10**(-0.4 * mag[i])
        elif band[i] == 'gSDSS':
            # AB mags - need to convert to AB if in Vegamag, see link above
            lameff[i] = np.polyval([4647.,312.,241.,-173.][::-1], theta) / 1.e4
            weff[i] = np.polyval([1156.,909.,-1424.,387.][::-1], theta) / 1.e4
            lamflamzp = 3631. * 3e-9 / 0.4788**2 * 0.4788
            flux[i] = lamflamzp * 10**(-0.4 * mag[i]-0.04) #  assuming Pogson mags instead of asinh
        elif band[i] == 'rSDSS':
            # AB mags - need to convert to AB if in Vegamag, see link above
            lameff[i] = np.polyval([6145.,139.,156.,-80.][::-1], theta) / 1.e4
            weff[i] = np.polyval([1255.,289.,-183.,-109.][::-1], theta) / 1.e4
            lamflamzp = 3631. * 3e-9 / 0.6242**2 * 0.6242
            flux[i] = lamflamzp * 10**(-0.4 * mag[i]-0.04) #  assuming Pogson mags instead of asinh
        elif band[i] == 'iSDSS':
            # AB mags - need to convert to AB if in Vegamag, see link above
            lameff[i] = np.polyval([7562.,101.,123.,-52.][::-1], theta) / 1.e4
            weff[i] = np.polyval([1310.,144.,-9.,-104.][::-1], theta) / 1.e4
            lamflamzp = 3631. * 3e-9 / 0.7704**2 * 0.7704
            flux[i] = lamflamzp * 10**(-0.4 * mag[i]-0.04) #  assuming Pogson mags instead of asinh
        elif band[i] == 'zSDSS':
            # AB mags - need to convert to AB if in Vegamag, see link above
            lameff[i] = np.polyval([8997.,88.,105.,-36.][::-1], theta) / 1.e4
            weff[i] = np.polyval([1357.,91.,24.,-76.][::-1], theta) / 1.e4
            lamflamzp = 3631. * 3e-9 / 0.9038**2 * 0.9038
            flux[i] = lamflamzp * 10**(-0.4 * mag[i]-0.04) #  assuming Pogson mags instead of asinh
        elif band[i] == 'WISE1':
            mag[i] += 2.683
            lameff[i] = 33526. / 1.e4
            weff[i] = 6626. / 1.e4
            lamflamzp = 3631. * 3e-9 / lameff[i]**2 * lameff[i]
            flux[i] = lamflamzp * 10**(-0.4 * mag[i])
        elif band[i] == 'WISE2':
            mag[i] += 3.319
            lameff[i] = 46028. / 1.e4
            weff[i] = 10423. / 1.e4
            lamflamzp = 3631. * 3e-9 / lameff[i]**2 * lameff[i]
            flux[i] = lamflamzp * 10**(-0.4 * mag[i])
        else:
            print('Invalid band selection!')
            sys.exit()

    fluxerr = flux * np.log(10) / 2.5 * merr

    return lameff, weff, flux, fluxerr, band

def InitializeExtinction(seddata):
    wave, kappa = np.loadtxt('extinction_law.ascii', unpack=True)
    interpfunc = interpolate.interp1d(wave, kappa, kind='cubic')
    extinct0 = interpfunc(0.55)
    kappanorm = kappa / extinct0 # so V = 1
    interpfunc = interpolate.interp1d(wave, kappanorm, kind='cubic')

    Npts = seddata['band_fluxes'].shape[0]
    ExtinctionBase = np.zeros(Npts)
    for i in range(Npts):
        start = seddata['band_wavelengths'][i]-seddata['band_widths'][i]/2.
        stop = seddata['band_wavelengths'][i]+seddata['band_widths'][i]/2.
        waverange = np.linspace(start, stop, 100)
        kapparange = interpfunc(waverange)
        ExtinctionBase[i] = np.mean(kapparange)

    return ExtinctionBase