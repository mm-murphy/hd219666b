import numpy, math, sys, progressbar
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy import interpolate
from astropy import constants as const
from astropy import units as u

def RVOrbit(time, pars):
	p = 10.**pars['logp']
	w = math.pi/2.
	ecc = 0.0

	tp = TctoTp(pars['tc'], p, ecc, w)

	n = 2.*math.pi / p
	M = n*(time-tp)

	E0 = M
	EA = fsolve(EccAnomRoot, E0, args=(M, ecc))

	cosf = (numpy.cos(EA)-ecc)/(1-ecc*numpy.cos(EA))
	sinf = (numpy.sqrt(1-ecc**2.)*numpy.sin(EA))/(1-ecc*numpy.cos(EA))

	f = numpy.arctan2(sinf, cosf)

	#sini = numpy.sqrt(1-pars['cosi']**2.)
	#a2rs = 10**pars['logars']
	#a2 = a2rs*rstar
	#a1 = pars['mpms']*a2 * (696000.) # km

	#psec = p*24.*60.*60.
	#K = (2.*math.pi*a1*sini)/(psec*numpy.sqrt(1-ecc**2.))

	vr = pars['K']*(numpy.cos(w+f)+ecc*numpy.cos(w)) + (pars['gamma']) + SlopeTerm(pars['RVslope'], time)

	return vr

def PlotRV(pars, data, outfile='RVplot.png'):
	period = 10.**pars['logp']

	phase = PhaseTimes(data['time'], pars['tc'], period, t0center=False)
	velphase = data['vel'] - SlopeTerm(pars['RVslope'], data['time'])

	timemodel = numpy.linspace(data['time'][0], data['time'][-1], 1000)
	vrmodel = RVOrbit(timemodel, pars)

	timeforphase = numpy.linspace(pars['tc'], pars['tc']+period, 100)
	phasemodel = PhaseTimes(timeforphase, pars['tc'], period, t0center=False)
	sortonphase = numpy.argsort(phasemodel)
	vrmodelphase = RVOrbit(timeforphase, pars)
	phasemodel = phasemodel[sortonphase] # to make it look nice
	vrmodelphase = vrmodelphase[sortonphase]

	fig = plt.figure()

	ax1 = plt.subplot2grid((2,1), (0,0))
	ax1.errorbar(data['time']-pars['tc'], data['vel'], yerr=data['error'], fmt="ok", ms=6, capsize=0, zorder=1)
	ax1.plot(timemodel-pars['tc'], vrmodel, '-r', zorder=0, lw=2)

	ax2 = plt.subplot2grid((2,1), (1,0), sharey=ax1)
	ax2.errorbar(phase, velphase, yerr=data['error'], fmt="ok", ms=6, capsize=0, zorder=1)
	ax2.plot(phasemodel, vrmodelphase, '-r', zorder=0, lw=2)

	#ax1.set_ylim(-50,50)
	ax1.set_ylabel('Velocity (m/s)', fontsize=16)
	ax2.set_ylabel('Velocity (m/s)', fontsize=16)
	ax1.set_xlabel('BJD$_{\mathrm{TDB}}$-Tc', fontsize=16)
	ax2.set_xlabel('Orbital Phase', fontsize=16)

	fig.subplots_adjust(hspace=0.3)

	plt.savefig(outfile)
	plt.close()
	plt.clf()

	model = RVOrbit(data['time'], pars)
	residuals = data['vel'] - model

	#model = RVOrbit(data['time'], K, gamma, ecc, w, tc, period, slope)
	#resid = data['vel'] - model
	#plt.errorbar(phase, resid, yerr=data['error'], fmt="ok", ms=6, capsize=0, zorder=1)
	#plt.show()
	#plt.close()
	#plt.clf()
	return residuals

def SlopeTerm(slope, time):
	return slope*(time - numpy.median(time))

def EccAnomRoot(EA, MA, ecc):
	return EA - ecc*numpy.sin(EA) - MA

def PhaseTimes(time, t0, period, t0center=False):
	if t0center: phase = numpy.mod((time-t0)-period/2.,period) / period
	else: phase = numpy.mod((time-t0),period) / period
	return phase

def TctoTp(Tc, period, e, w):
	nu = math.pi/2. - w # true anomaly for transit away from peri.
	E = 2.*numpy.arctan(numpy.sqrt((1.0-e)/(1.+e))*numpy.tan(nu/2.)) # ecc. anomaly
	M = E - e*numpy.sin(E) # mean anomaly
	periPhase = M/(2.*math.pi)
	Tp = Tc - period*periPhase # peri time
	return Tp

def TctoTs(Tc, period, e, w):
	nuTc = (math.pi/2.) - w # true anomaly for primary
	nuTs = (3.*math.pi/2.) - w # true anomaly for secondary
	E_Tc = 2. * numpy.arctan(numpy.sqrt((1.0 - e) / (1. + e)) * numpy.tan(nuTc / 2.))
	E_Ts = 2. * numpy.arctan(numpy.sqrt((1.0 - e) / (1. + e)) * numpy.tan(nuTs / 2.))
	M_Tc = E_Tc - e * numpy.sin(E_Tc)
	M_Ts = E_Ts - e * numpy.sin(E_Ts)
	secPhase = (M_Ts-M_Tc)/(2.*math.pi) + 1
	Ts = Tc + period*secPhase # sec time
	return Ts

def GetSEDModelFlux(pars, InterpFuncs, ExtinctionBase):
	plx = pars[16]
	rstar = pars[17]
	logg = pars[18]
	teff = pars[19]
	av = pars[20]

	rstarau = (rstar/215)
	dist = 1000. / plx
	distau = dist*206265
	r2d2 = (rstarau/distau)**2.

	flux = numpy.zeros(11)
	flux[0] = InterpFuncs['galFUV'](logg, teff)*r2d2
	flux[1] = InterpFuncs['galNUV'](logg, teff)*r2d2
	flux[2] = InterpFuncs['BT'](logg, teff)*r2d2
	flux[3] = InterpFuncs['VT'](logg, teff)*r2d2
	flux[4] = InterpFuncs['J'](logg, teff)*r2d2
	flux[5] = InterpFuncs['H'](logg, teff)*r2d2
	flux[6] = InterpFuncs['K'](logg, teff)*r2d2
	flux[7] = InterpFuncs['W1'](logg, teff)*r2d2
	flux[8] = InterpFuncs['W2'](logg, teff)*r2d2
	flux[9] = InterpFuncs['W3'](logg, teff)*r2d2
	flux[10] = InterpFuncs['W4'](logg, teff)*r2d2

	tau = (ExtinctionBase/1.086)*av
	extinction = numpy.exp(-tau)
	flux *= extinction

	return flux

def InitializeExtinction(seddata):
	wave, kappa = numpy.loadtxt('../extinction_law.ascii', unpack=True)
	interpfunc = interpolate.interp1d(wave, kappa, kind='cubic')
	extinct0 = interpfunc(0.55)
	kappanorm = kappa / extinct0 # so V = 1
	interpfunc = interpolate.interp1d(wave, kappanorm, kind='cubic')

	np = seddata['flux'].shape[0]
	ExtinctionBase = numpy.zeros(np)
	for i in range(np):
		start = seddata['wave'][i]-seddata['width'][i]/2.
		stop = seddata['wave'][i]+seddata['width'][i]/2.
		waverange = numpy.linspace(start, stop, 100)
		kapparange = interpfunc(waverange)
		ExtinctionBase[i] = numpy.mean(kapparange)

	return ExtinctionBase

def PlotSED(pars, seddata, InterpFuncs, ExtinctionBase, outfile='SEDplot.png'):
	flux = GetSEDModelFlux(pars, InterpFuncs, ExtinctionBase)

	modelwave, modelflux = numpy.loadtxt('../KuruczforPlot.dat', unpack=True)
	modelwave /= 10000.
	plx = pars[16]
	rstar = pars[17]
	logg = pars[18]
	teff = pars[19]
	av = pars[20]

	rstarau = (rstar/215)
	dist = 1000. / plx
	distau = dist*206265
	r2d2 = (rstarau/distau)**2.

	modelflux *= r2d2

	plt.plot(modelwave, modelflux, '-k', zorder=0)
	plt.plot(seddata['wave'], flux, 'sg', ms=8, zorder=1)
	plt.errorbar(seddata['wave'], seddata['flux'], yerr=seddata['error'], xerr=seddata['width']/2., fmt='or', ms=8, capsize=2, zorder=2, lw=2, barsabove=True)

	plt.xlim(0.1,40)
	plt.ylim(1E-15,1E-07)
	plt.xscale('log')
	plt.yscale('log')
	xticks = [0.1, 1, 10]
	ticklabels = ['0.1', '1.0', '10']
	plt.xticks(xticks, ticklabels)
	plt.ylabel('$\lambda$F$_\lambda$ (erg s$^{-1}$ cm$^{-2}$)', fontsize=16)
	plt.xlabel('Wavelength ($\mu$m)', fontsize=16)
	plt.savefig(outfile)
	plt.close()
	return

def TorresEnforcement(pars):
	teff = pars[13]
	logg = pars[12]
	feh = pars[14]

	rstar = pars[11]
	period = pars[7]
	aRs = 10.**pars[10]

	psec = period*24.*60.*60.
	g = 6.6725985e-08
	rhostar = aRs**3. * 3. * math.pi / (psec**2. * g)
	rhosun = rhostar / 1.41
	mstar = rhosun * rstar**3.

	a = [0,1.5689,1.3787,0.4243,1.139,-0.14250,0.01969,0.10100]
	b = [0,2.4427,0.6679,0.1771,0.705,-0.21415,0.02306,0.04173]
	X = numpy.log10(teff)-4.1

	logm_torres = a[1] + a[2]*X + a[3]*X**2. + a[4]*X**3. + a[5]*logg**2. + a[6]*logg**3. + a[7]*feh
	logr_torres = b[1] + b[2]*X + b[3]*X**2. + b[4]*X**3. + b[5]*logg**2. + b[6]*logg**3. + b[7]*feh
	logm = numpy.log10(mstar)
	logr = numpy.log10(rstar)

	lnpM = -0.5*((logm-logm_torres)**2./0.027**2.)
	lnpR = -0.5*((logr-logr_torres)**2./0.014**2.)

	return lnpM+lnpR

'''
def MasterLnLikelihood(pars, data, priors, normidx, gammaidx, uniformidx):
	sqrtecosw = pars[1]
	sqrtesinw = pars[2]
	ecc = sqrtecosw**2. + sqrtesinw**2.
	if ecc>=1.0: return -numpy.inf

	lnpdata = 0.0

	model = GetRVModel(pars, data)
	chisqr = numpy.sum((data['vel']-model)**2./data['error']**2.)
	lnpdata += -0.5*chisqr

	lnppriors = LnLikePriors(pars, priors, normidx, gammaidx, uniformidx)

	#lnptorres = TorresEnforcement(pars)

	lnp = lnpdata + lnppriors# + lnptorres

	return lnp
'''