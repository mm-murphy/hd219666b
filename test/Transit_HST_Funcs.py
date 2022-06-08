import numpy, math, batman, time, sys, progressbar, emcee, copy
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import astropy.units as units
import astropy.constants as const
import PyFAST as RVSEDFuncs
from scipy.stats import binned_statistic
from RECTE import RECTE
from RECTECorrector import rampProfile2

def Lightcurve(pars, time, Section='Phase', SpecNo=99, SansPhase=False, PhaseVar='Sinusoid', Telescope='HST', visit=''):
	t = numpy.copy(time).astype(float)

	params = batman.TransitParams
	params.w = 90.
	params.ecc = 0.0

	timeForbatman = numpy.array(t)

	if Section=='Phase':
		params.rp = pars['rprs']
		params.limb_dark = "linear"
		if SpecNo == 99:
			params.u = [pars['ulimb']]
			#params.u = [0.4497,  0.4273, -0.6251, 0.2451]  # this is for J log=4.5!
			#params.u = [0.6975, -0.1538, -0.2108, 0.1358]  # this is for H log=4.5!
		else:
			params.u = [pars['u1_' + str(SpecNo).zfill(2)], pars['u2_'+str(SpecNo).zfill(2)]]

		radians = 2. * math.pi * (timeForbatman - params.t0) / params.per

		if PhaseVar=='GP': params.fp = pars['fp']
		elif PhaseVar=='Sinusoid': params.fp = 1.0
		else:
			print('PhaseVar is set incorrectly')
			sys.exit()

		if SansPhase:
			if PhaseVar != 'GP':
				print('Why are you calling lightcurve with SansPhase but not a GP PhaseVar?')
				#sys.exit()
			eclipsemodel = batman.TransitModel(params, timeForbatman, transittype="secondary")
			fluxeclipse = eclipsemodel.light_curve(params)-(1.0+params.fp)
			#phasecurve = PhaseVariation(pars, radians, SpecNo=SpecNo)
			fluxplanet = fluxeclipse
		else:
			eclipsemodel = batman.TransitModel(params, timeForbatman, transittype="secondary")
			fluxeclipse = eclipsemodel.light_curve(params) - 1.0
			phasecurve = PhaseVariation(pars, radians, SpecNo=SpecNo)
			fluxplanet = phasecurve * fluxeclipse

		transitmodel = batman.TransitModel(params, timeForbatman)
		fluxtransit = transitmodel.light_curve(params)
		beaming = DopplerBeaming(pars, radians)
		ellipsoidal = EllipVariation(pars, radians, params.t0, params.ecc)
		fluxstar = fluxtransit + ellipsoidal + beaming

		fluxsystem = fluxplanet + fluxstar

	if Section=='Eclipse':
		params.rp = pars['rprs']
		params.limb_dark = "quadratic"
		params.u = [0.032460002, 0.29844000]
		if SpecNo == 99: params.fp = pars['secdepth']
		else: params.fp = pars['secdepth_' + str(SpecNo).zfill(2)]
		eclipsemodel = batman.TransitModel(params, timeForbatman, transittype="secondary")
		fluxeclipse = eclipsemodel.light_curve(params)-params.fp
		fluxsystem = fluxeclipse

	if Section=='Transit':
		if SpecNo == 99:
			params.t0 = pars['tc']
			params.per = 10. ** pars['logp']
			params.inc = (180. / math.pi) * numpy.arccos(numpy.fabs(pars['cosi']))
			params.a = 10. ** pars['logars']
			romerdelay = 2. * params.a * 1.03 * 696000. / 300000.  # light crossing time
			romerdelay /= (3600. * 24)  # secs to days
			params.t_secondary = params.t0 + params.per / 2. + romerdelay
			params.limb_dark = "quadratic"
			if Telescope=='HST':
				params.rp = pars['rprs_WFC3_'+visit]
				params.u = [pars['ulimb1'], pars['ulimb2']]  # this is for H!
			else:
				params.rp = pars['rprs_TESS']
				params.u = [0.34, 0.23]
		else:
			params.t0 = 2458329.200100647
			params.per = 10. ** 0.780638562833854
			params.inc = (180. / math.pi) * numpy.arccos(numpy.fabs(0.06440772705532975))
			params.a = 10. ** 1.1200598308963605
			romerdelay = 2. * params.a * 1.03 * 696000. / 300000.  # light crossing time
			romerdelay /= (3600. * 24)  # secs to days
			params.t_secondary = params.t0 + params.per / 2. + romerdelay
			params.rp = pars['rprs_'+str(SpecNo).zfill(2)]
			params.limb_dark = "quadratic"
			params.u = [pars['u1'], pars['u2']]
		params.fp = 0.0
		transitmodel = batman.TransitModel(params, timeForbatman)
		fluxtransit = transitmodel.light_curve(params)
		fluxsystem = fluxtransit

	return fluxsystem

def PhaseVariation(pars, radians, SpecNo=99):
	c1 = pars['c1']
	c2 = pars['c2'] * math.pi/180.
	c3 = pars['c3']
	c4 = pars['c4'] * math.pi/180.
	f0 = pars['phaseF0']

	phasecurve = f0 + c1*numpy.cos(radians+c2+math.pi) + c3*numpy.cos(2.*radians+c4+math.pi)

	return phasecurve

def EllipVariation(pars, radians, tP, e):
	# this fixes the grav. dark. coeff. to be 0.1
	beta = 0.15 * (15+pars['ulimb']) * (1+0.1) / (3-pars['ulimb'])

	tC = pars['tc']
	period = 10.**pars['logp']
	ars = 10.**pars['logars']
	MpMs = pars['mpms']
	i = numpy.arccos(numpy.fabs(pars['cosi']))

	nu = radians + 2. * math.pi * (tC-tP) / period
	nuTc = radians # 2. * math.pi * (time-tC) / period

	firstbit = beta * MpMs / ars**3.
	middlebit = ((1+e*numpy.cos(nu))/(1-e**2.))**3.
	EllipAmp = firstbit*middlebit*numpy.sin(i)**2.

	BackHeatAmp = pars['backheat']
	BackHeat = BackHeatAmp*numpy.cos(nuTc) + BackHeatAmp

	return -1.*EllipAmp*numpy.cos(2.*nuTc) + EllipAmp + BackHeat

def DopplerBeaming(pars, radians):
	alpha = 0.9

	ars = 10.**pars['logars']
	MpMs = pars['mpms']
	period = 10. ** pars['logp']

	astar = MpMs * ars * (1.46 * units.Rsun).to(units.m)
	period *= units.day
	KRV = (2 * math.pi * astar / period).to(units.m / units.s)

	BeamAmp = (3 - alpha) * KRV / const.c # Loeb & Gaudi 2007
	#BeamAmp = alpha * 4 * KRV / const.c # Mazeh and Faigler 2011

	#BeamAmp = 00.0e-6

	return BeamAmp * numpy.sin(radians)

def Lightcurve_indiv(pars, time, FitParams, SpecNo=99):
	t = numpy.copy(time).astype(float)

	params = batman.TransitParams
	params.t0 = FitParams['FixedPars']['tc']
	params.per = 10.**FitParams['FixedPars']['logp']
	params.inc = (180./math.pi) * numpy.arccos(FitParams['FixedPars']['cosi'])
	params.a = 10.**FitParams['FixedPars']['logars']

	params.w = numpy.arctan(FitParams['FixedPars']['sqrtesinw']/FitParams['FixedPars']['sqrtecosw'])*180./math.pi
	params.ecc = FitParams['FixedPars']['sqrtesinw']**2. + FitParams['FixedPars']['sqrtecosw']**2.

	params.limb_dark = "quadratic"

	roemerdelay = 2. * params.a * 1.46 * 696000. / 300000.  # light crossing time
	roemerdelay /= (3600.*24) # secs to days

	# Calculate the peri. time based on tC
	nu = math.pi/2. - params.w # true anomaly for transit away from peri.
	E = 2.*numpy.arctan(numpy.sqrt((1.0-params.ecc)/(1.+params.ecc))*numpy.tan(nu/2.)) # ecc. anomaly
	M = E - params.ecc*numpy.sin(E) # mean anomaly
	periPhase = M/(2.*math.pi)
	tP = params.t0 - params.per*periPhase # peri time

	nu = 3.*math.pi/2. - params.w # true anomaly for eclipse away from peri.
	E = 2.*numpy.arctan(numpy.sqrt((1.0-params.ecc)/(1.+params.ecc))*numpy.tan(nu/2.)) # ecc. anomaly
	M = E - params.ecc*numpy.sin(E) # mean anomaly
	periPhase = M/(2.*math.pi)
	params.t_secondary = tP + params.per*periPhase - params.per + roemerdelay

	timeForbatman = numpy.array(t)

	params.rp = FitParams['FixedPars']['rprs']
	params.u = [0.032460002, 0.29844000]#[pars['u1_' + str(SpecNo).zfill(2)], pars['u2_'+str(SpecNo).zfill(2)]]

	radians = 2. * math.pi * (timeForbatman - params.t0) / params.per

	params.fp = 1.0

	if FitParams['Section'] == 'Phase':
		eclipsemodel = batman.TransitModel(params, timeForbatman, transittype="secondary")
		fluxeclipse = eclipsemodel.light_curve(params) - 1.0
		phasecurve = PhaseVariation_Indiv(pars, radians, SpecNo=SpecNo)
		fluxplanet = phasecurve * fluxeclipse

		transitmodel = batman.TransitModel(params, timeForbatman)
		fluxtransit = transitmodel.light_curve(params)
		beaming = DopplerBeaming_Indiv(FitParams, radians)
		ellipsoidal = EllipVariation_Indiv(FitParams, pars, radians, tP, params.ecc)
		fluxstar = fluxtransit + ellipsoidal + beaming

		fluxsystem = fluxplanet + fluxstar

	if FitParams['Section']=='Transit':
		params.limb_dark = "quadratic"
		params.rp = 0.0766
		params.u = [0.032460002, 0.29844000]
		#params.u = [pars['ulimb'], pars['ulimb2']]
		#params.limb_dark = "linear"
		#params.u = [pars['ulimb']]
		transitmodel = batman.TransitModel(params, timeForbatman)
		fluxtransit = transitmodel.light_curve(params)

		paramsnight = copy.copy(params)
		paramsnight.t_secondary = params.t0
		paramsnight.t0 = params.t_secondary
		paramsnight.fp = pars['NightLevel']
		nightmodel = batman.TransitModel(paramsnight, timeForbatman, transittype="secondary")
		fluxnight = nightmodel.light_curve(paramsnight) - (1.0+paramsnight.fp)

		fluxsystem = fluxtransit - fluxnight

	return fluxsystem

def PhaseVariation_Indiv(pars, radians, SpecNo=99):
	c1 = pars['c1']
	c2 = pars['c2'] * math.pi/180.
	c3 = pars['c3']
	c4 = pars['c4'] * math.pi/180.
	f0 = pars['phaseF0']

	phasecurve = f0 + c1*numpy.cos(radians+c2+math.pi) + c3*numpy.cos(2.*radians+c4+math.pi)

	return phasecurve

def EllipVariation_Indiv(FitParams, pars, radians, tP, e):
	# this fixes the grav. dark. coeff. to be 0.1
	#beta = 0.15 * (15 + pars['ulimb']) * (1 + 0.1) / (3 - pars['ulimb'])
	beta = 0.15 * (15 + 0.17469382476730727) * (1 + 0.1) / (3 - 0.17469382476730727)
	#beta = 0.9 # baseline ulimb = 0.2

	tC = FitParams['FixedPars']['tc']
	period = 10.**FitParams['FixedPars']['logp']
	ars = 10.**FitParams['FixedPars']['logars']
	MpMs = FitParams['FixedPars']['mpms']
	i = numpy.arccos(FitParams['FixedPars']['cosi'])

	nu = radians + 2. * math.pi * (tC-tP) / period
	nuTc = radians # 2. * math.pi * (time-tC) / period

	firstbit = beta * MpMs / ars**3.
	middlebit = ((1+e*numpy.cos(nu))/(1-e**2.))**3.
	EllipAmp = firstbit*middlebit*numpy.sin(i)**3.

	BackHeatAmp = FitParams['FixedPars']['backheat']
	BackHeat = BackHeatAmp*numpy.cos(nuTc) + BackHeatAmp

	return -1.*EllipAmp*numpy.cos(2.*nuTc) + EllipAmp + BackHeat

def DopplerBeaming_Indiv(FitParams, radians):
	alpha = 0.9

	ars = 10.**FitParams['FixedPars']['logars']
	MpMs = FitParams['FixedPars']['mpms']
	period = 10.**FitParams['FixedPars']['logp']

	astar = MpMs * ars * (1.46 * units.Rsun).to(units.m)
	period *= units.day
	KRV = (2 * math.pi * astar / period).to(units.m / units.s)

	BeamAmp = (3 - alpha) * KRV / const.c # Loeb & Gaudi 2007
	#BeamAmp = alpha * 4 * KRV / const.c # Mazeh and Faigler 2011

	#BeamAmp = 00.0e-6

	return BeamAmp * numpy.sin(radians)

def PhaseTimes(time, t0, period, t0center=False):
	if t0center: phase = numpy.mod((time-t0)-period/2.,period) / period
	else: phase = numpy.mod((time-t0),period) / period
	return phase

def GelmanRubin(chains):
	nwalker = chains.shape[0]
	niter = chains.shape[1]
	npar = chains.shape[2]

	grarray = numpy.zeros(npar)

	for i in range(npar):
		sj2 = numpy.zeros(nwalker)
		chainmeans = numpy.zeros(nwalker)
		for j in range(nwalker):
			chainmeans[j] = numpy.mean(chains[j,:,i])
			sj2[j] = numpy.sum((chains[j,:,i]-chainmeans[j])**2.) / (niter-1)
		W = numpy.sum(sj2) / nwalker
		ThetaDoubleBar = numpy.sum(chainmeans) / nwalker
		B = numpy.sum((chainmeans-ThetaDoubleBar)**2.) * niter / (nwalker-1)
		VarTheta = (1-(1/niter))*W + (B/niter)
		grarray[i] = numpy.sqrt(VarTheta/W)
	return grarray

def GetETA(starttime, position, total):
	currenttime = time.time()
	deltat = currenttime-starttime
	rate = position / deltat
	remainingtime = (total-position)/rate
	totaltime = total / rate
	return remainingtime, totaltime

def RunMCMC(p0, sampler, nburn, nprod, nupdates, concordance, OnHPC=False, RunFile='running.out'):

	if OnHPC:
		outfile = open(RunFile, 'w')
		print("Running burn-in...", file=outfile)
		outfile.close()
		iterations = int(nburn/nupdates)
		starttime = time.time()
		for k in range(nupdates):
			p0, _, _ = sampler.run_mcmc(p0, iterations)
			eta, totaltime = GetETA(starttime, k+1, nupdates)
			outfile = open(RunFile, 'a')
			print(numpy.round(eta/60,3), numpy.round(totaltime/60,3), file=outfile)
			outfile.close()
		sampler.reset()

		outfile = open(RunFile, 'a')
		print("Running production...", file=outfile)
		outfile.close()
		iterations = int(nprod/nupdates)
		starttime = time.time()
		for k in range(nupdates):
			p0, _, _ = sampler.run_mcmc(p0, iterations)
			eta, totaltime = GetETA(starttime, k+1, nupdates)
			outfile = open(RunFile, 'a')
			print(numpy.round(eta/60,3), numpy.round(totaltime/60,3), file=outfile)
			outfile.close()
	else:
		if nburn>0:
			print("Running burn-in...")
			bar = progressbar.ProgressBar(maxval=nupdates, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(), ' ', progressbar.ETA()])
			bar.start()
			bar.update(0)
			iterations = int(nburn/nupdates)
			for k in range(nupdates):
				p0, _, _ = sampler.run_mcmc(p0, iterations)
				bar.update(k)
			bar.finish()
			sampler.reset()

		print("\n Running production...")
		bar = progressbar.ProgressBar(maxval=nupdates, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(), ' ', progressbar.ETA()])
		bar.start()
		bar.update(0)
		iterations = int(nprod/nupdates)
		for k in range(nupdates):
			p0, _, _ = sampler.run_mcmc(p0, iterations)
			bar.update(k)
		bar.finish()

	numpy.savez('./Emergency_MCMCSave.npz', chain=sampler.chain, lnprob=sampler.lnprobability, concordance=concordance)

	return sampler

def RampModel(pars, data, visit, FitParams, SpecNo=99):
	if FitParams['FitType']=='Spectra_Indiv':
		lc = Lightcurve_indiv(pars, data['time'], FitParams, SpecNo=SpecNo, Telescope='HST', visit=visit)
		#ramp = JustRamp(FitParams['FixedPars'], data, visit, SpecNo=SpecNo)
		ramp = JustRamp(pars, data, visit, SpecNo=SpecNo)
	else:
		lc = Lightcurve(pars, data['time'], Section=FitParams['Section'], SpecNo=SpecNo, Telescope='HST', visit=visit)
		ramp = JustRamp(pars, data, visit, SpecNo=SpecNo)
	model = ramp * lc
	return model

def JustRamp(pars, data, visit, SpecNo=99, VisitSlope=True):
	if VisitSlope:
		if SpecNo==99: slope = pars['Norm_V_'+visit] + pars['Slope_V_'+visit]*data['time_visit']
		else: slope = pars['Norm_V_'+visit] + pars['Slope_V_'+visit]*data['time_visit']
	else:
		slope = 1.
	# same ramp for everything
	orbitramp = 1 - pars['Amp_O_'+visit]*numpy.exp(data['time_orbit']/pars['Tau_O_'+visit])
	# different ramp for second orbit
	orbit2 = numpy.where(data['orbit_no']==1)
	orbitramp[orbit2] = 1 - pars['Amp_O2_'+visit] * numpy.exp(data['time_orbit'][orbit2] / pars['Tau_O2_'+visit])
	# different ramp for first orbit
	#orbit1 = numpy.where(data['orbit_no']==0)
	#orbitramp[orbit1] = 1 - pars['Amp_O1_'+visit] * numpy.exp(data['time_orbit'][orbit1] / pars['Tau_O1_'+visit])
	ramp = orbitramp * slope
	return ramp

def ParamModel(pars, data, visit, FitParams, SpecNo=99):
	if FitParams['FitType']=='Spectra_Indiv': lcmodel = Lightcurve_indiv(pars, data['time'], FitParams, SpecNo=SpecNo, visit=visit)
	else: lcmodel = Lightcurve(pars, data['time'], Section=FitParams['Section'], SpecNo=SpecNo, Telescope='HST', visit=visit)

	if FitParams['RampType']=='PolyRamp':
		fullmodel = RampModel(pars, data, visit, FitParams, SpecNo=SpecNo)
		detrendmodel = fullmodel / lcmodel
	elif FitParams['RampType']=='GPRamp':
		NoVisitSlope = JustRamp(pars, data, visit, VisitSlope=False)
		resid = data['flux'] - (lcmodel * NoVisitSlope)
		ToFit = data['time']
		'''
		kernel = pars['GP_Amp_' + visit] * kernels.ExpSquaredKernel(pars['GP_Gam_' + visit])
		gp = george.GP(kernel, mean=0.0)
		gp.compute(ToFit, data['error'])
		mu, cov = gp.predict(resid, ToFit)
		fullmodel = mu + (lcmodel * NoVisitSlope)
		detrendmodel = mu + NoVisitSlope
		'''

	return fullmodel, detrendmodel, lcmodel

def RECTE_ChargeModel(pars, data, visit, FitParams, SpecNo=99):
	if FitParams['FitType']=='Spectra_Indiv':
		lc = Lightcurve_indiv(pars, data['time'], FitParams, SpecNo=SpecNo, Telescope='HST', visit=visit)
		#ramp = JustRamp(FitParams['FixedPars'], data, visit, SpecNo=SpecNo)
		ramp = JustRamp(pars, data, visit, SpecNo=SpecNo)
	else:
		lc = Lightcurve(pars, data['time'], Section=FitParams['Section'], SpecNo=SpecNo, Telescope='HST', visit=visit)
		recte_correction = RECTE_JustChargeModel(pars, data, visit, SpecNo=SpecNo)
	model = recte_correction * lc
	return model

def RECTE_JustChargeModel(pars, data, visit, SpecNo=99):
	recte_time = data['time'] - 2458636.0
	recte_time = recte_time*24.*60.*60.
	chargemodel = rampProfile2(pars['norm_crate1_'+visit]*data['crate1'], pars['slope1_'+visit], pars['norm_crate2_'+visit]*data['crate2'], pars['slope2_'+visit], pars['dTrap_s_'+visit], pars['dTrap_f_'+visit], pars['trap_pop_s_'+visit], pars['trap_pop_f_'+visit], recte_time, data['texps'][0], data['scandir'])
	
	#visittime = data['time'] - numpy.median(data['time'])
	#chargemodel = chargemodel + pars['quad1_'+visit]*visittime**2.
	
	for orbit in range(4):
		idx = numpy.where(data['orbit_no'] == orbit)
		tdelta = (recte_time[idx] - numpy.median(recte_time[idx]))/(24.*60.*60.)
		chargemodel[idx] *= (1+pars['orbit_slope_'+str(orbit)+'_'+visit]*tdelta)
	return chargemodel

def RECTEModel(pars, data, visit, FitParams, SpecNo=99):
	if FitParams['FitType']=='Spectra_Indiv': lcmodel = Lightcurve_indiv(pars, data['time'], FitParams, SpecNo=SpecNo, visit=visit)
	else: lcmodel = Lightcurve(pars, data['time'], Section=FitParams['Section'], SpecNo=SpecNo, Telescope='HST', visit=visit)
	fullmodel = RECTE_ChargeModel(pars, data, visit, FitParams, SpecNo=SpecNo)
	detrendmodel = fullmodel / lcmodel
	return fullmodel, detrendmodel, lcmodel

def ParamLnLikelihood(pars, dataHST, dataTESS, dataRV, priors, priors_to_apply, FitParams):
	if FitParams['FitType'] != 'Spectra_Indiv':
		if pars['cosi'] < 0. or pars['cosi'] > 1.: return -numpy.inf
	if pars['ulimb1'] < 0.: return -numpy.inf
	if pars['ulimb2'] < 0.: return -numpy.inf

	lnprobmodel = 0.0

	if FitParams['FitType'] == 'Broad':
		if FitParams['RampType'] == 'PolyRamp':
			i = 1
			for visit in dataHST[0]:
				data = dataHST[i]
				model = RampModel(pars, data, visit, FitParams)
				if model[0] == -99: return -numpy.inf
				chiarr = ((data['flux'] - model) ** 2. / data['error'] ** 2.) + numpy.log(2. * math.pi * data['error'] ** 2.)
				lnprobmodel += -0.5 * numpy.sum(chiarr)
				i += 1

			model = Lightcurve(pars, dataTESS['time'], Section=FitParams['Section'], Telescope='TESS')
			if model[0] == -99: return -numpy.inf
			chiarr = ((dataTESS['flux'] - model) ** 2. / dataTESS['error'] ** 2.) + numpy.log(2. * math.pi * dataTESS['error'] ** 2.)
			lnprobmodel += -0.5 * numpy.sum(chiarr)

			model = RVSEDFuncs.RVOrbit(dataRV['time'], pars)
			if model[0] == -99: return -numpy.inf
			chiarr = ((dataRV['vel'] - model) ** 2. / dataRV['error'] ** 2.) + numpy.log(2. * math.pi * dataRV['error'] ** 2.)
			lnprobmodel += -0.5 * numpy.sum(chiarr)
			
	if FitParams['FitType'] == 'Spectra':
		if FitParams['RampType'] == 'PolyRamp':
			i = 1
			for visit in dataHST[0]:
				data = dataHST[i]
				model = RampModel(pars, data, visit, FitParams, SpecNo=1)
				if model[0] == -99: return -numpy.inf
				chiarr = ((data['flux'] - model) ** 2. / data['error'] ** 2.) + numpy.log(2. * math.pi * data['error'] ** 2.)
				lnprobmodel += -0.5 * numpy.sum(chiarr)
				i += 1

	else:
		print('"FitType" needs to be set correctly. Halting...')
		sys.exit()

	lnprobprior = 0.0
	for param in priors_to_apply['gaussian']:
		lnprobprior += -(pars[param] - priors[param][0]) ** 2. / (2 * priors[param][1] ** 2.) - numpy.log(
			math.sqrt(2 * priors[param][1] ** 2. * math.pi))

	lnprobtot = lnprobmodel + lnprobprior
	return lnprobtot

def ParamLnLikelihoodSingle(pars, dataV1, priors, priors_to_apply, FitParams):
	if FitParams['FitType'] != 'Spectra':
		if pars['cosi'] < 0. or pars['cosi'] > 1.: return -numpy.inf
	if pars['u1'] < 0.: return -numpy.inf
	if pars['u2'] < 0.: return -numpy.inf

	lnprobmodel = 0.0

	if FitParams['FitType'] == 'Broad':
		if FitParams['RampType'] == 'PolyRamp':
			model = RampModel(pars, dataV1, 'v1', FitParams)
			if model[0] == -99: return -numpy.inf
			chiarr = ((dataV1['flux'] - model) ** 2. / dataV1['error'] ** 2.) + numpy.log(2. * math.pi * dataV1['error'] ** 2.)
			lnprobmodel += -0.5 * numpy.sum(chiarr)
			
	if FitParams['FitType'] == 'Spectra':
		j = FitParams['SpecNo']
		if FitParams['RampType'] == 'PolyRamp':
			model = RampModel(pars, dataV1, 'v1', FitParams, SpecNo=j)
			if model[0] == -99: return -numpy.inf
			chiarr = ((dataV1['flux'][j] - model) ** 2. / dataV1['error'][j] ** 2.) + numpy.log(2. * math.pi * dataV1['error'][j] ** 2.)
			lnprobmodel += -0.5 * numpy.sum(chiarr)

	else:
		print('"FitType" needs to be set correctly. Halting...')
		sys.exit()

	lnprobprior = 0.0
	for param in priors_to_apply['gaussian']:
		lnprobprior += -(pars[param] - priors[param][0]) ** 2. / (2 * priors[param][1] ** 2.) - numpy.log(
			math.sqrt(2 * priors[param][1] ** 2. * math.pi))

	lnprobtot = lnprobmodel + lnprobprior
	return lnprobtot

def RECTELnLikelihood(pars, dataHST, dataTESS, dataRV, priors, priors_to_apply, FitParams):
	if FitParams['FitType'] != 'Spectra_Indiv':
		if pars['cosi'] < 0. or pars['cosi'] > 1.: return -numpy.inf
	if pars['ulimb1'] < 0.: return -numpy.inf
	if pars['ulimb2'] < 0.: return -numpy.inf

	lnprobmodel = 0.0

	if FitParams['FitType'] == 'Broad':
		i = 1
		for visit in dataHST[0]:
			data = dataHST[i]
			model = RECTE_ChargeModel(pars, data, visit, FitParams)
			if model[0] == -99: return -numpy.inf
			chiarr = ((data['flux'] - model) ** 2. / data['error'] ** 2.) + numpy.log(2. * math.pi * data['error'] ** 2.)
			lnprobmodel += -0.5 * numpy.sum(chiarr)
			i += 1

			model = Lightcurve(pars, dataTESS['time'], Section=FitParams['Section'], Telescope='TESS')
			if model[0] == -99: return -numpy.inf
			chiarr = ((dataTESS['flux'] - model) ** 2. / dataTESS['error'] ** 2.) + numpy.log(2. * math.pi * dataTESS['error'] ** 2.)
			lnprobmodel += -0.5 * numpy.sum(chiarr)

			model = RVSEDFuncs.RVOrbit(dataRV['time'], pars)
			if model[0] == -99: return -numpy.inf
			chiarr = ((dataRV['vel'] - model) ** 2. / dataRV['error'] ** 2.) + numpy.log(2. * math.pi * dataRV['error'] ** 2.)
			lnprobmodel += -0.5 * numpy.sum(chiarr)

	else:
		print('"FitType" needs to be set correctly. Halting...')
		sys.exit()

	lnprobprior = 0.0
	for param in priors_to_apply['gaussian']:
		lnprobprior += -(pars[param] - priors[param][0]) ** 2. / (2 * priors[param][1] ** 2.) - numpy.log(
			math.sqrt(2 * priors[param][1] ** 2. * math.pi))

	lnprobtot = lnprobmodel + lnprobprior
	return lnprobtot

def RECTELnLikelihoodSingle(pars, dataV1, priors, priors_to_apply, FitParams):
	if FitParams['FitType'] != 'Spectra_Indiv':
		if pars['cosi'] < 0. or pars['cosi'] > 1.: return -numpy.inf
	if pars['ulimb1'] < 0.: return -numpy.inf
	if pars['ulimb2'] < 0.: return -numpy.inf

	lnprobmodel = 0.0

	if FitParams['FitType'] == 'Broad':
		model = RECTE_ChargeModel(pars, dataV1, 'v1', FitParams)
		if model[0] == -99: return -numpy.inf
		chiarr = ((dataV1['flux'] - model) ** 2. / dataV1['error'] ** 2.) + numpy.log(2. * math.pi * dataV1['error'] ** 2.)
		lnprobmodel += -0.5 * numpy.sum(chiarr)

	lnprobprior = 0.0
	for param in priors_to_apply['gaussian']:
		lnprobprior += -(pars[param] - priors[param][0]) ** 2. / (2 * priors[param][1] ** 2.) - numpy.log(
			math.sqrt(2 * priors[param][1] ** 2. * math.pi))

	lnprobtot = lnprobmodel + lnprobprior
	return lnprobtot

def GetNightPriorLnp(pars, priors):
	fluxlevel = PhaseVariation_Indiv(pars, 0.0)
	lnpNight = -(fluxlevel - priors['NightLevel'][0]) ** 2. / (2 * priors['NightLevel'][1] ** 2.) - numpy.log(math.sqrt(2 * priors['NightLevel'][1] ** 2. * math.pi))
	return lnpNight

def GetSecPriorLnp(pars, priors):
	fluxlevel = PhaseVariation_Indiv(pars, math.pi)
	lnpNight = -(fluxlevel - priors['SecDepth'][0]) ** 2. / (2 * priors['SecDepth'][1] ** 2.) - numpy.log(math.sqrt(2 * priors['SecDepth'][1] ** 2. * math.pi))
	return lnpNight

def ParamMCMCFunc(parsarray, dataHST, dataTESS, dataRV, pars, priors, priors_to_apply, labels, concordance, FitParams):
	for i in range(len(labels)): pars[labels[i]] = parsarray[concordance[labels[i]]]
	lnp = ParamLnLikelihood(pars, dataHST, dataTESS, dataRV, priors, priors_to_apply, FitParams)
	return lnp

def ParamMCMCFuncSingle(parsarray, dataV1, pars, priors, priors_to_apply, labels, concordance, FitParams):
	for i in range(len(labels)): pars[labels[i]] = parsarray[concordance[labels[i]]]
	lnp = ParamLnLikelihoodSingle(pars, dataV1, priors, priors_to_apply, FitParams)
	return lnp

def RECTEMCMCFunc(parsarray, dataHST, dataTESS, dataRV, pars, priors, priors_to_apply, labels, concordance, FitParams):
	for i in range(len(labels)): pars[labels[i]] = parsarray[concordance[labels[i]]]
	lnp = RECTELnLikelihood(pars, dataHST, dataTESS, dataRV, priors, priors_to_apply, FitParams)
	return lnp

def RECTEMCMCFuncSingle(parsarray, dataV1, pars, priors, priors_to_apply, labels, concordance, FitParams):
	for i in range(len(labels)): pars[labels[i]] = parsarray[concordance[labels[i]]]
	lnp = RECTELnLikelihoodSingle(pars, dataV1, priors, priors_to_apply, FitParams)
	return lnp

def MedianAndClean(file, FitType='Broad', visit='v1', RECTE=False):
	loaded = numpy.load(file)
	time = loaded['time']
	flux_raw = loaded['flux']
	flux_err_raw = loaded['error']
	scandir = loaded['scandir']
	time_orbit = loaded['time_orbit']
	orbit2idxTemp = loaded['orbit2idx']
	orbit2idx = numpy.where(orbit2idxTemp)
	orbit_no = loaded['orbit_no'] - 1

	time_visit = time - time[0]  # OH BOY, this needs to be changed later. Later: does it? Why?

	down = numpy.where(scandir == 0)
	up = numpy.where(scandir == 1)

	down2 = numpy.where(scandir[orbit2idx] == 0)
	up2 = numpy.where(scandir[orbit2idx] == 1)

	if RECTE:
		if FitType == 'Broad':
			texps = loaded['texps']
			#upIndex, = numpy.where(scandir == 0)
			#downIndex, = numpy.where(scandir == 1)
			#divisor = 20000.
			if visit == 'v1':
				flux_raw = flux_raw / 20500.
				#flux_raw[down] /= 1.01
				flux_err_raw = flux_err_raw / 20500.
				#flux_err_raw[down] /= 1.01
			if visit == 'v2':
				flux_raw = flux_raw / 20600.
				#flux_raw[down] /= 1.012
				flux_err_raw = flux_err_raw / 20600.
				#flux_err_raw[down] /= 1.012
			crate1 = numpy.mean(flux_raw[down])
			crate2 = numpy.mean(flux_raw[up])
			data = {"time": time, "flux": flux_raw, "error": flux_err_raw, "scandir": scandir, "texps": texps, "crate1": crate1, "crate2": crate2, "orbit_no": orbit_no}
		if FitType == 'Spectra':  # I'm not 100% sure this works
			wavecenters = loaded['WaveCenters']
			texps = loaded['texps']
			flux = numpy.copy(flux_raw)
			error = numpy.copy(flux_err_raw)

			flux_raw = flux_raw / 3050.

			medflux_up = numpy.median(flux_raw[:, up[0]], axis=1)
			medflux_down = numpy.median(flux_raw[:, down[0]], axis=1)
			for idx in up[0]:
				flux[:, idx] /= medflux_up
				error[:, idx] /= medflux_up
				error[:, idx] /= 4.0 # This seems to be the case
			for idx in down[0]:
				flux[:, idx] /= medflux_down
				error[:, idx] /= medflux_down
				error[:, idx] /= 4.0 # This seems to be the case

			crate1 = medflux_down
			crate2 = medflux_up
			data = {"time": time, "flux": flux_raw, "error": flux_err_raw, "scandir": scandir, "texps": texps, "crate1": crate1, "crate2": crate2, "orbit_no": orbit_no, "wavecenters": wavecenters}
	else:
		if FitType == 'Broad':
			flux = numpy.copy(flux_raw)
			error = numpy.copy(flux_err_raw)
			medflux_up = numpy.median(flux_raw[up])
			flux[up] /= medflux_up
			error[up] /= medflux_up
			medflux_down = numpy.median(flux_raw[down])
			flux[down] /= medflux_down
			error[down] /= medflux_down

			#if visit=='v1':	flux[up2] += 0.00025

			data = {"time": time, "flux": flux, "error": error, "rawflux": flux_raw, "time_orbit": time_orbit, "time_visit": time_visit,
			                                                                    "orbit2idx": orbit2idx, "scandir": scandir, "orbit_no": orbit_no}
		if FitType == 'Spectra': # I'm not 100% sure this works
			wavecenters = loaded['WaveCenters']
			flux = numpy.copy(flux_raw)
			error = numpy.copy(flux_err_raw)
			medflux_up = numpy.median(flux_raw[:, up[0]], axis=1)
			medflux_down = numpy.median(flux_raw[:, down[0]], axis=1)
			for idx in up[0]:
				flux[:, idx] /= medflux_up
				error[:, idx] /= medflux_up
				error[:, idx] /= 4.0 # This seems to be the case
			for idx in down[0]:
				flux[:, idx] /= medflux_down
				error[:, idx] /= medflux_down
				error[:, idx] /= 4.0 # This seems to be the case
			data = {"time": time, "flux": flux, "error": error, "rawflux": flux_raw, "time_orbit": time_orbit, "time_visit": time_visit,
			                                                                    "orbit2idx": orbit2idx, "scandir": scandir, "orbit_no": orbit_no, "wavecenters": wavecenters}

	return data

def CombinedVisitPlots(pars, dataAll, FitParams):
	if FitParams['FitType'] == 'Spectra_Indiv':
		t0 = 2457306.97352
		period = 10. ** 0.0854663976564351
	else:
		t0 = pars['tc']
		period = 10.**pars['logp']

	if FitParams['FitType']=='Broad':
		fig = plt.figure()
		ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
		ax2 = plt.subplot2grid((4, 1), (3, 0), sharex=ax1)
		i=1
		for visit in dataAll[0]:
			data = dataAll[i]
			if FitParams['ModelType'] == 'RECTE':
				fullmodel, detrendmodel, lcmodel = RECTEModel(pars, data, visit, FitParams)
				resid = (data['flux'] - fullmodel) / numpy.median(data['flux'])
				error_plot = data['error'] / numpy.median(data['flux'])
			else:
				fullmodel, detrendmodel, lcmodel = ParamModel(pars, data, visit, FitParams)
				resid = (data['flux'] - fullmodel)
				error_plot = data['error']
			detrend = data['flux'] / detrendmodel
			phase = PhaseTimes(data['time'], t0, period, t0center=False)
			if FitParams['Section']=='Transit':
				beforetran = numpy.where(phase>0.5)
				phase[beforetran] -= 1.0
			#if visit=='v4':
			ax1.plot(phase, detrend, ls='None', marker='o', ms=2, zorder=1, label=visit)
			ax2.plot(phase, resid, 'o', ms=2)
			
			if FitParams['Section'] == 'Phase': modelphase = numpy.linspace(0.00, 1.00, 200)
			if FitParams['Section'] == 'Eclipse': modelphase = numpy.linspace(0.30, 0.70, 200)
			if FitParams['Section'] == 'Transit': modelphase = numpy.linspace(-0.03, 0.03, 200)
			modeltime = t0 + modelphase * period
			modellc = Lightcurve(pars, modeltime, Section=FitParams['Section'], visit=visit)
			ax1.plot(modelphase, modellc, '-', color='crimson', lw=1)
			
			i+=1

		if FitParams['Section']=='Phase': ax2.plot([0,1], [0,0], ":k")
		if FitParams['Section']=='Eclipse': ax2.plot([0.3,0.7], [0,0], ":k")
		if FitParams['Section']=='Transit': ax2.plot([-0.03,0.03], [0,0], ":k")

		#pars['rprs_WFC3'] = 0.0416
		#modellc = Lightcurve(pars, modeltime, Section=FitParams['Section'])
		#ax1.plot(modelphase, modellc, '--', color='crimson', lw=1, label='Expected Depth')

		ax1.legend(loc=2, fontsize=10, ncol=2)

		ax1.set_ylabel('Intensity', fontsize=16, labelpad=5)
		ax2.set_ylabel('O-C', fontsize=16, labelpad=0.5)
		ax2.set_xlabel('Orbital Phase', fontsize=16)

		ax1.set_ylim(0.997, 1.001)
		ax2.set_ylim(-0.0004, 0.0004)
		fig.subplots_adjust(hspace=0)
		plt.setp(ax1.get_xticklabels(), visible=False)

		plt.savefig('Detrend_'+FitParams['ModelType']+'.png', dpi=300)
		plt.clf()
		plt.close()

	elif FitParams['FitType']=='Spectra_Indiv':
		j = FitParams['SpecNo']
		fig = plt.figure()
		ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
		ax2 = plt.subplot2grid((4, 1), (3, 0), sharex=ax1)
		i = 1
		for visit in dataAll[0]:
			data = dataAll[i]
			if FitParams['ModelType'] == "Param": fullmodel, detrendmodel, lcmodel = ParamModel(pars, data, visit, FitParams, SpecNo=j)
			else:
				print('"ModelType" Is set incorrectly. Halting...')
				sys.exit()

			detrend = data['flux'][j] / detrendmodel
			resid = data['flux'][j] - fullmodel
			phase = PhaseTimes(data['time'], t0, period, t0center=False)
			if FitParams['Section'] == 'Transit':
				beforetran = numpy.where(phase > 0.5)
				phase[beforetran] -= 1.0
			ax1.plot(phase, detrend, ls='None', marker='o', ms=2, zorder=1, label=visit)
			ax2.plot(phase, resid, 'o', ms=2)
			i += 1

		ax1.legend(loc=2, fontsize=10, ncol=4)

		if FitParams['Section'] == 'Phase': modelphase = numpy.linspace(0.00, 1.00, 200)
		if FitParams['Section'] == 'Eclipse': modelphase = numpy.linspace(0.30, 0.70, 200)
		if FitParams['Section'] == 'Transit': modelphase = numpy.linspace(-0.20, 0.20, 200)
		modeltime = t0 + modelphase * period
		if FitParams['FitType'] == 'Spectra_Indiv': modellc = Lightcurve_indiv(pars, modeltime, FitParams, SpecNo=j)
		else: modellc = Lightcurve(pars, modeltime, Section=FitParams['Section'], SpecNo=j)
		ax1.plot(modelphase, modellc, '-', color='crimson', lw=1)

		ax1.set_ylabel('Intensity', fontsize=16, labelpad=18)
		ax2.set_ylabel('O-C', fontsize=16, labelpad=0.5)
		ax2.set_xlabel('Orbital Phase', fontsize=16)

		ax1.set_ylim(0.9935, 1.003)
		ax2.set_ylim(-0.0008, 0.0008)
		fig.subplots_adjust(hspace=0)
		plt.setp(ax1.get_xticklabels(), visible=False)

		plt.savefig('./Plots/Detrend_' + FitParams['ModelType'] + '_'+str(j).zfill(2)+'.png', dpi=120)
		plt.clf()
		plt.close()

	else:
		print('"FitType" needs to be set to either "Broad" or "Spectra_Indiv". Halting on CombinedVisitPlots...')
		sys.exit()

	return

def SingleVisitPlot(pars, data, FitParams, visit):
	if FitParams['FitType'] == 'Spectra':
		t0 = 2458329.200100647
		period = 10. ** 0.780638562833854
	else:
		t0 = pars['tc']
		period = 10.**pars['logp']

	fig = plt.figure()
	ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
	ax2 = plt.subplot2grid((4, 1), (3, 0), sharex=ax1)
	i=1

	if FitParams['ModelType']=='Broad':
		if FitParams['ModelType']=='RECTE':
			fullmodel, detrendmodel, lcmodel = RECTEModel(pars, data, visit, FitParams)
			resid = (data['flux'] - fullmodel) / numpy.median(data['flux'])
			error_plot = data['error'] / numpy.median(data['flux'])
		else:
			fullmodel, detrendmodel, lcmodel = ParamModel(pars, data, visit, FitParams)
			resid = (data['flux'] - fullmodel)
			error_plot = data['error']
		detrend = data['flux'] / detrendmodel
	else:
		fullmodel, detrendmodel, lcmodel = ParamModel(pars, data, visit, FitParams, SpecNo=FitParams['SpecNo'])
		resid = (data['flux'][FitParams['SpecNo']] - fullmodel)
		error_plot = data['error'][FitParams['SpecNo']]
		detrend = data['flux'][FitParams['SpecNo']] / detrendmodel
	phase = PhaseTimes(data['time'], t0, period, t0center=False)
	if FitParams['Section']=='Transit':
		beforetran = numpy.where(phase>0.5)
		phase[beforetran] -= 1.0
	#ax1.plot(data['time']-2458600, detrend, ls='None', marker='o', ms=2, zorder=1, label='Visit 1')
	ax1.errorbar(data['time']-2458600, detrend, yerr=error_plot, fmt="o", ms=2, capsize=0, zorder=2, mec='k', mew=0.5, lw=1, label=visit)
	#ax2.plot(data['time']-2458600, resid, 'o', ms=2)
	ax2.errorbar(data['time'] - 2458600, resid, yerr=error_plot, fmt="o", ms=2, capsize=0, zorder=2, mec='k', mew=0.5, lw=1)
	i+=1

	modeltime = numpy.linspace(data['time'][0],data['time'][-1], 200)
	modelphase = PhaseTimes(modeltime, t0, period, t0center=True)-0.5
	modellc = Lightcurve(pars, modeltime, Section=FitParams['Section'], visit=visit, SpecNo=FitParams['SpecNo'])

	#parsEarly = pars
	#parsEarly['tc'] = 2458329.1996
	#parsEarly['logp'] = numpy.log10(6.03607)
	#parsEarly['rprs_WFC3'] = 0.04192
	#modellcEarly = Lightcurve(parsEarly, modeltime, Section=FitParams['Section'])

	modeltime -= 2458600
	ax1.plot(modeltime, modellc, '-', color='crimson', lw=1, label='Fit')
	#ax1.plot(modeltime, modellcEarly, ':', color='crimson', lw=1, label='Predicted')
	ax2.plot([modeltime[0],modeltime[-1]], [0,0], ":k")

	#ax1.plot(modelphase, modellc, '-', color='crimson', lw=1)
	#ax2.plot([modelphase[0],modelphase[-1]], [0,0], ":k")

	ax1.set_ylabel('Intensity', fontsize=16, labelpad=5)
	ax2.set_ylabel('O-C', fontsize=16, labelpad=0.5)
	ax2.set_xlabel('BJD-2458600', fontsize=16)

	ax1.legend(loc=3, fontsize=10, ncol=4)

	ax1.set_ylim(0.997, 1.001)
	ax2.set_ylim(-0.0004, 0.0004)
	fig.subplots_adjust(hspace=0)
	plt.setp(ax1.get_xticklabels(), visible=False)
	
	if FitParams['ModelType'] == 'Broad':
		plt.savefig('Detrend_'+FitParams['ModelType']+'_'+visit+'.png', dpi=300)
	else:
		plt.savefig('./OutputPlots/Detrend_' + FitParams['ModelType'] + '_' + visit + '_'+str(FitParams['SpecNo']).zfill(2)+'.png', dpi=300)
	plt.clf()
	plt.close()

	fig = plt.figure()
	ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
	ax2 = plt.subplot2grid((4, 1), (3, 0), sharex=ax1)
	
	if FitParams['ModelType'] == 'Broad':
		ax1.plot(data['time'], data['flux'], ls='None', marker='o', ms=2, zorder=1, label=visit)
	else:
		ax1.plot(data['time'], data['flux'][FitParams['SpecNo']], ls='None', marker='o', ms=2, zorder=1, label=visit)
	ax1.plot(data['time'], fullmodel, '-', color='crimson', lw=1)

	ax1.legend(loc=2, fontsize=10, ncol=4)

	if FitParams['Section']=='Phase': ax2.plot([0,1], [0,0], ":k")
	if FitParams['Section']=='Eclipse': ax2.plot([0.3,0.7], [0,0], ":k")
	if FitParams['Section']=='Transit': ax2.plot([data['time'][0],data['time'][-1]], [0,0], ":k")
	if FitParams['ModelType']=='RECTE': ax2.plot(data['time'], resid*numpy.median(data['flux']), 'o', ms=2)
	else: ax2.plot(data['time'], resid, 'o', ms=2)

	ax1.set_ylabel('Intensity', fontsize=16, labelpad=5)
	ax2.set_ylabel('O-C', fontsize=16, labelpad=0.5)
	ax2.set_xlabel('BJD', fontsize=16)

	#ax1.set_ylim(0.9935, 1.003)
	#ax2.set_ylim(-0.0004, 0.0004)
	fig.subplots_adjust(hspace=0)
	plt.setp(ax1.get_xticklabels(), visible=False)

	
	if FitParams['ModelType'] == 'Broad':
		plt.savefig('Raw_'+FitParams['ModelType']+'.png', dpi=300)
	else:
		plt.savefig('./OutputPlots/Raw_' + FitParams['ModelType']+str(FitParams['SpecNo']).zfill(2)+'.png', dpi=300)
	plt.clf()
	plt.close()

	return

def TESSDataPlot(pars, data, FitParams):
	t0 = pars['tc']
	period = 10.**pars['logp']

	if FitParams['FitType']=='Broad':
		fig = plt.figure()
		ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
		ax2 = plt.subplot2grid((4, 1), (3, 0), sharex=ax1)
		i=1

		lcmodel = Lightcurve(pars, data['time'], Section=FitParams['Section'], Telescope='TESS')
		detrend = data['flux']
		resid = data['flux']-lcmodel
		phase = PhaseTimes(data['time'], t0, period, t0center=False)
		if FitParams['Section']=='Transit':
			beforetran = numpy.where(phase>0.5)
			phase[beforetran] -= 1.0
		timeplot = phase * period
		ax1.plot(timeplot, detrend, ls='None', marker='.', ms=0.3, zorder=1, color='gray')
		#ax1.errorbar(phase, detrend, yerr=data['error'], fmt="o", ms=2, capsize=0, zorder=2, mec='k', mew=0.5, lw=1, label='TESS')
		ax2.plot(timeplot, resid, ls='None', marker='.', ms=0.3, zorder=1, color='gray')
		#ax2.errorbar(phase, resid, yerr=data['error'], fmt="o", ms=2, capsize=0, zorder=2, mec='k', mew=0.5, lw=1)

		sortidx = numpy.argsort(timeplot)
		timeplot = timeplot[sortidx]
		detrend = detrend[sortidx]
		resid = resid[sortidx]
		error = data['error'][sortidx]

		timerange = (timeplot[-1] -timeplot[0]) * 24. * 60.  # in minutes
		binsize = 10.  # in minutes
		nbins = numpy.round(timerange / binsize)
		binarr = binned_statistic(timeplot, detrend, statistic='median', bins=nbins)
		binflux = binarr[0]
		binarr = binned_statistic(timeplot, resid, statistic='median', bins=nbins)
		binresid = binarr[0]
		binarr = binned_statistic(timeplot, timeplot, statistic='median', bins=nbins)
		bintime = binarr[0]
		binarr = binned_statistic(timeplot, error, statistic='mean', bins=nbins)
		errtemp = binarr[0]
		binarr = binned_statistic(timeplot, error, statistic='count', bins=nbins)
		binerr = errtemp / numpy.sqrt(binarr[0])

		ax1.errorbar(bintime, binflux, yerr=binerr, fmt="o", ms=2, capsize=0, zorder=2, mec='k', mew=0.5, lw=1, color='dodgerblue')
		ax2.errorbar(bintime, binresid, yerr=binerr, fmt="o", ms=2, capsize=0, zorder=2, mec='k', mew=0.5, lw=1, color='dodgerblue')

		modelphase = numpy.linspace(-0.5,0.5, 4000)
		modeltime = modelphase*period + t0
		modellc = Lightcurve(pars, modeltime, Section=FitParams['Section'], Telescope='TESS')

		modeltime -= t0
		ax1.plot(modeltime, modellc, '-', color='crimson', lw=1, label='Fit', zorder=3)
		ax2.plot([modeltime[0],modeltime[-1]], [0,0], ":k", zorder=3)

		ax1.set_ylabel('Intensity', fontsize=16, labelpad=5)
		ax2.set_ylabel('O-C', fontsize=16, labelpad=0.5)
		ax2.set_xlabel('Time From Transit Center (Days)', fontsize=16)

		ax1.set_ylim(0.996, 1.004)
		ax2.set_ylim(-0.0013, 0.0013)
		fig.subplots_adjust(hspace=0)
		plt.setp(ax1.get_xticklabels(), visible=False)

		plt.savefig('Detrend_TESS_All.png', dpi=300)
		plt.clf()
		plt.close()

		###############################################

		fig = plt.figure()
		ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
		ax2 = plt.subplot2grid((4, 1), (3, 0), sharex=ax1)

		ax1.plot(timeplot, detrend, ls='None', marker='.', ms=0.5, zorder=1, color='gray')
		ax2.plot(timeplot, resid, ls='None', marker='.', ms=0.5, zorder=1, color='gray')

		ax1.errorbar(bintime, binflux, yerr=binerr, fmt="o", ms=2, capsize=0, zorder=2, mec='k', mew=0.5, lw=1, color='dodgerblue')
		ax2.errorbar(bintime, binresid, yerr=binerr, fmt="o", ms=2, capsize=0, zorder=2, mec='k', mew=0.5, lw=1, color='dodgerblue')

		ax1.plot(modeltime, modellc, '-', color='crimson', lw=1, label='Fit', zorder=3)
		ax2.plot([modeltime[0],modeltime[-1]], [0,0], ":k", zorder=3)

		ax1.set_ylabel('Intensity', fontsize=16, labelpad=5)
		ax2.set_ylabel('O-C', fontsize=16, labelpad=0.5)
		ax2.set_xlabel('Time From Transit Center (Days)', fontsize=16)

		ax1.set_xlim(-0.15,0.15)
		ax1.set_ylim(0.996, 1.004)
		ax2.set_ylim(-0.0013, 0.0013)
		fig.subplots_adjust(hspace=0)
		plt.setp(ax1.get_xticklabels(), visible=False)

		plt.savefig('Detrend_TESS_Zoom.png', dpi=300)
		plt.clf()
		plt.close()

	return

def Phase_Spectra_Diag_Plots(inputpars, dataAll, FitParams, MCMC=False):
	if MCMC: # setup for output
		labels = list(inputpars.keys())
		pars = {}
		for i in range(len(labels)): pars[labels[i]] = inputpars[labels[i]][0]
	else: pars = inputpars

	t0 = pars['tc']
	period = 10.**pars['logp']
	nspec = dataAll[1]['flux'].shape[0]

	# LC detrend plots
	fig = plt.figure()
	for j in range(nspec):
		col = numpy.mod(j, 4)
		row = numpy.floor((j/4)).astype(int)
		ax = plt.subplot2grid ((4, 4), (row, col))
		i = 1
		color = iter(cm.viridis(numpy.linspace(0.0, 1.0, 4)))
		for visit in dataAll[0]:
			data = dataAll[i]
			phase = PhaseTimes(data['time'], t0, period, t0center=False)
			fullmodel, detrendmodel, lcmodel = ParamModel(pars, data, visit, Section='Phase', SpecNo=j, FitType=FitParams['FitType'])
			detrend = data['flux'][j] / detrendmodel
			coloruse = next(color)
			ax.plot(phase, detrend, 'o', color=coloruse, ms=1, zorder=2, mec='k', mew=0)

			modelphase = numpy.linspace(0.00, 1.00, 200)
			modeltime = t0 + modelphase * period
			modellc = Lightcurve(pars, modeltime, Section=FitParams['Section'], SpecNo=j)
			ax.plot(modelphase, modellc, '-', color='crimson', lw=1)
			i+=1
		plt.setp(ax.get_xticklabels(), visible=False)
		plt.setp(ax.get_yticklabels(), visible=False)
		ax.set_ylim(0.991, 1.003)
		ax.set_xlim(-0.05, 1.05)
	fig.subplots_adjust(hspace=0, wspace=0)
	plt.savefig('Diag_Detrended.png', dpi=200)
	plt.clf()
	plt.close()

	if MCMC:

		fig = plt.figure()
		param = 'c1'
		ax = plt.subplot2grid((2, 2), (0, 0))
		plot_arr = numpy.zeros(nspec)
		plot_err = numpy.zeros((2, nspec))
		for j in range(nspec):
			plot_arr[j] = 1e6*inputpars[param + '_' + str(j).zfill(2)][0]
			plot_err[0, j] = 1e6*inputpars[param + '_' + str(j).zfill(2)][1]
			plot_err[1, j] = 1e6*inputpars[param + '_' + str(j).zfill(2)][2]
		plt.errorbar(dataAll[1]['wavecenters'], plot_arr, yerr=plot_err, fmt='s', ecolor='black', markerfacecolor='black', mec='black',
	                 capsize=2, markeredgewidth=2, ms=4, zorder=2, lw=2)
		#ax.plot(dataAll[1]['wavecenters'], plot_arr, 'sk')
		param = 'c2'
		ax = plt.subplot2grid((2, 2), (0, 1))
		plot_arr = numpy.zeros(nspec)
		plot_err = numpy.zeros((2, nspec))
		for j in range(nspec):
			plot_arr[j] = inputpars[param + '_' + str(j).zfill(2)][0]
			plot_err[0, j] = inputpars[param + '_' + str(j).zfill(2)][1]
			plot_err[1, j] = inputpars[param + '_' + str(j).zfill(2)][2]
		plt.errorbar(dataAll[1]['wavecenters'], plot_arr, yerr=plot_err, fmt='s', ecolor='black', markerfacecolor='black', mec='black',
		             capsize=2, markeredgewidth=2, ms=4, zorder=2, lw=2)
		#ax.plot(dataAll[1]['wavecenters'], plot_arr, 'sk')
		param = 'c3'
		ax = plt.subplot2grid((2, 2), (1, 0))
		plot_arr = numpy.zeros(nspec)
		plot_err = numpy.zeros((2, nspec))
		for j in range(nspec):
			plot_arr[j] = 1e6*inputpars[param + '_' + str(j).zfill(2)][0]
			plot_err[0, j] = 1e6*inputpars[param + '_' + str(j).zfill(2)][1]
			plot_err[1, j] = 1e6*inputpars[param + '_' + str(j).zfill(2)][2]
		plt.errorbar(dataAll[1]['wavecenters'], plot_arr, yerr=plot_err, fmt='s', ecolor='black', markerfacecolor='black', mec='black',
		             capsize=2, markeredgewidth=2, ms=4, zorder=2, lw=2)
		#ax.plot(dataAll[1]['wavecenters'], plot_arr, 'sk')
		param = 'c4'
		ax = plt.subplot2grid((2, 2), (1, 1))
		plot_arr = numpy.zeros(nspec)
		plot_err = numpy.zeros((2, nspec))
		for j in range(nspec):
			plot_arr[j] = inputpars[param + '_' + str(j).zfill(2)][0]
			plot_err[0, j] = inputpars[param + '_' + str(j).zfill(2)][1]
			plot_err[1, j] = inputpars[param + '_' + str(j).zfill(2)][2]
		plt.errorbar(dataAll[1]['wavecenters'], plot_arr, yerr=plot_err, fmt='s', ecolor='black', markerfacecolor='black', mec='black',
		             capsize=2, markeredgewidth=2, ms=4, zorder=2, lw=2)
		#ax.plot(dataAll[1]['wavecenters'], plot_arr, 'sk')
		plt.savefig('Diag_CVals.png', dpi=200)
		plt.clf()
		plt.close()



		fig = plt.figure()
		param = 'phaseF0'
		ax = plt.subplot2grid((2, 2), (0, 0))
		plot_arr = numpy.zeros(nspec)
		plot_err = numpy.zeros((2, nspec))
		for j in range(nspec):
			plot_arr[j] = 1e6*inputpars[param + '_' + str(j).zfill(2)][0]
			plot_err[0, j] = 1e6*inputpars[param + '_' + str(j).zfill(2)][1]
			plot_err[1, j] = 1e6*inputpars[param + '_' + str(j).zfill(2)][2]
		plt.errorbar(dataAll[1]['wavecenters'], plot_arr, yerr=plot_err, fmt='s', ecolor='black', markerfacecolor='black', mec='black',
		             capsize=2, markeredgewidth=2, ms=4, zorder=2, lw=2)
		#ax.plot(dataAll[1]['wavecenters'], plot_arr, 'sk')
		param = 'u1'
		ax = plt.subplot2grid((2, 2), (0, 1))
		plot_arr = numpy.zeros(nspec)
		plot_err = numpy.zeros((2, nspec))
		for j in range(nspec):
			plot_arr[j] = inputpars[param + '_' + str(j).zfill(2)][0]
			plot_err[0, j] = inputpars[param + '_' + str(j).zfill(2)][1]
			plot_err[1, j] = inputpars[param + '_' + str(j).zfill(2)][2]
		plt.errorbar(dataAll[1]['wavecenters'], plot_arr, yerr=plot_err, fmt='s', ecolor='black', markerfacecolor='black', mec='black',
		             capsize=2, markeredgewidth=2, ms=4, zorder=2, lw=2)
		#ax.plot(dataAll[1]['wavecenters'], plot_arr, 'sk')
		param = 'u2'
		ax = plt.subplot2grid((2, 2), (1, 0))
		plot_arr = numpy.zeros(nspec)
		plot_err = numpy.zeros((2, nspec))
		for j in range(nspec):
			plot_arr[j] = inputpars[param + '_' + str(j).zfill(2)][0]
			plot_err[0, j] = inputpars[param + '_' + str(j).zfill(2)][1]
			plot_err[1, j] = inputpars[param + '_' + str(j).zfill(2)][2]
		plt.errorbar(dataAll[1]['wavecenters'], plot_arr, yerr=plot_err, fmt='s', ecolor='black', markerfacecolor='black', mec='black',
		             capsize=2, markeredgewidth=2, ms=4, zorder=2, lw=2)
		#ax.plot(dataAll[1]['wavecenters'], plot_arr, 'sk')
		plt.savefig('Diag_OVals.png', dpi=200)
		plt.clf()
		plt.close()

	return

def PhaseCurveComps_Plots(pars, FitParams):
	degrees = numpy.linspace(-220, 220, 400)
	radians = degrees * math.pi / 180.

	if FitParams['FitType'] == 'Spectra_Indiv':
		phasecurve = PhaseVariation_Indiv(pars, radians)
		ellipsoidal = EllipVariation_Indiv(FitParams, pars, radians, 0.0, 0.0)
		beaming = DopplerBeaming_Indiv(FitParams, radians)
		savefile = './Plots/Phasecurve_Comps_' + str(FitParams['SpecNo']).zfill(2) + '.png'
	else:
		phasecurve = PhaseVariation(pars, radians)
		ellipsoidal = EllipVariation(pars, radians, 0.0, 0.0)
		beaming = DopplerBeaming(pars, radians)
		savefile = 'Phasecurve_Comps.png'

	combined = phasecurve + beaming + ellipsoidal

	plt.plot(degrees, phasecurve * 10 ** 6., '--k', label='Planetary Phase')
	plt.plot(degrees, ellipsoidal * 10 ** 6., '--b', label='Ellipsoidal Def.')
	plt.plot(degrees, beaming * 10 ** 6., '--g', label='Doppler Beaming')
	plt.plot(degrees, combined * 10 ** 6., '-r', label='Combined')

	plt.plot([180,180], [-1000.,5000.], ':k')

	plt.legend()

	plt.xlabel('Orbital Phase (degrees)', fontsize=16)
	plt.ylabel('F$_\mathrm{p}$/F$_*$ (ppm)', fontsize=16, labelpad=3)

	plt.ylim(-200, 1500)
	plt.grid()

	plt.savefig(savefile, dpi=200)
	plt.clf()

	return