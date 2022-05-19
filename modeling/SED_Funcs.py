import numpy
import sys

def InitializeExtinction(seddata):
	wave, kappa = numpy.loadtxt('extinction_law.ascii', unpack=True)
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

def MagsToFluxes(filename, Teff=5040., PrintFluxes=False):

	# see mag2fluxconv for important notes and refs!

	band = numpy.loadtxt(filename, unpack=True, usecols=[0], dtype='object')
	mag, merr = numpy.loadtxt(filename, unpack=True, usecols=[1,2])

	theta = Teff / 5040.
	nband = band.shape[0]
	lameff = numpy.zeros(nband)
	weff = numpy.zeros(nband)
	flux = numpy.zeros(nband)

	for i in range(nband):
		if band[i] == 'U':
			lameff[i] = numpy.polyval([3476.,162.,86.,-63.][::-1], theta) / 1.e4
			weff[i] = numpy.polyval([612.,346.,-741.,269.][::-1], theta) / 1.e4
			lamflamzp = 1.51e-5
			flux[i] = lamflamzp * 10**(-0.4 * mag[i])
		elif band[i] == 'B':
			lameff[i] = numpy.polyval([4336., 201., 235., -115.][::-1], theta) / 1.e4
			weff[i] = numpy.polyval([863., 494., -833., 192.][::-1], theta) / 1.e4
			lamflamzp = 2.90e-5
			flux[i] = lamflamzp * 10**(-0.4 * mag[i])
		elif band[i] == 'V':
			print('Not implemented')
			sys.exit()
		elif band[i] == 'R':
			print('Not implemented')
			sys.exit()
		elif band[i] == 'RC':
			print('Not implemented')
			sys.exit()
		elif band[i] == 'I':
			print('Not implemented')
			sys.exit()
		elif band[i] == 'IC':
			print('Not implemented')
			sys.exit()
		elif band[i] == 'J':
			print('Not implemented')
			sys.exit()
		elif band[i] == 'H':
			print('Not implemented')
			sys.exit()
		elif band[i] == 'K':
			print('Not implemented')
			sys.exit()
		elif band[i] == 'J2M':
			lameff[i] = numpy.polyval([1.23,0.08,0.0,-0.01][::-1], theta)
			weff[i] = numpy.polyval([0.16,0.08,-0.23,0.11][::-1], theta)
			lamflamzp = 1594. * 3e-9 / 1.235**2 * 1.235
			flux[i] = lamflamzp * 10**(-0.4 * mag[i])
		elif band[i] == 'H2M':
			lameff[i] = numpy.polyval([1.64,0.05,0.02,-0.02][::-1], theta)
			weff[i] = numpy.polyval([0.24,0.09,-0.21,0.07][::-1], theta)
			lamflamzp = 1024. * 3e-9 / 1.662**2 * 1.662
			flux[i] = lamflamzp * 10**(-0.4 * mag[i])
		elif band[i] == 'K2M':
			lameff[i] = numpy.polyval([2.15,0.03,0.02,-0.01][::-1], theta)
			weff[i] = numpy.polyval([0.25,0.04,-0.06,0.01][::-1], theta)
			lamflamzp = 666.7 * 3.e-9 / 2.159**2 * 2.159
			flux[i] = lamflamzp * 10**(-0.4 * mag[i])
		elif band[i] == 'BT':
			print('Not implemented')
			sys.exit()
		elif band[i] == 'VT':
			print('Not implemented')
			sys.exit()
		elif band[i] == 'gSDSS':
			# AB mags - need to convert to AB if in Vegamag, see link above
			lameff[i] = numpy.polyval([4647.,312.,241.,-173.][::-1], theta) / 1.e4
			weff[i] = numpy.polyval([1156.,909.,-1424.,387.][::-1], theta) / 1.e4
			lamflamzp = 3631. * 3e-9 / 0.4788**2 * 0.4788
			flux[i] = lamflamzp * 10**(-0.4 * mag[i]-0.04) #  assuming Pogson mags instead of asinh
		elif band[i] == 'rSDSS':
			# AB mags - need to convert to AB if in Vegamag, see link above
			lameff[i] = numpy.polyval([6145.,139.,156.,-80.][::-1], theta) / 1.e4
			weff[i] = numpy.polyval([1255.,289.,-183.,-109.][::-1], theta) / 1.e4
			lamflamzp = 3631. * 3e-9 / 0.6242**2 * 0.6242
			flux[i] = lamflamzp * 10**(-0.4 * mag[i]-0.04) #  assuming Pogson mags instead of asinh
		elif band[i] == 'iSDSS':
			# AB mags - need to convert to AB if in Vegamag, see link above
			lameff[i] = numpy.polyval([7562.,101.,123.,-52.][::-1], theta) / 1.e4
			weff[i] = numpy.polyval([1310.,144.,-9.,-104.][::-1], theta) / 1.e4
			lamflamzp = 3631. * 3e-9 / 0.7704**2 * 0.7704
			flux[i] = lamflamzp * 10**(-0.4 * mag[i]-0.04) #  assuming Pogson mags instead of asinh
		elif band[i] == 'zSDSS':
			# AB mags - need to convert to AB if in Vegamag, see link above
			lameff[i] = numpy.polyval([8997.,88.,105.,-36.][::-1], theta) / 1.e4
			weff[i] = numpy.polyval([1357.,91.,24.,-76.][::-1], theta) / 1.e4
			lamflamzp = 3631. * 3e-9 / 0.9038**2 * 0.9038
			flux[i] = lamflamzp * 10**(-0.4 * mag[i]-0.04) #  assuming Pogson mags instead of asinh
		elif band[i] == 'galNUV':
			print('Not implemented')
			sys.exit()
		elif band[i] == 'galFUV':
			print('Not implemented')
			sys.exit()
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
		elif band[i] == 'WISE3':
			print('Not implemented')
			sys.exit()
		elif band[i] == 'WISE4':
			print('Not implemented')
			sys.exit()
		elif band[i] == 'Ch1':
			lameff[i] = 35500. / 1.e4
			weff[i] = 6626. / 1.e4
			lamflam = 7.57e-4 * 280.9 * 3e-5 / lameff[i] ** 2
			flux[i] = lamflam * 10 ** (-0.4 * mag[i])
		elif band[i] == 'Ch2':
			lameff[i] = 44930. / 1.e4
			weff[i] = 6626. / 1.e4
			lamflam = 6.93e-4 * 179.7 * 3e-5 / lameff[i] ** 2
			flux[i] = lamflam * 10 ** (-0.4 * mag[i])
		else:
			print('Invalid band selection!')
			sys.exit()

	fluxerr = flux * numpy.log(10) / 2.5 * merr

	return lameff, weff, flux, fluxerr