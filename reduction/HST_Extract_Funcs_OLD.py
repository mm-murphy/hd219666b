import numpy, math, sys, scipy, progressbar
import matplotlib.pyplot as plt
import astropy.time as time
from astropy.stats import sigma_clipped_stats
from astropy.io import fits
from astropy.convolution import convolve, Gaussian1DKernel
from astropy.coordinates import SkyCoord, EarthLocation
from astropy import units as units
from photutils import DAOStarFinder
from scipy.optimize import minimize
from scipy import interpolate

def CreateSubExps(file, WaveSol):
    hdu = fits.open(file)
    bjdtdb = []
    texp = []

    scanangle = hdu[0].header['SCAN_ANG']
    scandir = 0
    if scanangle > 180.: scandir = 1

    # setup for individual image
    imageidxs = numpy.arange(1, 76, 5)  # doesn't include 71
    nsubexp = imageidxs.shape[0]
    subexps = numpy.zeros((nsubexp, 522, 522))
    dqs = numpy.zeros((nsubexp, 522, 522))
    badpix_masks = numpy.full((nsubexp, 522, 522), False)

    # sub exposure shifts #
    side_angle = hdu[0].header['ANG_SIDE']
    x = numpy.arange(nsubexp)
    b = 0.0
    m = 5.85532565271 * numpy.cos(side_angle * math.pi / 180.)
    subexp_shifts = m * x + b

    # subtracting subsequent reads from other
    for i in range(nsubexp):
        preflat = hdu[imageidxs[i]].data
        subexps[i, :, :] = preflat

        dqs[i,:,:] = hdu[imageidxs[i]+2].data

        texp = numpy.append(texp, hdu[imageidxs[i]].header['SAMPTIME'])

        readtime = hdu[imageidxs[i]].header['ROUTTIME']
        if imageidxs[i] < 71:
            subexptime = (hdu[imageidxs[i]].header['DELTATIM']) / (24. * 3600.) 
        else:  # turns out the first multiaccum runs slightly long. Samptime is real time, deltatim is what HST thinks
            subexptime = (hdu[imageidxs[i]].header['SAMPTIME']) / (24. * 3600.)
        subexp_mjd = time.Time((readtime - subexptime / 2.), format='mjd', scale='utc')
        subexp_jd = time.Time(subexp_mjd.jd, format='jd', scale='utc')
        subexp_jd_tdb = subexp_jd.tdb
        Position = SkyCoord('23:29:13.6', '-56:54:14.0', unit=(units.hourangle, units.deg), frame='icrs')
        ObsLocation = EarthLocation.of_site('Kitt Peak') # not actually, but we don't care about the 1 ms error
        subexp_bjd_tdb = subexp_jd_tdb + subexp_jd_tdb.light_travel_time(Position, location=ObsLocation)
        bjdtdb = numpy.append(bjdtdb, subexp_bjd_tdb.value)

        # trim the edges
        badpix_masks[i,0:6,:] = True
        badpix_masks[i,516:522,:] = True
        badpix_masks[i,:,0:6] = True
        badpix_masks[i,:,516:522] = True
        # all non-OK pixels
        #badidx = numpy.where(dqs[i,:,:]!=0)
        #badpix_masks[i, badidx[0], badidx[1]] = True
        # decoding error
        #badidx = numpy.where(dqs[i,:,:]==1)
        #badpix_masks[i, badidx[0], badidx[1]] = True
        # data missing
        #badidx = numpy.where(dqs[i,:,:]==2)
        #badpix_masks[i, badidx[0], badidx[1]] = True
        # bad detector pixels
        badidx = numpy.where(dqs[i,:,:]==4)
        badpix_masks[i, badidx[0], badidx[1]] = True
        # non-zero bias
        badidx = numpy.where(dqs[i,:,:]==8)
        badpix_masks[i, badidx[0], badidx[1]] = True
        # hot pixels
        badidx = numpy.where(dqs[i,:,:]==16)
        badpix_masks[i, badidx[0], badidx[1]] = True
        # Unstable response
        badidx = numpy.where(dqs[i,:,:]==32)
        badpix_masks[i,badidx[0],badidx[1]] = True
        # warm pixel
        #badidx = numpy.where(dqs[i,:,:]==64)
        #badpix_masks[i, badidx[0], badidx[1]] = True
        # bad reference
        #badidx = numpy.where(dqs[i,:,:]==128)
        #badpix_masks[i, badidx[0], badidx[1]] = True
        # saturation
        #badidx = numpy.where(dqs[i,:,:]==256)
        #badpix_masks[i, badidx[0], badidx[1]] = True
        # bad flat (blobs)
        badidx = numpy.where(dqs[i,:,:]==512)
        badpix_masks[i, badidx[0], badidx[1]] = True
        # signal in zero read
        #badidx = numpy.where(dqs[i,:,:]==2048)
        #badpix_masks[i, badidx[0], badidx[1]] = True
        # CR by MD
        #badidx = numpy.where(dqs[i,:,:]==4096)
        #badpix_masks[i, badidx[0], badidx[1]] = True
        # cosmic ray
        #badidx = numpy.where(dqs[i,:,:]==8192)
        #badpix_masks[i, badidx[0], badidx[1]] = True
        # ghost
        badidx = numpy.where(dqs[i,:,:]==16384)
        badpix_masks[i, badidx[0], badidx[1]] = True
    for i in range(nsubexp-1):
        subexps[i,:,:] = subexps[i,:,:]-subexps[(i+1),:,:]
        texp[i] = texp[i] - texp[i+1]

    return subexps[0:14,:,:], bjdtdb[0:14], badpix_masks[0:14,:,:], dqs[0:14,:,:], scandir, texp[0:14], subexp_shifts[0:14]


def CreateSubExps_old(file, WaveSol):
	hdu = fits.open(file)
	bjdtdb = []
	texp = []

	scanangle = hdu[0].header['SCAN_ANG']
	scandir = 0
	if scanangle > 180.: scandir = 1

	# setup for individual image
	imageidxs = numpy.arange(1, 71, 5)  # doesn't include 71
	nsubexp = imageidxs.shape[0]
	subexps = numpy.zeros((nsubexp, 522, 522))
	dqs = numpy.zeros((nsubexp, 522, 522))
	badpix_masks = numpy.full((nsubexp, 522, 522), False)

	# sub exposure shifts #
	side_angle = hdu[0].header['ANG_SIDE']
	x = numpy.arange(nsubexp)
	b = 0.0
	m = 5.85532565271 * numpy.cos(side_angle * math.pi / 180.)
	subexp_shifts = m * x + b

	# subtracting subsequent reads from other
	for i in range(nsubexp):
		preflat = hdu[imageidxs[i]].data
		subexps[i, :, :] = preflat

		dqs[i,:,:] = hdu[imageidxs[i]+2].data

		texp = numpy.append(texp, hdu[imageidxs[i]].header['SAMPTIME'])

		readtime = hdu[imageidxs[i]].header['ROUTTIME']
		if imageidxs[i] < 66:
			subexptime = (hdu[imageidxs[i]].header['DELTATIM']) / (24. * 3600.)
		else:  # turns out the first multiaccum runs slightly long. Samptime is real time, deltatim is what HST thinks
			subexptime = (hdu[imageidxs[i]].header['SAMPTIME']) / (24. * 3600.)
		subexp_mjd = time.Time((readtime - subexptime / 2.), format='mjd', scale='utc')
		subexp_jd = time.Time(subexp_mjd.jd, format='jd', scale='utc').value
		# NEED TO CORRECT THIS TO BJD!
		bjdtdb = numpy.append(bjdtdb, subexp_jd)

		# trim the edges
		badpix_masks[i,0:6,:] = True
		badpix_masks[i,516:522,:] = True
		badpix_masks[i,:,0:6] = True
		badpix_masks[i,:,516:522] = True
		# bad detector pixels
		badidx = numpy.where(dqs[i,:,:]==4)
		badpix_masks[i, badidx[0], badidx[1]] = True
		# Unstable response
		badidx = numpy.where(dqs[i,:,:]==32)
		badpix_masks[i,badidx[0],badidx[1]] = True
		# bad flat (blobs)
		badidx = numpy.where(dqs[i,:,:]==512)
		badpix_masks[i, badidx[0], badidx[1]] = True
	for i in range(nsubexp-1):
		subexps[i,:,:] = subexps[i,:,:]-subexps[(i+1),:,:]
		texp[i] = texp[i] - texp[i+1]

	return subexps, bjdtdb, badpix_masks, dqs, scandir, texp, subexp_shifts

def BkgdSub(subexps, badpix_masks, scandir):
	nsubexp = subexps.shape[0]
	bkgdsubbed = numpy.copy(subexps)
	bkgd = []
	bkgd_err = []

	for isub in range(nsubexp):
		subexp = subexps[isub,:,:]
		badpix_mask = badpix_masks[isub,:,:]

	############## background estimation and subtraction ######################

		bkgd_mask = numpy.copy(badpix_mask)

		# locate and mask the star
		vertslice = numpy.sum(subexp, axis=1)
		yloc = int(numpy.rint(numpy.argmax(vertslice)))
		horzslice = numpy.sum(subexp, axis=0)
		xpixels = numpy.arange(0,522)
		xloc = int(numpy.rint(numpy.average(xpixels, weights=horzslice)))
		bkgd_mask[yloc-50:yloc+40,xloc-90:xloc+90] = True

		# locate and mask nearby stars
		# mask KELT-1 and see what remains on the right side
		otherstarid = numpy.ma.array(subexp, mask=bkgd_mask)
		vertslice = numpy.sum(otherstarid[:,230:], axis=1)
		yloc = int(numpy.rint(numpy.argmax(vertslice)))
		bkgd_mask[yloc-5:yloc+5,175:] = True

		# for downward scans, also look at the left side
		if scandir == 0:
			vertslice = numpy.sum(otherstarid[:,0:20], axis=1)
			yloc = int(numpy.rint(numpy.argmax(vertslice)))
			bkgd_mask[yloc-5:yloc+5,0:50] = True

		std = numpy.std(subexp[~bkgd_mask]-numpy.median(subexp[~bkgd_mask]))
		error_of_mean = std/numpy.sqrt(subexp[~bkgd_mask].size)

		#plt.hist(subexp[~bkgd_mask]-numpy.median(subexp[~bkgd_mask]), bins=100)
		#plt.xlim(-100,100)
		#plt.show()
		#plt.clf()

		bkgd = numpy.append(bkgd,numpy.median(subexp[~bkgd_mask]))
		bkgd_err = numpy.append(bkgd_err,error_of_mean)
		bkgdsubbed[isub,:,:] -= numpy.median(subexp[~bkgd_mask]) # subtract the background from the subexp
	return bkgdsubbed, bkgd, bkgd_err

def WavelengthSolution(directname):
	direct = fits.open(directname)
	mean, median, std = sigma_clipped_stats(direct['SCI'].data, sigma=3.0, maxiters=5)
	daofind = DAOStarFinder(fwhm=2.0, threshold=20. * std)
	sources = daofind(direct['SCI'].data)
	try:
		xcenSubArrTarg = sources['xcentroid'][0]  # + 1 # DS9 offset
	except:
		# visit3 orbit1 fails for some reason. Do it manually
		direct_sub = direct['SCI'].data[145:165, 25:40]
		daofind = DAOStarFinder(fwhm=1.0, threshold=4. * std, sharplo=0.1)
		sources = daofind(direct_sub)
		sources['xcentroid'][0] += 25
		sources['ycentroid'][0] += 145
		xcenSubArrTarg = sources['xcentroid'][0]  # + 1 # DS9 offset

	#ycenSubArrTarg = sources['ycentroid'][0]  # + 1 # DS9 offset
	xcenAbs = sources['xcentroid'][0] + 374.  # plus subarray center loc
	ycenAbs = sources['ycentroid'][0] + 374.  # plus subarray center loc
	dldp0 = 0.997*8.95431e+03 + 0.90*9.35925e-02 * xcenAbs
	dldp1 = 1.029*4.51423e+01 + 3.17239e-04 * xcenAbs + 2.17055e-03 * ycenAbs - 7.42504e-07 * xcenAbs ** 2. + 3.48639e-07 * xcenAbs * ycenAbs + 3.09213e-07 * ycenAbs ** 2.
	#xpixel = numpy.arange(522.0)
	#waveTarg = dldp0 + dldp1 * (xpixel - xcenSubArrTarg)
	#WaveInterpFunc = interpolate.interp1d(waveTarg, xpixel, kind='linear')
	return [dldp0, dldp1, xcenSubArrTarg]

def GetExtractRange(wavestart, wavestop, directname, imageoffset):
	wavesol = WavelengthSolution(directname)

	xpixel = numpy.arange(522)
	waveTarg = wavesol[0] + wavesol[1] * (xpixel - wavesol[2])
	waveinterpTarg = interpolate.interp1d(waveTarg, xpixel, kind='linear')
	ExtStart = waveinterpTarg(wavestart) - imageoffset
	ExtStop = waveinterpTarg(wavestop) - imageoffset
	return [ExtStart, ExtStop]

def FlatField(flatimagepath, subexps, wavesol, subexp_shifts):
	nsubexp = subexps.shape[0]
	flatted = numpy.copy(subexps)

	ffcube = fits.open(flatimagepath)
	xpixels = numpy.arange(522)
	lambdamax = ffcube[0].header['WMAX']
	lambdamin = ffcube[0].header['WMIN']
	offsetx = 246
	offsety = 246
	for i in range(nsubexp):
		wavelength = wavesol[0] + wavesol[1] * ((xpixels-subexp_shifts[i]) - wavesol[2])
		x = (wavelength - lambdamin) / (lambdamax - lambdamin)
		for col in xpixels:
			a0 = ffcube[0].data[offsety:offsety+522,col+offsetx]
			a1 = ffcube[1].data[offsety:offsety+522,col+offsetx]
			a2 = ffcube[2].data[offsety:offsety+522,col+offsetx]
			a3 = ffcube[3].data[offsety:offsety+522,col+offsetx]
			flatfield = a0 + a1*x[col] + a2*x[col]**2. + a3*x[col]**3.
			badvals = numpy.where(flatfield <= 0)
			flatfield[badvals] = 0.1
			flatted[i,:,col] = subexps[i,:,col] / flatfield
	return flatted

