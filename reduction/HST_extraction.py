import sys
#sys.path.append('~/./anaconda/lib/python3.8/site-packages')
#print(sys.path)
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

def WavelengthSolution(directname):
    ## This function computes the wavelength solution for an HST WFC3 direct image
    ## Input: the file path to the direct image
    ## Written by Thomas Beatty
    
	direct = fits.open(directname) # the direct image fits data
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

def CreateSubExps(file, WaveSol):
    ## This function reads in an HST 'ima' spectral image and pulls out relevant data for the
    ##   individual sub-exposures that make up the image.
    ## Inputs: 1- file path to the grism image, 2- wavelength solution for that image as obtained
    ##                from the corresponding orbit's direct image
    ## outputs:
    ## Written by Thomas Beatty
    
    hdu = fits.open(file) # fits data for this spectral image
    bjdtdb = []  # empty array that will contain the times of each sub-exposure [bjd tdb]
    texp = []    # empty array that will contain XXX

    ## Determining the scan direction of each sub-exposure
    #   scan direction is determined from the sub-exposure's scan angle
    scanangle = hdu[0].header['SCAN_ANG']
    scandir = 0
    if scanangle > 180.: scandir = 1

    ## setup for individual image
    imageidxs = numpy.arange(1, 76, 5)  # doesn't include 71
    nsubexp = imageidxs.shape[0]
    subexps = numpy.zeros((nsubexp, 522, 522))
    dqs = numpy.zeros((nsubexp, 522, 522))
    badpix_masks = numpy.full((nsubexp, 522, 522), False)

    # sub exposure shifts #
    side_angle = hdu[0].header['ANG_SIDE']
    scan_rate = hdu[0].header['SCAN_RAT']   # [arcsec/s]
    #scan_length_asec = hdu[0].header['SCAN_LEN'] # length of scan in [arcsec]
    #print(16*scan_rate, scan_length_asec)
    platescale = 0.13 # [arcsec/pixel] for WFC3
    scan_length_pix = (16 * scan_rate) / platescale         
    
    x = numpy.arange(nsubexp)
    b = 0.0
    #m = 5.85532565271 * numpy.cos(side_angle * math.pi / 180.)
    m = scan_length_pix * numpy.cos(side_angle * math.pi / 180.)
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

def BkgdSub(subexps, badpix_masks, scandir):
    ## Function to perform background subtraction on an HST/WFC3 2D spectrum image, which has 
    ##   already been split into its constituent sub-exposures
    ## Inputs: 1- the sub-exposure images (N_image long array of 2D spectra),
    ##         2 - the sub-exposure bad pixel masks (N_image long array of 2D masks),
    ##         3 - scan directions of each sub-exposure (N_image long array of values)
    
	nsubexp = subexps.shape[0]       # The number of sub-exposures
	bkgdsubbed = numpy.copy(subexps) # Background subtracted images (starts as copy, will be altered)
	bkgd = []       # Array containing the median background level of each sub-exposure
	bkgd_err = []   #  Uncertainty on the above
    
    ### Will loop through each sub-exposure
	for isub in range(nsubexp):
        ## Pull out this sub-exposure's image and bad pixel mask
		subexp = subexps[isub,:,:]
		badpix_mask = badpix_masks[isub,:,:]

        #### background estimation and subtraction ###########
		bkgd_mask = numpy.copy(badpix_mask) # mask of pixels to not get background level of

		## locate and mask the star using the centre of light
		vertslice = numpy.sum(subexp, axis=1)
		yloc = int(numpy.rint(numpy.argmax(vertslice)))
		horzslice = numpy.sum(subexp, axis=0)
		xpixels = numpy.arange(0,522)
		xloc = int(numpy.rint(numpy.average(xpixels, weights=horzslice)))
		bkgd_mask[yloc-50:yloc+40,xloc-90:xloc+90] = True  # Add star's position + buffer to the non-background mask

		## locate and mask nearby stars
		otherstarid = numpy.ma.array(subexp, mask=bkgd_mask)
		vertslice = numpy.sum(otherstarid[:,230:], axis=1)
		yloc = int(numpy.rint(numpy.argmax(vertslice)))
		bkgd_mask[yloc-5:yloc+5,175:] = True

		## for downward scans, also look at the left side for masking
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

import numpy as np
def cutout2Dspectrum(subexps, extractrange, height, sidebuffer, returnCoords=False, manualOverride=None):
    ## Function that takes full spectrum images and extracts only the part around the 2D target spectrum
    
    Nsubexp = subexps.shape[0]  # The number of sub-exposures
    dpix_left = extractrange[0] # left-most dispersion pixel in the wavelength range
    dpix_right = extractrange[1] # rightmost dispersion pixel in the wavelength range
    middle_dpixel = int(np.rint(np.median(extractrange)))
    
    box_halfheight = height
    box_widthbuffer = sidebuffer
    box_halfwidth = int(0.5*(dpix_right-dpix_left) + sidebuffer)
    
    twoDspectra = np.zeros((Nsubexp, int(2*box_halfheight), int(2*box_halfwidth)+1))
    boxCoords = np.zeros((Nsubexp, 4), dtype=int)  # returns [bottom, top, left, right] for each subexposure
    #### Looping through the sub-exposures
    for i in range(Nsubexp):
        subexp_fullimg = subexps[i]  # Get this sub-exposure's image
        
        # Determining the 'middle' of the spectrum along the scan (y) axis by weighted average along y at middle position in (x)
        
        middle_spixel_float = np.average(np.arange(subexp_fullimg.shape[0]), weights=subexp_fullimg[:, middle_dpixel])
        if manualOverride == 'o2i13':
            print('override engaged')
            middle_spixel_float = np.average(np.arange(subexp_fullimg.shape[0])[100:], weights=subexp_fullimg[100:, middle_dpixel])
        middle_spixel = int(np.rint(middle_spixel_float))
   
        # Defining the cut-out box
        bottom, top = int(middle_spixel - box_halfheight), int(middle_spixel + box_halfheight)
        left, right = int(dpix_left - box_widthbuffer), int(dpix_right + box_widthbuffer)
        
        # Cutting out the 2D spectrum
        subexp_2Dspec = subexp_fullimg[bottom:top, left:right]
        twoDspectra[i] = subexp_2Dspec
        boxCoords[i] = np.array([bottom, top, left, right])
        
    if returnCoords:
        return twoDspectra, boxCoords
    else:
        return twoDspectra
 
def CorrectBadPixels(subexps, DQimage, Niter=5, DQflags=[1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384]):
    corrected_subexps = np.copy(subexps)
    Nbad = np.zeros(subexps.shape[0])
    for i in range(subexps.shape[0]):
        correctedspec = np.copy(subexps[i])
        Nrows = correctedspec.shape[0]
        DQsubexp = DQimage[i]
        for iteration in range(Niter):
            for scanrow in range(Nrows):
                bad_idxs = np.where(DQsubexp[scanrow, :] != 0)[0]
                Nbad[i] = len(bad_idxs)
                for badpixel in bad_idxs:
                    if int(DQsubexp[scanrow, badpixel]) not in DQflags:
                        if int(DQsubexp[scanrow, badpixel]) == 48:
                            # There's a lot of non-standard flags of DQ = 48
                            # which aren't causing issues, so ignore them
                            pass
                        else:
                            try:
                                # Set pixel to the median of pixels above and below it
                                correctedspec[scanrow, badpixel] = np.median((correctedspec[scanrow, badpixel-1], 
                                                                        correctedspec[scanrow, badpixel+1]))
                            except IndexError:
                                # it'll throw an index error if on the edge of the cut-out image
                                # in which case, it's a pixel we don't care about
                                pass
                    else:
                        # If we get a standard flag, ignore it ...
                        pass
            corrected_subexps[i] = correctedspec
    Nbad_average = np.mean(Nbad)
    return corrected_subexps, Nbad_average
        
def FitSpectralTrace(subexps):
    
    scanpixels = np.arange(subexps.shape[1])
    disppixels = np.arange(subexps.shape[2])
    traces = np.zeros((subexps.shape[0], len(disppixels)))
    
    for i in range(subexps.shape[0]):
        spec2D = subexps[i]
        ## Determining the 'centre' of the spectrum at each dispersion column
        ##    via flux-weighted mean of that column
        center_spix_locs = np.zeros(len(disppixels))
        for dpixel in disppixels:
            center_spix_locs[dpixel] = np.average(scanpixels, weights=abs(spec2D[:,dpixel]))
        ## Fitting these flux-weighted centres with a polynomial
        p2, p1, p0 = np.polyfit(disppixels, center_spix_locs, deg=2)
        trace = p2*(disppixels**2) + p1*disppixels + p0
        traces[i] = trace
    
    return disppixels, traces        

def Extract1D(subexps, traces, extractionheight=15):
    
    disppixels = np.arange(subexps.shape[2])
    spectra1D = np.zeros((subexps.shape[0], len(disppixels)))
    spectrauncs1D = np.copy(spectra1D)
    for i in range(subexps.shape[0]):
        spectrace = traces[i]
        spec2D = subexps[i]
        spec2D_photnoise = np.sqrt(spec2D)
        # negative pixels will return nan, so need to correct them
        spec2D_photnoise[np.where(np.isnan(spec2D_photnoise))] = 0.
                
        box_bottoms = spectrace - extractionheight
        box_tops = spectrace + extractionheight
        subexp_spec1D = np.zeros(len(disppixels))
        subexp_spec1D_uncs = np.copy(subexp_spec1D)
        
        for j, dpixel in enumerate(disppixels):
            box_bottom = int(np.rint(box_bottoms[j]))
            if box_bottom < 0:
                # sometimes the rounding makes the index -1
                # in which case the sum will not work properly
                box_bottom = 0
            box_top = int(np.rint(box_tops[j]))
            
            box = spec2D[box_bottom:box_top, j]
            box_uncs = spec2D_photnoise[box_bottom:box_top, j]
            
            subexp_spec1D[j] = np.sum(box)
            box_sqr = box**2
            sum_sqr_box = np.sum(box_sqr)
            subexp_spec1D_uncs[j] = sum_sqr_box
            
        spectra1D[i] = subexp_spec1D
        spectrauncs1D[i] = subexp_spec1D_uncs
        
    return spectra1D, spectrauncs1D

def shift(spectrum, xshift=0, yzerolevel=0.):
    
    shiftedspectrum = np.zeros(len(spectrum))
    # finding the array indices where the 'signal' is
    nonzero_idxs = np.where(spectrum > yzerolevel)[0]
    # Doing the shifting, by index
    for idx in nonzero_idxs:
        shiftedspectrum[idx+xshift] = spectrum[idx]
    
#     # Now to delete and pad the pixels at the bounds
#     if (shift > 0):
#         # This is a shift to the right
#         # number of pixels to delete and pad = the shift value
#         Ntodo = xshift
        
    return shiftedspectrum

### A chi-2 comparison function to compare the true signal and a re-shifted signal
def compare(signal1, signal2):
    # add a vertical offset since having 0 messes it up
    buffsignal1 = signal1 + 1
    buffsignal2 = signal2 + 1
    
    differences = (buffsignal1 - buffsignal2)**2
    vals = differences / buffsignal1
    chi2 = np.sum(vals)
    return chi2
    
def crosscorrelate(truesignal, shiftedsignal, shiftguess=0, guesslimits=20, yzerolevel=0.):
    shift_values = np.arange(shiftguess-guesslimits, shiftguess+guesslimits, 1, dtype=int)
    chi2_values = np.ones(len(shift_values))*999.
    
    for i, x_shift in enumerate(shift_values):
        shifted_signal2 = shift(shiftedsignal, -x_shift, yzerolevel)
        chi2_values[i] = compare(truesignal, shifted_signal2)
    
    min_chi2_idx = np.where(chi2_values == min(chi2_values))[0]
    trueshift_guess = shift_values[min_chi2_idx]
    
    return trueshift_guess
    