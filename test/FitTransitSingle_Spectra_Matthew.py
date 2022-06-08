import numpy, math, sys, progressbar, emcee, corner
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from matplotlib import gridspec
from scipy.stats import anderson

sys.path.insert(0, '../')
import Transit_HST_Funcs as hst
import PyFAST as RVSEDFuncs

# Set up Thomas' data structure
dataV1 = hst.MedianAndClean('./Transit_Spectra_NoFirst_Visit3.npz', visit='v3', FitType='Spectra')
dataV1['error'] *= 1.0

# Load in my data
mydata = numpy.load('/home/mmmurphy/data/hd219666b/reduced_data/allorbits_rawSpectralLightcurves.npz')
times = numpy.concatenate((mydata['orbit2_times'], mydata['orbit3_times'], mydata['orbit4_times']))
rawfluxes = numpy.concatenate((mydata['orbit2_flux'], mydata['orbit3_flux'], mydata['orbit4_flux']))
# the arrays read in as shape (Npoints, Nwaves) but code needs (Nwaves, Npoints), so swap the axes ...
rawfluxes = numpy.swapaxes(rawfluxes, 0, 1)
rawerrors = numpy.concatenate((mydata['orbit2_error'], mydata['orbit3_error'], mydata['orbit4_error']))
rawerrors = numpy.swapaxes(rawerrors, 0, 1)
scandirs = numpy.concatenate((mydata['orbit2_scandirs'], mydata['orbit3_scandirs'], mydata['orbit4_scandirs']))

fluxesM = numpy.copy(rawfluxes)
errorsM = numpy.copy(rawerrors)
Nbins = rawfluxes.shape[0]
for wav in range(Nbins):
    sd1_idxs = numpy.where(scandirs == 1.)[0]
    sd0_idxs = numpy.where(scandirs == 0.)[0]
    # if normalizing by median flux of whole visit
    normval1 = numpy.median(fluxesM[wav, sd1_idxs])
    normval0 = numpy.median(fluxesM[wav, sd0_idxs])
    
    fluxesM[wav, sd1_idxs] /= normval1
    errorsM[wav, sd1_idxs] /= normval1
    fluxesM[wav, sd0_idxs] /= normval0
    errorsM[wav, sd0_idxs] /= normval0
    
errorsM *= 10.0  
# setting his fluxes and errors equal to mind
dataV1['flux'] = numpy.copy(fluxesM)
dataV1['error'] = numpy.copy(errorsM)





FitParams = {}
FitParams['Section'] = 'Transit'
FitParams['FitType'] = 'Spectra'
FitParams['RampType'] = 'PolyRamp'
FitParams['ModelType'] = 'Param'
FitParams['SpecNo'] = 1
DoPlot = True
MCMC = True
nburn = 1000
nprod = 4000
nupdates = 1000

FitOutput = {}
for i in range(15):
    print(i)
    FitParams['SpecNo'] = i
    
    priors = {}
    priors_to_apply = {}
    FuncNames = {}
    priors['rprs_'+str(FitParams['SpecNo']).zfill(2)] = [0.04254, 0.007]
    priors['u1'] = [0.058096833, 0.01/5]
    priors['u2'] = [0.34962683, 0.05/5]
    priors_to_apply['gaussian'] = ['u1', 'u2']
    
    # Ramp model priors
    priors['Norm_V_v1'] = [1.0000+0.0011, 0.01]
    priors['Slope_V_v1'] = [-0.0065, 0.01]
    priors['Amp_O_v1'] = [0.0012, 0.01]
    priors['Tau_O_v1'] = [-0.005,0.01]
    priors['Amp_O2_v1'] = [0.0022, 0.01]
    priors['Tau_O2_v1'] = [-0.007, 0.01]
    #priors['Amp_O1_v1'] = [0.016, 0.001]
    #priors['Tau_O1_v1'] = [-0.0055, 0.001]
    #priors['Off_O1_v1'] = [0.0, 0.001]
    priors_to_apply['gaussian'].extend(['Norm_V_v1', 'Slope_V_v1', 'Amp_O_v1', 'Tau_O_v1', 'Amp_O2_v1', 'Tau_O2_v1'])
    
    FuncNames['LnLike'] = hst.ParamLnLikelihoodSingle
    FuncNames['MCMC'] = hst.ParamMCMCFuncSingle
    FuncNames['Detrend'] = hst.ParamModel
    
    labels = list(priors.keys())
    pars = {name: priors[name][0] for name in labels}
    scales = {name: priors[name][1] for name in labels}
    
    concordance = {}
    parsarray = numpy.zeros(len(labels))
    scalesarray = numpy.zeros(len(labels))
    for i in range(len(labels)):
        concordance[labels[i]] = i
        parsarray[i] = pars[labels[i]]
        scalesarray[i] = scales[labels[i]]
    
#     PrevRun = numpy.load('FitTransitSingle_Spectra'+str(FitParams['SpecNo']).zfill(2)+'.npz', allow_pickle=True)
#     PrevBestfit = PrevRun['bestfit'].item()
#     for name in labels:
#        try:
#            pars[name] = PrevBestfit[name][0]
#        except KeyError:
#            pass
    
    InitMin = parsarray
    InitDic = {}
    for i in range(len(labels)): InitDic[labels[i]] = InitMin[concordance[labels[i]]]
    InitMinLnp = FuncNames['LnLike'](InitDic, dataV1, priors, priors_to_apply, FitParams)
    
    #print("Initial best fit:")
    #print(InitMin)
    print("Initial best lnp:", numpy.round(InitMinLnp,1))
    
    if MCMC:
        nwalkers = 2*len(parsarray)
        ndim = len(InitMin)
    
        MCMCscales = scalesarray / 100.
    
        p0 = emcee.utils.sample_ball(InitMin, MCMCscales, nwalkers)
    
        sampler = emcee.EnsembleSampler(nwalkers, ndim, FuncNames['MCMC'], args=[dataV1, pars, priors, priors_to_apply, labels, concordance, FitParams], live_dangerously=True)
        sampler = hst.RunMCMC(p0, sampler, nburn, nprod, nupdates, concordance)
    
        GR = hst.GelmanRubin(sampler.chain)
        print("Gelman-Rubin array:")
        print(GR)
    
        MCMCresults = sampler.flatchain
    
        rprs = MCMCresults[:, concordance['rprs_'+str(FitParams['SpecNo']).zfill(2)]]
        depth = rprs**2.
        MCMCresults = numpy.column_stack((MCMCresults, depth))
        labels = numpy.append(labels, ['depth_'+str(FitParams['SpecNo']).zfill(2)])
        concordance['depth_'+str(FitParams['SpecNo']).zfill(2)] = MCMCresults.shape[1] - 1
    
        outputdict = {}
        for param in labels:
            idx = concordance[param]
            medval = numpy.median(MCMCresults[:, idx])
            minus = medval - numpy.percentile(MCMCresults[:, idx], 16)
            plus = numpy.percentile(MCMCresults[:, idx], 84) - medval
            outputdict[param] = [medval, minus, plus]
    
        bestfit = {}
        for i in range(len(labels)): bestfit[labels[i]] = outputdict[labels[i]][0]
    
        numpy.savez('./FitTransitSingle_MatthewSpectra'+str(FitParams['SpecNo']).zfill(2)+'.npz', bestfit=outputdict, labels=labels, chain=sampler.chain, flatchain=MCMCresults, lnprob=sampler.lnprobability, flatlnprob=sampler.flatlnprobability,
                    priors=priors, priors_to_apply=priors_to_apply, concordance=concordance, fitparams=FitParams)
    
    else:
        bestfit = {}
        for i in range(len(labels)): bestfit[labels[i]] = InitMin[concordance[labels[i]]]
    
    if DoPlot:
    
        printout = ['rprs_'+str(FitParams['SpecNo']).zfill(2), 'u1', 'u2']
        if MCMC:
            for param in outputdict: print(param, outputdict[param])
            '''
            nsteps = MCMCresults.shape[0]
            cornermcmc = numpy.zeros((nsteps, len(printout)))
            cornervals = numpy.zeros(len(printout))
            nthin = 1
            for i in range(len(printout)):
                cornermcmc[:, i] = MCMCresults[:, concordance[printout[i]]]
                cornervals[i] = outputdict[printout[i]][0]
            figure = corner.corner(cornermcmc[::nthin, :], labels=printout, truths=cornervals, quantiles=[0.16, 0.5, 0.84], verbose=False)
            figure.savefig('CornerPlot.png', dpi=200)
            plt.close()
            '''
        else:
            for param in printout: print(param, bestfit[param])
    
        print(' ')
    
        fullmodel, detrendmodel, lcmodel = FuncNames['Detrend'](bestfit, dataV1, 'v1', FitParams, SpecNo=FitParams['SpecNo'])
        residV1 = dataV1['flux'][FitParams['SpecNo']]-fullmodel
        stdresid = numpy.round(numpy.std(residV1)*1e6,2)
        mederr = numpy.round(numpy.median(dataV1['error'][FitParams['SpecNo']])*1e6,2)
        print('V1 Resid. RMS / Median error:', stdresid, "/", mederr, "ppm")
        print('V1 Anderson statistic:', anderson(residV1)[0], anderson(residV1)[1][0])
    
        FinalLnp = FuncNames['LnLike'](bestfit, dataV1, priors, priors_to_apply, FitParams)
        print("Final Lnp:", numpy.round(FinalLnp, 1))
    
        hst.SingleVisitPlot(bestfit, dataV1, FitParams, 'v1')
        '''
        if MCMC:
            MCMCchains = sampler.chain
            Lnp = sampler.lnprobability
    
            trim = 0
            clip = 0
            if trim > 0 or clip > 0:
                MCMCchains = MCMCchains[:, trim:-1 * clip, :]
                Lnp = Lnp[:, trim:-1 * clip]
                npars = MCMCchains.shape[2]
                nstepfinal = MCMCchains.shape[1] * MCMCchains.shape[0]
                MCMCresults = numpy.zeros((nstepfinal, npars))
                for i in range(npars):
                    MCMCresults[:, i] = MCMCchains[:, :, i].flatten()
    
            #if ModelType=='GP': toshow = ['tc', 'secdepth', 'GPGam', 'GPOmg']
            toshow = ['rprs_'+str(FitParams['SpecNo']).zfill(2), 'u1', 'u2']
            #toshow = []
    
            for param in toshow:
                par = concordance[param]
                print(param)
                plt.plot(MCMCchains[0, :, par])
                plt.plot(MCMCchains[1, :, par])
                plt.plot(MCMCchains[2, :, par])
                plt.plot(MCMCchains[3, :, par])
                plt.plot(MCMCchains[4, :, par])
                plt.plot(MCMCchains[5, :, par])
                plt.show()
                plt.clf()
    
            plt.plot(Lnp[0, :])
            plt.plot(Lnp[1, :])
            plt.plot(Lnp[2, :])
            plt.plot(Lnp[3, :])
            plt.plot(Lnp[4, :])
            plt.plot(Lnp[5, :])
            plt.show()
            plt.clf()
        '''
        FitOutput['rprs_'+str(FitParams['SpecNo']).zfill(2)] = outputdict['rprs_'+str(FitParams['SpecNo']).zfill(2)]
        FitOutput['depth_' + str(FitParams['SpecNo']).zfill(2)] = outputdict['depth_' + str(FitParams['SpecNo']).zfill(2)]

numpy.savez('./FitTransitSingle_MatthewSpectra.npz', FitOutput=FitOutput)
