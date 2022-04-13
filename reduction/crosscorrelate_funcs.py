import numpy as np

def shift(spectrum, xshift=0, yzerolevel=0.):
#     if type(xshift) != int:
#         print('Shift value must be an integer')
#         return nan

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

def crosscorrelate(signal1, signal2, shiftguess=0, guesslimits=20):
    shift_values = np.arange(shiftguess-guesslimits, shiftguess+guesslimits, 1, dtype=int)
    chi2_values = np.ones(len(shift_values))*999.

    for i, x_shift in enumerate(shift_values):
        shifted_signal2 = shift(signal2, -x_shift)
        chi2_values[i] = compare(signal1, shifted_signal2)

    min_chi2_idx = np.where(chi2_values == min(chi2_values))[0]
    trueshift_guess = shift_values[min_chi2_idx]

    return trueshift_guess
