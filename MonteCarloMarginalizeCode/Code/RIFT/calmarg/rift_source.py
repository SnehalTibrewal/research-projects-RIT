"""
Source functions to match how RIFT generates waveforms
"""

import numpy as np
import lal
import lalsimulation as lalsim
import RIFT.lalsimutils as lalsimutils
import matplotlib
from bilby.core import utils
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt


def RIFT_lal_binary_black_hole(
        frequency_array, mass_1, mass_2, luminosity_distance, spin_1x, spin_1y, spin_1z,
        spin_2x, spin_2y, spin_2z, lambda_1, lambda_2, iota, phase, **kwargs):

    waveform_kwargs = dict(
        waveform_approximant='SEOBNRv4PHM', reference_frequency=15.0,
        minimum_frequency=15.0, maximum_frequency=frequency_array[-1], Lmax=4,
        sampling_frequency=2*frequency_array[-1])
    waveform_kwargs.update(kwargs)
    waveform_approximant = waveform_kwargs['waveform_approximant']
    reference_frequency = waveform_kwargs['reference_frequency']
    minimum_frequency = waveform_kwargs['minimum_frequency']
    maximum_frequency = waveform_kwargs['maximum_frequency']
    sampling_frequency = waveform_kwargs['sampling_frequency']
    Lmax = waveform_kwargs['Lmax']
    waveform_dictionary = waveform_kwargs.get(
        'lal_waveform_dictionary', lal.CreateDict()
    )

    approximant = lalsim.GetApproximantFromString(waveform_approximant)

    P = lalsimutils.ChooseWaveformParams()
    P.m1 = mass_1 * lal.MSUN_SI
    P.m2 = mass_2 * lal.MSUN_SI
    P.s1x = spin_1x; P.s1y = spin_1y; P.s1z = spin_1z
    P.s2x = spin_2x; P.s2y = spin_2y; P.s2z = spin_2z
    P.lambda1 = lambda_1; P.lambda2 = lambda_2
    P.deltaT = 1./(sampling_frequency)

    P.fmin = float(minimum_frequency)
    P.fmax = float(maximum_frequency)
    P.fref = float(reference_frequency)
    P.deltaF=frequency_array[1]-frequency_array[0]
    P.incl = iota
    P.phiref = phase
    P.dist=luminosity_distance*lal.PC_SI*1e6
    P.approx = approximant
    P.taper = lalsim.SIM_INSPIRAL_TAPER_START

    hlmT = lalsimutils.hlmoft(P,Lmax=Lmax)
    h22T = hlmT[(2,2)]
    hT = lal.CreateCOMPLEX16TimeSeries("hoft", h22T.epoch, h22T.f0, h22T.deltaT, h22T.sampleUnits, h22T.data.length)
    hT.data.data = np.zeros(hT.data.length)

    # combine modes
    phase_offset = 0#np.pi/2 # TODO this could be a different value i.e. np.pi/2
    for mode in hlmT:
        hT.data.data += hlmT[mode].data.data * lal.SpinWeightedSphericalHarmonic(
            P.incl,phase_offset - 1.0*P.phiref, -2, int(mode[0]), int(mode[1]))

    tvals = lalsimutils.evaluate_tvals(hT)
    t_max = tvals[np.argmax(np.abs(hT.data.data))]

    # end max is cutting the signal such that it ends 2s after merger
    n_max = np.argmax(np.abs(hT.data.data))

    hp = lal.CreateREAL8TimeSeries("h(t)", h22T.epoch, h22T.f0, h22T.deltaT, h22T.sampleUnits, h22T.data.length)
    hp.data.data = np.real(hT.data.data)
    hc = lal.CreateREAL8TimeSeries("h(t)", h22T.epoch, h22T.f0, h22T.deltaT, h22T.sampleUnits, h22T.data.length)
    hc.data.data = -np.imag(hT.data.data)

    lalsim.SimInspiralREAL8WaveTaper(hp.data, P.taper)
    lalsim.SimInspiralREAL8WaveTaper(hc.data, P.taper)

    h_plus = hp.data.data
    h_plus = np.concatenate([h_plus[n_max:], h_plus[:n_max]])
    h_cross = hc.data.data
    h_cross = np.concatenate([h_cross[n_max:], h_cross[:n_max]])

    hf_p, freqs = utils.nfft(h_plus, sampling_frequency)
    hf_c, freqs = utils.nfft(h_cross, sampling_frequency)

    #hf_p *= np.exp(2j*np.pi * freqs * tvals[0])
    #hf_c *= np.exp(2j*np.pi * freqs * tvals[0])

    return dict(plus=hf_p, cross=hf_c)


if __name__ == '__main__':
    # For testing purposes we evaluate one waveform and plot it here
    frequency_array = np.arange(15, 1024., 1./8)
    waveform_kwargs = {'Lmax':4, 'maximum_frequency':1024, 'minimum_frequency':10}
    waveform_polarizations = RIFT_lal_binary_black_hole(
        frequency_array, 60., 55., 400., 0.0, 0.0, 0.1,
        0.0, 0.0, 0.1, iota=np.pi/4, phase=np.pi/2, **waveform_kwargs)

    hf_p = waveform_polarizations['plus']
    hf_c = waveform_polarizations['cross']

    waveform_kwargs = {'Lmax':4, 'maximum_frequency':4096, 'minimum_frequency':10}
    waveform_polarizations = RIFT_lal_binary_black_hole(
        frequency_array, 60., 55., 400., 0.0, 0.0, 0.1,
        0.0, 0.0, 0.1, iota=np.pi/4, phase=np.pi/2, **waveform_kwargs)

    hf_p2 = waveform_polarizations['plus'][:int(len(waveform_polarizations['plus'])/4)]
    hf_c2 = waveform_polarizations['cross'][:int(len(waveform_polarizations['plus'])/4)]

    plt.axvline(15, color='k')

    plt.loglog(np.linspace(0,frequency_array[-1],len(hf_p)), np.abs(hf_p), color='C0')
    plt.loglog(np.linspace(0,frequency_array[-1],len(hf_p)), np.abs(hf_c), color='C0', linestyle='--')
    plt.loglog(np.linspace(0,frequency_array[-1],len(hf_p2)), np.abs(hf_p2), color='C1')
    plt.loglog(np.linspace(0,frequency_array[-1],len(hf_p2)), np.abs(hf_c2), color='C1', linestyle='--')
    plt.show()
    plt.clf()

    plt.semilogx(np.linspace(0,frequency_array[-1],len(hf_p)), np.angle(hf_p), color='C0')
    plt.semilogx(np.linspace(0,frequency_array[-1],len(hf_p)), np.angle(hf_c), color='C0', linestyle='--')
    plt.semilogx(np.linspace(0,frequency_array[-1],len(hf_p2)), np.angle(hf_p2), color='C1')
    plt.semilogx(np.linspace(0,frequency_array[-1],len(hf_p2)), np.angle(hf_c2), color='C1', linestyle='--')
    plt.show()
    plt.clf()
