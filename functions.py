import numpy as np

try:
    from scipy import signal
except ModuleNotFoundError:
    class _Signal:
        @staticmethod
        def boxcar(width):
            return np.ones(width)

    signal = _Signal()

if not hasattr(signal, 'boxcar') and hasattr(signal, 'windows'):
    signal.boxcar = signal.windows.boxcar
if not hasattr(signal, 'parzen') and hasattr(signal, 'windows'):
    signal.parzen = signal.windows.parzen
if not hasattr(signal, 'gaussian') and hasattr(signal, 'windows'):
    signal.gaussian = signal.windows.gaussian
if not hasattr(signal, 'flattop') and hasattr(signal, 'windows'):
    signal.flattop = signal.windows.flattop
if not hasattr(signal, 'hann') and hasattr(signal, 'windows'):
    signal.hann = signal.windows.hann


def smooth1d(f, N, win='boxcar', cutoff='power', oddflag=1, edgeflag=1):
    """
    Smooth a 1d array with a half-power (-3dB) or half-amplitude (-6dB) point correspnding to N data points 
    using the specified window.      
 
    Parameters
    ----------
    f : numeric
        So far I have only used a 1d Numpy.array.
    N : numeric
        Number of points for the half-power point of the smoothing window 
        (e.g., with N=26, the half-power point is at 26 points)
    win : string
        can be 'boxcar' (rectangle),'parzen','gauss', 'flattop', or 'hann' (Hanning squared cosine)
        default is 'boxcar'
    cutoff : string
        can be 'power' or 'amplitude'
        default is 'power'
    oddflag : numeric
        Can be 0 or 1; when set to 1 (default), the number of points in the smoothing window
        is forced to be odd (giving priority to making sure the filter does not introduce a phase shift)
        When set to 0, the number of points in the smoothing window is chosen so that the half-power
        point is as close as possible to the one specified by N (giving priority to making sure the
        filter has the specified half-power point)
    edgeflag : numeric
        Can be 0 or 1; when set to 1 (default), regions affected by edge effects are set to nan

    Returns
    -------
    fz : numeric
        smoothed version of f [same size as f].
    N_actual : numeric
        actual number of points for the half-power point of the smoothing window;
        1/N_actual is the actual half-power or half-amplitude frequency

    Reference
    -------
    Harris, F. J. (1978). On the use of windows for harmonic analysis with the discrete Fourier transform.
    Proceedings of the IEEE, 66(1), 51-83.

    Example
    -------
    >>> # Test smooth1d function by comparing spectra of a random 1D array before and after smoothing
    >>> N = 101
    >>> # make a random 1D array
    >>> x = np.random.randn(200000)
    >>> x_smooth, N_actual = functions.smooth1d(x,N,win='parzen')
    >>> # find indices of nan values in x_smooth and remove them from x and x_smooth
    >>> idx = np.where(np.isnan(x_smooth))
    >>> x = np.delete(x,idx)
    >>> x_smooth = np.delete(x_smooth,idx)
    >>> YY_avg, freq, EDOF = functions.spectrum_band_avg(x,dt=1,M=111,winstr=None,plotflag=0,ebarflag=0)
    >>> YY_smooth, freq, EDOF = functions.spectrum_band_avg(x_smooth,dt=1,M=111,winstr=None,plotflag=0,ebarflag=0)
    >>> plt.figure()
    >>> plt.plot(freq,YY_smooth/YY_avg)
    >>> plt.xlim([0,0.1])
    >>> plt.plot(1/N*np.ones(2),[0,1],'--k')
    >>> plt.plot(1/N_actual*np.ones(2),[0,1],'--r')
    >>> plt.grid()

    @author: jtomf
    jfarrar@whoi.edu
    """
    # check and set defaults
    if win == None:
        win = 'boxcar'
    if cutoff == None:
        cutoff = 'power'
    if oddflag == None:
        oddflag = 1
    if edgeflag == None:
        edgeflag = 1
    
    # I could make the coutoff be specified as
    # a number of points
    # a dimensional length
    # a dimensional frequency
    # I will go with a number of points for now

    fc=1/N # dimensionless cutoff frequency

    # Set width of each type of window
    if win == 'boxcar':
        if cutoff=='power':
            T=0.89/fc/2 #-3dB
        elif cutoff=='amplitude':
            T=1.21/fc/2 #-6 dB, Harris says 1.2, which gives 52% instead of 50% power; 1,21 gives 49.1%; 1.20001 gives 49.2%
        coh_gain = 1.000

    elif win == 'parzen':
        if cutoff=='power':
            T=1.82/fc/2 #-3 dB
        elif cutoff=='amplitude':
            T=2.55/fc/2 #-6 dB
        coh_gain=0.3750

    elif win == 'gauss': # this is for a=3 from Harris
        if cutoff=='power':
            T=1.55/fc/2 # -3 dB
        elif cutoff=='amplitude':
            T=2.26/fc/2 # -6 dB; Harris says 2.18, which gives 52.5% power
        coh_gain=0.4167

    elif win == 'flattop': # these parameters are for matlab's flattopwin
        if cutoff=='power':
            T=3.75/fc/2
        elif cutoff=='amplitude':
            T=4.595/fc/2
        coh_gain=1

    elif win == 'hann':
        if cutoff=='power':
            T=1.44/fc/2 # -3 dB
        elif cutoff=='amplitude':
            T=2.0/fc/2 # -6 dB
        coh_gain=0.5


    else:
        print('Error: win must be one of boxcar, parzen, gauss, flattop, or hann')
        return

    # Determine the width in points of the window
    # T is the (non-integer) width of the window in points
    # width is the integer width of the window in points
    # coh_gain is the coherence gain of the window
    if oddflag == 1:
        # make sure width is the nearest odd integer to T
        width = int(np.round(T/2)*2+1)
    else:
        # just let width be the nearest integer to T
        width = int(np.round(T))


    # Now make the window
    if win == 'boxcar':
        win = signal.boxcar(width)
    elif win == 'parzen':
        win = signal.parzen(width)
    elif win == 'gauss':
        a = 3
        # win=exp(-0.5*(a/(T/2))**2); n/sigma= a*n/(T/2); sigma = T/2/a
        win = signal.gaussian(width, std=T/2/a)
    elif win == 'flattop':
        win = signal.flattop(width)
    elif win == 'hann':
        win = signal.hann(width)

    # normalize window to have unit gain
    win = win/np.sum(win)
    #win = win/(width*coh_gain) # normalize window to have unit area

    # Initialize fz
    fz = np.empty(np.shape(f))
    fz= np.convolve(f, win, mode='same')

    if edgeflag == 1:
        fz[1:N] = np.nan
        fz[-1-N:-1] = np.nan

    # Actual half-power or half-amplitude point
    N_actual = N*width/T

    return fz, N_actual


def smooth2d(f, N, win='boxcar', cutoff='power', oddflag=1, edgeflag=1, dim=1):
    """
    Smooth a 2d array along one axis with a half-power (-3dB) or half-amplitude (-6dB)
    point correspnding to N data points using the specified window.      
 
    Parameters
    ----------
    f : numeric
        So far I have only used a 1d Numpy.array.
    N : numeric
        Number of points for the half-power point of the smoothing window 
        (e.g., with N=26, the half-power point is at 26 points)
    win : string
        can be 'boxcar' (rectangle),'parzen','gauss', 'flattop', or 'hann' (Hanning squared cosine)
        default is 'boxcar'
    cutoff : string
        can be 'power' or 'amplitude'
        default is 'power'
    oddflag : numeric
        Can be 0 or 1; when set to 1 (default), the number of points in the smoothing window
        is forced to be odd (giving priority to making sure the filter does not introduce a phase shift)
        When set to 0, the number of points in the smoothing window is chosen so that the half-power
        point is as close as possible to the one specified by N (giving priority to making sure the
        filter has the specified half-power point)
    edgeflag : numeric
        Can be 0 or 1; when set to 1 (default), regions affected by edge effects are set to nan

    Returns
    -------
    fz : numeric
        smoothed version of f [same size as f].
    N_actual : numeric
        actual number of points for the half-power point of the smoothing window;
        1/N_actual is the actual half-power or half-amplitude frequency

    Reference
    -------
    Harris, F. J. (1978). On the use of windows for harmonic analysis with the discrete Fourier transform.
    Proceedings of the IEEE, 66(1), 51-83.

    Example
    -------
    >>> # Test smooth1d function by comparing spectra of a random 1D array before and after smoothing
    >>> N = 101
    >>> # make a random 1D array
    >>> x = np.random.randn(200000)
    >>> x_smooth, N_actual = functions.smooth1d(x,N,win='parzen')
    >>> # find indices of nan values in x_smooth and remove them from x and x_smooth
    >>> idx = np.where(np.isnan(x_smooth))
    >>> x = np.delete(x,idx)
    >>> x_smooth = np.delete(x_smooth,idx)
    >>> YY_avg, freq, EDOF = functions.spectrum_band_avg(x,dt=1,M=111,winstr=None,plotflag=0,ebarflag=0)
    >>> YY_smooth, freq, EDOF = functions.spectrum_band_avg(x_smooth,dt=1,M=111,winstr=None,plotflag=0,ebarflag=0)
    >>> plt.figure()
    >>> plt.plot(freq,YY_smooth/YY_avg)
    >>> plt.xlim([0,0.1])
    >>> plt.plot(1/N*np.ones(2),[0,1],'--k')
    >>> plt.plot(1/N_actual*np.ones(2),[0,1],'--r')
    >>> plt.grid()

    @author: jtomf
    jfarrar@whoi.edu   
    """

    # check and set defaults
    if win == None:
        win = 'boxcar'
    if cutoff == None:
        cutoff = 'power'
    if oddflag == None:
        oddflag = 1
    if edgeflag == None:
        edgeflag = 1
    if dim == None:
        dim = 1
    
    # I could make the coutoff be specified as
    # a number of points
    # a dimensional length
    # a dimensional frequency
    # I will go with a number of points for now

    fc=1/N # dimensionless cutoff frequency

    # Set width of each type of window
    if win == 'boxcar':
        if cutoff=='power':
            T=0.89/fc/2 #-3dB
        elif cutoff=='amplitude':
            T=1.21/fc/2 #-6 dB, Harris says 1.2, which gives 52% instead of 50% power; 1,21 gives 49.1%; 1.20001 gives 49.2%
        coh_gain = 1.000

    elif win == 'parzen':
        if cutoff=='power':
            T=1.82/fc/2 #-3 dB
        elif cutoff=='amplitude':
            T=2.55/fc/2 #-6 dB
        coh_gain=0.3750

    elif win == 'gauss': # this is for a=3 from Harris
        if cutoff=='power':
            T=1.55/fc/2 # -3 dB
        elif cutoff=='amplitude':
            T=2.26/fc/2 # -6 dB; Harris says 2.18, which gives 52.5% power
        coh_gain=0.4167

    elif win == 'flattop': # these parameters are for matlab's flattopwin
        if cutoff=='power':
            T=3.75/fc/2
        elif cutoff=='amplitude':
            T=4.595/fc/2
        coh_gain=1

    elif win == 'hann':
        if cutoff=='power':
            T=1.44/fc/2 # -3 dB
        elif cutoff=='amplitude':
            T=2.0/fc/2 # -6 dB
        coh_gain=0.5


    else:
        print('Error: win must be one of boxcar, parzen, gauss, flattop, or hann')
        return

    # Determine the width in points of the window
    # T is the (non-integer) width of the window in points
    # width is the integer width of the window in points
    # coh_gain is the coherence gain of the window
    if oddflag == 1:
        # make sure width is the nearest odd integer to T
        width = int(np.round(T/2)*2+1)
    else:
        # just let width be the nearest integer to T
        width = int(np.round(T))


    # Now make the window
    if win == 'boxcar':
        win = signal.boxcar(width)
    elif win == 'parzen':
        win = signal.parzen(width)
    elif win == 'gauss':
        a = 3
        # win=exp(-0.5*(a/(T/2))**2); n/sigma= a*n/(T/2); sigma = T/2/a
        win = signal.gaussian(width, std=T/2/a)
    elif win == 'flattop':
        win = signal.flattop(width)
    elif win == 'hann':
        win = signal.hann(width)

    # normalize window to have unit gain
    win = win/np.sum(win)
    #win = win/(width*coh_gain) # normalize window to have unit area

    # Initialize fz
    fz = np.empty(np.shape(f))
    if dim == 1:
        for n in range(0, len(f[0, :])-1):
            fz[:, n] = np.convolve(f[:, n], win, mode='same') 
        if edgeflag == 1:
            fz[1:N,:] = np.nan
            fz[-1-N:-1,:] = np.nan

    elif dim == 2:
        for n in range(0, len(f[:, 0])-1):
           fz[n, :] = np.convolve(f[n, :], np.transpose(win), mode='same') 
        if edgeflag == 1:
            fz[:,1:N] = np.nan
            fz[:,-1-N:-1] = np.nan


    # Actual half-power or half-amplitude point
    N_actual = N*width/T

    return fz, N_actual
