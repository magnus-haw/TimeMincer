import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO



### Process steady state 
def eval_steady(time,vals,t0,t1):
    """Get stats for values in time interval 

    Args:
        time (np.array or list): time array, should not have nans
        vals (np.array or list): value array, can have nans
        t0 (float): start time
        t1 (float): end time

    Raises:
        Exception: if interval is invalid raise exception

    Returns:
        dict: stats in dictionary format. {'average':np.nanmean(svals),'stdev':np.nanstd(svals),
            'max':np.nanmax(svals),'min':np.nanmin(svals), 't0':t0, 't1':t1,
            'time':list(time),'vals':list(vals) }
    """
    time, vals = np.array(time), np.array(vals)
    try:
        assert(t1 != None and t0 != None)
        assert(t1>t0)
        inds = (time>t0)*(time<t1)*np.isfinite(vals)
        assert(inds.any())
        svals = vals[inds]
        stime = time[inds]
    except Exception as e:
        raise Exception('Time interval empty or invalid (t0=%f, t1=%f) for this series (%f< t < %f): '%(t0,t1,min(time),max(time))+str(e)) from e
        
    return {'average':np.nanmean(svals),'stdev':np.nanstd(svals),
            'max':np.nanmax(svals),'min':np.nanmin(svals), 't0':t0, 't1':t1,
            'time':list(stime),'vals':list(svals) }

### moving avg
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w

### extract moving net difference (DeltaY)
def moving_difference(x, w):
    window = np.zeros(w)
    window[0] = 1.
    window[-1] = -1.
    return np.convolve(x, window, 'same')

### extract moving local maximum
def moving_maxima(x,w):
    w2 = int(w/2.)
    w = w2*2.
    mx = max(x)
    weights = np.ones(len(x))*mx
    for i in range(w2,len(x)-w2):
        weights[i] = max(mx/3, max(x[i-w2:i+w2]))
    return weights

### getting intersection of 2 intervals
def get_intersection(a0,a1,b0,b1):
    t0,t1 = max(a0,b0),min(a1,b1)
    if t0 > t1:
        return None, None
    else:
        return t0,t1

### collate condition intervals
def collate_conditions(conds):
    """Combine condition segmentation from individual series 
       into global segmentation of conditions

    Args:
        conds (list dict): list of dictionaries describing series segmentation
        (e.g., [{'mask':[0,1,1,1,0,0...], 
                 'ilist':[{'interval':[1,3], 'mean':10, 'stdev':1, 'min':8, 'max':11, 'source_obj':DiagnosticSeriesObject},...] }...])

    Returns:
        list dict: list of dictionaries compiling global condition params
    """
    if len(conds)==0:
        return []

    # combine masks for all series
    masks = 1
    for cond in conds:
        masks *= cond['mask']

    # segment global mask into intervals
    global_intervals = start_stop(masks,1,len_thresh=25)

    time = conds[0]['time']

    # compile data for each global interval
    allconds = []
    for gi in global_intervals:
        start,stop = gi
        cond_dict ={'interval':gi,'start':time[start],'stop':time[stop],'sloped':False}
        #for each series, locate overlapping interval, value & type
        for condlist in conds:
            v1,v2 = condlist['values'][stop-5], condlist['values'][start+5]
            delta_norm =  abs(v1 - v2)/min(v1,v2)
            if delta_norm > .1:
                v3 = condlist['values'][int((start+stop)/2)]
                if abs(v1 - v3)/min(v1,v3) > .05:
                    cond_dict['sloped'] = True
                else:
                    start = int((start+stop)/2 )
                    cond_dict['start'] = time[start]
            for cond in condlist['ilist']:
                i1,i2 = cond['interval']
                if (start<=i2 and i2<=stop) or (start<=i1 and i1<=stop) or (i1<=start and stop<=i2):
                    s = cond['source_obj']
                    name = s.diagnostic.name.lower()
                    if 'current' in name:
                        cond_dict['current'] = cond
                    elif 'main' in name:
                        cond_dict['main_gas'] = cond
                    elif 'add' in name:
                        cond_dict['add_gas'] = cond
                    elif 'shield' in name or 'argon' in name:
                        cond_dict['shield_gas'] = cond
                    break
        if not cond_dict['sloped']:
            allconds.append(cond_dict)
    return allconds

### nan helper fcn
def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

### interpolate missing data
def interp_nans(y):
    nans, x= nan_helper(y)        
    y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    return y

### Normalize time series 
def normalize_time_series(y):
    """utility function to normalize a time series 

    Args:
        y (np array): 1d array or list of floats
    """
    ynorm = (y-min(y)) /(max(y))
    return ynorm

### get start/stop indicies of val==True intervals
def start_stop(a, trigger_val=1., len_thresh=2):
    """extract the indices of continuous interval of particular value

    Args:
        a (array floats): e.g. [1,1,1,1,1,0,1,1,1,1]
        trigger_val (float): True value (e.g. 1) 
        len_thresh (int, optional): minimum interval length. Defaults to 2.

    Returns:
        array of intervals: (e.g. [[0,4],[6,9]])
    """
    # "Enclose" mask with sentients to catch shifts later on
    mask = np.r_[False,np.equal(a, trigger_val),False]

    # Get the shifting indices
    idx = np.flatnonzero(mask[1:] != mask[:-1])

    # Get lengths
    lens = idx[1::2] - idx[::2]

    return idx.reshape(-1,2)[lens>len_thresh]-[0,1]

### smoothing helper fcn
def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
    
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also: 

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    x = np.array(x)
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    #s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    N = window_len
    spad = np.pad(x, (N//2, N-1-N//2), mode='edge')
    ret = np.convolve(w/w.sum(),spad,mode='valid')
    #ret = y[int(window_len/2)-1:-int(window_len/2)-1] 

    assert len(ret) == len(x)
    return ret

### get the top portion of tophat curve
def top_values(x, bins=25):
    """use distribution to isolate top values of tophat curve

    Args:
        x (array): array of tophat curve values greater than midpoint (zero values not included) 

    Returns:
        retdict: {'vals':[], 'average':avg, 'max':max, 'min':min, 'stdev':stdev}

    """
    hist,edges = np.histogram(x,bins=bins)
    imax = np.argmax(hist)
    low, high = edges[imax], edges[imax+1]

    inds = (x>low)*(x<high*1.05)
    vals = x[inds]
    return {'vals':list(vals), 'average':np.mean(vals), 'max':max(vals), 'min':min(vals), 'stdev':np.std(vals)}

def plot_to_string():
    """transform matplotlib plot to png (string fmt)

    Returns:
        png: png in string fmt
    """

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close()
    plt.close()

    return graph

def plot_to_bytes():
    """transform matplotlib plot to png (string fmt)

    Returns:
        png: png in string fmt
    """

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)

    return buffer

def first_falling_edge(t0,v0, threshold=-.5):
    """identify first falling edge in boolean time series

    Args:
        t0 (array): first array times
        v0 (array): first array values, assume values are booleans
        threshold (float, optional): falling edge threshold. Defaults to -.5.

    Returns:
        time0: time of first falling edge 
    """
    ind0 = np.argmax(np.diff(v0*1.)<threshold) +1
    return t0[ind0]

def last_rising_edge(t0,v0, threshold=.5):
    """identify last rising edge in boolean time series

    Args:
        t0 (array): times
        v0 (array): array values, assume values are booleans
        threshold (float, optional): rising edge threshold. Defaults to .5.

    Returns:
        time0: time of last rising edge
    """
    ind0 = np.argwhere(np.diff(v0*1.)>threshold)[-1][0] -1
    return t0[ind0]

### one hot encoding for labels
def encode_labels(labels, nclasses=3):
    encoded_labels = np.zeros((len(labels),nclasses))
    for i in range(0,nclasses):
        encoded_labels[:,i] = labels==i

    return encoded_labels

### one hot decoding for labels
def decode_labels(elabels, nclasses=3):
    n = len(elabels[0])
    retlabels = np.zeros(n)
    for i in range(0,n):
        retlabels[i] = np.argmax(elabels[0,i,:])
    return retlabels

   

