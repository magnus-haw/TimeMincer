import pickle 
import numpy as np
from pathlib import Path

from utils import encode_labels, smooth
from classes import TimeSegment1D, UNetSegment1D

from tensorflow import keras

MODELDATA_FOLDER = Path( __file__ ).parent.absolute() / "modeldata" / "1DCNN" / "Oct17_2023"

class MyTimeSegment1D(UNetSegment1D):

    ### Artificial data generation functions
    @staticmethod
    def gen_trace(n=512, smooth_window=5):
        """Generate top hat time series

        Args:
            n (int, optional): _description_. Defaults to 500.

        Returns:
            (1D float array): time series with random top hat start/stop/noise
        """
        # 
        rnd = sorted(np.random.random_sample((2,)))
        amp = np.random.random()/2. + .2
        noise_amp= np.random.random()/25.
        window = int((np.random.random() +1.2)*smooth_window/2.)
        start,stop = rnd[0],rnd[1]
        vals = np.zeros(n)
        t = np.linspace(-.5, 1.5, n)
        dt = 1./n
        
        # Label data
        vals[t>start]= amp
        vals[t>stop] = 0

        # add random peaks at start of some steps
        randnums = np.random.normal(0, 1, (2))
        if randnums[0]<.05:
             # Define Gaussian peak parameters
            peak_width = np.random.uniform(dt*10, dt*5)  # assuming peak width is less than 0.1 for narrowness
            peak_center = start  # center of the Gaussian peak
            peak_height = np.random.uniform(0.2, 1.0)  # height of the Gaussian peak

            # Add the Gaussian peak
            vals += peak_height * np.exp(-((t - peak_center) ** 2) / (2 * peak_width ** 2))

        vals = smooth(vals, window_len=window)
        noise = np.random.normal(0, noise_amp, (n))
        raw = vals + noise
        raw -= raw.min()

        # label fake top hat data
        labels = np.zeros(n)
        labels[t>start]= 2
        labels[t>stop] = 0
        labels[abs(t-start)<window/n] = 1
        labels[abs(t-stop)<window/n] = 1
        
        return  raw, encode_labels(labels, nclasses=3)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    npoints = 512
    tss = MyTimeSegment1D(n=npoints, modelfolder=MODELDATA_FOLDER)
    print(tss.modelname, tss.input_shape)
    print(tss.load_status)
    print(tss.print_model_summary())
    #data = tss.get_data_set(samples=20000,n=npoints,nsteps=1)
    #tss.train(data, epochs=1)

    for i in range(0,5):
        raw, labels = MyTimeSegment1D.gen_step_trace(n=npoints,nsteps=1)
        a,mask= tss.get_intervals(raw)
        out = tss.predict(raw)
        plt.plot(raw)
        #plt.plot(labels)
        # plt.plot(labels[:,0],'r--')
        # plt.plot(labels[:,1],'m--')
        # plt.plot(labels[:,2],'g--')

        plt.plot(out[0,:,0],'r-')
        plt.plot(out[0,:,1],'m-')
        plt.plot(out[0,:,2],'g-')
        
        plt.show()


    # ### Load normed traces
    # x_train,y_train=[],[]
    # with open('/Users/mhaw/Desktop/TimeMincer/normed_traces_dict_cleaned.pickle', 'rb') as fin:
    #     datadict = pickle.load(fin)
    # for key,val in datadict.items():
    #     raw,labels_ = val
    #     labels = encode_labels(labels_)
    #     x_train.append(raw)
    #     y_train.append(labels)

    #     out = tss.predict(raw)
    #     plt.plot(raw)
    #     #plt.plot(labels)
    #     # plt.plot(labels[:,0],'r--')
    #     # plt.plot(labels[:,1],'m--')
    #     # plt.plot(labels[:,2],'g--')

    #     plt.plot(out[0,:,0],'r-')
    #     plt.plot(out[0,:,1],'m-')
    #     plt.plot(out[0,:,2],'g-')
        
    #     plt.show()
    
    