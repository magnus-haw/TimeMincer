import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt 

# tensorflow
from tensorflow import keras
from keras.layers import Conv1D, Conv1DTranspose
from tensorflow.python.keras.layers.core import Dropout

# utility functions
from utils import start_stop, smooth, encode_labels

MODELDATA_FOLDER = Path( __file__ ).parent.absolute() / "modeldata" / "1DCNN"


class TimeSegment1D(object):
    """Helper class for 1D CNN model for time series segmentation

    Args:
        AbstractMLModel (class): base class for ml models 
    """
    def __init__(self, modelfolder=MODELDATA_FOLDER, modelname = 'ckpt', n=500):
        """Create 1D encoder CNN model using sequential API from Keras

        Args:
            modelfolder (_type_): _description_
            modelname (str, optional): _description_. Defaults to 'ckpt'.
            n (int, optional): _description_. Defaults to 500.
        """
        #
        model = keras.models.Sequential()
        model.add(Conv1D(filters= 32, kernel_size=7, activation='relu', input_shape= (n,1), strides=2, padding='same'))
        model.add(Dropout(0.25))
        model.add(Conv1D(filters=16, kernel_size=7, activation='relu', strides=2, padding='same'))
        model.add(Dropout(0.25))
        model.add(Conv1DTranspose(filters=16, kernel_size=7, activation='relu', strides=2, padding='same'))
        model.add(Dropout(0.25))
        model.add(Conv1DTranspose(filters=32, kernel_size=7, activation='relu', strides=2, padding='same'))
        model.add(Dropout(0.25))
        model.add(Conv1DTranspose(filters=3, kernel_size=3, activation='softmax',padding='same'))
        model.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate=0.001), loss=keras.losses.CategoricalCrossentropy())

        self.model = model
        self.input_shape = (n,1)
        self.modelname = modelname
        self.modelfolder = modelfolder
        self.load_status = False
        self.history = None
        self.load_weights()

    def load_weights(self):
        """load trained weights

        Returns:
            bool: load_status (True: success, False: failure)
        """
        ### 
        try:
            self.load_status = self.model.load_weights("%s/%s"%(self.modelfolder, self.modelname)).expect_partial()
            self.load_status = True
        except:
            self.load_status = False
        return self.load_status

    def predict(self, raw):
        """Predict/classify input values using trained model

        Args:
            raw (1d array of floats): times series values

        Returns:
            ndarray: output classifications
        """
        if self.load_status:
            # normalize inputs to ~[0,1]
            normed = raw/max(raw)
            normed = np.array([normed])
            normed = np.expand_dims(normed, axis=2)
            out = self.model.predict(normed)
            return out
        else:
            print('Model failed to load. Cannot predict')
            return None

    def get_intervals(self,raw, source_obj=None):
        """Get intervals from 1D-CNN output

        Args:
            raw (float ndarray): 1d array of floats
            source_obj (DiagnosticSeries): database model object, not processed but 
                                           returned in dictionary
        
        Returns:
            list dict: [{interval, mean, stdev, min, max}, {interval, mean, stdev, min, max} ...] 
        """
        modeloutput = self.predict(raw) #shape:(1,n,3) 
        trans = modeloutput[0,:,1]
        on = modeloutput[0,:,2]
        
        mask = (on>trans)*(on>.4)
        on_intervals = start_stop(mask,1,len_thresh=25)
        ret_list =[]
        for intv in on_intervals:
            start,stop = intv
            vals = raw[start:stop+1]
            ret_list.append({'interval':intv,'average':np.mean(vals),'stdev':np.std(vals),
            'max':np.max(vals),'min':np.min(vals), 'source_obj':source_obj})

        return ret_list,mask
    
    def train(self, input):
        """train member function

        Args:
            input (dict): {'values':ndarray,'labels':ndarray}
        """
        x_train, y_train = input['values'], input['labels']
        ### Train model and plot validation vs training error
        self.history = self.model.fit(
            x_train,
            y_train,
            epochs=5,
            batch_size=128,
            validation_split=0.2,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")
            ],
        )
        self.model.save_weights("%s/%s"%(self.modelfolder, self.modelname))
    
    def get_input_shape(self):
        return self.input_shape

    ### Artificial data generation functions
    @staticmethod
    def gen_trace(n=500, smooth_window=25):
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
        window = int((np.random.random() +1.5)*smooth_window/2.)
        start,stop = rnd[0],rnd[1]
        vals = np.zeros(n)
        t = np.linspace(-.5, 1.5, n)
        dt = 1./n
        
        # Label data
        vals[t>start]= amp
        vals[t>stop] = 0

        # add peak at start
        randnums = np.random.normal(0, 1, (2))
        if randnums[0]<.3:
             # Define Gaussian peak parameters
            peak_width = np.random.uniform(dt*10, dt*75)  # assuming peak width is less than 0.1 for narrowness
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
        
        return  raw, encode_labels(labels)

    @staticmethod
    def gen_step_trace(n=500, nsteps=3):
        """Generate multi-step time series

        Args:
            n (int, optional): length of sequence. Defaults to 500.
            nsteps (int, optional): number of top hat in each trace. Defaults to 3

        Returns:
            (1D float array): time series with random top hat start/stop/noise
        """
        rawsum=0;labels=[]
        final_labels = np.zeros((n,3))
        for i in range(0,nsteps):
            raw_,labels_ = TimeSegment1D.gen_trace(n=n)
            rawsum +=raw_
            labels.append(labels_)
        rawsum -= rawsum.min()

        for i in range(0,n):
            final_labels[i,0] = 1
            for j in labels:
                if j[i,1]:
                    final_labels[i,1] = 1
                    final_labels[i,2] = 0
                    final_labels[i,0] = 0
                    break
                elif j[i,2]:
                    final_labels[i,2] = 1
                    final_labels[i,0] = 0

        return  rawsum, final_labels

    def get_data_set(self,samples=600,n=1000,nsteps=3):
        """Generates dataset for training 1D CNN for top hat segmentation

        Args:
            samples (int, optional): number of samples in set. Defaults to 600.
            n (int, optional): length of sequences. Defaults to 500.
            nsteps (int, optional): number of steps in each sequence. Defaults to 3.

        Returns:
            dict: {"values":x_train, "labels":y_train}
        """
        x_train, y_train = [],[]
        for i in range(0,samples):
            raw,labels = TimeSegment1D.gen_step_trace(n=n, nsteps=nsteps)
            x_train.append(raw)
            y_train.append(labels)
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_train = np.expand_dims(x_train, axis=2)

        return {"values":x_train, "labels":y_train}

    ### Manual labeling function
    def manually_label_data(self, labels, raw):
        """Manually adjust 3-class labels for 1d time trace using GUI

        Args:
            labels (array int): array of integers defining class label (0,1,2)
            raw (array float): _description_

        Returns:
            array int: array of integers defining class label (0,1,2)
            bool: skip flag for bad data (default:False)
        """
        
        print(np.shape(labels), np.shape(raw))
        lt = LabelTrace(labels, raw)
        lt.connect()
        return lt.labels, lt.skip

 
class LabelTrace:
    """Matplotlib class for class labling of 1D normalized traces
    
    Plots trace (black) and labels (red,yellow,green lines): 
        red line: 'off' class, 0
        yellow line: 'transition' class, 1
        green line: stable 'on' class, 2
    
    Operation:
        - select class of interest by pressing key (0,1,2)
        - drag mouse over region to color, selected region will be labeled with current class
    """

    def __init__(self, labels, values):
        
        self.t0 = 0
        self.t1 = 0
        self.classlabel = 1  
        self.labels = labels
        self.values = values
        self.skip = False

        self.figure = plt.figure()
        self.axis = plt.gca()

        self.vline = self.axis.plot(values, 'k-')
        elabels = encode_labels(self.labels)
        self.line0, = self.axis.plot(elabels[:,0], 'r-')
        self.line1, = self.axis.plot(elabels[:,1], 'y-')
        self.line2, = self.axis.plot(elabels[:,2], 'g-')
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def connect(self):
        """Connect to all the events we need."""
        self.cidpress = self.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidrelease = self.figure.canvas.mpl_connect(
            'key_press_event', self.on_key)

    def on_press(self, event):
        """store some data."""
        self.press = True
        self.t0 = event.xdata

    def on_release(self, event):
        """store release location & update plot"""
        self.press = None
        self.t1 = event.xdata
        print(self.t0, self.t1)
        tmax = int(max(self.t0,self.t1))
        tmin = int(min(self.t0,self.t1))
        self.labels[tmin:tmax] = self.classlabel

        # redraw the full figure
        self.plot_values()

    def on_key(self, event):
        ''' 
            Handles predefined key-press events 
        ''' 
        print('Key press:\'%s\'' %(event.key))
        if event.key in ['0','1','2']: 
            self.classlabel = int(event.key)
        if event.key == 'n':
            self.skip = True
            plt.close(self.figure)
        print(self.classlabel)

    def plot_values(self):
        elabels = encode_labels(self.labels)
        self.line0.set_ydata(elabels[:,0])
        self.line1.set_ydata(elabels[:,1])
        self.line2.set_ydata(elabels[:,2])
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def disconnect(self):
        """Disconnect all callbacks."""
        self.figure.canvas.mpl_disconnect(self.cidpress)
        self.figure.canvas.mpl_disconnect(self.cidrelease)
        self.figure.canvas.mpl_disconnect(self.cidmotion)
