from pathlib import Path
import numpy as np
from utils import encode_labels, smooth
from classes import TimeSegment1D

from tensorflow import keras
import keras_tuner as kt

MODELDATA_FOLDER = Path( __file__ ).parent.absolute() / "modeldata" / "1DCNN" / "Oct17_2023"

class MyTimeSegment1D(TimeSegment1D):

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

        # add random peaks at start of some steps
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
        
        return  raw, encode_labels(labels, nclasses=3)

def build_model(hp: kt.HyperParameters) -> keras.Model:
    """Builds a model with hyperparameters given by the Keras Tuner's HyperParameters object."""
    
    # Define hyperparameters to tune
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    
    conv_layers = []
    for i in range(2):
        conv_layers.append({
            'filters': hp.Int(f'filters_{i}', min_value=16, max_value=64, step=16),
            'kernel_size': hp.Int(f'kernel_size_{i}', min_value=3, max_value=9, step=2),
            'activation': 'relu',
            'dropout_rate': hp.Float(f'dropout_rate_{i}', min_value=0.0, max_value=0.5, step=0.1),
            'depth': hp.Int(f'depth_{i}', min_value=1, max_value=3, step=1),

        })

    convtrans_layers = []
    for i in range(2):
        convtrans_layers.append({
            'filters': hp.Int(f'tfilters_{i}', min_value=16, max_value=64, step=16),
            'kernel_size': hp.Int(f'tkernel_size_{i}', min_value=3, max_value=9, step=2),
            'activation': 'relu',
            'dropout_rate': hp.Float(f'tdropout_rate_{i}', min_value=0.0, max_value=0.5, step=0.1),
            'depth': hp.Int(f'tdepth_{i}', min_value=1, max_value=3, step=1),
        })
    
    # Using the modified TimeSegment1D class to create a model with the specified hyperparameters
    model = TimeSegment1D(
        modelfolder=str(MODELDATA_FOLDER),  # specify the correct folder
        modelname='hyperparam',      # specify the model name
        n=500,                            # input size, change accordingly
        conv_layers=conv_layers,
        convtrans_layers=convtrans_layers,
        learning_rate=learning_rate
    ).model

    return model


if __name__ == '__main__':
    # import matplotlib.pyplot as plt
    # npoints = 500
    # tss = MyTimeSegment1D(n=npoints)
    # print(tss.modelname, tss.input_shape)
    # print(tss.load_status)
    # tss.train(tss.get_data_set(samples=20000,n=npoints,nsteps=3))

    # for i in range(0,5):
    #     raw, labels = MyTimeSegment1D.gen_step_trace(n=npoints,nsteps=3)
    #     a,mask= tss.get_intervals(raw)
    #     out = tss.predict(raw)
    #     plt.plot(raw)
    #     #plt.plot(labels)
    #     plt.plot(out[0,:,0],'r-')
    #     plt.plot(out[0,:,1],'m-')
    #     plt.plot(out[0,:,2],'g-')
        
    #     plt.show()
    
    tuner = kt.Hyperband(
        build_model,
        objective='val_loss',  # the goal you're optimizing for, change if needed
        max_epochs=10,  # maximum number of epochs to train, change if needed
        factor=3,
        directory=str(MODELDATA_FOLDER),  # specify the directory where tuning results will be saved
        project_name='hyperparam'  # specify the project name
    )

    # Assume you have your train_x, train_y, val_x, val_y datasets ready
    # Replace with your actual dataset
    input = MyTimeSegment1D.get_data_set(samples=5000, n=500)
    validation = MyTimeSegment1D.get_data_set(samples=1000, n=500)
    train_x, train_y = input['values'], input['labels']
    val_x, val_y = validation['values'], validation['labels']

    tuner.search(train_x, train_y, epochs=10, validation_data=(val_x, val_y))  # change epochs if needed

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"""
    The hyperparameters search is complete. The optimal learning rate for the optimizer is {best_hps.get('learning_rate')}, and the optimal configuration for each
    convolutional layer is:""")
    
    for i in range(2):
        print(f"""
        - Conv Layer {i+1}:
            filters: {best_hps.get(f'filters_{i}')},
            kernel size: {best_hps.get(f'kernel_size_{i}')},
            dropout rate: {best_hps.get(f'dropout_rate_{i}')},
            depth: {best_hps.get(f'depth_{i}')}
        """)

        print(f"""
        - ConvTranspose Layers {i+1}:
            filters: {best_hps.get(f'tfilters_{i}')},
            kernel size: {best_hps.get(f'tkernel_size_{i}')},
            dropout rate: {best_hps.get(f'tdropout_rate_{i}')},
            depth: {best_hps.get(f'tdepth_{i}')}
        """)

    
    

