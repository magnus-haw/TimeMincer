from classes import TimeSegment1D

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    npoints = 5000
    tss = TimeSegment1D(n=npoints)
    print(tss.modelname, tss.input_shape)
    print(tss.load_status)
    #tss.train(tss.get_data_set(samples=20000,n=npoints,nsteps=3))

    for i in range(0,5):
        raw, labels = TimeSegment1D.gen_step_trace(n=npoints,nsteps=3)
        a,mask= tss.get_intervals(raw)
        out = tss.predict(raw)
        plt.plot(raw)
        #plt.plot(labels)
        plt.plot(out[0,:,0],'r-')
        plt.plot(out[0,:,1],'m-')
        plt.plot(out[0,:,2],'g-')
        
        plt.show()
    

    
    

