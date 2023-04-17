    
    
    function PlotAS(in::Matrix; dy=0.004, fmax=100)

		d= in;
        xlabel = "Frequency"
	    xunits = "(Hz)"
	    ylabel = "Amplitude"
	    yunits = ""
        
        nx = size(d[:,:], 2)
	    df = 1/dy/size(d[:, :], 1)
	    FMAX = df*size(d[:, :], 1)/2
	    if fmax > FMAX
            fmax = FMAX
	    end
	    
        nf = convert(Int32, floor((size(d[:, :], 1)/2)*fmax/FMAX))
	    y = fftshift(sum(abs.(fft(d[:, :], 1)), dims=2))/nx
	    y = y[round(Int,end/2):round(Int, end/2)+nf]
	    norm = maximum(y[:])
	    #if (norm > 0.)
		 #   y = y/norm
	    #end
	    x = collect(0:df:fmax)
	    im = plt.plot(x, y)
	    plt.title(title)
	    plt.xlabel(join([xlabel " " xunits]))
	    plt.ylabel(join([ylabel " " yunits]))
	    #plt.axis([0, fmax, 0, 1.1])

    end
