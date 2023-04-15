function GetSpectrum(d;fmax=100,dt=0.002,dx=25.0)
                           
                           dk = 1/dx/size(d[:,:], 2)
	                       kmin = -dk*size(d[:,:], 2)/2
	                       kmax =  dk*size(d[:,:], 2)/2
	                       df = 1/dt/size(d[:,:], 1)
	                       FMAX = df*size(d[:,:], 1)/2
	                       if fmax > FMAX
                                fmax = FMAX
                            end
	                       nf = convert(Int32, floor((size(d[:, :], 1)/2)*fmax/FMAX))
	                       D = abs.(fftshift(fft(d[:, :])))
	                       D = D[round(Int,end/2):round(Int,end/2)+nf, :]
                           #D=D./maximum(D);
                           #imshow(hcat(D),aspect="auto",extent=[kmin,kmax,fmax,0],dy=df)

                           return D
                        end




function PlotFK(in; pclip=98, cmap="cubehelix_r", vmin="NULL", vmax="NULL",
    aspect="auto", interpolation="Hanning", fmax=100,
    xcur=1.2, scal="NULL",dt=0.002,dx=25.0)

    d=GetSpectrum(in;fmax=fmax,dt=dt,dx=dx)
   
    dk = 1/dx/size(d[:,:], 2)
    kmin = -dk*size(d[:,:], 2)/2
    kmax =  dk*size(d[:,:], 2)/2
    df = 1/dt/size(d[:,:], 1)
	FMAX = df*size(d[:,:], 1)/2
	if fmax > FMAX
        fmax = FMAX
    end


    im = plt.imshow(d, extent=[kmin,kmax,fmax,0],
                        aspect=aspect, interpolation=interpolation, vmin=vmin, vmax=vmax)


    #xlabel("Wavenumber [Cycles/m]", fontsize=17);
    #ylabel("Frequency [Hz]", fontsize=17);

    #plt.colobar(shrink=0.75);
            

end

    #xlabel"Wavenumber"
    #xunits = "(1/m)"
    #ylabel = "Frequency"
                #        yunits = "Hz"







