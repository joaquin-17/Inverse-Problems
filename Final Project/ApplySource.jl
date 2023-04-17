"""
         ApplySource(in, dt::Real,FB::Vector)

Function to model Disperse Source Arrays data by applying in the frequency domain each one of the ideal sources.
This function it is used inside the SourceOperator function.

# Arguments:
- `in`: up to 3D input data. First dimension is time.
- `dt :: Float=0.002` : sampling rate along the time axis (in seconds).
- `FB`:: Vector : Frequency band for modeling the ideal broad-band source.
"""
function ApplySource(in, dt::Float64,FB::Vector)
    
    #Get the parameters in some way that follows the structure of SeismicJulia
    #F1= parameters[:F1]
    #dt=parameters[:dt];
    dt=dt;
    #f1 = FB[1]; f2 = FB[2]; f3 = FB[3]; f4 = FB[4];

    
    number_dims = ndims(in);
    d_size= size(in);
    nt  = d_size[1];
    x = reshape(in,nt,prod(d_size[2:end])); #Reshape dimension to be a matrix;

    k=Int(log2(nextpow(2,nt))); #Zero padd!
    np= 4*(2^k);
    BF=SoftFreqBand(np,dt,FB) #Frequency Band!

    #Take the rehsaped data to frequency domain:
    xp = zeros(Float64,(np,prod(d_size[2:end])));
    xp[1:nt,:] .= x[:,:]; #Zero-padding!
    
    
    Nyquist = (np ÷ 2)+1 #Last independent frequency.
    X_symmetries=zeros(ComplexF64,size(xp)); #Pre-allocate spectrum

    aux=zeros(ComplexF64,(Nyquist, size(xp,2)))


    @inbounds for j=1:size(xp,2);
        aux[:, j]=rfft(xp[:,j])
    end


    X_symmetries[1:Nyquist,:]= aux[:,:]


    @inbounds for j =1:size(xp,2);
        @inbounds for i = Nyquist +1:np
            X_symmetries[i,j]=conj(aux[np-i+2,j])
        end
    end
    
    
    Y_symmetries=zeros(ComplexF64,size(X_symmetries));
    aux2=zeros(ComplexF64,(Nyquist, size(xp,2)))


    @inbounds for j=1:size(X_symmetries,2)
        aux2[:,j] .=  BF[1:Nyquist].*X_symmetries[1:Nyquist,j]
    end

    Y_symmetries[1:Nyquist,:]= aux2[:,:];


    @inbounds for j =1:size(X_symmetries,2);
        @inbounds for i = Nyquist +1:np
            Y_symmetries[i,j]=conj(aux2[np-i+2,j])
        end
    end
    
    #Reshape everything, re-arrange and combe back to the data domain
    out= ifft(Y_symmetries,1) #Go back to the time-space domain
    out=real(out[1:nt,:]); # get real part ( imag is zero) and the number of samples
    out=reshape(out, d_size) #Reshape to original size data
    
    
    return out
    
    
end


"""
         ApplySource(in, dt::Real,S_params::Vector)

Function to model Disperse Source Arrays data by applying in the frequency domain each one of the ideal sources.
This function it is used inside the SourceOperator function.

# Arguments:
- `in`: up to 3D input data. First dimension is time.
- `dt :: Float=0.002` : sampling rate along the time axis (in seconds).
- `FB`:: Vector : Frequency band for modeling the ideal broad-band source.
"""
function ApplySource(in, dt::Float64, Sparams::Float64)
    
    #Get the parameters in some way that follows the structure of SeismicJulia

    dt=dt;
    number_dims = ndims(in);
    d_size= size(in);
    nt  = d_size[1];
    x = reshape(in,nt,prod(d_size[2:end])); #Reshape dimension to be a matrix;

    k=Int(log2(nextpow(2,nt))); #Zero padd!
    np= 4*(2^k);
    
    wavelet=RickerWavelet(np,Sparams);
    #wavelet=circshift(wavelet,-1200)
    WAVELET= fft(wavelet);
    WAVELET = WAVELET./maximum(abs.(WAVELET))

    #Take the rehsaped data to frequency domain:
    xp = zeros(Float64,(np,prod(d_size[2:end])));
    xp[1:nt,:] .= x[:,:]; #Zero-padding!
    
    
    Nyquist = (np ÷ 2)+1 #Last independent frequency.
    X_symmetries=zeros(ComplexF64,size(xp)); #Pre-allocate spectrum

    aux=zeros(ComplexF64,(Nyquist, size(xp,2)))


    @inbounds for j=1:size(xp,2);
        aux[:, j]=rfft(xp[:,j])
    end


    X_symmetries[1:Nyquist,:]= aux[:,:]


    @inbounds for j =1:size(xp,2);
        @inbounds for i = Nyquist +1:np
            X_symmetries[i,j]=conj(aux[np-i+2,j])
        end
    end
    
    
    Y_symmetries=zeros(ComplexF64,size(X_symmetries));
    aux2=zeros(ComplexF64,(Nyquist, size(xp,2)))


    @inbounds for j=1:size(X_symmetries,2)
        aux2[:,j] .=  WAVELET[1:Nyquist].*X_symmetries[1:Nyquist,j]
    end

    Y_symmetries[1:Nyquist,:]= aux2[:,:];


    @inbounds for j =1:size(X_symmetries,2);
        @inbounds for i = Nyquist +1:np
            Y_symmetries[i,j]=conj(aux2[np-i+2,j])
        end
    end
    
    #Reshape everything, re-arrange and combe back to the data domain
    out= ifft(Y_symmetries,1) #Go back to the time-space domain
    out=real(out[1:nt,:]); # get real part ( imag is zero) and the number of samples
    out=reshape(out, d_size) #Reshape to original size data
    
    
    return out
    
    
end
