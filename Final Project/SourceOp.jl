""" 
         SourceOp(in,adj; <keyword argument>)

Source operator for the Disprese Source Arrays problem

# Arguments:
- `in`: up to 3D input data. First dimension is time.
- `adj`::Bool=true : Flag to apply the forward or adjoint source operator. Adjoint is set by default.
- `dt :: Float=0.002` : sampling rate along the time axis (in seconds).
- `SFB`:: Matrix : Frequency band for modeling the different narrow band-sources.
- `FB`:: Vector : Frequency band for modeling the ideal broad-band source.
- `k` :: Vector : Vector with the position of the different sources.

"""
function SourceOp(in,adj::Bool=true; dt::Real=0.002, SFB::Matrix, FB::Vector, k::Vector) 
    
    
    #get the parameters
    #F2=SFB; k= k; dt=dt; F1=FB;
    
    in=real(in);

    if adj == false  # inverse fourier transform the data and passit to apply source
        
        d_size=size(in);
        out=zeros(ComplexF64,d_size)
        @inbounds for is = 1:d_size[3]
            FB = SFB[k[is],:];
            aux2 =ApplySource(in[:,:,is],dt,FB);
            out[:,:,is] = aux2 # return data in time-space
        end
        
        return out
    
    elseif adj == true 
    
        
        d_size=size(in);
        aux2=zeros(ComplexF64,d_size)
        out=zeros(ComplexF64,d_size);
        
        @inbounds for is = 1:d_size[3]
            FB= SFB[k[is],:]
            aux1 =ApplySource(in[:,:,is],dt,FB);
            aux2[:,:,is] = aux1;
        end 
            
        out=aux2;
        
        return out
    
    else
        message="To apply the forward opeartor `adj= false` and to apply the adjoint operator `adj=true`"
        error(message)
    
    end

end




function SourceOp(in,adj::Bool=true; dt::Real=0.001, Wavelets::Vector, k::Vector) 
    
    
    #get the parameters
    #F2=SFB; k= k; dt=dt; F1=FB;
    
    in=real(in);

    if adj == false  # inverse fourier transform the data and passit to apply source
        
        d_size=size(in);
        out=zeros(ComplexF64,d_size)
        @inbounds for is = 1:d_size[3]
            wavelet = Wavelets[k[is]];
            aux2 =ApplySource(in[:,:,is],dt,wavelet);
            out[:,:,is] = aux2 # return data in time-space
        end
        
        return out
    
    elseif adj == true 
    
        
        d_size=size(in);
        aux2=zeros(ComplexF64,d_size)
        out=zeros(ComplexF64,d_size);
        
        @inbounds for is = 1:d_size[3]
            wavelet= Wavelets[k[is]]
            aux1 =ApplySource(in[:,:,is],dt,wavelet);
            aux2[:,:,is] = aux1;
        end 
            
        out=aux2;
        
        return out
    
    else
        message="To apply the forward opeartor `adj= false` and to apply the adjoint operator `adj=true`"
        error(message)
    
    end

end