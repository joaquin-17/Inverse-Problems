""" 
         DSAOp(in; <keyword argument>)

Function to mimic the acquisition of shot-gathers acquired with different sources. This function applies  SourceOp function and returns data that mimics the Dispersed Source Array (DSA) acquisition strategy with a given pre-established distribution of sources.

# Arguments:
- `in`: up to 3D input data. First dimension is time.
- `adj`::Bool=true : Flag to apply the forward or adjoint source operator. Adjoint is set by default.
- `dt :: Float=0.002` : sampling rate along the time axis (in seconds).
- `SFB`:: Matrix : Frequency band for modeling the different narrow band-sources.
- `FB`:: Vector : Frequency band for modeling the ideal broad-band source.
- `k` :: Vector : Vector with the position of the different sources.

"""
function DSAOp(in; dt::Real, SFB::Matrix, FB::Vector, k::Vector);

    d_obs=SourceOp(in, true, dt= dt, SFB= SFB, FB=FB,k=k);
    
    return d_obs
end


function DSAOp(in; dt::Real, Wavelets::Vector, k::Vector);

    d_obs=SourceOp(in, true, dt= dt,Wavelets=Wavelets,k=k);
    
    return d_obs
end
