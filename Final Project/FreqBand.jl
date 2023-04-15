"""
          FreqBand(N::Int,dt::Real,FB::Vector{Real})

Function to model a frequency filter that it is used to model different sources 
This function it is used inside the SourceOperator function.

# Arguments:
- `N`:: Number of points in the filter.
- `dt :: Float=0.002` : sampling rate along the time axis (in seconds).
- `FB`:: Vector{Real} : Frequency band for modeling the different types of sources.
"""
function FreqBand(N::Int,dt::Float64,FB::Vector)

    
    f1=FB[1];
    f2=FB[2];
    f3=FB[3];
    f4=FB[4];
    
    
    
    Mb= (N ÷2) +1
    BF=zeros(Float64,N)

    #Frequency index:
    n1 = round(Int, f1*dt*N +1)
    n2 = round(Int, f2*dt*N +1)
    
    n3 = round(Int, f3*dt*N +1)
    n4 = round(Int, f4*dt*N +1)

    for i=n1:n2
        BF[i] = (i-n1)/(n2-n1)
    end

    for i=n2+1:n3
        BF[i] = 1.0
    end

    for i=n3+1:n4
        BF[i] = -1*(i-n4)/(n4-n3)
    end

    # ---> Symmetries <---
    for k= Mb+2:N
        BF[k]= conj(BF[N-k+2])
    end
    
    #return the frequency band to build the wavelet
    return BF

end




