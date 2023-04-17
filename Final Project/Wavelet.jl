
using DSP # Replace this with my own hamming functions to avoid load the namespace.


"""
            Wavelet(;FB::Vector, ϕ::Real= 0.0 ,taper::Vector= hamming(length(FB)))
            
            
Function to model a wavelets to use into the DSA forward modeling. For now, only a generic wavelet
is provided built from a frequency band vector. Also the phase rotation and the taper can be provided.

This function it is used inside the SourceOperator function.

# Keywords arguments:
-`FB`:: Vector : Frequency band for modeling the different types of sources.
-`ϕ`:: Real= 0.0 : Given Phase rotation for the wavelet.
-`taper` :: Vector = hamming(length(FB))

"""
function Wavelet(;FB::Vector, ϕ::Real= 0.0 ,taper::Vector= hamming(length(FB)))
 

N=length((FB))
M= (N ÷ 2) +1;
Hilbert =zeros(ComplexF64,N)
Hilbert[1] = 0                       
Hilbert[2:M] = FB[2:M]*1im        
Hilbert[M] = 0.0                     
Hilbert[M+1:N] = FB[M+1:N]*(-1im) 
    
w=real(ifft(FB))
h = real(ifft(Hilbert));
    
rotation = (ϕ)*(π/180)
w = cos(rotation)*w + sin(rotation)*h
taper=hamming(length(w));
    
w= fftshift(w).*taper;
W=fft(w);
#should I normalize my wavelet?
#w= w/maximum(w);
#W= W/maximum(abs.(W))


return W;
end
