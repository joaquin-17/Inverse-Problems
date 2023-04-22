using PyPlot, FFTW, DSP, SeisProcessing, SeisPlot, LinearAlgebra, HDF5 #SeisReconstruction



include("/home/aacedo/Desktop/GitHub/Inverse-Problems/Final Project/Tools.jl")

include("CG.jl");
include("DFT.jl");
include("cond_number.jl")

#Create synthetic data:

dt= 0.002;
d = SeisLinearEvents(;nt=128, nx1=64,dx1=12.5, nx2=64,dx2=12.5,tau=[0.25,0.4],p1=[-0.00005,-0.00015],p2=[0.00005,0.00015]);
shot=d[:,64,:];
shot=shot*(10^2)
nt,nr=size(shot);
d_obs = copy(shot);

for i=1:nr
       p = rand()
            if   p < 0.6
                d_obs[:,i] .= 0.0
            end
end

  
S = CalculateSampling(d_obs);
d_obs = S.*shot; 



Ft=DFTmatrix(size(shot,1));
Fx=DFTmatrix(size(shot,2));
kronFxFt=kron(Fx,Ft); #Fx x Ft 

R=SamplingMatrix(r,type="cs")
T=diagm(ones(Nt));
kronRT= kron(R,T);
d=reshape(d_obs,length(d_obs)); # shot as a vector
A=kronRT*kronFxFt;
G=A'*A; 
ρ=0.0

λ,cn=k(G,ρ)


#=

# Do 1D Fourier
temp1=zeros(ComplexF64,size(shot));

for j=1:nr

    temp1[:,j]=Ft*d_obs[:,j]

end
=#


#=
# Do 1D Fourier
temp2=zeros(ComplexF64,size(shot));

for i=1:nt

    temp2[i,:]=Fx*temp1[i,:]

end
=#

