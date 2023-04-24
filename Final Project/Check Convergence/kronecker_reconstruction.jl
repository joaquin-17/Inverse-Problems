using PyPlot, FFTW, DSP, SeisProcessing, SeisPlot, LinearAlgebra, HDF5 #SeisReconstruction



include("/home/aacedo/Desktop/GitHub/Inverse-Problems/Final Project/Tools.jl")

include("CG.jl");
include("DFT.jl");
include("SamplingMatrix.jl")
include("cond_number.jl")

#Create synthetic data:

dt= 0.002;
d = SeisLinearEvents(;nt=128, nx1=64,dx1=12.5, nx2=64,dx2=12.5,tau=[0.25,0.4],p1=[-0.00015,-0.00015],p2=[0.00015,0.00015]);
shot=d[:,64,:];
shot=shot*(10^2)
nt,nr=size(shot);
d_obs = copy(shot);

for i=1:nr
       p = rand()
            if   p < 0.5
                d_obs[:,i] .= 0.0
            end
end

  
S = CalculateSampling(d_obs);
d_obs = S.*shot; 


#for j=1:4:size(d_obs,2)
 #   d_obs[:,j] .= 0.0
#end


r=SamplingVector(d_obs[64,:])

Ft=DFTmatrix(size(shot,1));
Fx=DFTmatrix(size(shot,2));
kronFxFt=kron(Fx,Ft); #Fx x Ft 

R=SamplingMatrix(r,type="seismic")
Nt=size(shot,1);
T=diagm(ones(Nt));
kronRT= kron(R,T);
d=reshape(d_obs,length(d_obs)); # shot as a vector
A=kronRT*kronFxFt;
y=A*d;


#G=A'*A; 
#ρ=1.0
#λ,cn=k(G,ρ)

#xls= (G+ ρ*diagm(ones(size(G,1))))\ A'*y;
#m0=zeros(length(d_obs));
#z=copy(m0);
#Id = diagm(ones(size(A,2)));
#Ac= vcat(A,sqrt(ρ)*Id);
#yc=vcat(y, sqrt(ρ)* (z))
#xcg, Jcg= CG(Ac,yc; x0=m0, Ni=50, tol=1.0e-15)

m, J=  ADMM(A,y; x0= 0.0, ρ= 1.0, λ=0.5, Ni=150, Ne=50, tol=1.0e-8)


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

