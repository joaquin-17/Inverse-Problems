using PyPlot, FFTW, DSP, SeisProcessing, SeisPlot, LinearAlgebra, HDF5 #SeisReconstruction



#include("/home/aacedo/Desktop/GitHub/Inverse-Problems/Final Project/Tools.jl")
include("C:\\Users\\Joaquin\\Desktop\\Research\\Codes\\Inverse-Problems\\Final Project\\Tools.jl")


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


#=

for i=1:nr
       p = rand()
            if   p < 0.5
                d_obs[:,i] .= 0.0
            end
end

  
S = CalculateSampling(d_obs);
d_obs = S.*shot; 
=#


for j=1:4:size(d_obs,2)
    d_obs[:,j].= 0.0
end

d_obs[:,15:32] .= 0.0


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
y=kronRT*d;
ρ=1.0
G=A*A'; 


#=
ρ=[0.0, 0.001, 0.01, 0.1, 1,10, 100]
λ1,cn1=k(G,ρ[1]);
λ2,cn2=k(G,ρ[2]);
λ3,cn3=k(G,ρ[3]);
λ4,cn4=k(G,ρ[4]);
λ5,cn5=k(G,ρ[5]);
λ6,cn6=k(G,ρ[6]);
λ7,cn7=k(G,ρ[7]);
#λp5,cn=k(G,0.0);
cn=[cn1,cn2,cn3,cn4,cn5,cn6,cn7]



figure(1, figsize=(10,5));

plot(sort(λ1,rev=true),label="ρ=0.0");
plot(sort(λ2,rev=true),label="ρ=0.001");
plot(sort(λ3,rev=true),label="ρ=0.01");
plot(sort(λ4,rev=true),label="ρ=0.1");;
plot(sort(λ5,rev=true),label="ρ=1.0");;
plot(sort(λ6,rev=true),label="ρ=10.0");
plot(sort(λ7,rev=true),label="ρ=100.0");

grid("True");
ylabel("λ",fontsize=15);
xlabel("n",fontsize=15);
#ylim([-10, 150])
legend()



figure(2,figsize=(8,5));

plot( ρ[1],round(cn1,digits=3),"o",label="ρ=0.0");
plot(ρ[2],round(cn2,digits=3),"o",label="ρ=0.001");
plot(ρ[3],round(cn3,digits=3),"o",label="ρ=0.01");
plot(ρ[4],round(cn4,digits=3),"o",label="ρ=0.1");;
plot(ρ[5],round(cn5,digits=3),"o",label="ρ=1.0");;
plot(ρ[6],round(cn6,digits=3),"o",label="ρ=10.0");
plot(ρ[7],round(cn7,digits=3),"o",label="ρ=100.0");
#plot(ρ,round.(cn,digits=5),"o")
xlabel("ρ",fontsize=15)
ylabel("κ(ρ)",fontsize=15)
grid("True")
legend()
=#


#xls= (G+ ρ*diagm(ones(size(G,1))))\ A'*y;
#m0=zeros(length(d_obs));
#z=copy(m0);
#Id = diagm(ones(size(A,2)));
#Ac= vcat(A,sqrt(ρ)*Id);
#yc=vcat(y, sqrt(ρ)* (z))

xcg, Jcg= CG(Ac,yc; x0=m0, Ni=50, tol=1.0e-15)
m, J=  ADMM(A,y; x0= 0.0, ρ= 3.0, λ=6.0, Ni=150, Ne=50, tol=1.0e-8)

aux=kronFxFt'*m;
d_rec=real(reshape(aux,size(shot)));
SeisPlotTX(d_rec)

#m=reshape(m, size(shot));
=#

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

