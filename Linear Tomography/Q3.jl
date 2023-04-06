
using DelimitedFiles, PyPlot, Statistics, LinearAlgebra

include("LinearTomography.jl")

tomo_data=readdlm("tomo_data.txt"); # data set
xs=tomo_data[:,1]; # x coordinates of sources
zs=tomo_data[:,2]; # z coordinates of sources
xr=tomo_data[:,3]; # x coordinates of receiver
zr=tomo_data[:,4]; # z coordinates of receiver
t=tomo_data[:,5]; # Travel times;


#Show how to compute sources adn receivers:

ns=50;
nr=50;
#Grid points:
grid=(100,100)
gz,gx= grid

#Vectors for sources and receivers!

sx=zeros(Float64,ns);
sz=LinRange(minimum(zs),maximum(zs),ns);

rx=300.0*ones(Float64,ns);
rz=LinRange(minimum(zr),maximum(zr),nr);

sources=[sx sz]; receivers=[rx rz];


D=DistanceMatrix(sources, receivers, grid)


#Velocity or Slowness initial model
s=1/1000; 

M0=s*ones(Float64,(grid[1],grid[2]));
m0=zeros(Float64,grid[1]*grid[2]);

#Vectorize the model
for ix=1:grid[2]
    for iz=1:grid[1]
        k=(ix-1)*size(M0,1) +iz
        m0[k]=M0[iz,ix]
    end
end




μ=10; # Define a new μ for problem 3.

u=Matrix(μ*I, gz*gx, gz*gx)

#Initial slowness model:
slo=reshape(m0,(gz,gx))
(Dx,Dz)=Derivatives(slo,1);  


#=
βx=100.0;
βz=βx;

Dxm0=βx*(Dx*m0);
Dzm0=βz*(Dz*m0);
#Build aumented forward:

A=vcat(D,u,βx*Dx,βz*Dz); #Concatenation of forward models. Left side of the equation.
y=vcat(t,μ*m0,Dxm0,Dzm0) #right side of the equation

#Inversion using CG:
nt=100;
m=cgaq(A,y,nt)
m=reshape(m,(gz,gx));
=#
#Examples to fit mu to the level of noise variance.


p =1*collect(-2:0.1:1.5);
β = 10.0 .^(p);;
misfit=zeros(Float64,length(β));
modelnorm=zeros(Float64,length(β));
chi2=zeros(Float64,length(β));


for i=1:length(β)
   

    #β .= 6.8;    
    
    Dxm0=β[i].*(Dx*m0);
    Dzm0=β[i].*(Dz*m0);
    A=vcat(D,u,β[i]*Dx,β[i]*Dz); 
    y=vcat(t,μ*m0,Dxm0,Dzm0); 

    nt=250;
    m=cgaq(A,y,nt)
    m=reshape(m,(gz,gx));


    tp=A[1:length(t),:]*m[:]; #t predicted
    σ=4e-4 #Noise variance;
    misfit[i]= (norm(t .- tp,2)).^2;
    modelnorm[i] = norm( m[:] .- m0 ,2).^2;
    chi2[i]=misfit[i]/(σ^2);


end




fig, axs = plt.subplots(2, 2)
axs[1, 1].plot(β,chi2,c="r");
axs[1,1].plot(β,chi2,"ko");
#axs[1,1].set_title("χ² vs β",)
axs[1,1].set_xlabel("β",)
axs[1,1].set_ylabel("χ² ≈ N")
axs[1,1].grid("True")

axs[1,2].plot(β,misfit,c="b");
axs[1,2].plot(β,misfit,"ko");
#axs[1,2].set_title("|| t⃗ - t⃗ₚ ||² vs β",)
axs[1,2].set_xlabel("β",)
axs[1,2].set_ylabel("||t⃗ - t⃗ₚ ||²")
axs[1,2].grid("True")

axs[2,1].plot(β,modelnorm,c="g");
axs[2,1].plot(β,modelnorm,"ko");
#axs[1,3].set_title("|| t⃗ - t⃗ₚ ||² vs β",)
axs[2,1].set_xlabel("β",)
axs[2,1].set_ylabel("|| m⃗ - m⃗₀ ||²")
axs[2,1].ticklabel_format(style="sci", axis="y")
axs[2,1].grid("True")

axs[2,2].plot(modelnorm,misfit,c="Purple");
axs[2,2].plot(modelnorm,misfit,"ko");
#axs[1,3].set_title("|| t⃗ - t⃗ₚ ||² vs β",)
axs[2,2].set_xlabel("|| m⃗ - m⃗₀ ||²",)
axs[2,2].set_ylabel("||t⃗ - t⃗ₚ ||²")
axs[2,2].ticklabel_format(style="sci", axis="y")
axs[2,2].grid("True")


tight_layout()


