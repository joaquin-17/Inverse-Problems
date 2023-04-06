
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


D, is, js, vs=DistanceMatrix(sources, receivers, grid)
Ds=sparse(is,js,vs)

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



βx= 6.8;
βz=βx;

Dxm0=βx*(Dx*m0);
Dzm0=βz*(Dz*m0);
#Build aumented forward:

A=vcat(D,u,βx*Dx,βz*Dz); #Concatenation of forward models. Left side of the equation.
y=vcat(t,μ*m0,Dxm0,Dzm0) #right side of the equation

As=sparse(A);
ys=sparse(y);

#Inversion using CG:
nt=100;
println("CG no sparsity")
@time m=cgaq(A,y,nt)
println("CG sparsity")
 @time m=cgaq(As,ys,nt)
m=reshape(m,(gz,gx));
m=reshape(m,(gz,gx));

v = 1 ./ m;


tp=A[1:length(t),:]*m[:]; #t predicted
σ=4e-4 #Noise variance;
misfit= (norm(t .- tp,2)).^2;
modelnorm = norm( m[:] .- m0 ,2).^2;
chi2=misfit/(σ^2);



clf()
figure(1, figsize=(10,20))
subplot(121);
title("Slowness, Inversion with CG and SMF ", fontsize=15)
imshow(m, extent=[0.0, 300.0, 400.0, 0.0], cmap="jet", interpolation="bilinear")
xlabel("x[m]", fontsize=13)
ylabel("z[m]", fontsize=13)
colorbar(shrink=0.65, orientation="vertical", label="[s/m]")
subplot(122);
title("Velocity Inversion with CG and SMF  ", fontsize=15)
imshow(v,extent=[0.0, 300.0, 400.0,0.0] ,cmap="jet", interpolation="bilinear")
xlabel("x[m]", fontsize=13)
ylabel("z[m]", fontsize=13)
colorbar(shrink=0.65, orientation="vertical", label="[m/s]")
tight_layout()