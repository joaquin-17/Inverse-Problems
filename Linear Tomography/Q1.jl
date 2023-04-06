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

tm = D*m0;
δt = t .- tm;


μ=2;

m,v=DLS(D,t,μ,m0,grid[2],grid[1]);

tp=D*m[:]; #t predicted

σ=4e-4 #Noise variance;
misfit= (norm(t .- tp,2)).^2
modelnorm = norm( m[:] .- m0 ,2).^2
chi2=misfit/(σ^2)



clf()
figure(1, figsize=(10,15))
subplot(121);
title("Slowness", fontsize=15)
imshow(m, extent=[0.0, 300.0, 400.0, 0.0], cmap="jet", interpolation="bilinear")
xlabel("x [m]", fontsize=13)
ylabel("z [m]", fontsize=13)
colorbar(shrink=0.65, orientation="vertical", label="[s/m]")
subplot(122);
title("Velocity", fontsize=15)
imshow(v,extent=[0.0, 300.0, 400.0,0.0] ,cmap="jet", interpolation="bilinear")
xlabel("x [m]", fontsize=13)
ylabel("z [m]", fontsize=13)
colorbar(shrink=0.65, orientation="vertical", label="[m/s]")
tight_layout()






#gcf()
#println("misfit:")
#println( misfit)
#println("modelnorm:")
#println( modelnorm)
#println("chi2:")
#println( chi2)


#axs[1, 1].plot(μ[1:50],chi2[1:50],c="r");
#axs[1,1].plot(μ[1:50],chi2[1:50],"ko");
#axs[1,1].set_title("χ² vs μ",)
#axs[1,1].set_xlabel("μ")
#axs[1,1].set_ylabel("χ² ≈ N")
#axs[1,1].grid("True")
