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
#δt = t .- tm;


#μ=[0.001,0.01,0.1,1.0,10.0,100.0,200.0,400.0,600.0];
p =1*collect(-2:0.1:3.5);
μ = 10.0 .^(p);;

misfit=zeros(Float64,length(μ));
modelnorm=zeros(Float64,length(μ));
chi2=zeros(Float64,length(μ));

for i=1:length(μ)
    
    m,v=DLS(D,t,μ[i],m0,grid[2],grid[1]);

    tp=D*m[:]; #t predicted

    σ=4e-4 #Noise variance;
    misfit[i]= (norm(t .- tp,2)).^2
    modelnorm[i] = norm( m[:] .- m0 ,2).^2
    chi2[i]=misfit[i]/(σ^2)

end




fig, axs = plt.subplots(2, 2)
axs[1, 1].plot(μ,chi2,c="r");
axs[1,1].plot(μ,chi2,"ko");
#axs[1,1].set_title("χ² vs μ",)
axs[1,1].set_xlabel("μ",)
axs[1,1].set_ylabel("χ² ≈ N")
axs[1,1].grid("True")

axs[1,2].plot(μ,misfit,c="b");
axs[1,2].plot(μ,misfit,"ko");
#axs[1,2].set_title("|| t⃗ - t⃗ₚ ||² vs μ",)
axs[1,2].set_xlabel("μ",)
axs[1,2].set_ylabel("||t⃗ - t⃗ₚ ||²")
axs[1,2].grid("True")


axs[2,1].plot(μ,modelnorm,c="g");
axs[2,1].plot(μ,modelnorm,"ko");
#axs[1,3].set_title("|| t⃗ - t⃗ₚ ||² vs μ",)
axs[2,1].set_xlabel("μ",)
axs[2,1].set_ylabel("|| m⃗ - m⃗₀ ||²")
axs[2,1].ticklabel_format(style="sci", axis="y")
axs[2,1].grid("True")

axs[2,2].plot(modelnorm,misfit,c="Purple");
axs[2,2].plot(modelnorm,misfit,"ko");
#axs[1,3].set_title("|| t⃗ - t⃗ₚ ||² vs μ",)
axs[2,2].set_xlabel("|| m⃗ - m⃗₀ ||²",)
axs[2,2].set_ylabel("||t⃗ - t⃗ₚ ||²")
axs[2,2].ticklabel_format(style="sci", axis="y")
axs[2,2].grid("True")
tight_layout()