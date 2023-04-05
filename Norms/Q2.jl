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
β=6.8;


Dxm0=βx*(Dx*m0);
Dzm0=βz*(Dz*m0);
nt=35
β= 100;
A=vcat(D,u,β*Dx,β*Dz); #Concatenation of forward models. Left side of the equation.
y=vcat(t,μ*m0,Dxm0,Dzm0); #right side of the equation
m=cgaq(A,y,nt)
m=reshape(m,(gz,gx));
v= 1 ./ m;


# Promote sparsity in the model derivatives
#δ=1.0; nt=100; Ne=10;ϵ=0.004;

function IRLSCG(t::Vector,D::Matrix,m0::Vector; ϵ::Real=1e-9, β::Real=25.0 , δ::Real=1.0,Ne::Int=15, Ni::Int=25)
    

    m=m0;
    slo=reshape(m,(gz,gx))
    (Dx,Dz)=Derivatives(slo,1); 
   # β= 25.0
    λ=(2*β)/(δ^2)
    Dxm0=β*(Dx*m);
    Dzm0=β*(Dz*m);
    wx=zeros(Float64,length(Dxm0));
    wz=zeros(Float64,length(Dzm0));
    h=zeros(Float64,length(Dxm0));
    v=zeros(Float64,length(Dzm0));
   # βDx= β*Dx;
    #βDz= β*Dz;

    for i=1:Ne
        h=Dx*m; v=Dz*m;
        wx=(1 ./ (1  .+ (h/δ).^2 .+ ϵ ));
        wz=(1 ./(1 .+ (v/δ).^2  .+ ϵ ));
        #wx=wx./maximum(wx); wz=wz./maximum(wz);
        println(wz);
        A=vcat(D, λ*wx.*Dx, λ*wz.*Dz); #Concatenation of forward models. Left side of the equation.
        y=vcat(t,Dxm0,Dzm0); #right side of the equation
        #m=(A'*A + λ*Dx'*Wx*Dx + λ*Dz'*Wz*Dz)\(A'*y);
        m=cgaq(A,y,Ni)
        println("Outer loop i=$i")
    end

    return m, wx,wz
end


msol, wx, wz=IRLSCG(t,D,m0,β=25,δ=0.0001,Ne=10);
#dh=reshape(dh,(gz,gx-1));
#dv= reshape(dv,(gz-1,gx));
msol=reshape(msol,(gz,gx));
vsol = 1 ./ msol;



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




figure(2, figsize=(10,20))
subplot(121);
title("Slowness, Inversion with CG and SMF ", fontsize=15)
imshow(msol, extent=[0.0, 300.0, 400.0, 0.0], cmap="jet", interpolation="bilinear")
xlabel("x[m]", fontsize=13)
ylabel("z[m]", fontsize=13)
colorbar(shrink=0.65, orientation="vertical", label="[s/m]")
subplot(122);
title("Velocity Inversion with CG and SMF  ", fontsize=15)
imshow(vsol,extent=[0.0, 300.0, 400.0,0.0] ,cmap="jet", interpolation="bilinear")
xlabel("x[m]", fontsize=13)
ylabel("z[m]", fontsize=13)
colorbar(shrink=0.65, orientation="vertical", label="[m/s]")
tight_layout()
#=

#Examples to fit mu to the level of noise variance.


p =1*collect(-2:0.1:2);
β = 10.0 .^(p);
misfit=zeros(Float64,length(β));
modelnorm=zeros(Float64,length(β));
chi2=zeros(Float64,length(β));
#k=0;

for i=1:length(β)
   

    msol=IRLSCG(t,D,m0,β= β[i],δ=0.0001,Ne=5,Ni=100);
    #dh=reshape(dh,(gz,gx-1));
    #dv= reshape(dv,(gz-1,gx));
    #msol=reshape(msol,(gz,gx));
    #vsol = 1 ./ msol;


    tp=A[1:length(t),:]*msol[:]; #t predicted
    σ=4e-4 #Noise variance;
    σ=4e-4 #Noise variance;
    misfit[i]= (norm(t .- tp,2)).^2;
    modelnorm[i] = norm( msol[:] .- m0 ,2).^2;
    chi2[i]=misfit[i]/(σ^2);
    #k= k +1
   # print("β=$k")
end

=#

#=
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
=#