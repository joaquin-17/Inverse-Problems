using PyPlot, LinearAlgebra, SparseArrays, Statistics, DelimitedFiles

include("FunctionsLinTomo.jl")


tomo_data=readdlm("tomo_data.txt"); # data set


gx=100; gz=100; scrd=0; rcrd=300; ymax=400

nel=50; 
xrwell=fill(rcrd,nel); xswell=fill(scrd,nel);
yrwell=LinRange(1.25,399.25,nel); yswell=LinRange(0.6,399,nel);
rwell=[xrwell yrwell]; swell=[xswell yswell];            #OJO changing data type

#Grid definition

xgrid=LinRange(0,300,gx+1);
dx=xgrid[2]-xgrid[1]
xgrid=collect(xgrid[2:end-1])
ygrid=LinRange(0,400,gz+1);
dz=ygrid[2]-ygrid[1]
ygrid=collect(ygrid[2:end-1])

L=L_build(rwell,swell,gx,gz,scrd,rcrd,xgrid,ygrid,ymax,dx,dz)
#(Js,Is,Vs)=L_build(rwell,swell,gx,gz,scrd,rcrd,xgrid,ygrid,ymax,dx,dz)
#Lsparse=sparse(Is,Js,Vs);

#tomofile=readdlm("tomo_data.txt")
tt=tomo_data[:,5]
m0=fill(1/1000,gz*gx)
mu=10
u=Matrix(mu*I, gz*gx, gz*gx)
#@time begin
#minv=(inv(L'*L.+u))*(L'*tt+mu.*m0)
#end
#minvres1=reshape(minv,(gz,gx))
slo=reshape(m0,(gz,gx))
(hf,vf)=diffaq(slo,1);
muh=muv=6.8;                         #FLATNESS
dhm0=muh.*(hf*m0)
dvm0=muv.*(hf*m0)
LF=sparse(vcat(L,u,muh.*hf,muv.*vf));
tf=sparse(vcat(tt,mu.*m0,dhm0,dvm0));
#LF=vcat(L,u,muh.*hf,muv.*vf);
#tf=vcat(tt,mu.*m0,dhm0,dvm0)
@time begin
minvder, error=cgaq(LF,tf,250)
end

m=minvder;
m=reshape(m,(gz,gx));s

v=1 ./m;

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