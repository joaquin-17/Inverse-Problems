using Plots, PyPlot, LinearAlgebra, SparseArrays, Statistics, DelimitedFiles

include("FunctionsLinTomo.jl")

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

#L=sparse(Is,Js,Vs);
#Building the traveltimes vector
#(hf,vf)=diffaq(slo,1);                         #FLATNESS
#(hs,vs)=diffaq(slo,2);                       #SMOOTHNESS
###################################################################################################
#                                    TRAVEL TIME calculator                                       #
###################################################################################################
#tt=zeros(ns*nr);
#tt=L*sarr                                          #Forward Model
#mu=0.00001;
#u=Matrix(mu*I,size(L))
#L=L.+u
#u=Matrix(mu*I, gz*gx, gz*gx)
#########################         #Solution by CG
#Adding noise to the travel times 
#function noise_tt(tt)
#    dist=randn(Float16,length(tt))
#    dist_tt=(dist.*std(tt)).+mean(tt)
#    ntt=tt.+(0.01*(dist_tt))
#    return ntt
#end
#ntt=noise_tt(tt) 
#muh=1; muv=1;
#LF=vcat(L,muh.*hf,muv.*vf); tf=spzeros(length(LF[:,1])-length(ntt)); tf=vcat(ntt,tf)
#LS=vcat(L,muh.*hs,muv.*vs); ts=spzeros(length(LS[:,1])-length(ntt)); ts=vcat(ntt,ts)
tomofile=readdlm("tomo_data.txt")
tt=tomofile[:,5]
m0=fill(1/1000,gz*gx)
mu=0.00001
u=Matrix(mu*I, gz*gx, gz*gx)
@time begin
minv=(inv(L'*L.+u))*(L'*tt+mu.*m0)
end
minvres=reshape(minv,(gz,gx))


tmodel=L*minv

dmisfit=sum((tt.-tmodel).^2)/length(tt)
println("dmisfit: ",dmisfit)

modnorm=sum((minv.-m0).^2)/length(minv)
println("modnorm: ",modnorm)


#chisq=(sum((tt.-tmodel).^2))/(4e-8)


#println(chisq)
#heatmap(sin)
#sinv0=cgaq(L,tt)
#sinv0=reshape(sinv0,(gz,gx));
#sinv1=cgaq(LF,tf)
#sinv1=reshape(sinv1,(gz,gx));
#sinv2=cgaq(LS,ts)
#sinv2=reshape(sinv2,(gz,gx));
#########################################################################################################
#                                               PLOT

#clf()

#subplot(121); title("DAMPED-LS (mu=0.001)"); imshow(minvres1.^(-1), cmap="gray_r",extent=[0,300,400,0]); cbar=colorbar(fraction=0.046, pad=0.04); xlabel("Distance(m)");
#ylabel("Depth(m)");cbar.set_label("Velocity (m/s)");
#subplot(122); title("DAMPED-LS (mu=100)"); imshow(minvres2.^(-1), cmap="gray_r",extent=[0,300,400,0]); cbar=colorbar(fraction=0.046, pad=0.04); xlabel("Distance(m)");
#ylabel("Depth(m)");cbar.set_label("Velocity (m/s)");tight_layout()
#subplot(132); title("FLATNESS"); imshow(sinv1, cmap="gray_r", vmin=0.25, vmax=1); colorbar()
#subplot(133); title("SMOOTHNESS"); imshow(sinv2, cmap="gray_r", vmin=0.25, vmax=1); colorbar()
#gcf()

