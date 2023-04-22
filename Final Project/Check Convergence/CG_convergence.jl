#CG_convergence.jl

using PyPlot, FFTW, DSP, SeisProcessing, SeisPlot, LinearAlgebra, HDF5 #SeisReconstruction

include("/home/aacedo/Desktop/GitHub/Inverse-Problems/Final Project/Tools.jl")


include("CGLS-ADMM.jl"); #include("ADMM.jl")

#println("1) load data and generate observed data")

#=
data = h5read("BB_ricker15.0.h5", "BB_ricker15.0");


ns, nr, nt= size(data);
dt=0.001;
fs=1/dt;
df= fs/nt;
f=df*collect(0:1:nt-1);
line2D=zeros(Float32,(nt,nr,ns));

#Fill line2D: 
for ishots in 1:ns
    aux=transpose(data[ishots,:,:])
    line2D[:,:,ishots]= aux;
end


line2D=line2D[:,1:512,:]
shot=line2D[:,:,256];
shot= shot*(10^9) # Increase the amplitudes
dobs=copy(shot)
=#



dt= 0.002;
d = SeisLinearEvents(;nt=250, nx1=128,dx1=12.5, nx2=128,dx2=12.5,tau=[0.25,0.4],p1=[-0.00005,-0.00015],p2=[0.00005,0.00015]);
shot=d[:,64,:];
shot=shot*(10^2)

nt,nr=size(shot);
d_obs = copy(shot);


for i=1:nr
     #for i2=1:nx2
       p = rand()
            if   p < 0.5
                d_obs[:,i] .= 0.0
            end
   # end
end

  
S = CalculateSampling(d_obs);
d_obs = S.*shot; 
#d_obs = SeisAddNoise(d_obs,0.5,L=5);




#println("3) Get parameters:")

m0 = zeros(Float64,size(shot)); #      ,spec_size(dobs,patch_size,Noverlap));
operators=[WeightingOp, FFTOp];
parameters= [Dict(:w =>S),Dict(:normalize=>true)];


m, Ji, Je= ADMM_CGLS(m0,d_obs,operators,parameters, ρ=1.5, μ= 6.0, Ni=25, Ne=1,tol=1.0e-8)


d_rec=real(FFTOp(m,false));


clf()
figure(1)
subplot(131);
SeisPlotTX(shot,fignum=1); colorbar()
subplot(132);
SeisPlotTX(d_obs, fignum=1); colorbar()
subplot(133)
SeisPlotTX(d_rec, fignum=1); colorbar()


figure(2)
subplot(131);
SeisPlotFK(shot,dy=dt,fignum=2); colorbar()
subplot(132);
SeisPlotFK(d_obs,dy=dt, fignum=2); colorbar()
subplot(133)
SeisPlotFK(d_rec,dy=dt, fignum=2); colorbar()