using SeisPlot, PyPlot, SeisProcessing,LinearAlgebra, DelimitedFiles, HDF5, FFTW

#Examples to:


# Define criterias to convergence and tolerance for the lagorithms.
#  Check what happen with different iterations of CG inside ADMM.
# Plots the different costs functions and evaluate


include("ADMM.jl")
include("Tools.jl")
include("LocalFFTOp3d-padding.jl")



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

nt,nr=size(shot)
for i=1:nr
     #for i2=1:nx2
       p = rand()
            if   p < 0.3
                dobs[:,i] .= 0.0
            end
   # end
end


  
S = CalculateSampling(dobs);
dobs = S.*shot; 
patch_size=(64,32,32); #Patch size in LocalFourier Operator
Noverlap=(32,16,16); #Overlap of patches in LocalFourier Operator
dims=size(shot); #Parameter to  ensure right dimensions with the LocalFourier transform
    


println("3) Get parameters:")

x0 = randn(Float64,size(shot)); #      ,spec_size(dobs,patch_size,Noverlap));
operators=[WeightingOp, FFTOp];
parameters= [Dict(:w =>S),Dict(:normalize=>true)];

println("4) Reconstruction of the data: Inversion of the coefficients")

m, J , Ji= ADMM(x0,dobs,operators,parameters, ρ= 1.0, μ= 1.25,Ne=25, Ni=10,tolerance=1.0e-10);
d_rec=FFTOp(m,false);
