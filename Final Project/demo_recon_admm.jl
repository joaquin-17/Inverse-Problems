# Reconstruction Demo

using PyPlot, FFTW, DSP, SeisProcessing, SeisPlot, LinearAlgebra #SeisReconstruction


include("ADMM.jl")
include("Tools.jl")
include("LocalFFTOp3d-padding.jl")



println("1) Generate synthetic data:")

dt= 0.002;
#d = SeisLinearEvents(;nt=250, nx1=128,dx1=12.5, nx2=128,dx2=12.5,tau=[0.25,0.4],p1=[-0.0001,-0.00015],p2=[0.0001,0.00015]);
#shot=d;
#shot=d[:,64,:];

d=SeisParabEvents(tau=[0.256, 0.512, 0.768] ,amp=[1.0, 0.5, -1.0], p2=[0.45,0.0, -0.45], 
p1=[0.015,0.0, -0.015], dt=dt, nt=512,  
dx1=10.0, nx1=128, dx2=10.0, nx2=128, f0= 25.0 );

dobs = copy(d);

#shot=d;
#shot=d[:,32,:];

nt,nx1,nx2=size(d)
for i1=1:nx1
    for i2=1:nx2
       p = rand()
            if   p < 0.5
                dobs[:,i1,i2] .= 0.0
            end
    end
end

    
    
S = CalculateSampling(dobs);
dobs = S.*d; 
patch_size=(64,32,32); #Patch size in LocalFourier Operator
Noverlap=(32,16,16); #Overlap of patches in LocalFourier Operator
dims=size(d); #Parameter to  ensure right dimensions with the LocalFourier transform
    


println("3) Get parameters:")

x0 = zeros(Float64,spec_size(dobs,patch_size,Noverlap));
operators=[WeightingOp, LocalFFTOp];
parameters= [Dict(:w =>S),
Dict(:patch_size=>patch_size, :Noverlap=>Noverlap, :dims=>dims, :normalize=>true, :padd=>false)];

println("4) Reconstruction of the data: Inversion of the coefficients")

m, J = ADMM(x0,dobs,operators,parameters, ρ= 0.5, μ= 1.5,Ne=25,tolerance=1.0e-4);
d_rec= real(LocalFFTOp(m,false; patch_size, Noverlap, dims, normalize=true, padd=true));
