# Reconstruction Demo

using PyPlot, FFTW, DSP, SeisProcessing, SeisPlot, LinearAlgebra #SeisReconstruction


include("ADMM.jl")
include("Tools.jl")
include("LocalFFTOp3d-padding.jl")



println("1) Generate synthetic data:")
dt= 0.002;

d=SeisParabEvents(tau=[0.256, 0.512, 0.768] ,amp=[1.0, 0.5, -1.0], p2=[0.3,0.0, -0.45], 
p1=[0.15,0.0, -0.15], dt=dt, nt=512,  
dx1=10.0, nx1=64, dx2=10.0, nx2=64, f0= 20.0 );


#d = SeisLinearEvents(;nt=512, nx1=64 ,dx1=12.5, nx2=64,dx2=12.5,
#tau=[0.15,0.25,0.4],p1=[-0.0001,-0.00015,-0.0002],p2=[0.0001,0.00015,0.0002]);

#shot=d;
#shot=d[:,32,:];
dobs = copy(d);
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
patch_size=(32,16,16); #Patch size in LocalFourier Operator
Noverlap=(16,8,8); #Overlap of patches in LocalFourier Operator
dims=size(d); #Parameter to  ensure right dimensions with the LocalFourier transform
    


println("3) Get parameters:")

x0 = zeros(Float64,spec_size(dobs,patch_size,Noverlap));
operators=[WeightingOp, LocalFFTOp];
parameters= [Dict(:w =>S),
Dict(:patch_size=>patch_size, :Noverlap=>Noverlap, :dims=>dims, :normalize=>true, :padd=>false)];

#PARAMETERS FOR FFT
#x0 = randn(Float64,size(d)); #      ,
#operators=[WeightingOp, FFTOp];
#parameters= [Dict(:w =>S),Dict(:normalize=>true)];

println("4) Reconstruction of the data: Inversion of the coefficients")

m, J= ADMM(x0,dobs,operators,parameters, ρ= 0.25, μ= 2.5,Ne=25, Ni=15,tolerance=1.0e-4);
d_rec= real(LocalFFTOp(m,false; patch_size, Noverlap, dims, normalize=true, padd=true));
diff= d .- d_rec;


println("4) Reconstruction of the data: Inversion of the coefficients")

figure(1);
subplot(131);
SeisPlotTX(d[:,54,:],dy=0.002, cmap="gist_gray", fignum=1, vmin=minimum(d), vmax=maximum(d));  colorbar()
subplot(132);
SeisPlotTX(d_rec[:,54,:],dy=0.002, cmap="gist_gray", fignum=1, vmin=minimum(d), vmax=maximum(d));  colorbar()
subplot(133);
SeisPlotTX(diff[:,54,:],dy=0.002, cmap="gist_gray", fignum=1, vmin=minimum(d), vmax=maximum(d));  colorbar()
gcf()


figure(2);

subplot(131);
SeisPlotFK(d[:,54,:],dy=0.002, fignum=2); colorbar()
subplot(132);
SeisPlotFK(d_rec[:,54,:],dy=0.002, fignum=2) ;colorbar();
subplot(133);
SeisPlotFK(diff[:,54,:],dy=0.002, fignum=2);  colorbar();
gcf()