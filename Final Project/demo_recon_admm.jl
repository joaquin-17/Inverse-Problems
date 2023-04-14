# Reconstruction Demo

using PyPlot, FFTW, DSP, SeisProcessing, SeisPlot, LinearAlgebra,SeisReconstruction


include("ADMM.jl")



println("1) Generate synthetic data:")

dt= 0.002;

d = SeisLinearEvents(;nt=250, nx1=128,dx1=12.5, nx2=128,dx2=12.5,tau=[0.25,0.4],p1=[-0.0001,-0.00015],p2=[0.0001,0.00015]);

shot=d;

shot=d[:,64,:];

dobs = copy(shot);

nt,nx1=size(d)
for i1=1:nx1
    #for i2=1:nx2
       p = rand()
            if   p < 0.5
                dobs[:,i1] .= 0.0
            end
   # end
end


S = CalculateSampling(dobs);
dobs = S.*shot; 


println("3) Get parameters:")


operators=[WeightingOp, FFTOp];
parameters=[Dict(:w=>S), Dict(:normalize=>true)];


println("4) Reconstruction of the data: Inversion of the coefficients")

m, J = ADMM_CGLS(dobs,operators,parameters, ρ= 0.5, μ= 1.5,Ne=150,tolerance=1.0e-4);

d_rec= real(FFTOp(m,false));