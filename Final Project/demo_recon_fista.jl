# Reconstruction Demo

using PyPlot, FFTW, DSP, SeisProcessing, SeisPlot, LinearAlgebra,SeisReconstruction


include("FISTA.jl")



println("1) Generate synthetic data:")

dt= 0.002;

#d=SeisParabEvents(tau=[0.256, 0.512, 0.768] ,amp=[1.0, 0.5, -1.0], p2=[0.45,0.0, -0.45], 
#p1=[0.015,0.0, -0.015], dt=dt, nt=512,  
#dx1=10.0, nx1=128, dx2=10.0, nx2=128, f0= 25.0 );


d = SeisLinearEvents(;nt=250, nx1=128,dx1=12.5, nx2=128,dx2=12.5,tau=[0.25,0.4],p1=[-0.0001,-0.00015],p2=[0.0001,0.00015]);

shot=d;

shot=d[:,32,:];

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
x0=randn(size(dobs));
λ=0.1;
Ne=100;


println("4) Reconstruction of the data: Inversion of the coefficients")


m1, J = ISTA(x0,dobs,operators,parameters,μ=λ,Ni=Ne,tolerance=0.00005)
d_rec1= real(FFTOp(m1,false));



m2, J = FISTA(x0,dobs,operators,parameters,λ,Ne,0.0001);
d_rec2= real(FFTOp(m2,false));
