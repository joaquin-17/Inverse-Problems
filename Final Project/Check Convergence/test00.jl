
using PyPlot, FFTW, DSP, SeisProcessing, SeisPlot, LinearAlgebra, HDF5, Printf #SeisReconstruction

include("/home/aacedo/Desktop/GitHub/Inverse-Problems/Final Project/Tools.jl")


include("CG_files.jl");

function MatrixMultiplyOp(in,adj;matrix=1)

	if (adj)
		out = matrix'*in
	else
		out = matrix*in
	end

	return out
end


m=40; n=30;
A=randn(m,n); 
x=randn(n); 
y=A*x;
ρ=1.2;




m0 = zeros(Float64,size(x));
operators=[MatrixMultiplyOp];
parameters= [Dict(:matrix =>A)];


xls= (A'*A + ρ*diagm(n,n))\A'*y 

xcg, Jcg= CG(A,y; x0=m0, Ni=50, tol=1.0e-15)



#xcgls1,Jcgls1=CGLS(y, operators,parameters, x0=m0, μ=ρ, Ni=200, tol=1.0e-15)

#xcgls2, Jcgls2=ConjugateGradients(y,operators,parameters,x0=m0,Ni=200,μ=ρ,tol=1.0e-15)


#xcgls3, Jcgls3= ConjugateGradients2(y,operators,parameters,;Niter=200,mu=ρ,tol=1.0e-15)