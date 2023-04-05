
using FFTW, LinearAlgebra, Statistics, DelimitedFiles, PyPlot

data=readdlm("/home/aacedo/Desktop/GEOPH531/Inverse-Problems/Fourier Reconstruction/data/data_to_reconstruct.txt");
t=data[:,1]; s_real=data[:,2]; s_imag=data[:,3];

signal= s_real .+ im*s_imag;

function DFT_matrix(N::Int64)
    
    #Check that n is equal to kernel

    k=N;
    n=N;

    F=zeros(ComplexF64,(k,n));
    
        
        for k=1:k
            for n=1:n
                F[k,n] = (1/sqrt(N))*exp((-im*2π*(k-1)*(n-1))/N) # Fourier kernel that depends on N. #(1/sqrt(N))*
            end
        end

    
    return F
end



function SamplingOp(x::Vector, Ni::Int)

    No=length(x);
    T=zeros(Int,(No,Ni));

    for i=1:size(T,1);
        T[i,Int(t[i])+1] = 1
    end
    return T
    
end



function IRLS(A,d_obs;Niter=15,λ=0.5,ϵ= 0.0001)

    m= zeros(ComplexF64,512)
    J = zeros(Float64,Niter);
    
    G = A'*A  #  FT'TF' 
     #= A'*y =# 
    
     for i in 1:Niter
        q = 1.0./ (abs.(m).+ ϵ)
       	Q = diagm(0 => q)
        m = vec(( G + λ*Q) \ A'*d_obs)
        x=(A*m-d_obs)'*(A*m-d_obs);
        J[i] = 0.5*(x^2) + λ*sum(abs.(m))
     end
        return m, J
end



y=signal;
No=length(y);
Ni=512;
F=DFT_matrix(Ni);
T=SamplingOp(signal,Ni)
A= T*F'
m, J =IRLS(A,y,Niter=15,λ=0.31);
d_obs=T'*y;
m0=F*(d_obs);
d_rec=F'*(m);


p =1*collect(-2:0.5:2);
λ = 10.0 .^(p);
misfit=zeros(Float64,length(λ));
modelnorm=zeros(Float64,length(λ));
chi2=zeros(Float64,length(λ));
σ=0.05;
#k=0;

for i=1:length(λ)
    m,J=IRLS(A,y,Niter=45,λ= λ[i])
    d_pred=T*(F'*(m));
    m=DFT_matrix(length(y))*(d_pred);
    m0=DFT_matrix(length(y))*(y);
    misfit[i]= (norm(y - d_pred,2))^2;
    modelnorm[i] = norm( m - m0 ,2)^2;
    chi2[i]=misfit[i]/(σ^2);
end

