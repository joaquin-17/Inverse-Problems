
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




y=signal;
No=length(y);
Ni=512;
F=DFT_matrix(Ni);
T=SamplingOp(signal,Ni)
A= T*F'
#m, J =IRLS(A,y,Niter=15,λ=0.1);
d_obs=T'*y;
m0=zeros(length(d_obs));
#m0=F*(d_obs);
#d_rec=F'*(m);



# Alternative Direction Method of Multipliers:

SoftThresholding(x,ρ,λ) = sign(x)*max(abs(x)- λ*ρ,0)


function LASSO_ADMM(A,y,m0; ρ= 1.0, λ=0.5,tol=1e-8, Ni=200)

    
    u=zeros(length(m0)); 
    z=copy(m0);

    G=A'*A;
    I = diagm( ones(size(G,1)));

    
    for k=1:Ni;

        x= inv(G +ρ*I)*(A'*(y) .+ ρ*(z - u)); #x-update
        z= SoftThresholding.(x.+u ,ρ,λ)  # z-update
        u= u + (x-z); #dual update

    end

    return z;

end





function LASSO_ADMM(y,m0, operators,parameters; ρ= 1.0, λ=0.5,tol=1e-8, Ni=200)

    
    u=zeros(length(m0)); 
    z=copy(m0);

    G=A'*A;
    I = diagm( ones(size(G,1)));

    
    for k=1:Ni;

        x= inv(G +ρ*I)*(A'*(y) .+ ρ*(z - u)); #x-update
        z= SoftThresholding.(x.+u ,ρ,λ)  # z-update
        u= u + (x-z); #dual update

    end

    return z;

end

