using FFTW, LinearAlgebra, Statistics, DelimitedFiles, PyPlot


include("CG.jl")

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
m0=randn(length(d_obs));
#m0=F*(d_obs);
#d_rec=F'*(m);

SoftThresholding(x,ρ,λ) = sign(x)*max(abs(x)- (λ/ρ),0)


# Alternative Direction Method of Multipliers: Inveting Matrices


SoftThresholding(x,ρ,λ) = sign(x)*max(abs(x)- (ρ/(2*λ)),0)


function ADMM(A,y,m0; ρ= 1.0, λ=0.5,tol=1e-8, Ne=150)

    
    u=zeros(length(m0)); 
    z=copy(m0);

    G=A'*A;
    I = diagm( ones(size(G,1)));
    Je=zeros(Ne); # ADMM cost function

    
    for k=1:Ne;

        x= inv(G +ρ*I)*(A'*(y) .+ ρ*(z - u)); #x-update
        z= SoftThresholding.(x.+u ,ρ,λ)  # z-update
        u= u + (x-z); #dual update, lagrange multiploier

        residual= T'*y .- (F*z); # residual between the data and reconstructed
        aux=abs.(residual)
      
        Je[k] =  sum(aux.^2) + λ*sum(abs.(z[:]))
    end

    return z, Je;

end






# Alternative Direction Method of Multipliers: With explicit matrices CG




function ADMM(A,y; x0=0.0, ρ= 1.0, λ=0.5, Ni=150, Ne=50, tol=1.0e-8)

    #Check initial model
    if x0 ≠ 0.0
        x0= x0
    else
        x0=zeros(size(A,2));
    end

    # Initialize cost functions
    Ji=Float64[]; # CG cost function. 
    Je=zeros(Ne); # ADMM cost function
    norms0= norm(y .- (A*x0),2);  #||∇J||₂ => norm of the gradient at the beginining.  
  
    #Initialize variables
    ucg=zeros(length(x0)); 
    zcg=copy(x0);
    I = diagm( ones(size(A,2)));
    Ac= vcat(A, sqrt(ρ)*I);
    yc=vcat(y, sqrt(ρ)* (zcg .- ucg))
    xcg=zero(ucg)
    
    # ADDM's loops

    for k=1:Ne;
        yc=vcat(y, sqrt(ρ)* (zcg-ucg));
        xcg, Ji= CG(Ac,yc,x0=x0,Ni=Ni,tol=1.0e-15); #x-update
        zcg= SoftThresholding.(xcg .+ ucg ,ρ,λ)  # z-update
        ucg= ucg .+ (xcg-zcg); #dual update, lagrange multiploier;
        
        residual= T'*y .- (F*zcg); # residual between the data and reconstructed
        aux=abs.(residual)
      
        Je[k] =  sum(aux.^2) + λ*sum(abs.(zcg[:]))
     
        norms=norm(aux,2);

    
        if norms <= norms0*tol
            println("ADMM Iterations")
            println("At iteration k=$k is ||∇Jₖ||₂² ≤ $tol* ||∇J₀||₂²")
           println("Outer loop ended.")

            break;
        end
        
            

    end

    return zcg, Ji, Je
end





m1, J= ADMM(A,y,m0, ρ= 1.0 , λ=1.8,Ne=50) # This worlks
m,Ji,Je= ADMM(A,y,x0=m0, ρ= 1.0 , λ= 1.3, Ni=10, Ne=50, tol=1.0e-5) # This worlks
