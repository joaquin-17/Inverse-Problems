
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




function cgaq(A,y,it)
    
     error=zeros(it)
     x=zeros(size(A,2));
     #it=length(x)
     s=y.-(A*x);              #s represents the residual error, this lives in the space of the data
     p=A'*s;                   #By applying A', we are moving the residual back to the model space. Getting a model error(?)
     r=p;                      #    ??
     q=A*p                     #We are taking the model error(?) and we are projecting it back to the data space.
     old=r'*r;                 #norm squared of the model error(?) are we looking to make this small as well?
    #Conjugate gradient loop
    for k in 1:it
         alpha=(r'*r)/(q'*q);   #Ratio between the sq norm of the model error(?) and the sqnorm of its projection onto data space
         x=x+alpha.*p;          #We update our initial model guess multiplying alpha by the original model error (?)
         s=s-alpha.*q;          #We update the data error subtracting alpha*model error projected on data
         r= A'*s;               #We project the updated data error into the model space.
         new=r'*r;
         beta=new/old;          #Ratio between the new and old norm of the model error (?)
         old=new;               #Variable update
         p = r.+(beta.*p);      #Updating the model error by advancing using the ratio between new and old norm.
         q= A*p;
         #println("Iteration",k)
         #error[k]= new                #Taking the new model error and projecting it into the data space.
    end
    return x
end




function IRLS(A,y,m0;Ne=25,Ni=150,λ=0.5,ϵ= 0.0001)

    m=ones(ComplexF64,length(m0));
    J = zeros(Float64,Ne);
   # G = A'*A  #  FT'TF' 

     for i in 1:Ne
        q = sqrt.(1.0./ (abs.(m).+ ϵ))
       	Q = diagm(0 => q);
        Ac=vcat(A,sqrt(λ)*Q);
        yc=vcat(y,zeros(length(m)));
        m=cgaq(Ac,yc,Ni)               #m = vec(( G + λ*Q) \ A'*y)
        #x=(A*m-y)'*(A*m-y);
        #J[i] = 0.5*(x.^2) + λ*sum(abs.(m))
     end
        return m, J
end



y=signal;
No=length(y);
Ne=15;
Ni=512;
F=DFT_matrix(Ni);
T=SamplingOp(signal,Ni)
A= T*F';
m0=zeros(Ni);
m, J =IRLS(A,y,m0,Ne=50,Ni=100,λ=0.31);
d_obs=T'*y;
m0=F*(d_obs);
d_rec=F'*(m);

#=
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

=#