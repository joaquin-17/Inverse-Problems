
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

SoftThresholding(x,ρ,λ) = sign(x)*max(abs(x)- (ρ/(2*λ)),0)


function ADMM(A,y,m0; ρ= 1.0, λ=0.5,tol=1e-8, Ni=150)

    
    u=zeros(length(m0)); 
    z=copy(m0);

    G=A'*A;
    I = diagm( ones(size(G,1)));

    
    for k=1:Ni;

        x= inv(G +ρ*I)*(A'*(y) .+ ρ*(z - u)); #x-update
        z= SoftThresholding.(x.+u ,ρ,λ)  # z-update
        u= u + (x-z); #dual update, lagrange multiploier

    end

    return z;

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
        println("Iteration",k)
        #error[k]= new                #Taking the new model error and projecting it into the data space.
   end
   return x
end



function ADMM_CG(A,y,m0; ρ= 1.0, λ=0.5,tol=1e-8, Ni=150, Ne=50)

    
    u=zeros(length(m0)); 
    z=copy(m0);
    I = diagm( ones(size(A,2)));
    Ac= vcat(A, sqrt(ρ)*I);
    yc=vcat(y, sqrt(ρ)* (z .- u))
    #G=A'*A;
    #I = diagm( ones(size(G,1)));

    
    for k=1:Ni;


        Ac= vcat(A, sqrt(ρ)*I);
        yc=vcat(y, sqrt(ρ)* (z-u));
        x= cgaq(Ac,yc,Ne) #nv(G +ρ*I)*(A'*(y) .+ ρ*(z - u)); #x-update
        z= SoftThresholding.(x .+ u ,ρ,λ)  # z-update
        u= u .+ (x-z); #dual update, lagrange multiploier

    end

    return z;

end




m1= ADMM(A,y,m0, ρ= 1.0 , λ=1.8,Ni=50) # This worlks
m2= ADMM_CG(A,y,m0, ρ= 1.0 , λ= 1.8, Ni=50, Ne=50) # This worlks