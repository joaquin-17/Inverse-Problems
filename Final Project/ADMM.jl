
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

SoftThresholding(x,ρ,λ) = sign(x)*max(abs(x)- (ρ/(2*λ)),0)

#=
# Alternative Direction Method of Multipliers: Inveting Matrices


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

=#


# Alternative Direction Method of Multipliers: With explicit matrices CG


function cgaq(A,y,it)
    
    error=zeros(it)
    x=zeros(size(A,2));

    #it=length(x)
    s=y.-(A*x);              #s represenats the residual error, this lives in the space of the data
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
    x=zero(u)
    for k=1:Ne;
        yc=vcat(y, sqrt(ρ)* (z-u));
        x= cgaq(Ac,yc,Ni) #nv(G +ρ*I)*(A'*(y) .+ ρ*(z - u)); #x-update
        z= SoftThresholding.(x .+ u ,ρ,λ)  # z-update
        u= u .+ (x-z); #dual update, lagrange multiploier
    end

    return z;

end




# # Alternative Direction Method of Multipliers: With Operators


function Sampling(in::Vector)

    cutoff = 1e-10
    i = 1

    wd = zeros(Real,length(in))
    for i = 1 : length(in)
        a =(in[i])^2;
        if (abs(a) > cutoff)
            wd[i] = 1.0;
        end
    end
    return wd;
end

S=Sampling(d_obs);

function SamplingOp(in:: Vector, adj; t::Vector, Ni=512)


    No=length(t);
    T=zeros(Int,(No,Ni));

    for i=1:size(T,1);
        T[i,Int(t[i])+1] = 1
    end
       
    if (adj)

        return T'*in;
    else
        return T*in;

    end
end

function FFTOp(in,adj;normalize=true)
	norm = normalize ? sqrt(length(in[:])) : 1.
	if (adj)
		out = fft(in)/norm
	else
		out = bfft(in)/norm
	end
	return out
end	



function WeightingOp(in,adj;w=1.0)

        return in.*w;
end



function InnerProduct(in1,in2)
    
    return convert(Float32,real(sum(conj(in1[:]).*in2[:])))

end


function LinearOperator(in,operators,parameters;adj=true)
	if adj
		d = copy(in)
		m = [];
		for j = 1 : 1 : length(operators)
			op = operators[j]
			m = op(d,true;parameters[j]...)
			d = copy(m)
		end
		return m
	else
		m = copy(in)
		d = [];
		for j = length(operators) : -1 : 1
			op = operators[j]
			d = op(m,false;parameters[j]...)
			m = copy(d)
		end
		return d
	end

end



function cgaq_op(y, operators, parameters,  it)
    
    A=LinearOperator;


    error=zeros(it)
    x=zeros(length(y));

    #it=length(x)
    s=y.-A(x,operators,parameters, adj=false);   
               #s represenats the residual error, this lives in the space of the data
     p=A(s,operators,parameters, adj=true);                   #By applying A', we are moving the residual back to the model space. Getting a model error(?)
    r=p;                      #    ??
    q=A(p,operators,parameters,adj=false)                     #We are taking the model error(?) and we are projecting it back to the data space.
    old=r'*r;                 #norm squared of the model error(?) are we looking to make this small as well?
   #Conjugate gradient loop
   for k in 1:it
        alpha=(r'*r)/(q'*q);   #Ratio between the sq norm of the model error(?) and the sqnorm of its projection onto data space
        x=x+alpha.*p;          #We update our initial model guess multiplying alpha by the original model error (?)
        s=s-alpha.*q;          #We update the data error subtracting alpha*model error projected on data
        r= A(s,operators,parameters, adj=true);               #We project the updated data error into the model space.
        new=r'*r;
        beta=new/old;          #Ratio between the new and old norm of the model error (?)
        old=new;               #Variable update
        p = r.+(beta.*p);      #Updating the model error by advancing using the ratio between new and old norm.
        q= A(p,operators,parameters,adj=false);
        println("Iteration",k)
        #error[k]= new                #Taking the new model error and projecting it into the data space.
   end
   return x
end


"""
%CGLS: Solves for the minimum of J = || A x - b ||_2^2  + mu ||x||_2^2 
%      via the method of conjugate gradients for least-squares problems. The 
%      matrix A is given via an  operator  and apply on the flight by
%      user-defined function "operator" with parameters "Param".

"""

function CGLS(d_obs, operators,parameters; μ=0.5, Ni=100, tol=1.0e-15)

    m=zeros(length(d_obs));
    r= d_obs - LinearOperator(m,operators,parameters,adj=false);
    s =  LinearOperator(r,operators,parameters,adj=true) - μ*m;
    p=copy(s);

    gamma= InnerProduct(s,s);
    norms0=sqrt(gamma); #norm of the gradient is used to stop.
    k=0;
    flag=0;
    J=zeros(Ni);
    while k < Ni && flag == 0
        
        q = LinearOperator(p,operators,parameters,adj=false);
        delta= InnerProduct(q,q) + μ*InnerProduct(p,p);

        if delta == 0
            delta=1.e-10;
        end

        alpha= gamma/delta;
        m = m + alpha*p;
        r = r - alpha*q;
        s =  LinearOperator(r,operators,parameters,adj=true)  - μ*m;
        gamma1  = InnerProduct(s,s);
        norms  = sqrt(gamma1);
        beta = gamma1/gamma;
        gamma = gamma1;
        p = s + beta*p;
        #if norms <= norms0*tol
         #   println("Loop ended causde tolerance was reached",k)
          #  break;
        #end
        k = k+1;
        println(k)
        error = LinearOperator(m,operators,parameters,adj=false) - d_obs ;
        J[k] = sum( abs.(error[:]).^2 ) + μ*sum( (abs.(m)).^2);

    end

    return m, J
end


function ADMM_CGLS(d_obs,operators,parameters; ρ= 1.0, λ= 1.8,tol=1e-8, Ni=50,Ne=50)
    
    ρ=ρ;
    u=zeros(length(m0)); 
    z=copy(m0);
    w=zero(u);
    for k=1:Ne;  
        b=  -1*LinearOperator(z.- u ,operators, parameters, adj=false) .+ d_obs; # thi is the problem
        #d_obs= LinearOperator(b,operators,parameters, adj=false) 
        w, J= CGLS(b,operators, parameters; μ= ρ, Ni=Ni, tol=1.0e-15)
        x= w .+z .-u;
        z= SoftThresholding.( x .+ u,ρ,λ)  # z-update
        u= u .+ (x .-z); #dual update, lagrange multiploier
    end
     
    return z;

end



operators=[WeightingOp, FFTOp];
parameters=[Dict(:w=>S), Dict(:normalize=>true)];


#m1= ADMM(A,y,m0, ρ= 1.0 , λ=1.8,Ni=50) # This worlks
m2= ADMM_CG(A,y,m0, ρ= 1.0 , λ= 1.8, Ni=50, Ne=50) # This worlks
m3= ADMM_CGLS(d_obs,operators,parameters, ρ= 1.0 , λ= 1.8, Ni=50, Ne=50)
