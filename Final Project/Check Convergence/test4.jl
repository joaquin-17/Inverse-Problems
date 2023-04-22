
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
d_obs=T'*y;
m0=zeros(length(d_obs));

SoftThresholding(x,ρ,λ) = sign(x)*max(abs(x)- (ρ/(2*λ)),0)


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

function ConjugateGradients(d,operators,parameters;Niter=10,mu=0,tol=1.0e-15)

    cost = Float64[]
    r = copy(d)
    g = LinearOperator(r,operators,parameters,adj=true)
    m = zero(g)
    s = copy(g)
    gamma = InnerProduct(g,g)
    gamma00 = gamma
    cost0 = InnerProduct(r,r)
    push!(cost,1.0)
    for iter = 1 : Niter
	t = LinearOperator(s,operators,parameters,adj=false)
	delta = InnerProduct(t,t) + mu*InnerProduct(s,s)
	if delta <= tol
#	    println("delta reached tolerance, ending at iteration ",iter)
	    break;
	end
	alpha = gamma/delta
	m = m + alpha*s
	r = r - alpha*t
	g = LinearOperator(r,operators,parameters,adj=true)
	g = g - mu*m
	gamma0 = copy(gamma)
	gamma = InnerProduct(g,g)
        cost1 = InnerProduct(r,r) + mu*InnerProduct(m,m)
        push!(cost,cost1/cost0)
	beta = gamma/gamma0
	s = beta*s + g
	if (sqrt(gamma) <= sqrt(gamma00) * tol)
	    println("tolerance reached, ending at iteration ",iter)
	    break;
	end
    end

    return m, cost
end


function CGLS(d_obs, operators,parameters; μ=0.5, Ni=100, tol=1.0e-15)

    #J=Float64[];
    m=zeros(length(d_obs));
    r= d_obs - LinearOperator(m,operators,parameters,adj=false);
    s =  LinearOperator(r,operators,parameters,adj=true) - μ*m;
    p=copy(s);

    gamma= InnerProduct(s,s);
    norms0=sqrt(gamma); #norm of the gradient is used to stop.
    k=0;
    flag=0;
    J=zeros(Ni);
    J[1]=norm(r,2).^2
    while k < Ni && flag == 0


        k = k+1;
        println(k)
        
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
        if norms <= norms0*tol
            println("Loop ended causde tolerance was reached",k)
            break;
        end
        k = k+1;
        println(k)
        error = LinearOperator(m,operators,parameters,adj=false) - d_obs ;
        J[k] = sum( abs.(error[:]).^2 ) + μ*sum( (abs.(m)).^2);

    end

    return m, J
end


function ADMM_CGLS(d_obs,operators,parameters; ρ= 1.0, λ= 1.8,tol=1e-8, Ni=50,Ne=50)
    
    J=zeros(Ni)
    ρ=ρ;
    u=zeros(length(m0)); 
    z=copy(m0);
    w=zero(u);
    for k=1:Ne;  
        b=  -1*LinearOperator(z.- u ,operators, parameters, adj=false) .+ d_obs; # thi is the problem
        #d_obs= LinearOperator(b,operators,parameters, adj=false) 
        w, J=ConjugateGradients(b,operators,parameters, Niter=Ni,mu=ρ,tol=1.0e-15)
        # w, J= CGLS(b,operators, parameters; μ= ρ, Ni=Ni, tol=1.0e-10)
        x= w .+z .-u;
        z= SoftThresholding.( x .+ u,ρ,λ)  # z-update
        u= u .+ (x .-z); #dual update, lagrange multiploier
    end
     
    return z,J

end



operators=[WeightingOp, FFTOp];
parameters=[Dict(:w=>S), Dict(:normalize=>true)];
m0=zeros(size(shot))

#m1= ADMM(A,y,m0, ρ= 1.0 , λ=1.8,Ni=50) # This worlks
#m2 = ADMM_CG(A,y,m0, ρ= 1.0 , λ= 1.8, Ni=50, Ne=50) # This worlks
m3, J= ADMM_CGLS(m0,d_obs,operators,parameters, ρ= 1.0, λ= 1.8, Ni=50, Ne=50)