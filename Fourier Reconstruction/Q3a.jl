
using FFTW, LinearAlgebra, Statistics, DelimitedFiles, PyPlot

data=readdlm("C:\\Users\\Joaquin\\Desktop\\IP\\Inverse-Problems\\Inverse-Problems\\Fourier Reconstruction\\data\\data_to_reconstruct.txt");
t=data[:,1]; s_real=data[:,2]; s_imag=data[:,3];

signal= s_real .+ im*s_imag;




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

Operator(in::)




function cgaq(d_obs,m0,operators, parameters,it)

    x=zeros(size(m0));
    s=d_obs .- LinearOperator(x,operators,parameters,adj=false);              #s represents the residual error, this lives in the space of the data
    p=LinearOperator(s,operators,parameters,adj=true);                   #By applying A', we are moving the residual back to the model space. Getting a model error(?)
    r=p;                      #    ??
    q=LinearOperator(p,operators,parameters,adj=false)                    #We are taking the model error(?) and we are projecting it back to the data space.
    old=r'*r;                 #norm squared of the model error(?) are we looking to make this small as well?
   #Conjugate gradient loop
   for k in 1:it
        alpha=(r'*r)/(q'*q);   #Ratio between the sq norm of the model error(?) and the sqnorm of its projection onto data space
        x=x+alpha.*p;          #We update our initial model guess multiplying alpha by the original model error (?)
        s=s-alpha.*q;          #We update the data error subtracting alpha*model error projected on data
        r= LinearOperator(s,operators,parameters,adj=true);               #We project the updated data error into the model space.
        new=r'*r;
        beta=new/old;          #Ratio between the new and old norm of the model error (?)
        old=new;               #Variable update
        p = r.+(beta.*p);      #Updating the model error by advancing using the ratio between new and old norm.
        q= LinearOperator(p,operators,parameters,adj=false) ;
        #println("Iteration",k)
        #error[k]= new                #Taking the new model error and projecting it into the data space.
   end
   return x
end





#=

function CGLS(m0,d_obs,operators,parameters; μ=0.5, Ni=100, tol=1.0e-15)

    m=copy(m0);
    r= d_obs - LinearOperator(m,operators,parameters,adj=false);
    s =  LinearOperator(r,operators,parameters,adj=true) -μ*m;
    p=copy(s);

    gamma= InnerProduct(s,s);
    norms0=sqrt(gamma); #norm of the gradient is used to stop.
    k=0;
    flag=0;
    J=zeros(Ni);


    while k < Ni #&& flag == 0
        
        q = LinearOperator(p,operators,parameters,adj=false);
        delta= InnerProduct(q,q) + μ*InnerProduct(p,p);

        if delta == 0
            delta=1.e-10;
        end

        alpha= gamma/delta;
        m = m + alpha*p;
        r = r - alpha*q;
        s =  LinearOperator(r,operators,parameters,adj=true) -μ*m;
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



function IRLS(d,operators,parameters;Ni=10,Ne=5,μ=0.5)

	cost = Float64[]
	weights = ones(Float64,size(d))
	parameters[end][:w] = weights
	m = []
	for iter = 1 : Ne 
		v,cost1 = CGLS(m0,d,operators,parameters,Ni=Ni,μ=μ,tol=1.0e-15)
		append!(cost,cost1)
		m = v .* weights
		#weights= 1.0 ./ (abs.(m) .+ 0.0001);
        weights = abs.(m./(abs.(m[:])))
		parameters[end][:w] = weights
	end

	return m, cost

end
=#

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


function SamplingMatrix(x::Vector, Ni::Int)

    No=length(x);
    T=zeros(Int,(No,Ni));

    for i=1:size(T,1);
        T[i,Int(t[i])+1] = 1
    end
    return T
    
end

T=SamplingMatrix(t,512)
y=signal;
d_obs=T'y;
S=Sampling(d_obs);
operators=[WeightingOp, FFTOp]
parameters=[Dict(:w=> S), Dict(:normalize=>true)];
μ=1.0;
Ne=10;
Nint=100;
m0=zero(d_obs)

#m, J= IRLS(d_obs,operators,parameters; Ni=Nint,Ne=Ne,μ=μ)

#y=signal;
#No=length(y); Ni=512; # observed and ideal lengths.
#A= T*F'
#m, J =IRLS(A,y,Niter=15,λ=0.1);
#d_obs=T'*y;
#m0=F*(d_obs);
#d_rec=F'*(m);
