
using FFTW, LinearAlgebra, Statistics, DelimitedFiles, PyPlot

data=readdlm("/home/aacedo/Desktop/GEOPH531/Inverse-Problems/Fourier Reconstruction/data/data_to_reconstruct.txt");
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



function WeightingOp(in,adj;w=1)

	return in.*w

end

function WeightingOp(m::AbstractString,d::AbstractString,adj;w="NULL")

	if (adj==true)
		d1,h1,e1 = SeisRead(d)
		d2,h2,e2 = SeisRead(w)
		SeisWrite(m,d1[:,:].*d2[:,:],h1,e1)
	else
		d1,h1,e1 = SeisRead(m)
		d2,h2,e2 = SeisRead(w)
		SeisWrite(d,d1[:,:].*d2[:,:],h1,e1)
	end

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



function Sampling(in::Vector)

    cutoff = 1e-10
    i = 1
    
    #    n=size(in)
    #   in=reshape(in,n[1],:)
    wd = zeros(Int,length(in))
    #    n2=size(in,2)
    for i = 1 : length(in)
        a =(in[i])^2;
        if (abs(a) > cutoff)
            wd[i] = 1;
        end
    end
    return wd;
end




SoftThresholding(u,η,λ) = sign(u)*max(abs(u)- η*λ,0)


    
function PowerMethod(x0,operators,parameters)
    
    x= x0;
    α=0.0;
    for k = 1:10;
        aux=LinearOperator(x,operators,parameters,adj=false)
        y=LinearOperator(aux,operators,parameters,adj=true)
        n = norm(y,2);
        x = y/n;
        α = n;
    end
    return α
end


function SamplingOp(x::Vector, Ni::Int; flag="forward")

    No=length(x);
    T=zeros(Int,(No,Ni));

    for i=1:size(T,1);
        T[i,Int(t[i])+1] = 1
    end

    if flag == "forward"

        return T

    elseif flag == "adjoint"

        return copy(T')


    else 
        error("Specify flag= 'forward' or flag= 'adjoint'.")
    end

    
end


#=

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





function Power_Iteration(A)

    
    N,M = size(A)
    
    if N > M 
        H = A'*A;
        b = randn(M,1)
    else
        H = A*A';
        b = randn(N,1)
    end
        e = 1.
        
        for k = 1:20
            tmp = H*b
            e = norm(tmp)
            tmp = tmp/e
            b = tmp
        end
        
        return e
    end
=#

    



function ISTA(x0,y,operators,parameters,λ,Niter)

    x0=randn(size(x0));
    α = PowerMethod(x0,operators,parameters); #x0, operators, parameters
    η= 0.95/α;
    J=zeros(Float64, Niter);
    m = zeros(Float64,size(x0))

     
    k=0;
    
    while k < Niter # && err > tolerance
        
        k=k+1
        mk = copy(m);
        aux1= LinearOperator(mk,operators,parameters,adj=false) #A*x
        aux2= aux1 .-y; #(A*x.-y)
        aux3= LinearOperator(aux2,operators,parameters,adj=true) #A'*(A*x.-y)==> this is the gradient!;
        ∇fₖ= aux3; #change name
        u= mk  .-  η* ∇fₖ;
        m=SoftThresholding.(u,η,λ);
        aux4=  LinearOperator(m,operators,parameters,adj=false)
        aux5= aux4 .- y;
        aux6= (aux5')*(aux5);
        J[k] = sum(aux6^2) + λ*sum(abs.(m))
    
    end
    return m, J
end




y=signal;
No=length(y);
Ni=512;
#F=DFT_matrix(Ni);
T=SamplingOp(signal,Ni, flag="forward")
#A= T*F'
#d_obs=T'*y;
#m0=F*(d_obs);
#d_rec=F'*(m);


y=signal;
d_obs=T'y;
S=Sampling(d_obs);
operators=[WeightingOp, FFTOp]
parameters=[Dict(:w=> S), Dict(:normalize=>true)];
λ=0.35;
Ne=350;
x0=randn(512)


m, J = ISTA(x0,d_obs,operators,parameters,λ,Ne)

d_rec= FFTOp(m,false);


figure(figsize=(10,10))

subplot(221)
plot(d_obs);
subplot(222);
plot(d_rec)
subplot(223)
plot(abs.(FFTOp(d_obs,true)))
subplot(224);
plot(abs.(m));
















#=
function Sampling(in::Vector)

    cutoff = 1e-10
    i = 1
    
    #    n=size(in)
    #   in=reshape(in,n[1],:)
    wd = zeros(Int,length(in))
    #    n2=size(in,2)
    for i = 1 : length(in)
        a =(in[i])^2;
        if (abs(a) > cutoff)
            wd[i] = 1;
        end
    end
    return wd;
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



function WeightingOp(in,adj;w=1)

	return in.*w

end

function WeightingOp(m::AbstractString,d::AbstractString,adj;w="NULL")

	if (adj==true)
		d1,h1,e1 = SeisRead(d)
		d2,h2,e2 = SeisRead(w)
		SeisWrite(m,d1[:,:].*d2[:,:],h1,e1)
	else
		d1,h1,e1 = SeisRead(m)
		d2,h2,e2 = SeisRead(w)
		SeisWrite(d,d1[:,:].*d2[:,:],h1,e1)
	end

end



y=signal;
d_obs=T'y;
S=Sampling(d_obs);
operators=[WeightingOp, FFTOp]
parameters=[Dict(:w=> S), Dict(:normalize=>true)];
μ=0.5;
#Ne=5;
#Nint=100;

#=

function ISTA(m0,d_obs,operators,parameters,μ,Ni,tolerance)




function ISTA(A,y,Niter,λ)

    # Compute power method   
    m0 = randn(size(m0)); 
    α = 1.05*PowerMethod(x0,operators,parameters);
    
    
    #Initialize:    
    J=zeros(Float64, Niter);
    m=zeros(ComplexF64,size(x0));
    t=1.0;
    err= 1e4;
    T=μ/(2*α);

   # Start ISTA
    
    k=0;
    while k < Ni  && err > tol;

        k=k+1
        mk = copy(m); #update model
        forward=  LinearOperator(yk,operators,parameters,adj=false) #aplica ifft, me da el dato
        aux= yobs .-forward;
        adjoint= LinearOperator(aux,operators,parameters,adj=true) # applica fft, me da los coeficients.
        m=copy(forward);  

=#

=#