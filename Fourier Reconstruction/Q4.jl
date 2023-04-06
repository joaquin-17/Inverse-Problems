
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


function SamplingMatrix(x::Vector, Ni::Int)

    No=length(x);
    T=zeros(Int,(No,Ni));

    for i=1:size(T,1);
        T[i,Int(t[i])+1] = 1
    end
    return T
    
end




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


    



function ISTA(x0,y,operators,parameters,λ,Niter)

    x0=randn(size(x0));
    α = PowerMethod(x0,operators,parameters); #x0, operators, parameters
    η= 0.95/α;
    J=zeros(Float64, Niter);
    m = zeros(Float64,size(x0))

     
    k=0;
    
    while k < Niter  #&& err > tolerance
        
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
F=DFT_matrix(Ni);
T=SamplingMatrix(t,Ni)

y=signal;
d_obs=T'y;
S=Sampling(d_obs);
operators=[WeightingOp, FFTOp]
parameters=[Dict(:w=> S), Dict(:normalize=>true)];
λ=0.1;
Ne=500;
x0=randn(512)


m, J = ISTA(x0,d_obs,operators,parameters,λ,Ne)

d_rec= FFTOp(m,false);
dt=1;
tp=dt*collect(0:1:length(d_obs)-1);

error= T*d_rec .- y;

rel_error = norm(error,2)/ norm(y,2);


#=
p =1*collect(-2:0.1:2);
λ = 10.0 .^(p);
misfit=zeros(Float64,length(λ));
modelnorm=zeros(Float64,length(λ));
chi2=zeros(Float64,length(λ));
σ=0.05;
#k=0;


for i=1:length(λ)
    m,J=ISTA(x0,d_obs,operators,parameters,λ[i],Ne)
    d_pred=T*(F'*(m));
    m=DFT_matrix(length(y))*(d_pred);
    m0=DFT_matrix(length(y))*(y);
    misfit[i]= (norm(y - d_pred,2))^2;
    modelnorm[i] = norm( m - m0 ,2)^2;
    chi2[i]=misfit[i]/(σ^2);
end
=#









figure(1);

subplot(221);
plot(tp,d_obs,label="d_obs")
xlabel("Time [sec]")
ylabel("Amplitude")
ylim([-0.4,0.4])

title("Observed")
plt.grid("True")

subplot(222); 
plot(abs.(FFTOp(d_obs,true)),label="d_obs")
xlabel("k")
ylabel("Amplitude")
title("Observed: Amplitude Spectrum")
plt.grid("True")


subplot(223);
plot(tp,d_rec,label="d_obs",c="purple")
xlabel("Time [sec]")
ylabel("Amplitude")
ylim([-0.4,0.4])
title("Recovered")
plt.grid("True")

subplot(224);
plot(abs.(FFTOp(d_rec,true)),label="d_obs",c="purple")
xlabel("k")
ylabel("Amplitude")
title("Recovered : Amplitude Spectrum")
plt.grid("True")

tight_layout()



figure(2);
plot(tp,d_obs, label="d_obs");
plot(tp,d_rec, label="d_rec", c="purple");

xlabel("Time [sec]")
ylabel("Amplitude")
title("Comparison: ISTA")
plt.grid("True")
legend()

markers_on= chi2[18];
figure(3);
loglog(λ,chi2,c="k");
loglog(λ,chi2,"o");
xlabel("λ")
ylabel("χ²≈ N")
ylim([0,1200]) ; xlim([0,10])
title("χ² test: ISTA")
aux1= ones(length(λ))*chi2[18];
aux2= ones(length(chi2))*λ[18];
plot(λ,aux1,c="purple");
plot(aux2,chi2,c="purple");
plt.grid("True")


