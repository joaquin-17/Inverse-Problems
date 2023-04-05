#Problem 1):



"""
DtFT= DtFT (Nk,dt, xn )
Compute the Discrete Time Fourier Transform (Frequency Response) of a signal `xn` using a `dt` time sampling interval
and Nk samples.
Returns `DtFT` which is a `Vector{ComplexF64}` that carries the Discrete time Fourier Transform of the signal.
Returns the digital angular frequency ω, the angular frequency Ω and the frequency f in Hertz.
# Arguments
- ` Nk :: Int64`: the output numer of samples.
- `dt :: Float64`: sampling interval for the discrete signal.
- ` xn :: Vector{Float64}`: discrete time signal.
"""
function DtFT(N :: Int64, dt:: Float64, xn :: Vector{Float64})

 ω=zeros(Float64,N)
 ω= [ -π + 2.0*π*(k)/(N) for k=0:N-1]
 # dω=2π/N;
 # ω= dω*collect(0:1:N-1)
 Ω= ω/dt;
 f= Ω/2π;

 DtFT=zeros(ComplexF64,length(ω))

 for k in 1:length(ω)-1;
 for n in 1:length(xn)-1
 DtFT[k]= DtFT[k] + xn[n]*exp(-1im*ω[k]*(n))
 end
 end

 return DtFT
end



function DFT_matrix(N::Int64,k::Int64,n::Int64; flag="forward")
    
    #Check that n is equal to kernel


    F=zeros(ComplexF64,(k,n));
    
    if flag == "adjoint"
        
        for k=1:k
            for n=1:n
                F[k,n] = exp((-im*2π*(k-1)*(n-1))/N) # Fourier kernel that depends on N.
            end
        end

    elseif flag == "forward"

        for k=1:k
            for n=1:n
                F[k,n] = (exp((im*2π*(k-1)*(n-1))/N)) # Fourier kernel that depends on N.
            end
        end
    end
    
    return F
end




function CalculateSampling(in)

    cutoff = 1e-10
        itrace = 1
    
        n=size(in)
        in=reshape(in,n[1],:)
            wd = zeros(Float32,size(in))
        n2=size(in,2)
        for itrace = 1 : n2
            a = sum(in[:,itrace].*in[:,itrace])
            if (a > cutoff)
                wd[:,itrace] .= 1.
            end
        end
        wd=reshape(wd,n)
        return wd;
    
    end


# Example
dt=0.001;
fs=1/dt
fn=fs/2;
Ns=2^8; # Power of 2.
Ms=Int(Ns/2)+1
t = dt*collect(0:1:Ns-1);
s = sin.(2π*25*t) .+ sin.(2π*50*t) .+ sin.(2π*150*t) .+ sin.(2π*250*t);
figure(1, figsize=(8,5));
plt.plot(t[1:100],s[1:100], c="red");
plt.grid("True");
plt.xlabel("Time [sec]", labelpad= 10.0);
plt.ylabel("Amplitude", labelpad=10.0);
plt.title("Signal: 4 Frequencies",pad=10.0);



# Dot product test example using operators rather than matrices
# Fourier DFT matrices and its Hermitian Transpose are replaced by
# on-the-flight FFTs
M = 512;
Fadj=DFT_matrix(M,k,n);
x1=randn(M);
y1= Fadj*x1;

y2=randn(M);
forF=Fadj' # DFT_matrix(M,k,n, flag="forward");
x2= forF*y2;

dot_x = x1'*x2
dot_y = y1'*y2

println("y₁*y₂ = x₁*x₂ ? ", round(dot_x,digits=5) == round(dot_y,digits=5))



M = 512
x1 = randn(M)
y1 = fft(x1)
y2 = randn(M)
x2 = bfft(y2)
dot_x = x1'*x2
dot_y = y1'*y2
println(dot_x)
println(dot_y)
println("y₁*y₂ = x₁*x₂ ? ", round(dot_x,digits=5) == round(dot_y,digits=5))




data=readdlm("/home/aacedo/Desktop/GEOPH531/Inverse-Problems/Fourier Reconstruction/data/data_to_reconstruct.txt");
t=data[:,1]; s_real=data[:,2]; s_imag=data[:,3];

signal= s_real .+ im*s_imag;


subplot(121);
plot(t,s_real);
subplot(122);
plot(t,s_imag);




sampling=ones(Float64,length(t))
t=Int.(t)

for i=1:length(t)-1
    
    diff=1

    if t[i+1] - t[i] == 1;
        #sampling[i] = diff;
    else
        ns= t[i+1] - t[i]
        
        #sampling[i:dif] .= 0

        #println(t[i]);
         #println(t[i+1]);
    end

end


s_obs= [signal[1:28] ; zeros(ComplexF64,100-28); signal[29:219];zeros(271-220);signal[221:end]]

sampling=zeros(Float64,length(s_obs))

for i=1:length(s_obs)

    if s_obs[i] ≠ 0.0
        
        sampling[i]= 1

    else

        sampling[i]= 0

    end
end





F=DFT_matrix(512,512,512,flag="adjoint");
S_obs= F(s_obs);
Fadj=copy(F');



function IRLS(A,y,Niter,λ)

    x = zeros(Float64,M)
  
    J = zeros(Niter)
    G = A'*A 
    b = A'*y 
     for k in 1:Niter
        q = 1.0./(abs.(x).+0.0001)
       	Q = diagm(0 => q)
        x = vec((G + λ*Q)\b)
        J[k] = 0.5*sum((A*x-y).^2) + λ*sum(abs.(x))
     end
        return x, J
end

        


αr=IRLS(S_obs,F,zeros(ComplexF64,length(s_obs)), norma=L1weights,ϵ=1e-4, Ni=5)

#singal[30:30+72].=0.0;
#signal[139:139+52] =0.0;

#sampling=ones(Int64,512);
#sampling[30:30+72]=0; sampling[139:149+52]
