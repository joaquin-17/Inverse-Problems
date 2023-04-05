#=

GEOPH 531 : Geophysical inverse problems
Department of Physics
2023

Assignment3: Different Norms

=#


include("IRLS.jl")

using LinearAlgebra, DataFrames,DelimitedFiles, PyPlot

data=readdlm("data.txt");
df=DataFrame("x"=>data[:,1], "y"=>data[:,2]);

y=df[!,"y"]; x=df[!,"x"]; #data
A=[ones(Float64, length(x)) x ];  #matrix model
m0=[0.0,0.0]; #Initial model

ml2=IRLS(y,A,[0.0,0.0], norma=L2weights,ϵ=1e-4, Ni=5)
#dl2=A*ml2;
ml1=IRLS(y,A,[0.0,0.0], norma=L1weights,ϵ=1e-4, Ni=5)
#dl1=A*ml1;
mcn=IRLS(y,A,[0.0,0.0], norma=Cauchyweights,ϵ=1e-4, Ni=5)

mhn=IRLS(y,A,[0.0,0.0], norma=Huberweights,ϵ=1e-4, Ni=5)




# f(u) - Loss functions
aux=collect(-100:1:100);
f1=L2Loss(aux);
f2=L1Loss(aux);
f3=CauchyLoss(aux,δ=1)
f4=HuberLoss(aux,δ=10.0)


#df(u)/du => influence functions
df1=L2influence(aux);
df2=L1influence(aux);
df3=Cauchyinfluence(aux,δ=1)
df4=Huberinfluence(aux,δ=10.0)

#weights w(u)

w1=Weights(aux,norma=L2weights,ϵ=0.0001, δ=5.0)
w2=Weights(aux,norma=L1weights,ϵ=0.0001, δ=5.0)
w3=Weights(aux,norma=Cauchyweights,ϵ=0.0, δ=5.0)
w4=Weights(aux,norma=Huberweights,ϵ=0.0, δ=5.0)













#=
clf()
figure(1, figsize=(7,7))
subplot(121)
scatter(df[!,"x"], df[!,"y"],color="b",label="Data")
plot(x,A*ml2[:,end],"r", label="L2");
xlabel("x", fontsize=15)
ylabel("y", fontsize=15)
xlim([0,6]); ylim([0,12])
grid("True")
legend()
subplot(122)
scatter(df[!,"x"], df[!,"y"],color="b",label="Data")
plot(x,A*ml1[:,end],"g", label="L1");
xlabel("x", fontsize=15)
ylabel("y", fontsize=15)
xlim([0,6]); ylim([0,12])
grid("True")
legend()
=#

#=
scatter(df[!,"x"], df[!,"y"],color="b",label="Data")
plot(x,A*ml2[:,end],"r", label="L2");
plot(x,A*ml1[:,end],"g", label="L1");
plot(x,A*mcn[:,end],"y", label="Cauchy Norm");
plot(x,A*mhn[:,end],"k",label="Huber Loss");
plt.legend()
=#