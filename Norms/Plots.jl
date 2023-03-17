#Plots Assignment 3:





# Plot different loss functions 
figure(1, figsize=(5,5))
scatter(df[!,"x"], df[!,"y"],color="b",label="Data")
plot(x,A*ml2[:,end],"r", label="L2-fit");
plot(x,A*ml1[:,end],"g", label="L1-fit");
plot(x,A*mcn[:,end],"y", label="Cauchy-fit");
plot(x,A*mhn[:,end],"b",label="Huber-fit");
#legend()
xlabel("x", fontsize=15);
ylabel("y", fontsize=15);
title("Robust-Regression")





#Plot f(u) loss function


figure(2, figsize=(5,5))
plot(aux,f1./maximum(f1),"r", label="L2-Loss");
plot(aux,f2./maximum(f2),"g", label="L1-Loss");
plot(aux,f3./maximum(f3),"y", label="Cauchy-Loss");
plot(aux,f4./maximum(f4),"b",label="Huber-Loss");
#plt.legend()
xlabel("x");
ylabel("y")
#plt.grid("True")



#Plot df(u) influence function



figure(3, figsize=(5,5))
plot(aux,df1./maximum(df1),"r", label="L2-Influence");
plot(aux,df2./maximum(df2),"g", label="L1-Influence");
plot(aux,df3./maximum(df3),"y", label="Cauchy-Influence");
plot(aux,df4./maximum(df4),"b",label="Huber-Influence");
#plt.legend()
xlabel("x");
ylabel("y")
#plt.grid("True")


#Plot w(u) weigths function
figure(4, figsize=(5,5))
plot(aux,w1./maximum(w1),"r", label="L2-Influence");
plot(aux,w2./maximum(w2),"g", label="L1-Influence");
plot(aux,w3./maximum(w3),"y", label="Cauchy-Influence");
plot(aux,w4./maximum(w4),"b",label="Huber-Influence");
#plt.legend()
xlabel("x");
ylabel("y")
#plt.grid("True")include0