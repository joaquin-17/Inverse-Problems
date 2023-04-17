
function SoftFreqBand(N::Int,dt::Float64,FB::Vector)
    
    
    f1=FB[1];
    f2=FB[2];
    f3=FB[3];
    f4=FB[4];
    
    
    Mb= (N ÷2) +1
    BF=zeros(Float64,N)

    #Frequency index:   
    n1 = round(Int, f1*dt*N +1);
    n2 = round(Int, f2*dt*N +1);
    n3 = round(Int, f3*dt*N +1);
    n4 = round(Int, f4*dt*N +1);


    Nh= length(n1:n4)
    N12=2*(n2-n1) +1    
    N34= 2*(n4-n3)+1
#
#M12= div(N12,2)

    for i=n1:n2
        BF[i] = (1.0 .- cos.(2π*(i-n1)/((N12-1))))/2;
    end


    for i=n2+1:n3
        BF[i] = 1.0#maximum(BF)
    end

    for i=n3+1:n4
        BF[i] = 1 - (1.0 .- cos.(2π*(i-n3)/((N34-1))))/2;
    end


# ---> Symmetries <---

    for k= Mb+1:N
        BF[k]= conj(BF[N-k+2])
    end


    return BF


end

#=
#BF= BF/maximum(BF);




#BF[n3+1:n4]=reverse(BF[n1:n2])

#for i=n3+1:n4
 #   BF[i] = (1.0 .- cos.(2π*(i-n1)/((Nh-1))))/2;
#end
    

# ---> Symmetries <---

for k= Mb+1:N
    BF[k]= conj(BF[N-k+2])
end
=#