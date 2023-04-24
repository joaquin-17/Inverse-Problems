
using Printf

function CG(A,y; x0=0.0, Ni=160, tol=1.0e-8)
    
    if x0 ≠ 0.0
        x= x0
    else
        x=zeros(size(A,2));
    end


    
    J=zeros(Float64,Ni); #Initialize cost function
    #x=randn(length(A[1,:]));
    #x=zeros(length(A[1,:]));
    #x=zeros(size(A,2)); #Initial point or model
    s= y .- (A*x);          #Initial residual
    p=A'*s;               #Model error
    r=p;                    
    q=A*p                     
    old=r'*r;
    
    #gamma=  InnerProduct(p,p);#  ||∇J||²₂  
    norms0= sqrt(real(old));  #||∇J||₂ => norm of the gradient at the beginining.  
    norms=norms0;
    k=0;
    J[1]= (norm(s,2))^2
    
    
    #Conjugate gradient loop

   @printf "==============================================\n";
   @printf "================= CG =======================\n";
   @printf "==============================================\n";

   @printf ("k         ||∇J0||²       ||∇Jk||²    ||∇Jk||²/||∇J0||²      J   \n"); 

   @printf(" %3.0f  %12.5g   %12.5g    %8.3g    %8.3g \n", k,norms0,norms,norms/norms0,J[1])


   while k < Ni


        k=k+1
        alpha=(r'*r)/(q'*q);   #Ratio between the sq norm of the model error(?) and the sqnorm of its projection onto data space
        x=x+alpha.*p;          #We update our initial model guess multiplying alpha by the original model error (?)
        s=s-alpha.*q;          #We update the data error subtracting alpha*model error projected on data
        r= A'*s; 
       # gamma1  = InnerProduct(r,r); norms  = sqrt(gamma1);     #We project the updated data error into the model space.
        new=r'*r;
        norms = sqrt(real(new)); 
        beta=new/old;          #Ratio between the new and old norm of the model error (?)
        old=new;               #Variable update
        p = r.+(beta.*p);      #Updating the model error by advancing using the ratio between new and old norm.
        q= A*p;


        if norms <= norms0*tol
            println("---------------------------------------------------")
            println("CG Iterations:")
            println("At iteration k=$k is ||∇Jₖ||₂² ≤ $tol* ||∇J₀||₂²")
            println("Inner loop ended.")
            println("---------------------------------------------------")

            break;
        end
        

        J[k]= (norm(s,2))^2
       
        @printf(" %3.0f      %12.5g       %12.5g    %8.3g    %8.3g  \n", k,norms0,norms, norms/norms0,J[k])



   end
   return x, J
end




SoftThresholding(x,ρ,λ) = sign(x)*max(abs(x)- (ρ/(2*λ)),0)





function ADMM(A,y; x0=0.0, ρ= 1.0, λ=0.5, Ni=150, Ne=50, tol=1.0e-8)

    #Check initial model
    if x0 ≠ 0.0
        x0= x0
    else
        x0=zeros(size(A,2));
    end

    # Initialize cost functions
    Ji=Float64[]; # CG cost function. 
    Je=zeros(Ne); # ADMM cost function
    norms0= norm(y .- (A*x0),2);  #||∇J||₂ => norm of the gradient at the beginining.  
  
    #Initialize variables
    ucg=zeros(length(x0)); 
    zcg=copy(x0);
    I = diagm( ones(size(G,1)));
    Ac= vcat(A, sqrt(ρ)*I);
    yc=vcat(y, sqrt(ρ)* (zcg .- ucg))
    xcg=zero(ucg)
    
    # ADDM's loops

    for k=1:Ne;
        yc=vcat(y, sqrt(ρ)* (zcg-ucg));
        xcg, Ji= CG(Ac,yc,x0=x0,Ni=Ni,tol=1.0e-15); #x-update
        zcg= SoftThresholding.(xcg .+ ucg ,ρ,λ)  # z-update
        ucg= ucg .+ (xcg-zcg); #dual update, lagrange multiploier;
        
        residual= T'*y .- (F*zcg); # residual between the data and reconstructed
        aux=abs.(residual)
      
        Je[k] =  sum(aux.^2) + λ*sum(abs.(zcg[:]))
     
        norms=norm(aux,2);

    
        if norms <= norms0*tol
            println("ADMM Iterations")
            println("At iteration k=$k is ||∇Jₖ||₂² ≤ $tol* ||∇J₀||₂²")
           println("Outer loop ended.")

            break;
        end
        
            

    end

    return zcg, Ji, Je
end
