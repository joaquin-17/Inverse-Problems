

function CG(A,y; x0=0.0, Ni=160, tol=1.0e-15)
    
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



function ConjugateGradients(b,operators,parameters;x0=0.0,Ni=10,μ=0.5,tol=1.0e-18)


    if x0 ≠ 0.0
        x0 = x0
    else
        x0=zeros(size(b));
    end



    J = Float64[]
    r= b - LinearOperator(x0,operators,parameters,adj=false); #Initial residual.   
    g = LinearOperator(r,operators,parameters,adj=true) -μ*x0; # ∇J= A'(Ax-y) ==> gradient
    m = zero(g) #New model.
    s = copy(g) #Copu  the gradient
    gamma = InnerProduct(g,g) # || ∇J||₂
    gamma00 = gamma;
    J0 = InnerProduct(r,r) # Initalize cost function.
    push!(J,1.0) # push value 1.0 into J
    k=0;




    @printf "==============================================\n";
    @printf "================= CGLS- Fernanda =======================\n";
    @printf "==============================================\n";

    @printf ("k              |grad|             |grad|/|grad_0|     J  \n"); 

    @printf(" %3.0f        %12.5g    %8.3g   %8.3g   \n", k,sqrt(gamma00),1, J0)
 
    while k < Ni

        k=k+1

        t = LinearOperator(s,operators,parameters,adj=false); #Transform the gradient.
	    delta = InnerProduct(t,t) + μ*InnerProduct(s,s) #Compute delta.
	    if delta <= tol
	    println("delta reached tolerance, ending at iteration ",k)
	        break;
	    end
	    
        alpha = gamma/delta #Update alpha
 	    m = m + alpha*s #Update model with alfa and the gradient
	    r = r - alpha*t #Update the residual with alfa and projection of the gradient in the data
    	g = LinearOperator(r,operators,parameters,adj=true) #Compute a new gradient.
	    g = g - μ*m # 
	    
        gamma0 = copy(gamma) # ||∇J||
	    gamma = InnerProduct(g,g) #Compute || ∇Jk||₂
        Jk = InnerProduct(r,r) + μ*InnerProduct(m,m) #Update value of J for the iteration.
        push!(J,Jk/J0) #Push Jk into the vector J and normalize by J0
	    
        beta = gamma/gamma0 #Compue beta for the iteration.
	    s = beta*s + g #Update the gradient and goto t variable.

        norms0=sqrt(gamma00); norms=sqrt(gamma);
        @printf(" %3.0f          %12.5g      %8.3g   %8.3g \n", k,norms0,norms/norms0,Jk)

	
        #Stopping criteria: If the norm of the gradient is small enough, end.
        
        #if sqrt(gamma) <= sqrt(gamma00)*tol
         #    println("---------------------------------------------------")
          #   println("CG Iterations:")
           #  println("At iteration k=$k is ||∇Jₖ||₂² ≤ $tol* ||∇J₀||₂²")
            # println("Inner loop ended.")
            #println("---------------------------------------------------")

	       # break;
       # end
        
    end

    return m, J
end


function ConjugateGradients2(d,operators,parameters;Niter=10,mu=0,tol=1.0e-15)

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
	    println("delta reached tolerance, ending at iteration ",iter)
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
#	if (sqrt(gamma) <= sqrt(gamma00) * tol)
	#    println("tolerance reached, ending at iteration ",iter)
	#    break;
    #end
    end

    return m, cost
end


function CGLS(d_obs, operators,parameters; x0=0.0 ,μ=0.5, Ni=100, tol=1.0e-15)



    if x0 ≠ 0.0
        x0 = x0
    else
        x0=zeros(size(b));
    end

    #J=Float64[];
    #m=zeros(size(d_obs));
    m=x0;
    r= d_obs - LinearOperator(m,operators,parameters,adj=false);
    s =  LinearOperator(r,operators,parameters,adj=true) - μ*m;
    p=copy(s);

    gamma= InnerProduct(s,s);
    norms0=sqrt(gamma); #norm of the gradient is used to stop.
    norms=norms0;
    k=0;
    flag=0;
    J=zeros(Ni);
    J[1]=norm(r,2).^2

    @printf "==============================================\n";
    @printf "================= CGLS- joaquin =======================\n";
    @printf "==============================================\n";

    @printf ("k         ||∇J0||²       ||∇Jk||²    ||∇Jk||²/||∇J0||²      J   \n"); 

    @printf(" %3.0f  %12.5g   %12.5g    %8.3g    %8.3g \n", k,norms0,norms,norms/norms0,J[1])
  

    while k < Ni 


       
        k = k+1;

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
       # if norms <= norms0*tol
        #    println("Loop ended causde tolerance was reached",k)
        #    break;
        #end
        error = LinearOperator(m,operators,parameters,adj=false) - d_obs ;
        J[k] = sum( abs.(error[:]).^2 ) + μ*sum( (abs.(m)).^2);
       # J[k]=J[k]/J[1];
       # println(J[k])

       
       @printf(" %3.0f      %12.5g       %12.5g    %8.3g    %8.3g  \n", k,norms0,norms, norms/norms0,J[k])


    end

    return m, J
end
