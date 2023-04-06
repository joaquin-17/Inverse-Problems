
function cgaq(A,y,it)
    #We have the output (y) and the forward operator A.
    #We will try to invert for x, which is known to be a vector of ones.
     #Initial guess of zeros, true answer are ones.
     error=zeros(it)
     x=zeros(length(A[1,:]));
     #it=length(x)
     s=y.-(A*x);              #s represents the residual error, this lives in the space of the data
     p=A'*s;                   #By applying A', we are moving the residual back to the model space. Getting a model error(?)
     r=p;                      #    ??
     q=A*p                     #We are taking the model error(?) and we are projecting it back to the data space.
     old=r'*r;                 #norm squared of the model error(?) are we looking to make this small as well?
    #Conjugate gradient loop
    for k in 1:it
         alpha=(r'*r)/(q'*q);   #Ratio between the sq norm of the model error(?) and the sqnorm of its projection onto data space
         x=x+alpha.*p;          #We update our initial model guess multiplying alpha by the original model error (?)
         s=s-alpha.*q;          #We update the data error subtracting alpha*model error projected on data
         r= A'*s;               #We project the updated data error into the model space.
         new=r'*r;
         beta=new/old;          #Ratio between the new and old norm of the model error (?)
         old=new;               #Variable update
         p = r.+(beta.*p);      #Updating the model error by advancing using the ratio between new and old norm.
         q= A*p;
         println("Iteration",k)
         error[k]= new                #Taking the new model error and projecting it into the data space.
    end
    return x, error
end


function diffaq(M,n)
    #n=2
    (a,b)=size(M);
    if n==1
        global Dz=zeros(a-1,a); global Dh=zeros(b,b-1);
    #                                      1ST DERIVATIVE
        Dh[diagind(Dh,0)].=1; Dh[diagind(Dh,-1)].=-1; Dz[diagind(Dz,0)].=1; Dz[diagind(Dz,1)].=-1
    elseif n==2
        global Dz=zeros(a-2,a); global Dh=zeros(b,b-2);
    #                                      2ND DERIVATIVE
        Dh[diagind(Dh,0)].=1; Dh[diagind(Dh,-1)].=-2; Dh[diagind(Dh,-2)].=1; 
        Dz[diagind(Dz,0)].=1; Dz[diagind(Dz,1)].=-2; Dz[diagind(Dz,2)].=1
    end
    ##########################################################################################
    m=vec(M);
    Iz=Matrix(I,length(M[1,:]),length(M[1,:]));
    Ih=Matrix(I,length(M[:,1]),length(M[:,1]))
    vdm=Dz*M;
    hdm=M*Dh;
    BDz=kron(Iz,Dz);
    BDh=kron(Dh',Ih);
    if n==1
        vdvec=reshape(BDz*m,length(M[:,1])-1,length(M[1,:]));
        hdvec=reshape(BDh*m,length(M[1,:])-1,length(M[:,1]));
    elseif n==2
        vdvec=reshape(BDz*m,length(M[:,1])-2,length(M[1,:]));
        hdvec=reshape(BDh*m,length(M[:,1]),length(M[1,:])-2);
    end
    return BDh, BDz
end

function L_build(rwell,swell,gx,gz,scrd,rcrd,xgrid,ygrid,ymax,dx,dz)
    L=zeros(length(xrwell)*length(xswell),gx*gz);
    
    Js=Float64[];cellnum=Int[];Is=Float64[];Vs=Float64[]
    rayn=0;spV=Int[];
    for s in 1:length(swell[:,1])
        for r in 1:length(rwell[:,1])
                p1=[rwell[r,2] swell[s,2]];
                #println("p1 ",p1)
                p2=[rwell[r,1] swell[s,1]];
                #println("p2 ",p2)
                m=(rwell[r,2]-swell[s,2])/(rwell[r,1]-swell[s,1]); #Slope of the ray
                b=rwell[r,2]-(rwell[r,1]*m); #Intercept of the ray
                #println(m," ",b)
                #Determination grid points involved in distance calculation
                xcross=((ygrid.-b)./m); # Variable X at the fixed Y of the grid OJO Round
                ycross=((m.*xgrid).+b); # Variable Y at the fixed X of the grid OJO Round
                #println(xcross)
                #println(ycross)
                t1=[ycross' ygrid' p1;  xgrid' xcross' p2];
                #println(t1)
                t2=unique(sortslices(round.(t1;digits=2),dims=2,by=x->x[2],rev=false),dims=2); #Sorted array with all the points
                #println("T2 ", t2)
                #x=0;
                #tt=(findall(x->x>rcrd||x<scrd||isequal(x,NaN),t2));
                tt1= findall(x->x>rcrd||x<scrd,t2[2,:])
                tt2= findall(x->x>ymax||x<scrd,t2[1,:])
                tt=[tt1;tt2]
                #println(tt)
                #println(typeof(tt))
                #tt=vcat(tt1,tt2)
                #println("Index to delete :",tt1,tt2)
                a=zeros(length(tt))
                for i in 1:length(tt)
                    a[i]=tt[i][1]; #loop to extract the indices of the collumns that are out of bounds
                end
                crossmat =t2[:, 1:end .∉[a]]; #Final Cross matrix without undesired value
                #println("crossmat ",crossmat)
                #distance calculator
                d=zeros(length(crossmat[1,:])-1);
                for i in 1:length(crossmat[1,:])-1
                    d[i]=sqrt(((crossmat[1,i+1]-crossmat[1,i])^2)+(crossmat[2,i+1]-crossmat[2,i])^2);
                end
                #println("Distances :", d)
                td=sum(d);
                #Cell definition
                    c=zeros(2,length(crossmat[1,:])-1);
                for i in 1:2
                    for j in 1:length(crossmat[1,:])-1
                            c[i,j]=((crossmat[i,j+1]-crossmat[i,j])/2)+crossmat[i,j] #Calculation of segment center
                    end
                end
                #println("Centers of segments", c)            
                cellnum=zeros(length(d))
                for i in 1:length(d)
                    cellnum[i]=ceil(c[1,i]/dz)+(((ceil(c[2,i]/dx))-1)*gz)#+(gz*(temp3[2,i]-1)); #Calculation of cell number
                end
                #println("Numero de celda :",cellnum)
        #Up to here I am creating Distance vector d, and cell number cellnum for non-repeated stations
        #Now we will create an if for the cases at the borders, in the boundaries between cells
            #end
        ################################################################################################
        #It is necessary to polish what happens at the boundary between cells. According to USask Lab, #
        #The distance should be equally distributed between the two cells that share the boundary.     #
        ################################################################################################
        #STEP 3, Sparse Tomographic Matrix Creation
            rayn=rayn+1
            for i in 1:length(d)
                L[rayn,Int(cellnum[i])]=d[i]
            end
            #l=spzeros(length(sarr)); cellnum=round.(cellnum);
            #for i in 1:length(cellnum)
                #Js=push!(Js,cellnum[i]);
                #Is=push!(Is,rayn);
                #Vs=push!(Vs,d[i]);
            #end 
            #println("FIN iteration ", r)  
        end
    end
    return L
end

            #=if (s==r) && (s!=0)&&(s!=ns) && yswell[s] in [ygrid]==true
                cn1=0;cn2=0;cn3=0
                global cn1=LinRange(s,(gz*gx)-gz+s,gx)
                global cn2=cn1.-1;
                global cn3=[cn1 ; cn2]                       #Case horizontal ray traveling through interface
                cellnum=sort(cn3)
                d=zeros(length(cellnum)).+(dx/2)
            elseif (s==r) && (s==1)
                cellnum=LinRange(1,(gz*gx)-gz+1,gx);         #Case horizontal layer in the lower boundary
                d=zeros(length(cellnum)).+dx;                 #####################
            elseif (s==r) && (s==ns)                          #BOUNDARY CONDITIONS#
                cellnum=LinRange(gz,(gz*gx),gx);              #####################
                d=zeros(length(cellnum)).+dx;                #Case horizontal layer in the upper boundary
            else=#