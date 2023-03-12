#=

GEOPH 531 : Geophysical inverse problems
Department of Physics
2023

Assignment3: Norms

=#


using LinearAlgebra, DataFrames, Plots,DelimitedFiles

data=readdlm("data.txt");
df=DataFrame("x"=>data[:,1], "y"=>data[:,2]);

scatter(df[!,"x"], df[!,"y"],color=:green, seriestype=:scatter)
xlabel!("x")
ylabel!("y")