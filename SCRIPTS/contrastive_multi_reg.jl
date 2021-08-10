using Plots
using LaTeXStrings
using SpecialFunctions
using Distributions
using Flux
using LinearAlgebra


#generate data and set known parameters
σ = 0.25
τ = 1.0
n=5 #degree of the polynom
x1 = rand(Normal(1,σ^2*τ^2), 5)
x2 = rand(Normal(2,σ^2*τ^2), 5)
x = vcat(x1,x2) 
y = 3 .*x .+ 1 + rand(Normal(0, σ^2*τ^2), 10)
α = 0:0.1:1
Θ = zeros(n+1,length(α)) #preallocated space 
θ = zeros(n+1) #preallocated space 

#define model
f(θ,x,y) =  .-(((y'.- hcat([(x).^o for o = 0:n]...)*θ).^2) ./(2σ^2)) .- ((x.^2)./(2σ^2τ^2)) 
    
    
for alfa = 1:11 #cycle over different alpha
        #first q-factor for least squares -> L2 norm
        q1(θ,x,y) = α[alfa]*sum((y.- hcat([(x).^o for o = 0:n]...)*θ).^2) #.-
       # .- (α[alfa]*logsumexp(f(θ,x,y), dims=2))'

        #second q-factor with sum over x
        q2(θ,x,y) = (1-α[alfa])*diag((f(θ,x,y))) .-
        .- ((1-α[alfa])*logsumexp(f(θ,x,y), dims=1))'

        #all together with E_{p_data} -> define loss
        loss(x,y,θ) = q1(θ,x,y) .+ -mean(q2(θ,x,y))

        #train
        data = Iterators.repeated((x,y),10000)
        Flux.train!((x,y)->loss(x,y,θ),[θ],data,ADAM())

        #save trained parameters for plot
        Θ[:,alfa] = θ
 end


#Plots
gr()
P=scatter(x,y, label=L"data~points", legend=:topleft,markershape=:cross, color=:black, markersize=6)
x3 = 0.9:0.01:2.1
W2 = [(x3).^i for i = 0:n]
X2 = hcat(W2...)
Y2 = X2*Θ
#ψ = inv(X'*X)*X'*y unstable
X=hcat([(x).^o for o = 0:n]...)
ψ1 = pinv(X'*X)*X'*y #Moore-Penrose inverse

plot!(x3,Y2, 
    label=[L"$\alpha=0.0$" L"$\alpha=0.1$" L"$\alpha=0.2$" L"$\alpha=0.3$" L"$\alpha=0.4$" L"$\alpha=0.5$" L"$\alpha=0.6$" L"$\alpha=0.7$" L"$\alpha=0.8$" L"$\alpha=0.9$" L"$\alpha=1.0$"],
    lw = [3 1 1 1 1 3 1 1 1 1 3],
    color=[:red :grey :grey :grey :grey :blue :grey :grey :grey :grey :green],
    linestyle=[:solid :dash :dash :dash :dash :solid :dash :dash :dash :dash :solid],
    legend=:topleft)
Θ
#plot!(x3,X2*ψ1, label=L"simple~reg", lw = 3, color=:purple)

