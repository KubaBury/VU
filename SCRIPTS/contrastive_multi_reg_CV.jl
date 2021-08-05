using Plots
using LaTeXStrings
using SpecialFunctions
using Distributions
using Flux
using LinearAlgebra


#generate data and set known parameters
σ = 0.2
τ = 1.0
α = 0:0.1:1
x1 = rand(Normal(1,σ^2*τ^2), 5)
x2 = rand(Normal(2,σ^2*τ^2), 5)
x_train = vcat(x1,x2)
y_train = 2 .*x_train .+ 1 + rand(Normal(0, 0.2), 10)
x_test = 1:0.1:2
y_test = 2 .*x_test .+ 1 


k = 9  #rad polynomu


L1= zeros(k, length(α))
L2= zeros(k, length(α))

for n = 1:k
Θ=zeros(n+1,length(α))

#model
f(θ,x,y) =  exp.( .-(((y'.- hcat([(x).^o for o = 0:n]...)*θ).^2) ./(2σ^2)) .- ((x.^2)./(2σ^2τ^2)) )

    for alpha = 1:11
        θ = zeros(n+1)
     #first q-factor with sum over y
        q1(θ,x,y) = α[alpha]*diag(f(θ,x,y)) .-
        .- (α[alpha]*log.(sum(f(θ,x,y), dims=1)))'

        #second q-factor with sum over x
        q2(θ,x,y) = (1-α[alpha])*diag(f(θ,x,y)) .-
        .- (1-α[alpha])*log.(sum(f(θ,x,y), dims=2))

        #all together with E_{p_data} ?
        loss(x,y,θ) = -mean(q1(θ,x,y) + q2(θ,x,y))

        #train
        data = Iterators.repeated((x_train,y_train),10000)
        Flux.train!((x_train,y_train)->loss(x_train,y_train,θ), [θ],data, ADAM())

        #save loss
        L1[n,alpha] = loss(x_train,y_train,θ)
        L2[n,alpha] = loss(x_test,y_test,θ)
    end
end
  L1
  L2


plot(1:k,L2)
plot(1:k,L1)