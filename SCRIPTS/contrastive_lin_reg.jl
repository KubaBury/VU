using Plots
using LaTeXStrings
using SpecialFunctions
using Distributions
using Flux
using LinearAlgebra

#generate data and set known parameters
σ = 1.0
τ = 1.0
α = 0.5
x = rand(Normal(1,σ^2*τ^2), 100)
θ0_true = [1.0]
θ1_true = [5.0]
y = θ1_true .*x .+ θ0_true .+ rand(Normal(0,σ^2*τ^2), 100)

#model
f(θ0,θ1,x,y) =  exp.( .-(((y'.-θ0.-θ1.*x).^2) ./(2σ^2)) .- ((x.^2)./(2σ^2τ^2)) )


#first q-factor with sum over y
q1(θ0,θ1,x,y) = α*diag(f(θ0,θ1,x,y)) .-
 .- (α*log.(sum(f(θ0,θ1,x,y), dims=1)))'

#second q-factor with sum over x
q2(θ0,θ1,x,y) = (1-α)*diag(f(θ0,θ1,x,y)) .-
 .- (1-α)*log.(sum(f(θ0,θ1,x,y), dims=2))


#all together with E_{p_data} ?
loss(x,y,θ0,θ1) = -mean(q1(θ0,θ1,x,y) + q2(θ0,θ1,x,y))

#intital values of unknown parameters 
θ0=[0.0]
θ1=[0.0]

#train
data = Iterators.repeated((x,y),10000)
Flux.train!((x,y)->loss(x,y,θ0,θ1), [θ0,θ1],data, ADAM())

#compare
println("θ0_true: $θ0_true, \n θ0: $θ0, \n difference: $(θ0_true.-θ0)")
println("θ1_true: $θ1_true, \n θ1: $θ1, \n difference: $(θ1_true.-θ1)")