
using Plots
using LaTeXStrings
using SpecialFunctions
using Distributions
using Flux


σ = 2.0
τ = 2.0
α = 0.5
X = rand(Normal(0,σ^2*τ^2), 50)
Y = 2 .*X .+ 1 .+ rand(Normal(0,σ^2*τ^2), 50)


suma_y(θ0, θ1) = sum(exp.(.-(y[k].-θ0.-θ1.*x).^2 ./(2σ^2) .- (x.^2)./(2σ^2τ^2)) for k in length(x))
suma_x(θ0, θ1) = sum(exp.(.-(y.-θ0.-θ1.*x).^2 ./(2σ^2) .- (x[k].^2)./(2σ^2τ^2)) for k in length(x))

q1(θ0, θ1,x,y) = -α*( (y.-θ0.-θ1.*x).^2 ./(2σ.^2)  .+ (x.^2)./(2σ^2τ^2)   ) -
  α*log.(suma_y(θ0, θ1))

q2(θ0, θ1, x,y) = -(1-α)*( (y.-θ0.-θ1.*x).^2 ./(2σ.^2)  .+ (x.^2)./(2σ^2τ^2)   ) -
 - (1-α)*log.(suma_x(θ0, θ1))



q(x,y,θ0, θ1) = mean(q1(θ0, θ1,x,y) + q2(θ0, θ1,x,y))
#q(X,Y,1,1)

θ0=[1.0]
θ1=[1.0]
data = Iterators.repeated((X,Y),1000)
Flux.train!((X,Y)-> q(x,y,θ0,θ1), Flux.params([θ0,θ1]),data,ADAM())

θ0, θ1