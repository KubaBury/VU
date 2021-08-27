using DelimitedFiles, Mill, StatsBase, Flux
using FileIO, JLD2, Statistics, Mill, Flux
using Flux: throttle, @epochs
using Mill: reflectinmodel
using Base.Iterators: repeated
using LaTeXStrings
using Plots

problem  = "Tiger"


function seqids2bags(bagids)
	c = countmap(bagids)
	Mill.length2bags([c[i] for i in sort(collect(keys(c)))])
end

function csv2mill(problem)
	x=readdlm("$(problem)/data.csv",'\t',Float32)
	bagids = readdlm("$(problem)/bagids.csv",'\t',Int)[:]
	bags = seqids2bags(bagids)
	y = readdlm("$(problem)/labels.csv",'\t',Int)
	y = map(b -> maximum(y[b]), bags)
	(samples = BagNode(ArrayNode(x), bags), labels = y)
end

data = "D:/VU/SCRIPTS/DataSets/Tiger"
(x,y) = csv2mill(data)


K=5
A = length(x.data.data[:,1]) # for dense layer
b = length(y) #number of bags
n = floor(Int, length(y)/K) # number of bags in each fold
L1 = zeros(W,K) # initial position of loss array
L2 = zeros(W,K)
L3 = zeros(K)
L4 = zeros(K)
tr_set = zeros(Int, b-n, K)
te_set = zeros(Int, n, K)
### define random training and testing samples
for  j =1:K
	r1 = shuffle(1:b)
	r2 = sample(1:b, n, replace = false)
	q = symdiff(r1, r2)
	tr_set[:,j] = q 
	te_set[:,j] = r2
end

# define loss function as a combination of standard cross entropy loss and end-to-end version of 
#contrastive learning a.k.a. equation 2.14. Parameter α is set at the begining of the sript.

loss(x, y_oh) = Flux.logitcrossentropy(model(x).data, y_oh)

function hybridloss(ŷ, y, α; agg=mean)
	agg(.-sum( α.*(y .* logsoftmax(ŷ; dims = 1)) + (1-α).*(y .* logsoftmax(ŷ; dims = 2)); dims = 1))   
end 

opt = Flux.ADAM()
α=0:0.1:1
H=zeros(length(α), K)
E=zeros(length(α), K)

for a = 1:11
	for j = 1:K
		model = BagModel(
    ArrayModel(Dense(A, 10, Flux.tanh)),                      # model on the level of Flows
    meanmax_aggregation(10),                                       # aggregation
    ArrayModel(Chain(Dense(21, 10, Flux.tanh), Dense(10, 2)))) 
	loss1(x,y_oh) = hybridloss(model(x).data, y_oh, α[a])
	Flux.train!(loss1, Flux.params(model), repeated((x[tr_set[:,j]], y_oh[:, tr_set[:,j]]), 1000), opt)#,cb=evalcb)
	H[a,j]=loss(x[te_set[:,j]], y_oh[:, te_set[:,j]])
	E[a,j]=loss(x[tr_set[:,j]], y_oh[:, tr_set[:,j]])
	end
end

plot(α, mean(H, dims=2), lw=3, color=:red, xlabel=L"$\alpha$", ylabel="Prediction Error", label="Testing data",
xtickfont=font(11), 
    ytickfont=font(11),
    guidefont=font(14),
    legendfont=font(11)	)
plot!(α, mean(E, dims=2), color=:blue, lw=3, label="Training data")

# calculate the error on the training set (no testing set right now)

mean(mapslices(argmax, model(x).data, dims=1)' .!= y.+1)