using DelimitedFiles, Mill, StatsBase, Flux
using FileIO, JLD2, Statistics, Mill, Flux
using Flux: throttle, @epochs
using Mill: reflectinmodel
using Base.Iterators: repeated

problem  = "Musk1"


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

data = "D:/VU/SCRIPTS/DataSets/Musk1"
(x,y) = csv2mill(data)
y_oh = Flux.onehotbatch((y.+1)[:],1:2)


# create the model
model = BagModel(
    ArrayModel(Dense(166, 10, Flux.tanh)),                      # model on the level of Flows
    meanmax_aggregation(10),                                       # aggregation
    ArrayModel(Chain(Dense(21, 10, Flux.tanh), Dense(10, 2))))  # model on the level of bags


#function logitcrossentropy(ŷ, y; dims = 1, agg = mean)
#    agg(.-sum(y .* logsoftmax(ŷ; dims = dims); dims = dims))
#end


#function hybridloss1(ŷ, y, α; agg=mean)
#	agg(.-α.*sum(y .* logsoftmax(ŷ; dims = 1); dims = 1) .- (1-α).* sum(y .* logsoftmax(ŷ; dims = 2); dims = 1))
#end

# define loss function as a combination of standard cross entropy loss and end-to-end version of 
#contrastive learning a.k.a. equation 2.14. Parameter α is set at the begining of the sript.

function hybridloss(ŷ, y, α; agg=mean)
	α * agg(.-sum(y .* logsoftmax(ŷ; dims = 1); dims=1)) + (1-α)*agg(.-sum(y.*logsoftmax(ŷ; dims = 2); dims = 1))   
end


#loss(x, y_oh) = Flux.logitcrossentropy(model(x).data, y_oh)

	

# the usual way of training
evalcb = throttle(() -> @show(loss1(x, y_oh)), 1)
opt = Flux.ADAM()

α=0.5
loss1(x,y_oh) = hybridloss(model(x).data, y_oh, α)
@epochs 20  Flux.train!(loss1, Flux.params(model), repeated((x, y_oh), 1000), opt, cb=evalcb)



# calculate the error on the training set (no testing set right now)
mean(mapslices(argmax, model(x).data, dims=1)' .!= y.+1)