using DelimitedFiles, Mill, StatsBase, Flux
using FileIO, JLD2, Statistics
using Flux: throttle, @epochs
using Mill: reflectinmodel
using Base.Iterators: repeated
using DataFrames
using Random

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


data = "D:/VU/SCRIPTS_/DataSets/Musk1"
(x,y) = csv2mill(data)
y_oh = Flux.onehotbatch((y.+1)[:],1:2)


for i = 2:40

	# create the model
	model = BagModel(
    ArrayModel(Dense(166, i, Flux.tanh)),                      # model on the level of Flows
    SegmentedMeanMax(i),                                       # aggregation
    ArrayModel(Chain(Dense(2*i, i, Flux.tanh), Dense(i, 2)))) ; # model on the level of bags


	# define loss function
	loss(x, y_oh) = Flux.logitcrossentropy(model(x).data, y_oh);

	# the usual way of training

	opt = Flux.ADAM();
	Flux.train!(loss, params(model), repeated((x[1:74], y_oh[:,1:74]), 1000), opt);
	# calculate the error on the testing set
	m1 = mean(mapslices(argmax, model(x).data[:,75:92], dims=1)' .!= y[75:92]) ;
	 Flux.train!(loss, params(model), repeated((x[19:92], y_oh[:,19:92]), 1000), opt);
	# calculate the error on the testing set;
	m2 = mean(mapslices(argmax, model(x).data[:,1:18], dims=1)' .!= y[1:18]);
   m = (m1 + m2)/2;
   println(m)
end
