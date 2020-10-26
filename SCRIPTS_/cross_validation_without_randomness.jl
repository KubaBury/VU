using DelimitedFiles, Mill, StatsBase, Flux
using FileIO, JLD2, Statistics
using Flux: throttle, @epochs
using Mill: reflectinmodel
using Base.Iterators: repeated

data = "D:/VU/SCRIPTS_/DataSets/Musk1";
###BrownCreeper, CorelAfrican,CorelBeach, Elephant,Fox, Musk1, Musk2, Mutagenesis1(2)
###Newsgroups1, Newsgroups2, Newsgroups3, Protein, Tiger, Web1(2,3,4), WinterWren


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

###### load data ######

(x,y) = csv2mill(data);
y_oh = Flux.onehotbatch((y.+1)[:],1:2);

###### cross-validation - specify number of folds and dense layer ###### 
K = 10; # number of folds
W = 1; # specify dense layer in bagmodel, steps are length of 5 ;

b = length(y); #number of bags
A = length(x.data.data[:,1]); # for dense layer
n = floor(Int, length(y)/K); # number of bags in each fold
L2 = zeros(W,K); # initial position of loss matrix
train_sets = zeros(Int, b-n, K);
validation_sets = zeros(Int, n, K);

opt = Flux.ADAM();

### define random training and validation samples
for j = 1:K
	l1 = (1:b)
	l2 = (1+n*(j-1):n+n*(j-1))
	q = symdiff(l1,l2)
	train_sets[:,j] = q
	validation_sets[:,j] = l2
end

for i = 1:W
	# create the model
	model = BagModel(
    ArrayModel(Dense(A, 5*i, Flux.tanh)),                      		# model on the level of Flows
    SegmentedMeanMax(5*i),                                      		# aggregation
    ArrayModel(Chain(Dense(10*i, 5*i, Flux.tanh), Dense(5*i, 2)))) ; 	# model on the level of bags

	for j = 1:K
		#loss function
		loss(x, y_oh) = Flux.logitcrossentropy(model(x).data, y_oh);
		#train
		Flux.train!(loss, params(model), repeated((x[train_sets[:,j]], y_oh[:,train_sets[:,j]]), 1000), opt);
		# calculate loss for each fold and dense layer
		L2[i,j] = loss(x[validation_sets[:,j]], y_oh[:,validation_sets[:,j]])
	end;
end

L2

