using DelimitedFiles, Mill, StatsBase, Flux
using FileIO, JLD2, Statistics
using Flux: throttle, @epochs
using Mill: reflectinmodel
using Base.Iterators: repeated
using Plots
using Random

data = "D:/VU/SCRIPTS_/DataSets/Tiger";
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

W = 50
h = 1
b = length(y); #number of bags
A = length(x.data.data[:,1]); # for dense layer
L_valid = zeros(W,1); # initial position of loss matrix
L_test1 = zeros(W,1)
L_test2 = zeros(W,1)
    
opt = Flux.ADAM(0.0001)

### define random training and validation samples
    
for i = 1:W
	# create the model
	model = BagModel(
    ArrayModel(Dense(A, h*i, Flux.tanh)),                      		# model on the level of Flows
    SegmentedMeanMax(h*i),                                      		# aggregation
    ArrayModel(Chain(Dense(2*h*i, h*i, Flux.tanh), Dense(h*i, 2)))) ; 	# model on the level of bags
    
    l1 = sample(1:100, 80, replace = false)
    l2 = sample(101:200, 5, replace = false)
    a1 = sample(symdiff(1:100, l1), 8, replace = false)
    a2 = sample(symdiff(101:200, l1),1, replace = false)
    b1 = symdiff((1:100), vcat(l1,a1))
    b2 = symdiff((101:200), vcat(l2,a2))
    train_set = vcat(l1,l2)
    validation_set = vcat(a1,a2)
    test_set1 = vcat(b1,b2)
    test_set2 = symdiff(shuffle(1:b),train_set)
	    
	#loss function
	loss(x, y_oh) = Flux.logitcrossentropy(model(x).data, y_oh);
	#train
	Flux.train!(loss, params(model), repeated((x[train_set], y_oh[:,train_set]), 1000), opt);
	# calculate loss for each fold and dense layer
	L_valid[i] = loss(x[validation_set], y_oh[:,validation_set])
    L_test1[i] = loss(x[test_set1], y_oh[:,test_set1])
    L_test2[i] = loss(x[test_set2], y_oh[:,test_set2])
end

L_valid;
L_test1;
L_test2;

p1 = plot(1:W, L_valid, xlabel = "dense layer", ylabel = "loss", label = "validation set", legend =:topleft, color =:red, linewidth = 3);
p2 = plot!(p1, 1:W, L_test1, label = "test set (1)", color = :blue, linewidth = 3);
p3 = plot!(p2, 1:W, L_test2, label = "test set (2)", color = :green, linewidth = 3)
