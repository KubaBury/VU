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

data = "D:/VU/SCRIPTS/DataSets/Musk1"
(x,y) = csv2mill(data)
K=5
	b = length(y) #number of bags
	n = floor(Int, length(y)/K); # number of bags in each fold
	train_sets = zeros(Int, b-n, K);
	test_sets = zeros(Int, n, K);

	opt = Flux.ADAM()

	### define random training and validation samples
	for j = 1:K
		l1 = (1:b)
		l2 = (1+n*(j-1):n+n*(j-1))
		q = symdiff(l1,l2)
		train_sets[:,j] = q
		test_sets[:,j] = l2
	end



x_train=x[1:52]	
y_train=y[1:52]
x_test=x[53:92]	
y_test=y[53:92]
y_ohtrain = Flux.onehotbatch((y_train.+1)[:],1:2)
y_ohtest = Flux.onehotbatch((y_test.+1)[:],1:2)
A = length(x.data.data[:,1])
# create the model
 # model on the level of bags


#function logitcrossentropy(ŷ, y; dims = 1, agg = mean)
#    agg(.-sum(y .* logsoftmax(ŷ; dims = dims); dims = dims))
#end


#function hybridloss1(ŷ, y, α; agg=mean)
#	agg(.-α.*sum(y .* logsoftmax(ŷ; dims = 1); dims = 1) .- (1-α).* sum(y .* logsoftmax(ŷ; dims = 2); dims = 1))
#end

# define loss function as a combination of standard cross entropy loss and end-to-end version of 
#contrastive learning a.k.a. equation 2.14. Parameter α is set at the begining of the sript.

function hybridloss(ŷ, y, α; agg=mean)
	 agg(.-sum( α.*(y .* logsoftmax(ŷ; dims = 1)) + (1-α).*(y .* logsoftmax(ŷ; dims = 2)); dims = 1))   
end


#loss(x, y_oh) = Flux.logitcrossentropy(model(x).data, y_oh)

	

# the usual way of training
#evalcb = throttle(() -> @show(loss1(x, y_oh)), 1)
opt = Flux.ADAM()

α=0:0.1:1
H=zeros(length(α))
E=zeros(length(α))

for a = 11:-1:1
	function hybridloss(ŷ, y, α; agg=mean)
		agg(.-sum( α.*(y .* logsoftmax(ŷ; dims = 1)) + (1-α).*(y .* logsoftmax(ŷ; dims = 2)); dims = 1))   
	end 

	model = BagModel(
    ArrayModel(Dense(A, 10, Flux.tanh)),                      # model on the level of Flows
    meanmax_aggregation(10),                                       # aggregation
    ArrayModel(Chain(Dense(21, 10, Flux.tanh), Dense(10, 2)))) 

loss1(x,y_oh) = hybridloss(model(x).data, y_oh, α[a])
Flux.train!(loss1, Flux.params(model), repeated((x_train, y_ohtrain), 1000), opt)#,cb=evalcb)
H[a]=loss1(x_test, y_ohtest)
E[a]=loss1(x_train,y_ohtrain)

end

H
E
plot(α, H, lw=3, color=:red, xlabel=L"$\alpha$", ylabel="loss", label="Testing data"	)
plot!(α, E, color=:blue, lw=3, label="Training data")
# calculate the error on the training set (no testing set right now)
#mean(mapslices(argmax, model(x).data, dims=1)' .!= y.+1)