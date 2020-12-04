using DelimitedFiles, Mill, StatsBase, Flux
using FileIO, JLD2, Statistics
using Flux: throttle, @epochs
using Mill: reflectinmodel
using Base.Iterators: repeated
using Random: shuffle
using Plots
using Distributed
using SharedArrays

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

file = "D:/Point Cloud Mnist/train.csv"

function PCM2mill(number1::Int64, number2::Int64)
    data = readdlm(file, ',', header=true, Int64)[1]
	data = data[sortperm(data[:,1]), :]
	all_labels = data[:,1]
	data = data[:, 1:end .!= 1]
    c = countmap(all_labels)
    b = Mill.length2bags([c[i] for i in sort(collect(keys(c)))])
    class1 = data[b[number1 + 1], :]
	class2 = data[b[number2 + 1], :]
	x = vcat(class1, class2)
	k = sample(3:9, 1600)
	bagids = vcat([sample(j:j, k[j]) for j = 1:1600]...) 
	bags = seqids2bags(bagids)
	labels = sample(0:1, length(bagids))
	y = labels[sortperm(labels[:,1]), :]	
	y = map(b -> maximum(y[b]), bags)
	(samples = BagNode(ArrayNode(x'), bags), labels = y)
end


(x,y) = PCM2mill(4,5)
y_oh = Flux.onehotbatch((y.+1)[:],1:2)

function CVmill(W::Int64, F::Int64, h::Int64=1)
	# This function performs [F]-times cross-validation on any mill dataset [data], for
	# every number of dense layer from 1 to [W]. Step [h] in dense layer is set 
	# to 1 by default, but if you choose for example h = 2  
	
	###### load data ######
	
	
	########################
	bag_size = length(y); #number of bags
	instance_length = length(x.data.data[:,1]); # for dense layer
	percent80 = round(Int, 0.8*floor(Int, bag_size/2))
	percent5 = round(Int, 0.05*floor(Int,bag_size/2))
	percent85 = percent80+percent5
	percent115 = bag_size - percent80 - percent5
	percent20 = round(Int, 0.2*floor(Int, bag_size/2))
	percent2 = round(Int, 0.02*floor(Int, bag_size/2))
	percent22 = percent20+percent2
	
	L_valid = zeros(W,F); #vector for loss values of validation set
	
	L_test = zeros(W,F); #vector for loss values of test set 
	
	train_sets = zeros(Int, percent85,F)
	
	validation_sets = zeros(Int,percent22,F)
	
	test_sets = zeros(Int,percent115,F)
	
	opt = Flux.ADAM();
	
			for k =1:F
			l1 = sample(1:floor(Int, bag_size/2), percent80, replace = false)
			l2 = sample((floor(Int, bag_size/2)+1):bag_size,percent5, replace = false)
			a1 = sample(symdiff(1:floor(Int, bag_size/2), l1), percent20, replace = false)
			a2 = sample(symdiff((floor(Int, bag_size/2)+1):bag_size, l1),percent2 , replace = false)
			train_set = vcat(l1,l2)
			validation_set = vcat(a1,a2)
			test_set = symdiff(1:bag_size,train_set)
			train_sets[:,k] = train_set
			validation_sets[:,k] = validation_set
			test_sets[:,k] = test_set
		end
	
			for i = 1:W
			# create the model
			model = BagModel(
			ArrayModel(Dense(instance_length, h*i, Flux.tanh)),                      		# model on the level of Flows
			SegmentedMeanMax(h*i),                                      		# aggregation
			ArrayModel(Chain(Dense(2*h*i, h*i, Flux.tanh), Dense(h*i, 2)))) ; 	# model on the level of bags
		
			### define random training, validation and testing samples
	
			for j = 1:F
			#loss function
			loss(x, y_oh) = Flux.logitcrossentropy(model(x).data, y_oh);
			#train
			Flux.train!(loss, params(model), repeated((x[train_sets[:,j]], y_oh[:,train_sets[:,j]]), 1000), opt);
			# calculate loss for each set and dense layerp
			L_valid[i,j] = loss(x[validation_sets[:,j]], y_oh[:,validation_sets[:,j]])
			L_test[i,j] = loss(x[test_sets[:,j]], y_oh[:,test_sets[:,j]])
			end
		end
		L_validCV = mean(L_valid , dims = 2);
		L_testCV = mean(L_test, dims = 2);
		rozdil = - min(L_validCV...) + min(L_testCV...)
	
		#plot all together
		
		p1 = plot(1:W, L_validCV, xlabel = "dense layer", ylabel = "loss", label = "validation set", legend =:topleft, color =:red, linewidth = 5);
		p2 = plot!(p1, 1:W, L_testCV, label = "test set", color = :blue, linewidth = 5)
		return rozdil, p2
end
k1 = CVmill(20,3)








	
