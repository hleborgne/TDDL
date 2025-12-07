import torch
import torch.nn as nn

N = 16 # data size
C = 3 # number of classes

# input      : size NxC, float
# target     : size N  , integer value in [0,C-1]
# oh_target : size NxC, float values in [0,1]

target = torch.empty(N, dtype=torch.long).random_(C)
input = torch.randn(N, C) # on peut ajouter 'requires_grad=True' pour BackProp
oh_target = torch.nn.functional.one_hot(target).float()


#######################################################
### multiclass exclusive (softmax)
###    nn.CrossEntropyLoss() 
###    (naive)LogSoftmax + nn.NLLLoss()
### doc:
# https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss
# https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss
# https://pytorch.org/docs/stable/generated/torch.nn.LogSoftmax.html#torch.nn.LogSoftmax


criterion = nn.CrossEntropyLoss()
print("CrossEntropyLoss   {:1.4f}".format(criterion(input, target)))

criterion = nn.NLLLoss()
exp_in = torch.exp(input)
sum_exp_in = torch.sum(torch.exp(input),1).unsqueeze(1).expand(-1,C)
# equivalent with below (but memory is managed differently)
# sum_exp_in = torch.sum(torch.exp(input),1).unsqueeze(1).repeat(1,C)
# sum_exp_in = torch.sum(torch.exp(input),1).unsqueeze(1).repeat_interleave(C,dim=1)
# if sei=torch.sum(torch.exp(input),1) you can use e.g sei[:,None].expand(-1,C)
criterion(torch.log(torch.mul(1/sum_exp_in,exp_in)), target)
print("NLLLoss+LogSoftmax (naive) {:1.4f}".format(criterion(torch.log(torch.mul(1/sum_exp_in,exp_in)), target)))
# the (non naive implementation) LogSoftMax can be obtained with:
# note: dim is mandatory now. It is the dimension along which log_softmax iscomputed,
#       that is, the the dimension C corresponding to classes (not N, that are samples!)
lsm = nn.LogSoftmax(dim=1)
print("NLLLoss+LogSoftmax (class) {:1.4f}".format(criterion(lsm(input), target)))
# or with the "functional" version:
print("NLLLoss+LogSoftmax (func.) {:1.4f}".format(criterion(torch.nn.functional.log_softmax(input,dim=1)
, target)))

# in detail the (naive) NLLLoss is computed as
lsm_input=lsm(input) # from the LogSoftMax inputs
neg_lsm_input = -lsm_input
cumul=0
for i in range(N):
    cumul += neg_lsm_input[i,target[i]]
print("NLLLoss+LogSoftmax (doube naive) {:1.4f}".format(cumul/N))

#######################################################
### multiclass non-exclusive (multi-sigmoides)
###    nn.BCEWithLogitsLoss()
###    nn.Sigmoid() + nn.BCELoss()
### doc:
# https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html#torch.nn.BCELoss
# https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html#torch.nn.BCEWithLogitsLoss

criterion = torch.nn.BCEWithLogitsLoss()
print("BCEWithLogitsLoss  {:1.4f}".format(criterion(input, oh_target)))

criterion = nn.BCELoss()
sigmoid=nn.Sigmoid()
print("BCELoss+Sigmoid    {:1.4f}".format(criterion(sigmoid(input), oh_target)))
