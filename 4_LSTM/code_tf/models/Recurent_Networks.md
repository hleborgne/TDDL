What you should have in mind to understand how to build a RNN (Recurent Neural Network) ?
=========================================================================================

RNN are not composed of 'layers' but are composed of 'cells' The RNN can then be considered as a layer. With Keras, A RNN cell is a layer that takes two inputs and return two things it's 'state' and an 'output'. Here we use a LSTMCell:

```python 
self.rnn = K.layers.RNN(
    cell = K.layers.LSTMCell(units=128)
    input_shape=[28,28]
)
```

The 'state' of a cell can be considered as a sort of memory and the 'output' is more like "what the cell is saying" at an instant t. but other cells architectures also exists. For example GRU cells cobines the 'state' and the 'output' into a single 'output' as if the cell was telling 'it's memory and what it "think"' et the same time. A cell is like an agent that have to do some actions on an given input. Thus using a 'cell' without a 'simulator' makes little to no sense. Here the `K.layers.RNN` is our simulator and our `K.layers.LSTMCell` is well... our cell or agent. That is why `K.layers.RNN` takes an optionnal `input_lenght` argument.