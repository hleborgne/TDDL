import aidge_core
import aidge_backend_cpu ### indispensable, utilisé implicitement
import aidge_onnx
import matplotlib.pyplot as plt
import numpy as np

model_name = "circle"
onnx_filename = "model_G_"+model_name+".onnx"
model_G = aidge_onnx.load_onnx(onnx_filename)

batch_size = 1000
latent_dim = 2 # il faut le savoir, info non stockée dans le modèle!
x =np.random.randn(batch_size,latent_dim)
# originellement: x = torch.FloatTensor(torch.randn(batch_size,latent_dim))
# mais on n'a plus besoin de torch désormais!
input_tensor = aidge_core.Tensor(x)

# Create Producer Node for the Graph
input_node = aidge_core.Producer(input_tensor, "X")
# Configuration for input (optional)
input_node.get_operator().set_datatype(aidge_core.DataType.Float32)
input_node.get_operator().set_backend("cpu")
#Link Producer to the Graph
input_node.add_child(model_G)

# Configure the model for inference
model_G.set_datatype(aidge_core.DataType.Float32)
model_G.set_backend("cpu")

# Create a scheduler and run inference
scheduler = aidge_core.SequentialScheduler(model_G)
scheduler.forward(verbose=True)

for outNode in model_G.get_output_nodes():
    output_aidge = np.array(outNode.get_operator().get_output(0))
    plt.plot(output_aidge[:,0],output_aidge[:,1],'b.')
    plt.title('Inference with Aidge')
    plt.show()

