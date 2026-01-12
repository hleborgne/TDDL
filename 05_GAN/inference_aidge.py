import aidge_core
import aidge_backend_cpu ### indispensable, utilisé implicitement
import aidge_onnx
import matplotlib.pyplot as plt
import numpy as np
import time

model_name = "circle"
# model_name = "double_sin"


#================================
# load model
model_G = aidge_onnx.load_onnx(f'model_G_{model_name}.onnx')

# input data
batch_size = 1000
latent_dim = 2 # il faut le savoir, info non stockée dans le modèle!
x = np.random.randn(batch_size,latent_dim)
input_tensor = aidge_core.Tensor(x)

# Configure the model for inference
model_G.compile("cpu", aidge_core.dtype.float32, dims=[[batch_size, 1, 1, 2]])

# Create a scheduler and run inference
scheduler = aidge_core.SequentialScheduler(model_G)

tps1 = time.time()
scheduler.forward(data=[input_tensor])
tps2 = time.time()
print(f'temps inférence (Aidge CPU) {1000*(tps2 - tps1):4.2f} ms')

output_aidge = np.array(list(model_G.get_output_nodes())[0].get_operator().get_output(0))
plt.plot(output_aidge[:,0],output_aidge[:,1],'b.')
plt.title('Inference with Aidge')
plt.show()

np.save(model_name+"_aidge_data.npy",output_aidge)

