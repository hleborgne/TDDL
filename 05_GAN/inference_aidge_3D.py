import aidge_core
import aidge_backend_cpu ### indispensable, utilisé implicitement
import aidge_onnx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time

from mpl_toolkits.mplot3d.art3d import Poly3DCollection # dessin cube (ground truth)

model_name = "saddle_point"
# model_name = "full_cube"
# model_name = "helix"

#================================
# to display ground truth of the cube
def get_face_cube():
  faces = []
  for i in range(6):
    faces.append(np.zeros([5,3]))
  # Bottom face
  faces[0][:,0] = [-2,-2,2,2,-2]
  faces[0][:,1] = [-2,2,2,-2,-2]
  faces[0][:,2] = [-2,-2,-2,-2,-2]
  # Top face
  faces[1][:,0] = [-2,-2,2,2,-2]
  faces[1][:,1] = [-2,2,2,-2,-2]
  faces[1][:,2] = [2,2,2,2,2]
  # Left Face
  faces[2][:,0] = [-2,-2,-2,-2,-2]
  faces[2][:,1] = [-2,2,2,-2,-2]
  faces[2][:,2] = [-2,-2,2,2,-2]
  # Left Face
  faces[3][:,0] = [2,2,2,2,2]
  faces[3][:,1] = [-2,2,2,-2,-2]
  faces[3][:,2] = [-2,-2,2,2,-2]
  # front face
  faces[4][:,0] = [-2,2,2,-2,-2]
  faces[4][:,1] = [-2,-2,-2,-2,-2]
  faces[4][:,2] = [-2,-2,2,2,-2]
  # front face
  faces[5][:,0] = [-2,2,2,-2,-2]
  faces[5][:,1] = [2,2,2,2,2]
  faces[5][:,2] = [-2,-2,2,2,-2]
  return faces

# load model
model_G = aidge_onnx.load_onnx(f'model_G_{model_name}.onnx')

# input data
batch_size = 1000
latent_dim = 3 # il faut le savoir, info non stockée dans le modèle!
x = np.random.randn(batch_size,latent_dim)
input_tensor = aidge_core.Tensor(x)

# Configure the model for inference
model_G.compile("cpu", aidge_core.dtype.float32, dims=[[batch_size, 1, 1, 3]])

# Create a scheduler and run inference
scheduler = aidge_core.SequentialScheduler(model_G)

tps1 = time.time()
scheduler.forward(data=[input_tensor])
tps2 = time.time()
print(f'temps inférence (Aidge CPU) {1000*(tps2 - tps1):4.2f} ms')

output_aidge = np.array(list(model_G.get_output_nodes())[0].get_operator().get_output(0))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
if model_name == "full_cube":
    ax.add_collection3d(Poly3DCollection(get_face_cube(), facecolors='cyan', linewidths=1, edgecolors='k', alpha=.25))
ax.plot(output_aidge[:,0],output_aidge[:,1],output_aidge[:,2],'b.')
# plt.plot(output_aidge[:,0],output_aidge[:,1],'b.')
plt.title('Inference with Aidge')
plt.show()

np.save(model_name+"_aidge_data.npy",output_aidge)
