from kp import Manager, Tensor, OpTensorSyncDevice, OpTensorSyncLocal, OpAlgoDispatch
from pyshader import python2shader, ivec3, f32, Array

mgr = Manager()

# Can be initialized with List[] or np.Array
tensor_in_a = mgr.tensor([2, 2, 2])
tensor_in_b = mgr.tensor([1, 2, 3])
tensor_out = mgr.tensor([0, 0, 0])

sq = mgr.sequence()
sq.eval(OpTensorSyncDevice([tensor_in_a, tensor_in_b, tensor_out]))

# Define the function via PyShader or directly as glsl string or spirv bytes
@python2shader
def compute_shader_multiply(index=("input", "GlobalInvocationId", ivec3),
                            data1=("buffer", 0, Array(f32)),
                            data2=("buffer", 1, Array(f32)),
                            data3=("buffer", 2, Array(f32))):
    i = index.x
    data3[i] = data1[i] * data2[i]

#print binary  
#print(compute_shader_multiply.to_spirv())

algo = mgr.algorithm([tensor_in_a, tensor_in_b, tensor_out], compute_shader_multiply.to_spirv())

# Run shader operation synchronously
sq.eval(OpAlgoDispatch(algo))
sq.eval(OpTensorSyncLocal([tensor_out]))

assert tensor_out.data().tolist() == [2.0, 4.0, 6.0]
print("works")
