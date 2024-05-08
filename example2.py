import os
import sys
import kp
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from kp import Manager, Tensor


def compile_source(source):
    open("tmp_kp_shader.comp", "w").write(source)
    os.system("glslangValidator -V tmp_kp_shader.comp -o tmp_kp_shader.comp.spv")
    return open("tmp_kp_shader.comp.spv", "rb").read()

shader = """#version 450
// precision highp float

//#define M 1024
//#define N 1024
//#define K 1024
//#define NBY4 (N>>2)
//#define KBY4 (K>>2)

layout(local_size_x = 2, local_size_y = 8, local_size_z = 1) in;
layout(binding=0) buffer inputABuffer {vec4 A[];};
layout(binding=1) buffer inputBBuffer {vec4 B[];};
layout(binding=2) buffer outputCBuffer {vec4 C[];};

int M = 1024;
int N = 1024;
int K = 1024;
int NBY4 = (N>>2);
int KBY4 = (K>>2);

void main(){
const float alpha = 1.0f;
const float beta = 0.0f;
int globalRow = int(gl_GlobalInvocationID.y); // Row ID of C (0..M)
int globalCol = int(gl_GlobalInvocationID.x); // Row ID of C (0..N)

int aOffset = globalRow * KBY4;
int bOffset = globalCol;

vec4 sum = vec4(0.0f);

for(int k = 0; k<KBY4; k++){
vec4 vecA = A[aOffset++];

sum += B[bOffset] * vecA.x;
bOffset += NBY4;

sum += B[bOffset] * vecA.y;
bOffset += NBY4;

sum += B[bOffset] * vecA.z;
bOffset += NBY4;

sum += B[bOffset] * vecA.w;
bOffset += NBY4;
}

C[globalRow * NBY4 + globalCol] = alpha * sum + beta * C[globalRow * NBY4 + globalCol];

}
"""

b = compile_source(shader)

mgr = kp.Manager()
SIZE = 32

temp_zeros = np.zeros((SIZE,SIZE), dtype=np.uint32)
temp_eye = np.eye(SIZE, dtype=np.uint32) * 90
temp_ones = np.ones((SIZE,SIZE), dtype=np.uint32) * 80

tensor_in_a = mgr.tensor(temp_eye)
tensor_in_b = mgr.tensor(temp_ones)

# Explicit type constructor supports int, in32, double, float and int
tensor_out_a = mgr.tensor_t(temp_zeros)

params = [tensor_in_a, tensor_in_b, tensor_out_a]

algorithm = mgr.algorithm(params, b)

(mgr.sequence()
    .record(kp.OpTensorSyncDevice(params))
    .record(kp.OpAlgoDispatch(algorithm))
    .eval())

sq = mgr.sequence()
sq.eval_async(kp.OpTensorSyncLocal(params))

sq.eval_await()


data  = tensor_out_a.data()

np.set_printoptions(threshold=sys.maxsize)

print(data)
print(type(data))
print(data.shape)

m = np.max(data)
print(m)
data = data / m
data = np.reshape(data, (SIZE, SIZE))
data = data * 255

plt.matshow(data)
plt.show()
