import tensorflow as tf
import time

tf.config.optimizer.set_jit(False)
gpus = tf.config.experimental.list_physical_devices('GPU')
print('gpus', gpus)
if gpus:
    try:
        # Enable memory growth for each GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

mat_size = 512
num_layers = 20

W0 = [tf.Variable(tf.random.normal([mat_size, mat_size]), name='weight0_%d'%i) for i in range(num_layers+1)]
b0 = [tf.Variable(tf.random.normal([mat_size]), name='bias0_%d'%i) for i in range(num_layers+1)]
layer_norms0 = [tf.keras.layers.LayerNormalization(axis=-1, center=True, scale=True) for i in range(num_layers)]
def branch0(matrix0):
    matrix0 = tf.matmul(matrix0, W0[num_layers]) + b0[num_layers]  # Matrix multiplication
    for i in range(num_layers):
        matrix0 = tf.subtract(matrix0, tf.eye(tf.shape(matrix0)[0]))  # Subtract identity matrix
        matrix0 = tf.transpose(matrix0) * W0[i] - b0[i]  # Dot multiplication
        matrix0 = tf.add(matrix0, tf.eye(tf.shape(matrix0)[0]))  # Add identity matrix
        matrix0 = matrix0/100
        matrix0 = layer_norms0[i](matrix0)
    matrix0 = tf.matmul(matrix0, tf.transpose(matrix0))  # Matrix multiplication
    return matrix0

W1 = [tf.Variable(tf.random.normal([mat_size, mat_size]), name='weight1_%d'%i) for i in range(num_layers+1)]
b1 = [tf.Variable(tf.random.normal([mat_size]), name='bias1_%d'%i) for i in range(num_layers+1)]
layer_norms1 = [tf.keras.layers.LayerNormalization(axis=-1, center=True, scale=True) for i in range(num_layers)]
def branch1(matrix1):
    matrix1 = tf.matmul(matrix1, W1[num_layers]) + b1[num_layers]  # Matrix multiplication
    for i in range(num_layers):
        matrix1 = tf.add(matrix1, tf.eye(tf.shape(matrix1)[0]))  # Add identity matrix
        matrix1 = tf.transpose(matrix1) * W1[i] - b1[i]  # Dot multiplication
        matrix1 = tf.subtract(matrix1, tf.eye(tf.shape(matrix1)[0]))  # Subtract identity matrix
        matrix1 = matrix1/100
        matrix1 = layer_norms1[i](matrix1)
    matrix1 = tf.matmul(matrix1, tf.transpose(matrix1))  # Matrix multiplication
    return matrix1

W2 = [tf.Variable(tf.random.normal([mat_size, mat_size]), name='weight2_%d'%i) for i in range(num_layers+1)]
b2 = [tf.Variable(tf.random.normal([mat_size]), name='bias2_%d'%i) for i in range(num_layers+1)]
layer_norms2 = [tf.keras.layers.LayerNormalization(axis=-1, center=True, scale=True) for i in range(num_layers)]
def branch2(matrix2):
    matrix2 = tf.matmul(tf.transpose(matrix2), W2[num_layers]) + b2[num_layers]  # Matrix multiplication
    for i in range(num_layers):
        matrix2 = tf.square(matrix2)  # Element-wise square
        matrix2 = matrix2 * W2[i] - b2[i]  # Dot multiplication
        matrix2 = tf.sqrt(tf.abs(matrix2))  # Element-wise square root
        matrix2 = matrix2/100
        matrix2 = layer_norms2[i](matrix2)
    matrix2 = tf.reduce_sum(matrix2, axis=1, keepdims=True)  # Sum along columns
    return matrix2

W3 = [tf.Variable(tf.random.normal([mat_size, mat_size]), name='weight3_%d'%i) for i in range(num_layers+1)]
b3 = [tf.Variable(tf.random.normal([mat_size]), name='bias3_%d'%i) for i in range(num_layers+1)]
layer_norms3 = [tf.keras.layers.LayerNormalization(axis=-1, center=True, scale=True) for i in range(num_layers)]
def branch3(matrix3):
    matrix3 = tf.matmul(tf.transpose(matrix3), W3[num_layers]) + b3[num_layers]  # Matrix multiplication
    for i in range(num_layers):
        matrix3 = tf.sqrt(tf.abs(matrix3))  # Element-wise square root
        matrix3 = matrix3 * W3[i] - b3[i]  # Dot multiplication
        matrix3 = tf.square(matrix3)  # Element-wise square
        matrix3 = matrix3/100
        matrix3 = layer_norms3[i](matrix3)
    matrix3 = tf.reduce_sum(matrix3, axis=1, keepdims=True)  # Sum along columns
    return matrix3

W = tf.Variable(tf.random.normal([mat_size*2+2, mat_size]), name='weight')
b = tf.Variable(tf.random.normal([mat_size]), name='bias')
layer_norm = tf.keras.layers.LayerNormalization(axis=-1, center=True, scale=True)
def main_branch(matrix0, matrix1, matrix2, matrix3):
    merged_result = tf.concat([matrix0, matrix1, matrix2, matrix3], axis=1)
    merged_result = tf.matmul(merged_result, W) + b
    final_result = layer_norm(merged_result)
    return final_result

# Define the multi-stream execution function. Note that defining the branch1/branch2/... in a
# tf.function is not supported, because the stream assignment cannot be pass into the function graph,
# so nodes will be executed on stream 0. Also, using "@tf.function(jit_compile=True)" will also
# lead to all the nodes inside the function be executed on stream 0.
@tf.function
def multi_stream(matrix0, matrix1, matrix2, matrix3):
    with tf.name_scope('branch_0'), tf.cuda.stream_scope(tf.cuda.get_stream(0), include_grad=True):
        matrix0 = branch0(matrix0)
    with tf.name_scope('branch_1'), tf.cuda.stream_scope(tf.cuda.get_stream(1), include_grad=True):
        matrix1 = branch1(matrix1)
    with tf.name_scope('branch_2'), tf.cuda.stream_scope(tf.cuda.get_stream(2), include_grad=True):
        matrix2 = branch2(matrix2)
    with tf.name_scope('branch_3'), tf.cuda.stream_scope(tf.cuda.get_stream(3), include_grad=True):
        matrix3 = branch3(matrix3)
    final_result = main_branch(matrix0, matrix1, matrix2, matrix3)
    return final_result, [matrix1, matrix2, matrix3]

@tf.function
def single_stream(matrix0, matrix1, matrix2, matrix3):
    with tf.name_scope('branch_0'):
        matrix0 = branch0(matrix0)
    with tf.name_scope('branch_1'):
        matrix1 = branch1(matrix1)
    with tf.name_scope('branch_2'):
        matrix2 = branch2(matrix2)
    with tf.name_scope('branch_3'):
        matrix3 = branch3(matrix3)
    final_result = main_branch(matrix0, matrix1, matrix2, matrix3)
    return final_result

loss_fn = tf.losses.CategoricalCrossentropy(from_logits=True)
optimizer = tf.optimizers.Adam(learning_rate=0.0001)
variables = [W, b] + W1 + W2 + W3 + b1 + b2 + b3

# Define the training step
@tf.function
def train_step_multistream(matrix0, matrix1, matrix2, matrix3):
    y = tf.one_hot(indices=list(range(mat_size)), depth=mat_size)
    with tf.GradientTape() as tape:
        result, hold_tensors = multi_stream(matrix0, matrix1, matrix2, matrix3)
        loss = loss_fn(y, result)
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return loss, hold_tensors

@tf.function
def train_step_singlestream(matrix0, matrix1, matrix2, matrix3):
    y = tf.one_hot(indices=list(range(mat_size)), depth=mat_size)
    with tf.GradientTape() as tape:
        result = single_stream(matrix0, matrix1, matrix2, matrix3)
        loss = loss_fn(y, result)
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return loss


# Example usage
matrix0 = tf.zeros([mat_size, mat_size], dtype=tf.float32)
matrix1 = tf.zeros([mat_size, mat_size], dtype=tf.float32)
matrix2 = tf.zeros([mat_size, mat_size], dtype=tf.float32)
matrix3 = tf.zeros([mat_size, mat_size], dtype=tf.float32)

# Warmup
for i in range(20):
    loss = train_step_singlestream(matrix0, matrix1, matrix2, matrix3)
    loss_np = loss.numpy()
    print('loss: ', loss_np)
    loss, hold_tensors = train_step_multistream(matrix0, matrix1, matrix2, matrix3)
    loss_np = loss.numpy()
    print('loss: ', loss_np)
    # Hold tensors that should work across stream groups to prevent them from being
    # released early. Unlike normal tensors, they cannot be released until all
    # operations to them on the GPU are finished. However, we cannot know when it is.
    # So we hold the tensors until we've fetched the loss value from GPU to the host
    # side, which is a sync point where we know all the operations in one training
    # step are finished on the GPU side.
    del hold_tensors
time.sleep(1)
print("Warmup finished, start benchmarking...")

# Benchmarking
start = time.time()
for i in range(100):
    loss = train_step_singlestream(matrix0, matrix1, matrix2, matrix3)
    loss_np = loss.numpy()
end = time.time()
print('Single-Stream execution time: ', end-start)

start = time.time()
for i in range(100):
    loss, hold_tensors = train_step_multistream(matrix0, matrix1, matrix2, matrix3)
    loss_np = loss.numpy()
    del hold_tensors
end = time.time()
print('Multi-Stream execution time: ', end-start)
