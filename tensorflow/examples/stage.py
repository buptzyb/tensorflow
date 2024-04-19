import tensorflow as tf
from tensorflow.python.ops import data_flow_ops
import numpy as np

tf.config.threading.set_inter_op_parallelism_threads(5)

class MyModel(tf.Module):
    def __init__(self, num_samples, sample_shape, num_steps):
        # Create a dataset of random data
        dataset = tf.data.Dataset.from_tensor_slices(tf.random.uniform(
            shape=(num_samples,) + sample_shape, minval=0, maxval=256, dtype=tf.float32))
        dataset = dataset.batch(40)
        # dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        self.dataset_iter = iter(dataset.take(num_steps))

        # StagingArea of Tensors that will be accessed from the CPU side in training
        cpu_data_shape = [None, 512, 1]  # Example shape
        self.cpu_tensors_queue = data_flow_ops.StagingArea(dtypes=[tf.float32], shapes=[cpu_data_shape])

        # StagingArea of Tensors that will be accessed from the GPU side in training
        gpu_data_shape = [None, 512, 256]  # Example shape
        self.gpu_tensors_queue = data_flow_ops.StagingArea(dtypes=[tf.float32], shapes=[gpu_data_shape])

    @tf.function
    def process_data(self, data):
        # CPU data process
        with tf.device('/CPU:0'):
            output_cpu = tf.reduce_mean(data, axis=2, keepdims=True)
            input_gpu = tf.tile(output_cpu, [1, 1, 512])

        # GPU data process
        with tf.device('/GPU:0'):
            w1 = tf.constant(np.random.normal(loc=0.0, scale=1.0, size=(512, 256)), dtype=tf.float32)
            output_gpu = tf.matmul(input_gpu, w1)
            w2 = tf.constant(np.random.normal(loc=0.0, scale=1.0, size=(256, 256)), dtype=tf.float32)
            for _ in range(50):
                output_gpu = tf.matmul(output_gpu*0.01, w2)

        return output_cpu, output_gpu

    @tf.function
    def train_step(self, data_cpu, data_gpu):
        with tf.device('/CPU:0'):
            data_cpu = tf.pow(data_cpu, 2)
        with tf.device('/GPU:0'):
            data = data_cpu + data_gpu
            w1 = tf.constant(np.random.normal(loc=0.0, scale=1.0, size=(256, 64)), dtype=tf.float32)
            w2 = tf.constant(np.random.normal(loc=0.0, scale=1.0, size=(64, 512)), dtype=tf.float32)
            w3 = tf.constant(np.random.normal(loc=0.0, scale=1.0, size=(512, 256)), dtype=tf.float32)
            for _ in range(200):
                data = tf.matmul(data, w1)
                data = tf.matmul(data, w2)
                data = tf.matmul(data, w3)
                data = data*0.0001 + 1
            output = tf.transpose(data)
        return output

    @tf.function
    def prefetch(self):
        # Prefetch data for the next step.
        data = self.dataset_iter.get_next()
        output_cpu, output_gpu = self.process_data(data)
        with tf.device('/CPU:0'):
            self.cpu_tensors_queue.put(output_cpu)
        with tf.device('/GPU:0'):
            self.gpu_tensors_queue.put(output_gpu)

    @tf.function
    def train(self):
        # Get data from StagingArea and train.
        with tf.device('/CPU:0'):
            data_cpu = self.cpu_tensors_queue.get()
        with tf.device('/GPU:0'):
            data_gpu = self.gpu_tensors_queue.get()
        out = self.train_step(data_cpu, data_gpu)
        return out

    @tf.function
    def prefetch_and_train(self):
        self.prefetch()
        out = self.train()
        return out

    def save_pb(self):
        # data_signature = tf.TensorSpec(shape=[None], dtype=tf.float32)
        concrete_func = self.prefetch_and_train.get_concrete_function()
        frozen_graph_def = concrete_func.graph.as_graph_def()
        tf.io.write_graph(frozen_graph_def,
                  logdir="./",
                  name="frozen_model.pbtxt",
                  as_text=True)

if __name__ == '__main__':
    # Define input parameters
    num_samples = 1000
    sample_shape = (512, 512)
    num_steps = 25
    
    # Define model
    model = MyModel(num_samples, sample_shape, num_steps)
    model.save_pb()

    # Benchmark the input pipeline
    for i in range(num_steps+1):
        if i == 0:
            model.prefetch()
            print('first step, just prefetch data.')
        elif i == num_steps:
            output = model.train()
            print('last step, ', i, output[0, 0, 0])
        else:
            output = model.prefetch_and_train()
            print('step ', i, output[0, 0, 0])
