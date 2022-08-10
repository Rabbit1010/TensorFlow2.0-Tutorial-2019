# TensorFlow2.0-Tutorial-2019
An updated TensorFlow2.0 tutorial for teaching purpose with PowerPoint explanations. (in progress)

## Schedule
1. Topic 1
	* Build sequential model using tf.keras.Sequential()
	* Set optimizer and loss function using model.compile()
	* Simple image classification example (MINST)
	* Simple text classifcation example (IMDB)
2. Topic 2
	* Build model using TensorFlow Keras functional API
	* Simple ResNet example
	* Simple U-Net example
3. Topic 3
	* Data Input Pipeline using tf.data.Dataset
	* Online data augmentation using map()
	* Arbitrary Python functions using tf.py_function()
4. Topic 4
	* Custom training loop
	* Anime face generation using DCGAN

## Specify which GPU(s) to use
You can either set Linux environment variable:
```bash
export CUDA_VISIBLE_DEVICES = 1 # use the 2nd GPU
```
or in Python script
```Python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1' # use the 1st and 2nd GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # use only GPU
import tensorflow as tf # environment variable has to be changed before importing TensorFlow
```
or in Tensorflow (Also controls TensorFlow GPU memory behaviour)
```Python
# Tensorflow GPU control
gpu_idx = 0
limit_memory = True

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[gpu_idx], 'GPU')
        if limit_memory == True:
        	tf.config.experimental.set_memory_growth(gpus[gpu_idx], True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        print('Using GPU num: {}'.format(gpu_idx))
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)
```

## tf.keras.utils.plot_model() Issues
In Windows, try installing pydot and graphviz using conda:
```bash
conda install -c https://conda.binstar.org/t/TOKEN/j14r pydot
conda install -c https://conda.binstar.org/t/TOKEN/j14r graphviz
```

In Linux, try installing the following:
```bash
pip install pydot-ng
conda install graphviz
```
