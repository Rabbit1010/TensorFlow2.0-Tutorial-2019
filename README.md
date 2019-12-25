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