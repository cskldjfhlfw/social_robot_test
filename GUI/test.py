#检查python、tensorflow、pyqt5、numpy、pandas的版本
import sys
import tensorflow as tf
import numpy
import pandas
import PyQt5.QtCore
import sklearn
print('sklearn',sklearn.__version__)
print('PyQt5版本',PyQt5.QtCore.PYQT_VERSION_STR)
print("Python version:", sys.version)
print("TensorFlow version:", tf.__version__)
print("Numpy version:", numpy.__version__)
print("Pandas version:", pandas.__version__)