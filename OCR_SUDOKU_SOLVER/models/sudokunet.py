import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Sequential



class SudokuNet:
    @staticmethod
    def build(width,height,depth,classes):
        model=Sequential()

        inp_shape=(height,width,depth)

        model.add(Conv2D(32,(5,5),padding="same",input_shape=inp_shape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D((2,2)))

        model.add(Conv2D(32, (3,3), padding="same", input_shape=inp_shape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D((2, 2)))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))


        model.add(Dense(64))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))

        model.add(Dense(classes))
        model.add(Activation("softmax"))


        return model