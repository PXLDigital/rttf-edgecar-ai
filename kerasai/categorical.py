from tensorflow.python import keras
from tensorflow.python.keras import Model, Input
from tensorflow.python.keras.layers import Convolution2D, Dropout, Flatten, Dense


from kerasai import KerasPilot


def linear_unbin(arr, N=15, offset=-1, R=2.0):
    '''
    preform inverse linear_bin, taking
    one hot encoded arr, and get max value
    rescale given R range and offset
    '''
    b = np.argmax(arr)
    a = b * (R / (N + offset)) + offset
    return a


class KerasCategorical(KerasPilot):
    '''
    The KerasCategorical pilot breaks the steering and throttle decisions into discreet
    angles and then uses categorical cross entropy to train the network to activate a single
    neuron for each steering and throttle choice. This can be interesting because we
    get the confidence value as a distribution over all choices.
    This uses the dk.utils.linear_bin and dk.utils.linear_unbin to transform continuous
    real numbers into a range of discreet values for training and runtime.
    The input and output are therefore bounded and must be chosen wisely to match the data.
    The default ranges work for the default setup. But cars which go faster may want to
    enable a higher throttle range. And cars with larger steering throw may want more bins.
    '''
    def __init__(self, input_shape=(120, 160, 3), throttle_range=0.5, roi_crop=(0, 0), *args, **kwargs):
        super(KerasCategorical, self).__init__(*args, **kwargs)
        self.model = default_categorical(input_shape, roi_crop)
        self.compile()
        self.throttle_range = throttle_range

    def compile(self):
        self.model.compile(optimizer=self.optimizer, metrics=['acc'],
                           loss={'angle_out': 'categorical_crossentropy',
                                 'throttle_out': 'categorical_crossentropy'},
                           loss_weights={'angle_out': 0.5, 'throttle_out': 1.0})

    def run(self, img_arr):
        if img_arr is None:
            print('no image')
            return 0.0, 0.0

        img_arr = img_arr.reshape((1,) + img_arr.shape)
        angle_binned, throttle = self.model.predict(img_arr)
        N = len(throttle[0])
        throttle = linear_unbin(throttle, N=N, offset=0.0, R=self.throttle_range)
        angle_unbinned = linear_unbin(angle_binned)
        return angle_unbinned, throttle


def adjust_input_shape(input_shape, roi_crop):
    height = input_shape[0]
    new_height = height - roi_crop[0] - roi_crop[1]
    return (new_height, input_shape[1], input_shape[2])


def default_categorical(input_shape=(120, 160, 3), roi_crop=(0, 0)):
    opt = keras.optimizers.Adam()
    drop = 0.2

    # we now expect that cropping done elsewhere. we will adjust our expeected image size here:
    input_shape = adjust_input_shape(input_shape, roi_crop)

    img_in = Input(shape=input_shape,
                   name='img_in')  # First layer, input layer, Shape comes from camera.py resolution, RGB
    x = img_in
    x = Convolution2D(24, (5, 5), strides=(2, 2), activation='relu', name="conv2d_1")(
        x)  # 24 features, 5 pixel x 5 pixel kernel (convolution, feauture) window, 2wx2h stride, relu activation
    x = Dropout(drop)(x)  # Randomly drop out (turn off) 10% of the neurons (Prevent overfitting)
    x = Convolution2D(32, (5, 5), strides=(2, 2), activation='relu', name="conv2d_2")(
        x)  # 32 features, 5px5p kernel window, 2wx2h stride, relu activatiion
    x = Dropout(drop)(x)  # Randomly drop out (turn off) 10% of the neurons (Prevent overfitting)
    if input_shape[0] > 32:
        x = Convolution2D(64, (5, 5), strides=(2, 2), activation='relu', name="conv2d_3")(
            x)  # 64 features, 5px5p kernal window, 2wx2h stride, relu
    else:
        x = Convolution2D(64, (3, 3), strides=(1, 1), activation='relu', name="conv2d_3")(
            x)  # 64 features, 5px5p kernal window, 2wx2h stride, relu
    if input_shape[0] > 64:
        x = Convolution2D(64, (3, 3), strides=(2, 2), activation='relu', name="conv2d_4")(
            x)  # 64 features, 3px3p kernal window, 2wx2h stride, relu
    elif input_shape[0] > 32:
        x = Convolution2D(64, (3, 3), strides=(1, 1), activation='relu', name="conv2d_4")(
            x)  # 64 features, 3px3p kernal window, 2wx2h stride, relu
    x = Dropout(drop)(x)  # Randomly drop out (turn off) 10% of the neurons (Prevent overfitting)
    x = Convolution2D(64, (3, 3), strides=(1, 1), activation='relu', name="conv2d_5")(
        x)  # 64 features, 3px3p kernal window, 1wx1h stride, relu
    x = Dropout(drop)(x)  # Randomly drop out (turn off) 10% of the neurons (Prevent overfitting)
    # Possibly add MaxPooling (will make it less sensitive to position in image).  Camera angle fixed, so may not to
    # be needed

    x = Flatten(name='flattened')(x)  # Flatten to 1D (Fully connected)
    x = Dense(100, activation='relu', name="fc_1")(x)  # Classify the data into 100 features, make all negatives 0
    x = Dropout(drop)(x)  # Randomly drop out (turn off) 10% of the neurons (Prevent overfitting)
    x = Dense(50, activation='relu', name="fc_2")(x)  # Classify the data into 50 features, make all negatives 0
    x = Dropout(drop)(x)  # Randomly drop out 10% of the neurons (Prevent overfitting)
    # categorical output of the angle
    angle_out = Dense(15, activation='softmax', name='angle_out')(
        x)  # Connect every input with every output and output 15 hidden units. Use Softmax to give percentage. 15 categories and find best one based off percentage 0.0-1.0

    # continous output of throttle
    throttle_out = Dense(20, activation='softmax', name='throttle_out')(x)  # Reduce to 1 number, Positive number only

    model = Model(inputs=[img_in], outputs=[angle_out, throttle_out])
    return model
