import tensorflow as tf
from tensorflow import keras
l=tf.keras.layers
from params import *

@tf.keras.utils.register_keras_serializable()
class YoloConv(keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding='same', bias=False, activation='swish', **kwargs):
        super(YoloConv, self).__init__(**kwargs)
        self.activation = activation
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.bias = bias
        self.cv = l.Conv2D(filters=self.filters, 
            kernel_size=self.kernel_size, 
            strides=self.strides,
            padding=self.padding,
            data_format='channels_first',
            use_bias=self.bias,
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))
        self.bn = l.BatchNormalization(axis=1)
        self.swish = l.Activation('swish')

    def call(self, inputs, training):
        output = self.cv(inputs)
        output = self.bn(output, training)
        if self.activation=='swish':
            output = self.swish(output)
        else:
            output = output
        return output

    def get_config(self):
        config = super(YoloConv, self).get_config()
        config.update({
            "activation": self.activation,
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "bias": self.bias
        })
        return config

@tf.keras.utils.register_keras_serializable()
class Elan(keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super(Elan, self).__init__(**kwargs)
        self.filters = filters
        self.cv1 = YoloConv(self.filters, 1, 1)
        self.cv2 = YoloConv(self.filters, 1, 1)
        self.cv3 = YoloConv(self.filters, 3, 1)
        self.cv4 = YoloConv(self.filters, 3, 1)
        self.cv5 = YoloConv(self.filters, 3, 1)
        self.cv6 = YoloConv(self.filters, 3, 1)
        self.cv7 = YoloConv(self.filters*4, 1, 1)
        self.concat = l.Concatenate(axis=1)
        
    def call(self, inputs, training):
        output1 = self.cv1(inputs, training)
        output2 = self.cv2(inputs, training)
        output3 = self.cv4(self.cv3(output2, training), training)
        output4 = self.cv6(self.cv5(output3, training), training)
        output = self.concat([output1, output2, output3, output4])
        output = self.cv7(output, training)
        return output

    def get_config(self):
        config = super(Elan, self).get_config()
        config.update({
            "filters": self.filters
        })
        return config

@tf.keras.utils.register_keras_serializable()
class MP(keras.layers.Layer):
    def __init__(self, filters, k=2):
        super(MP, self).__init__()
        self.filters = filters
        self.k = k
        self.cv1 = YoloConv(filters, 1, 1)
        self.cv2 = YoloConv(filters, 1, 1)
        self.cv3 = YoloConv(filters, 3, 2)
        self.pool = l.MaxPool2D(pool_size=self.k, strides=self.k, padding='same', data_format='channels_first')
        self.concat = l.Concatenate(axis=1)
        
    def call(self, inputs, training):
        output1 = self.pool(inputs)
        output1 = self.cv1(output1, training)
        output2 = self.cv2(inputs, training)
        output2 = self.cv3(output2, training)
        output = self.concat([output1, output2])
        return output

    def get_config(self):
        config = super(MP, self).get_config()
        config.update({
            "filters": self.filters,
            "k": self.k
        })
        return config

@tf.keras.utils.register_keras_serializable()    
class SPPCSPC(keras.layers.Layer):
    def __init__(self, filters, e=0.5, k=(5,9,13)):
        super(SPPCSPC, self).__init__()
        self.filters = filters
        self.e = e
        self.k = k
        c_ = int(2 * self.filters * self.e)
        self.cv1 = YoloConv(c_, 1, 1)
        self.cv2 = YoloConv(c_, 1, 1)
        self.cv3 = YoloConv(c_, 3, 1)
        self.cv4 = YoloConv(c_, 1, 1)
        self.m = [l.MaxPool2D(pool_size=x, strides=1, padding='same', data_format='channels_first') for x in k]
        self.cv5 = YoloConv(c_, 1, 1)
        self.cv6 = YoloConv(c_, 3, 1)
        self.cv7 = YoloConv(filters, 1, 1)
        self.concat = l.Concatenate(axis=1)
        
    def call(self, inputs, training):
        output1 = self.cv4(self.cv3(self.cv1(inputs, training), training), training)
        output2 = self.concat([output1] + [m(output1) for m in self.m])
        output2 = self.cv6(self.cv5(output2, training), training)
        output3 = self.cv2(inputs, training)
        output = self.cv7(self.concat([output2, output3]), training)
        return output
    
    def get_config(self):
        config = super(SPPCSPC, self).get_config()
        config.update({
            "filters": self.filters,
            "k": self.k,
            "e": self.e
        })
        return config

@tf.keras.utils.register_keras_serializable()
class Elan_A(keras.layers.Layer):
    def __init__(self, filters):
        super(Elan_A, self).__init__()
        self.filters = filters
        self.cv1 = YoloConv(filters, 1, 1)
        self.cv2 = YoloConv(filters, 1, 1)
        self.cv3 = YoloConv(filters//2, 3, 1)
        self.cv4 = YoloConv(filters//2, 3, 1)
        self.cv5 = YoloConv(filters//2, 3, 1)
        self.cv6 = YoloConv(filters//2, 3, 1)
        self.cv7 = YoloConv(filters, 1, 1)
        self.concat = l.Concatenate(axis=1)
        
    def call(self, inputs, training):
        output1 = self.cv1(inputs, training)
        output2 = self.cv2(inputs, training)
        output3 = self.cv3(output2, training)
        output4 = self.cv4(output3, training)
        output5 = self.cv5(output4, training)
        output6 = self.cv6(output5, training)
        output7 = self.concat([output1, output2, output3, output4, output5, output6])
        output = self.cv7(output7, training)
        return output
    
    def get_config(self):
        config = super(Elan_A, self).get_config()
        config.update({
            "filters": self.filters,
        })
        return config

@tf.keras.utils.register_keras_serializable()
class RepConv(keras.layers.Layer):
    def __init__(self, filters):
        super(RepConv, self).__init__()
        self.filters = filters
        self.cv1 = YoloConv(filters, 3, 1, activation=None)
        self.cv2 = YoloConv(filters, 1, 1, activation=None)
        self.swish = l.Activation('swish')
        
    def call(self, inputs, training):
        output1 = self.cv1(inputs, training)
        output2 = self.cv2(inputs, training)
        output = self.swish(output1+output2)
        return output
    
    def get_config(self):
        config = super(RepConv, self).get_config()
        config.update({
            "filters": self.filters,
        })
        return config

@tf.keras.utils.register_keras_serializable()
class IDetect(keras.layers.Layer):
    def __init__(self, shape, no, na, grids):
        super(IDetect, self).__init__()
        #self.a = tf.random.normal((1,shape,1,1), mean=0.0, stddev=0.02, dtype=tf.dtypes.float16)
        self.a = tf.Variable(tf.random.normal((1,shape,1,1), mean=0.0, stddev=0.02, dtype=tf.dtypes.float16))
        self.m = tf.Variable(tf.random.normal((1,no*na,1,1), mean=0.0, stddev=0.02, dtype=tf.dtypes.float16))
        #self.a = keras.initializers.RandomNormal(mean=0., stddev=0.02)(shape=(1,shape,1,1))
        #self.m = keras.initializers.RandomNormal(mean=0., stddev=0.02)(shape=(1,no*na,1,1))
        self.cv = YoloConv(no*na, 1, 1, bias=True, activation=None)
        self.shape = shape
        self.no = no
        self.na = na
        self.grids = grids
        self.reshape = l.Reshape([self.na, self.no, self.grids*self.grids])
        #self.permute = l.Permute([1,3,4,2])
        self.permute = l.Permute([1,3,2])
        self.activation = l.Activation('linear', dtype='float32')
    
    def call(self, inputs, training):
        #output = l.Add()([inputs, self.a])
        output = inputs + self.a
        output = self.cv(output, training)
        output = self.m * output
        #output = self.cv(inputs)
        #output = tf.reshape(output, [-1, self.na, self.no, self.grids, self.grids])
        output = self.reshape(output)
        #output = tf.transpose(output, perm=[0,1,3,4,2])
        output = self.permute(output)
        output = self.activation(output)
        return output

    def get_config(self):
        config = super(IDetect, self).get_config()
        config.update({
            "no": self.no,
            "na": self.na,
            "grids": self.grids,
            "shape": self.shape
        })
        return config

def create_model():
    inputs = keras.Input(shape=(3, img_size, img_size))
    x = YoloConv(32, 3, 1)(inputs)    #[32, img_size, img_size]
    x = YoloConv(64, 3, 2)(x)         #[64, img_size/2, img_size/2]
    x = YoloConv(64, 3, 1)(x)         #[64, img_size/2, img_size/2]
    x = YoloConv(128, 3, 2)(x)        #[128, img_size/4, img_size/4]
    x = Elan(64)(x)                   #11
    x = MP(128)(x)                    #16
    route1 = Elan(128)(x)             #24
    x = MP(256)(route1)               #29
    route2 = Elan(256)(x)             #37
    x = MP(512)(route2)               #42
    x = Elan(256)(x)                  #50
    route3 = SPPCSPC(512)(x)          #51
    x = YoloConv(256, 1, 1)(route3)
    x = l.UpSampling2D(size=(2, 2), data_format='channels_first', interpolation='nearest')(x)
    x = l.Concatenate(axis=1)([x, YoloConv(256, 1, 1)(route2)])
    route4 = Elan_A(256)(x)           #63
    x = YoloConv(128, 1, 1)(route4)
    x = l.UpSampling2D(size=(2, 2), data_format='channels_first', interpolation='nearest')(x)
    x = l.Concatenate(axis=1)([x, YoloConv(128, 1, 1)(route1)])
    route5 = Elan_A(128)(x)           #75, Connect to Detector 1
    x = MP(128)(route5)  
    x = l.Concatenate(axis=1)([x, route4])
    route6 = Elan_A(256)(x)           #88, Connect to Detector 2
    x = MP(256)(route6)   
    x = l.Concatenate(axis=1)([x, route3])
    route7 = Elan_A(512)(x)           #101, Connect to Detector 3
    detect1 = RepConv(256)(route5)
    detect2 = RepConv(512)(route6)
    detect3 = RepConv(1024)(route7)
    output1 = IDetect(256, 85, 3, 80)(detect1)
    output2 = IDetect(512, 85, 3, 40)(detect2)
    output3 = IDetect(1024, 85, 3, 20)(detect3)
    output = l.Concatenate(axis=-2)([output1, output2, output3])
    output = l.Activation('linear', dtype='float32')(output)
    model = keras.Model(inputs=inputs, outputs=output, name="yolov7_model")
    return model