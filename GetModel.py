#-*- coding : utf-32 -*-
import os
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.keras import backend as K

import MyConfig
import DrawLog
import SSIM_Loss
import SSIM_Metrics

modelJsonPath = r".\ModelStruct.json"   #存模型结构文件
modelWeightsPath = r".\ModelWeights.h5"     #存模型权重文件

def Squeeze_And_Excitation_Blocks(x, channel, rate=4):
    squeeze = tf.keras.layers.GlobalAveragePooling2D()(x)
    excitation = tf.keras.layers.Dense(units=channel // rate, activation = 'relu')(squeeze)
    excitation = tf.keras.layers.Dense(units=channel, activation = 'sigmoid')(excitation)
    excitation = tf.keras.layers.Reshape((1, 1, channel))(excitation)
    scale = tf.keras.layers.multiply([x, excitation])
    return scale

def atan2_layer(inputs):
    y, x = inputs
    return tf.math.atan2(y, x)

def make_model():
    tf.keras.mixed_precision.set_global_policy('mixed_float16')  #在模型里面设置数据精度
    K.set_floatx('float16') # supports float16, float32, float64

    inputs = tf.keras.layers.Input((MyConfig.IMAGE_SIZE[0], MyConfig.IMAGE_SIZE[1], 1))

    real = tf.keras.layers.Lambda(lambda x: tf.math.cos(x))(inputs)
    imag = tf.keras.layers.Lambda(lambda x: tf.math.sin(x))(inputs)

    conv1_1 = tf.keras.layers.Conv2D(32, 3, activation = 'tanh', padding = 'same')(real)
    conv1_1 = tf.keras.layers.Conv2D(32, 3, activation = 'relu', padding = 'same')(conv1_1)
    conv1_2 = tf.keras.layers.Conv2D(32, 3, activation = 'tanh', padding = 'same')(imag)
    conv1_2 = tf.keras.layers.Conv2D(32, 3, activation = 'relu', padding = 'same')(conv1_2)
    merge1 = tf.keras.layers.concatenate([conv1_1, conv1_2], axis = 3)
    se1 = Squeeze_And_Excitation_Blocks(merge1, 64, 2)
    
    conv1 = tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same')(se1)
    conv1 = tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same')(conv1)
    norm1 = tf.keras.layers.BatchNormalization()(conv1)
    drop1 = tf.keras.layers.Dropout(rate=0.25)(norm1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(drop1)
    conv1_out = tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same')(drop1)

    conv2 = tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same')(pool1)
    conv2 = tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same')(conv2)
    norm2 = tf.keras.layers.BatchNormalization()(conv2)
    drop2 = tf.keras.layers.Dropout(rate=0.25)(norm2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(drop2)
    up2_out = tf.keras.layers.UpSampling2D(size = (2, 2))(drop2)
    conv2_out = tf.keras.layers.Conv2D(64, 2, activation = 'relu', padding = 'same')(up2_out)

    conv3 = tf.keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same')(pool2)
    conv3 = tf.keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same')(conv3)
    norm3 = tf.keras.layers.BatchNormalization()(conv3)
    drop3 = tf.keras.layers.Dropout(rate=0.25)(norm3)
    up3_out = tf.keras.layers.UpSampling2D(size = (2, 2))(drop3)
    conv3_out = tf.keras.layers.Conv2D(128, 2, activation = 'relu', padding = 'same')(up3_out)
    up3_out = tf.keras.layers.UpSampling2D(size = (2, 2))(conv3_out)
    conv3_out = tf.keras.layers.Conv2D(64, 2, activation = 'relu', padding = 'same')(up3_out)

    merge4 = tf.keras.layers.concatenate([conv1_out, conv2_out, conv3_out], axis = 3)
    se4 = Squeeze_And_Excitation_Blocks(merge4, 192, 3)
    conv4 = tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same')(se4)
    conv4 = tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same')(conv4)
    norm4 = tf.keras.layers.BatchNormalization()(conv4)
    drop4 = tf.keras.layers.Dropout(rate=0.25)(norm4)
    conv4_1 = tf.keras.layers.Conv2D(64, 3, activation = 'linear', padding = 'same')(drop4)
    conv4_1 = tf.keras.layers.Conv2D(32, 2, activation = 'tanh', padding = 'same')(conv4_1)
    conv4_1 = tf.keras.layers.Conv2D(1, 1, activation = 'tanh', padding = 'same', name="real")(conv4_1)
    conv4_2 = tf.keras.layers.Conv2D(64, 3, activation = 'linear', padding = 'same')(drop4)
    conv4_2 = tf.keras.layers.Conv2D(32, 2, activation = 'tanh', padding = 'same')(conv4_2)
    conv4_2 = tf.keras.layers.Conv2D(1, 1, activation = 'tanh', padding = 'same', name="imag")(conv4_2)

    phase = tf.keras.layers.Lambda(lambda x: tf.math.atan2(x[0], x[1]), name="phase")([conv4_2, conv4_1])

    model = tf.keras.Model(inputs, [conv4_1, conv4_2, phase])   #卷积层也作为输出层

    model.build(input_shape=(None, MyConfig.IMAGE_SIZE[0], MyConfig.IMAGE_SIZE[1], 1))

    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 5e-4),
                  loss = [SSIM_Loss.ssim_l1_loss, SSIM_Loss.ssim_l1_loss, SSIM_Loss.ssim_psnr_loss],
                  metrics = [SSIM_Metrics.ssim_metrics, SSIM_Metrics.ssim_metrics_2pi])

    #tf.keras.utils.plot_model(model, "End2End_SE_MSCNN.png", show_shapes=True)

    return model

def get_model():
    model = None
    if(os.path.exists(modelJsonPath) and os.path.exists(modelWeightsPath)):
        custon_objs = {'ssim_metrics':SSIM_Metrics.ssim_metrics,
                       'ssim_metrics_2pi':SSIM_Metrics.ssim_metrics_2pi,
                       'ssim_l1_loss':SSIM_Loss.ssim_l1_loss,
                       'ssim_psnr_loss':SSIM_Loss.ssim_psnr_loss,
                       }
        with open(modelJsonPath, "r+") as js_file:
            model = tf.keras.models.model_from_json(js_file.read(),custom_objects=custon_objs)
        model.load_weights(modelWeightsPath)
        model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 5e-4),
                      loss = [SSIM_Loss.ssim_l1_loss, SSIM_Loss.ssim_l1_loss, SSIM_Loss.ssim_psnr_loss],
                      metrics = [SSIM_Metrics.ssim_metrics, SSIM_Metrics.ssim_metrics_2pi])
    else:
        print("Model Doesn't Exist!")
        model = make_model()
        DrawLog.init_log()
    model.summary()

    return model

def save_model(model):
    model.save_weights(modelWeightsPath)
    with open(modelJsonPath, "w+") as js_file:
        js_file.write(model.to_json())
