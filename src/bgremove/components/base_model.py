##################################################################
#
# Importing MOdules
#
###############################################################
from bgremove import logger
from bgremove.constants import PARAMS_FILE_PATH
from bgremove.utils.common import read_yaml
from bgremove.entity.config_entity import PrepareBaseModelConfig
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPool2D, UpSampling2D, Concatenate, Add
from tensorflow.keras.optimizers import Adam
############################################################################
# 
# Model Class The Fully MOdel Archietectur
#
################################################################################

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
        
    def conv_block(self, inputs, out_ch, rate=1):
        x = Conv2D(out_ch, 3, padding="same", dilation_rate=1)(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x
    
    def RSU_L(self, inputs, out_ch, int_ch, num_layers, rate=2):
#         initial Conv
        x = self.conv_block(inputs, out_ch)
        init_features = x
        
#         Encoders
        skip = []
        x = self.conv_block(x, int_ch)
        skip.append(x)
        
        for i in range(num_layers-2):
            x = MaxPool2D((2,2))(x)
            x = self.conv_block(x, int_ch)
            skip.append(x)

#         bridge
        x = self.conv_block(x, int_ch, rate=rate)
    
#         decoder
        skip.reverse()
        
        x = Concatenate()([x, skip[0]])
        x = self.conv_block(x, int_ch)

        for i in range(num_layers-3):
            x = UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
            x = Concatenate()([x, skip[i+1]])
            x = self.conv_block(x, int_ch)

        x = UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
        x = Concatenate()([x, skip[-1]])
        x = self.conv_block(x, out_ch)

        """ Add """
        x = Add()([x, init_features])
        return x
    

    def RSU_4F(self, inputs, out_ch, int_ch):
        """ Initial Conv """
        x0 = self.conv_block(inputs, out_ch, rate=1)

        """ Encoder """
        x1 = self.conv_block(x0, int_ch, rate=1)
        x2 = self.conv_block(x1, int_ch, rate=2)
        x3 = self.conv_block(x2, int_ch, rate=4)

        """ Bridge """
        x4 = self.conv_block(x3, int_ch, rate=8)

        """ Decoder """
        x = Concatenate()([x4, x3])
        x = self.conv_block(x, int_ch, rate=4)

        x = Concatenate()([x, x2])
        x = self.conv_block(x, int_ch, rate=2)

        x = Concatenate()([x, x1])
        x = self.conv_block(x, out_ch, rate=1)

        """ Addition """
        x = Add()([x, x0])
        return x
    
    def u2net(self,input_shape, out_ch, int_ch, num_classes):
        """ Input Layer """
        inputs = Input(input_shape)
        s0 = inputs

        """ Encoder """
        s1 = self.RSU_L(s0, out_ch[0], int_ch[0], 7)
        p1 = MaxPool2D((2, 2))(s1)

        s2 = self.RSU_L(p1, out_ch[1], int_ch[1], 6)
        p2 = MaxPool2D((2, 2))(s2)

        s3 = self.RSU_L(p2, out_ch[2], int_ch[2], 5)
        p3 = MaxPool2D((2, 2))(s3)

        s4 = self.RSU_L(p3, out_ch[3], int_ch[3], 4)
        p4 = MaxPool2D((2, 2))(s4)

        s5 = self.RSU_4F(p4, out_ch[4], int_ch[4])
        p5 = MaxPool2D((2, 2))(s5)

        """ Bridge """
        b1 = self.RSU_4F(p5, out_ch[5], int_ch[5])
        b2 = UpSampling2D(size=(2, 2), interpolation="bilinear")(b1)

        """ Decoder """
        d1 = Concatenate()([b2, s5])
        d1 = self.RSU_4F(d1, out_ch[6], int_ch[6])
        u1 = UpSampling2D(size=(2, 2), interpolation="bilinear")(d1)

        d2 = Concatenate()([u1, s4])
        d2 = self.RSU_L(d2, out_ch[7], int_ch[7], 4)
        u2 = UpSampling2D(size=(2, 2), interpolation="bilinear")(d2)

        d3 = Concatenate()([u2, s3])
        d3 = self.RSU_L(d3, out_ch[8], int_ch[8], 5)
        u3 = UpSampling2D(size=(2, 2), interpolation="bilinear")(d3)

        d4 = Concatenate()([u3, s2])
        d4 = self.RSU_L(d4, out_ch[9], int_ch[9], 6)
        u4 = UpSampling2D(size=(2, 2), interpolation="bilinear")(d4)

        d5 = Concatenate()([u4, s1])
        d5 = self.RSU_L(d5, out_ch[10], int_ch[10], 7)

        """ Side Outputs """
        y1 = Conv2D(num_classes, 3, padding="same")(d5)

        y2 = Conv2D(num_classes, 3, padding="same")(d4)
        y2 = UpSampling2D(size=(2, 2), interpolation="bilinear")(y2)

        y3 = Conv2D(num_classes, 3, padding="same")(d3)
        y3 = UpSampling2D(size=(4, 4), interpolation="bilinear")(y3)

        y4 = Conv2D(num_classes, 3, padding="same")(d2)
        y4 = UpSampling2D(size=(8, 8), interpolation="bilinear")(y4)

        y5 = Conv2D(num_classes, 3, padding="same")(d1)
        y5 = UpSampling2D(size=(16, 16), interpolation="bilinear")(y5)

        y6 = Conv2D(num_classes, 3, padding="same")(b1)
        y6 = UpSampling2D(size=(32, 32), interpolation="bilinear")(y6)

        y0 = Concatenate()([y1, y2, y3, y4, y5, y6])
        y0 = Conv2D(num_classes, 3, padding="same")(y0)

        y0 = Activation("sigmoid", name="y0")(y0)
        y1 = Activation("sigmoid", name="y1")(y1)
        y2 = Activation("sigmoid", name="y2")(y2)
        y3 = Activation("sigmoid", name="y3")(y3)
        y4 = Activation("sigmoid", name="y4")(y4)
        y5 = Activation("sigmoid", name="y5")(y5)
        y6 = Activation("sigmoid", name="y6")(y6)

        model = tf.keras.models.Model(inputs, outputs=[y0, y1, y2, y3, y4, y5, y6])
        return model
    
    def get_base_model(self, params_filepath = PARAMS_FILE_PATH):
        params_file_path = params_filepath
        params = read_yaml(params_file_path)
        model = self.u2net(params.IMAGE_SIZE, params.OUT_CH, params.INT_CH, num_classes=params.CLASSES)
        model.compile(loss="binary_crossentropy", optimizer=Adam(params.LEARNING_RATE))
        return model
    
    
    