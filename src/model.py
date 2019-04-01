import keras
import resnet
import keras.backend as K

global weight_decay
weight_decay = 1e-4


def Vggface2_ResNet50(input_dim=(224, 224, 3), nb_classes=8631, optimizer='sgd', mode='train'):
    # inputs are of size 224 x 224 x 3
    inputs = keras.layers.Input(shape=input_dim, name='base_input')
    x = resnet.resnet50_backend(inputs)

    # AvgPooling
    x = keras.layers.AveragePooling2D((7, 7), name='avg_pool')(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(512, activation='relu', name='dim_proj')(x)

    if mode == 'train':
        y = keras.layers.Dense(nb_classes, activation='softmax',
                               use_bias=False, trainable=True,
                               kernel_initializer='orthogonal',
                               kernel_regularizer=keras.regularizers.l2(weight_decay),
                               name='classifier_low_dim')(x)
    else:
        y = keras.layers.Lambda(lambda x: keras.backend.l2_normalize(x, 1))(x)

    # Compile
    model = keras.models.Model(inputs=inputs, outputs=y)
    if optimizer == 'sgd':
        opt = keras.optimizers.SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=True)
    else:
        opt = keras.optimizers.Adam()
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    return model

