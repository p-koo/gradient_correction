from tensorflow import keras
from tfomics.layers import MultiHeadAttention


def deepsea(input_shape, num_labels, activation='relu'):
    from tensorflow.keras.constraints import max_norm
    l1_l2_reg = keras.regularizers.L1L2(l1=1e-8, l2=5e-7)
    l2_reg = keras.regularizers.L2(5e-7)

    inputs = keras.layers.Input(shape=input_shape)

    # layer 1
    nn = keras.layers.Conv1D(filters=320, kernel_size=8, use_bias=True, padding='same',
                             kernel_constraint=max_norm(0.9), kernel_regularizer=l2_reg)(inputs)      
    nn = keras.layers.Activation(activation)(nn)
    nn = keras.layers.MaxPool1D(pool_size=4)(nn)
    nn = keras.layers.Dropout(0.2)(nn)

    # layer 2
    nn = keras.layers.Conv1D(filters=480, kernel_size=8, use_bias=True, padding='same', activation='relu',
                             kernel_constraint=max_norm(0.9), kernel_regularizer=l2_reg)(nn)      
    nn = keras.layers.MaxPool1D(pool_size=4)(nn)
    nn = keras.layers.Dropout(0.2)(nn)

    # layer 3
    nn = keras.layers.Conv1D(filters=960, kernel_size=8, use_bias=True, padding='same', activation='relu',
                             kernel_constraint=max_norm(0.9), kernel_regularizer=l2_reg)(nn)      
    nn = keras.layers.MaxPool1D(pool_size=4)(nn)
    nn = keras.layers.Dropout(0.5)(nn)

    nn = keras.layers.Flatten()(nn)

    # layer 4 - Fully-connected 
    nn = keras.layers.Dense(925, activation='relu', use_bias=True, 
                            kernel_constraint=max_norm(0.9), kernel_regularizer=l2_reg)(nn)      

    # Output layer
    logits = keras.layers.Dense(num_labels, activation='linear', use_bias=True, kernel_regularizer=l1_l2_reg)(nn)
    outputs = keras.layers.Activation('sigmoid')(logits)

    # create keras model
    return keras.Model(inputs=inputs, outputs=outputs)



def danq(input_shape, num_labels, activation='relu'):
      
    inputs = keras.layers.Input(shape=input_shape)

    # layer 1
    nn = keras.layers.Conv1D(filters=320, kernel_size=26, use_bias=False, padding='valid')(inputs)      
    nn = keras.layers.Activation(activation)(nn)
    nn = keras.layers.MaxPool1D(pool_size=13)(nn)
    nn = keras.layers.Dropout(0.2)(nn)

    # layer 2 - bi-LSTM
    forward_layer = keras.layers.LSTM(320, return_sequences=True)
    backward_layer = keras.layers.LSTM(320, activation='relu', return_sequences=True, go_backwards=True)
    nn = keras.layers.Bidirectional(forward_layer, backward_layer=backward_layer)(nn)
    nn = keras.layers.Dropout(0.5)(nn)

    # layer 3 - Fully-connected 
    nn = keras.layers.Dense(75*640, activation='relu', use_bias=True)(nn)      

    # layer 4 - Fully-connected 
    nn = keras.layers.Dense(925, activation='relu', use_bias=True)(nn)      

    # Output layer
    logits = keras.layers.Dense(num_labels, activation='linear', use_bias=True)(nn)
    outputs = keras.layers.Activation('sigmoid')(logits)

    # create keras model
    return keras.Model(inputs=inputs, outputs=outputs)



def basset(input_shape, num_labels, activation='relu'):

    inputs = keras.layers.Input(shape=input_shape)

    # layer 1
    nn = keras.layers.Conv1D(filters=300, kernel_size=19, use_bias=False, padding='same')(inputs)  
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation(activation)(nn)
    nn = keras.layers.MaxPool1D(pool_size=3)(nn)

    # layer 2
    nn = keras.layers.Conv1D(filters=200, kernel_size=11, use_bias=False, padding='same', activation=None)(nn)      
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.MaxPool1D(pool_size=4)(nn)

    # layer 3
    nn = keras.layers.Conv1D(filters=200, kernel_size=7, use_bias=False, padding='same', activation=None)(nn)      
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.MaxPool1D(pool_size=4)(nn)

    nn = keras.layers.Flatten()(nn)

    # layer 4 - Fully-connected 
    nn = keras.layers.Dense(1000, activation=None, use_bias=True)(nn)      
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.Dropout(0.3)(nn)

    # layer 5 - Fully-connected 
    nn = keras.layers.Dense(1000, activation=None, use_bias=True)(nn)      
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.Dropout(0.3)(nn)

    # Output layer
    logits = keras.layers.Dense(num_labels, activation='linear', use_bias=True)(nn)
    outputs = keras.layers.Activation('sigmoid')(logits)

    # create keras model
    return keras.Model(inputs=inputs, outputs=outputs)


def deepatt(input_shape, num_labels, activation='relu'):

    inputs = keras.layers.Input(shape=input_shape)

    # layer 1
    nn = keras.layers.Conv1D(filters=1024, kernel_size=30, use_bias=True, padding='valid')(inputs)  
    nn = keras.layers.Activation(activation)(nn)
    nn = keras.layers.MaxPool1D(pool_size=15)(nn)
    nn = keras.layers.Dropout(0.2)(nn)

    # layer 2 - bi-LSTM
    forward_layer = keras.layers.LSTM(512, return_sequences=True)
    backward_layer = keras.layers.LSTM(512, return_sequences=True, go_backwards=True)
    nn = keras.layers.Bidirectional(forward_layer, backward_layer=backward_layer)(nn)

    # layer 3
    category_encoding = tf.eye(num_labels)[tf.newaxis, :, :]
    query = tf.tile(category_encoding, multiples=[tf.shape(nn)[0], 1, 1])
    nn = MultiHeadAttention(400, 4)(query, k=nn, v=nn)
    nn = keras.layers.Dropout(0.2)(nn)

    # Output layer
    nn = keras.layers.Dense(100, activation='relu', use_bias=True)(nn)      
    logits = keras.layers.Dense(1, activation='sigmoid', use_bias=True)(nn)      
    outputs = tf.reshape(logits, [-1, num_labels])

    # create keras model
    return keras.Model(inputs=inputs, outputs=outputs)



def cnn_att(input_shape, num_labels, activation='relu'):

    inputs = keras.layers.Input(shape=input_shape)

    # layer 1
    nn = keras.layers.Conv1D(filters=300, kernel_size=30, use_bias=True, padding='valid')(inputs)  
    nn = keras.layers.Activation(activation)(nn)
    nn = keras.layers.MaxPool1D(pool_size=25)(nn)
    nn = keras.layers.Dropout(0.2)(nn)

    # layer 2 - bi-LSTM
    forward_layer = keras.layers.LSTM(256, return_sequences=True)
    backward_layer = keras.layers.LSTM(256, return_sequences=True, go_backwards=True)
    nn = keras.layers.Bidirectional(forward_layer, backward_layer=backward_layer)(nn)
    nn = keras.layers.Dropout(0.2)(nn)

    # layer 3
    nn = MultiHeadAttention(256, 8)(nn, nn, nn)
    nn = keras.layers.Dropout(0.2)(nn)

    # layer 4
    nn = keras.layers.Dense(1000, activation=None, use_bias=True)(nn)      
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.Dropout(0.5)(nn)

    # Output layer
    logits = keras.layers.Dense(num_labels, activation='linear', use_bias=True)(nn)
    outputs = keras.layers.Activation('sigmoid')(logits)

    # create keras model
    return keras.Model(inputs=inputs, outputs=outputs)


def cnn_lstm_trans(input_shape, num_labels, activation='relu', num_layers=1):
    key_size = 512
    heads = 8

    inputs = Input(shape=in_shape)

    nn = keras.layers..Conv1D(filters=300, kernel_size=19, use_bias=True, padding='same')(inputs)
    nn = keras.layers..Activation(activation, name='conv_activation')(nn)
    nn = keras.layers..MaxPool1D(pool_size=25)(nn)
    nn = keras.layers..Dropout(0.1)(nn)
    
    forward = keras.layers..LSTM(256, return_sequences=True)
    backward = keras.layers..LSTM(256, activation='relu', return_sequences=True, go_backwards=True)
    nn = keras.layers..Bidirectional(forward, backward_layer=backward)(nn)
    nn = keras.layers..Dropout(0.1)(nn)
    
    nn = keras.layers..LayerNormalization(epsilon=1e-6)(nn)
    for i in range(num_layers):
        nn2,_ = MultiHeadAttention(d_model=key_size, num_heads=heads)(nn, nn, nn)
        nn2 = keras.layers..Dropout(0.1)(nn2)
        nn = keras.layers..LayerNormalization(epsilon=1e-6)(nn + nn2)
        nn2 = keras.layers..Dense(64, activation='relu')(nn)
        nn2 = keras.layers..Dense(key_size)(nn2)
        nn2 = keras.layers..Dropout(0.1)(nn2)
        nn = keras.layers..LayerNormalization(epsilon=1e-6)(nn + nn2)
    
    nn = keras.layers..Flatten()(nn)
    nn = keras.layers..Dropout(0.5)(nn)

    nn = keras.layers..Dense(1000, use_bias=False)(nn)
    nn = keras.layers..BatchNormalization()(nn)
    nn = keras.layers..Activation('relu')(nn)
    nn = keras.layers..Dropout(0.5)(nn)

    outputs = keras.layers..Dense(num_out, activation='sigmoid')(nn)

    return Model(inputs=inputs, outputs=outputs)


def cnn_trans(input_shape, num_labels, activation='relu', num_layers=1):
    key_size = 512
    heads = 8

    inputs = Input(shape=in_shape)

    nn = keras.layers..Conv1D(filters=300, kernel_size=19, use_bias=True, padding='same')(inputs)
    nn = keras.layers..Activation(activation, name='conv_activation')(nn)
    nn = keras.layers..MaxPool1D(pool_size=10)(nn)
    nn = keras.layers..Dropout(0.2)(nn)

    nn = keras.layers..Conv1D(filters=300, kernel_size=7, use_bias=False, padding='same')(inputs)
    nn = keras.layers..BatchNormalization()(nn)
    nn = keras.layers..Activation(activation, name='conv_activation')(nn)
    nn = keras.layers..MaxPool1D(pool_size=5)(nn)
    nn = keras.layers..Dropout(0.2)(nn)
    
    nn = keras.layers..LayerNormalization(epsilon=1e-6)(nn)
    for i in range(num_layers):
        nn2,_ = MultiHeadAttention(d_model=key_size, num_heads=heads)(nn, nn, nn)
        nn2 = keras.layers..Dropout(0.1)(nn2)
        nn = keras.layers..LayerNormalization(epsilon=1e-6)(nn + nn2)
        nn2 = keras.layers..Dense(64, activation='relu')(nn)
        nn2 = keras.layers..Dense(key_size)(nn2)
        nn2 = keras.layers..Dropout(0.1)(nn2)
        nn = keras.layers..LayerNormalization(epsilon=1e-6)(nn + nn2)
    
    nn = keras.layers..Flatten()(nn)
    nn = keras.layers..Dropout(0.5)(nn)

    nn = keras.layers..Dense(1000, use_bias=False)(nn)
    nn = keras.layers..BatchNormalization()(nn)
    nn = keras.layers..Activation('relu')(nn)
    nn = keras.layers..Dropout(0.5)(nn)

    outputs = keras.layers..Dense(num_out, activation='sigmoid')(nn)

    return Model(inputs=inputs, outputs=outputs)


def deepsea_custom(input_shape, num_labels, activation='relu'):
    from tensorflow.keras.constraints import max_norm
    l1_l2_reg = keras.regularizers.L1L2(l1=1e-8, l2=5e-7)
    l2_reg = keras.regularizers.L2(5e-7)

    inputs = keras.layers.Input(shape=input_shape)

    # layer 1
    nn = keras.layers.Conv1D(filters=320, kernel_size=8, use_bias=True, padding='same',
                             kernel_constraint=max_norm(0.9), kernel_regularizer=l2_reg)(inputs)      
    nn = keras.layers.Activation(activation)(nn)
    nn = keras.layers.MaxPool1D(pool_size=4)(nn)
    nn = keras.layers.Dropout(0.2)(nn)

    # layer 2
    nn = keras.layers.Conv1D(filters=480, kernel_size=8, use_bias=False, padding='same', activation=None,
                             kernel_constraint=max_norm(0.9), kernel_regularizer=l2_reg)(nn)        
    nn = keras.layers.BatchNormalization()(nn)      
    nn = keras.layers.Activation(activation)(nn)
    nn = keras.layers.MaxPool1D(pool_size=4)(nn)
    nn = keras.layers.Dropout(0.2)(nn)

    # layer 3
    nn = keras.layers.Conv1D(filters=960, kernel_size=8, use_bias=False, padding='same', activation=None,
                             kernel_constraint=max_norm(0.9), kernel_regularizer=l2_reg)(nn)      
    nn = keras.layers.BatchNormalization()(nn)        
    nn = keras.layers.Activation(activation)(nn)
    nn = keras.layers.MaxPool1D(pool_size=4)(nn)
    nn = keras.layers.Dropout(0.5)(nn)

    nn = keras.layers.Flatten()(nn)

    # layer 4 - Fully-connected 
    nn = keras.layers.Dense(925, activation=None, use_bias=False, 
                            kernel_constraint=max_norm(0.9), kernel_regularizer=l2_reg)(nn)    
    nn = keras.layers.BatchNormalization()(nn)        
    nn = keras.layers.Activation(activation)(nn)
    nn = keras.layers.Dropout(0.5)(nn)  

    # Output layer
    logits = keras.layers.Dense(num_labels, activation='linear', use_bias=True, kernel_regularizer=l1_l2_reg)(nn)
    outputs = keras.layers.Activation('sigmoid')(logits)

    # create keras model
    return keras.Model(inputs=inputs, outputs=outputs)



def danq_custom(input_shape, num_labels, activation='relu'):
      
    inputs = keras.layers.Input(shape=input_shape)

    # layer 1
    nn = keras.layers.Conv1D(filters=320, kernel_size=26, use_bias=True, padding='valid')(inputs)       
    nn = keras.layers.Activation(activation)(nn)
    nn = keras.layers.MaxPool1D(pool_size=13)(nn)
    nn = keras.layers.Dropout(0.2)(nn)

    # layer 2 - bi-LSTM
    forward_layer = keras.layers.LSTM(320, return_sequences=True)
    backward_layer = keras.layers.LSTM(320, activation='relu', return_sequences=True, go_backwards=True)
    nn = keras.layers.Bidirectional(forward_layer, backward_layer=backward_layer)(nn)
    nn = keras.layers.Dropout(0.5)(nn)

    # layer 3 - Fully-connected 
    nn = keras.layers.Dense(75*640, activation=None, use_bias=False)(nn)         
    nn = keras.layers.BatchNormalization()(nn)   
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.Dropout(0.5)(nn)

    # layer 4 - Fully-connected 
    nn = keras.layers.Dense(925, activation=None, use_bias=True)(nn)      
    nn = keras.layers.BatchNormalization()(nn)   
    nn = keras.layers.Activation('relu')(nn)

    # Output layer
    logits = keras.layers.Dense(num_labels, activation='linear', use_bias=True)(nn)
    outputs = keras.layers.Activation('sigmoid')(logits)

    # create keras model
    return keras.Model(inputs=inputs, outputs=outputs)



def basset_custom(input_shape, num_labels, activation='relu'):
    l2_reg = keras.regularizers.L2(1e-6)

    inputs = keras.layers.Input(shape=input_shape)

    # layer 1
    nn = keras.layers.Conv1D(filters=300, kernel_size=19, use_bias=True, padding='same',
                             kernel_regularizer=l2_reg)(inputs)  
    nn = keras.layers.Activation(activation)(nn)
    nn = keras.layers.MaxPool1D(pool_size=3)(nn)
    nn = keras.layers.Dropout(0.1)(nn)

    # layer 2
    nn = keras.layers.Conv1D(filters=200, kernel_size=11, use_bias=False, padding='same', activation=None,
                             kernel_regularizer=l2_reg)(nn)      
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.MaxPool1D(pool_size=4)(nn)
    nn = keras.layers.Dropout(0.2)(nn)

    # layer 3
    nn = keras.layers.Conv1D(filters=200, kernel_size=7, use_bias=False, padding='same', activation=None,
                             kernel_regularizer=l2_reg)(nn)      
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.MaxPool1D(pool_size=4)(nn)
    nn = keras.layers.Dropout(0.2)(nn)

    nn = keras.layers.Flatten()(nn)

    # layer 4 - Fully-connected 
    nn = keras.layers.Dense(1000, activation=None, use_bias=False, 
                            kernel_regularizer=l2_reg)(nn)  
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn) 
    nn = keras.layers.Dropout(0.5)(nn)

    # Output layer
    logits = keras.layers.Dense(num_labels, activation='linear', use_bias=True)(nn)
    outputs = keras.layers.Activation('sigmoid')(logits)

    # create keras model
    return keras.Model(inputs=inputs, outputs=outputs)

