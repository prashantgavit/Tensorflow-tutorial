import tensorflow as tf

def create_placeholder(n_H0, n_W0, n_C0):
    X_o = tf.placeholder(tf.float32, shape = [None, n_H0, n_W0, n_C0])
    X_c = tf.placeholder(tf.float32, shape = [None, n_H0, n_W0, n_C0])
    Y = tf.placeholder(tf.float32, shape = [None , n_H0, n_W0])
    return X_o, X_c, Y

def initialize_parameter():
    tf.set_random_seed(1)
    
    w1 = tf.get_variable('w1', [3,3,3,64], initializer = tf.contrib.layers.xavier_initializer(seed =0))
    w2 = tf.get_variable('w2', [3,3,64,64], initializer = tf.contrib.layers.xavier_initializer(seed =1))
    w3 = tf.get_variable('w3', [5,5,64,64], initializer = tf.contrib.layers.xavier_initializer(seed =2))
    w4 = tf.get_variable('w4', [5,5,64,32], initializer = tf.contrib.layers.xavier_initializer(seed =3))
    w5 = tf.get_variable('w5', [1,1,32,16], initializer = tf.contrib.layers.xavier_initializer(seed =4))
    
    parameters = { 'w1': w1,
                   'w2': w2,
                   'w3': w3,
                   'w4': w4,
                   'w5': w5,
                 
                 }
    return(parameters)


def forward_propogation(X, parameters):
    w1 = parameters['w1']
    w2 = parameters['w2']
    w3 = parameters['w3']
    w4 = parameters['w4']
    w5 = parameters['w5']
    Z1 = tf.nn.conv2d(X, w1, strides=[1,1,1,1], padding = 'SAME')
    A1 = tf.nn.relu(Z1)
    Z2 = tf.nn.conv2d(A1, w2, strides=[1,1,1,1], padding = 'SAME')
    A2 = tf.nn.relu(Z2)
    Z3 = tf.nn.conv2d(A2, w3, strides=[1,1,1,1], padding = 'SAME')
    A3 = tf.nn.relu(Z3)
    Z4 = tf.nn.conv2d(A3, w4, strides=[1,1,1,1], padding = 'SAME')
    A4 = tf.nn.relu(Z4)
    Z5 = tf.nn.conv2d(A4, w5, strides=[1,1,1,1], padding = 'SAME')
    A5 = tf.nn.relu(Z5)

    return(A5)


def get_Euclidian_Distnace(X_o,X_c):
    return(tf.norm(X_o-X_c, ord= 'euclidean', axis =3))

def get_Lu(D):
    return (0.5*tf.square(D))

def get_Lc(D):
    return (0.5*tf.maximum(tf.zeros_like(D), 1-tf.square(D)))

def get_loss(Y,Lu,Lc):
    return tf.reduce_sum(tf.multiply(1-Y,Lu)+tf.multiply(Y,Lc))
    



    
    
                         

