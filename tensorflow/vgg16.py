
# coding: utf-8

# In[1]:

import os
import numpy as np
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# In[2]:

def conv_op(input_op, name, kh, kw, n_out, dh, dw):  
    
    input_op = tf.convert_to_tensor(input_op)  
    n_in = input_op.get_shape()[-1].value  
    
    with tf.name_scope(name) as scope:  
        kernel = tf.get_variable(scope+"w",  
                                shape = [kh, kw, n_in, n_out],  
                                dtype = tf.float32,  
                                initializer = tf.contrib.layers.xavier_initializer_conv2d())  
        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding = 'SAME')  
        bias_init_val = tf.constant(0.0, shape = [n_out], dtype = tf.float32)  
        biases = tf.Variable(bias_init_val, trainable = True, name = 'b')  
        z = tf.nn.bias_add(conv, biases)  
        activation = tf.nn.relu(z, name = scope)  
        
        return activation  
    
def fc_op(input_op, name, n_out):  
    
    n_in = input_op.get_shape()[-1].value 
    
    with tf.name_scope(name) as scope:  
        kernel = tf.get_variable(scope+'w',  
                                shape = [n_in, n_out],  
                                dtype = tf.float32,  
                                initializer = tf.contrib.layers.xavier_initializer())  
        biases = tf.Variable(tf.constant(0.1, shape = [n_out], dtype = tf.float32), name = 'b')  

        activation = tf.nn.relu_layer(input_op, kernel, biases, name = scope)  
        
        return activation  

def max_pool_op(input_op, name, kh, kw, dh, dw):  
    
    return  tf.nn.max_pool(input_op,  
                           ksize = [1, kh, kw, 1],  
                           strides = [1, dh, dw, 1],  
                           padding = 'SAME',  
                           name = name)  


# In[3]:

def vgg_16(input_op, num_classes, keep_prob):  
    
    end_point = {}
    # block 1 -- outputs 112x112x64  
    conv1_1 = conv_op(input_op, name="conv1_1", kh=3, kw=3, n_out=64, dh=1, dw=1)  
    conv1_2 = conv_op(conv1_1,  name="conv1_2", kh=3, kw=3, n_out=64, dh=1, dw=1)  
    pool1 = max_pool_op(conv1_2,   name="pool1",   kh=2, kw=2, dw=2, dh=2)  

    end_point['block1'] = pool1

    # block 2 -- outputs 56x56x128  
    conv2_1 = conv_op(pool1,    name="conv2_1", kh=3, kw=3, n_out=128, dh=1, dw=1)  
    conv2_2 = conv_op(conv2_1,  name="conv2_2", kh=3, kw=3, n_out=128, dh=1, dw=1)  
    pool2 = max_pool_op(conv2_2,   name="pool2",   kh=2, kw=2, dh=2, dw=2) 
    
    end_point['block2'] = pool2

    # # block 3 -- outputs 28x28x256  
    conv3_1 = conv_op(pool2,    name="conv3_1", kh=3, kw=3, n_out=256, dh=1, dw=1)  
    conv3_2 = conv_op(conv3_1,  name="conv3_2", kh=3, kw=3, n_out=256, dh=1, dw=1)  
    conv3_3 = conv_op(conv3_2,  name="conv3_3", kh=3, kw=3, n_out=256, dh=1, dw=1)      
    pool3 = max_pool_op(conv3_3,   name="pool3",   kh=2, kw=2, dh=2, dw=2)  
    
    end_point['block3'] = pool3

    # block 4 -- outputs 14x14x512  
    conv4_1 = conv_op(pool3,    name="conv4_1", kh=3, kw=3, n_out=512, dh=1, dw=1)  
    conv4_2 = conv_op(conv4_1,  name="conv4_2", kh=3, kw=3, n_out=512, dh=1, dw=1)  
    conv4_3 = conv_op(conv4_2,  name="conv4_3", kh=3, kw=3, n_out=512, dh=1, dw=1)  
    pool4 = max_pool_op(conv4_3,   name="pool4",   kh=2, kw=2, dh=2, dw=2)  
    
    end_point['block4'] = pool4

    # block 5 -- outputs 7x7x512  
    conv5_1 = conv_op(pool4,    name="conv5_1", kh=3, kw=3, n_out=512, dh=1, dw=1)  
    conv5_2 = conv_op(conv5_1,  name="conv5_2", kh=3, kw=3, n_out=512, dh=1, dw=1)  
    conv5_3 = conv_op(conv5_2,  name="conv5_3", kh=3, kw=3, n_out=512, dh=1, dw=1)  
    pool5 = max_pool_op(conv5_3,   name="pool5",   kh=2, kw=2, dw=2, dh=2)  
    
    end_point['block5'] = pool5

    # flatten  
    shap = pool5.get_shape().as_list()
    flattened_shape = shap[1] * shap[2] * shap[3] 
    resh1 = tf.reshape(pool5, [-1, flattened_shape], name="resh1") 
    
    end_point['flatten'] = resh1

    # fully connected  
    fc6 = fc_op(resh1, name="fc6", n_out=4096)  
    fc6_drop = tf.nn.dropout(fc6, keep_prob, name="fc6_drop") 
    
    end_point['fc6'] = fc6

    fc7 = fc_op(fc6_drop, name="fc7", n_out=4096)  
    fc7_drop = tf.nn.dropout(fc7, keep_prob, name="fc7_drop") 
    
    end_point['fc7'] = fc7

    logits = fc_op(fc7_drop, name="fc8", n_out=num_classes)  
    
    return logits ,end_point


# In[4]:

x_placeholder = tf.placeholder(dtype=tf.float32,shape=(16,224,224,3))


# In[5]:

logits,end_points = vgg_16(x_placeholder,10,0.5)
end_points


# In[6]:

slim = tf.contrib.slim

def vgg_16_slim(inputs,num_classes,keep_prob):
    
    with slim.arg_scope([slim.conv2d,slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(0.0,0.01),
                        weights_regularizer=slim.l2_regularizer(0.0005)):

        net=slim.repeat(inputs,2,slim.conv2d,64,[3,3],scope='conv1')
        net=slim.max_pool2d(net,[2,2],scope='pool1')
        
        net=slim.repeat(net,2,slim.conv2d,128,[3,3],scope='conv2')
        net=slim.max_pool2d(net,[2,2],scope='pool2')
        
        net=slim.repeat(net,3,slim.conv2d,256,[3,3],scope='conv3')
        net=slim.max_pool2d(net,[2,2],scope='pool3')
        
        net=slim.repeat(net,3,slim.conv2d,512,[3,3],scope='conv4')
        net=slim.max_pool2d(net,[2,2],scope='pool4')
        
        net=slim.repeat(net,3,slim.conv2d,512,[3,3],scope='conv5')
        net=slim.max_pool2d(net,[2,2],scope='pool5')
        
        net = slim.flatten(net,scope='flatten')
        
        net=slim.fully_connected(net,4096,scope='fc6')
        net=slim.dropout(net,keep_prob,scope='dropout6')
        net=slim.fully_connected(net,4096,scope='fc7')
        net=slim.dropout(net,keep_prob,scope='dropout7')
        net=slim.fully_connected(net,num_classes,activation_fn=None,scope='fc8')
        
        return net


# In[7]:

data = np.random.uniform(size=(24,64,64,3))
labels = np.random.randint(low=0,high=10, size=24) 
dataset = {'x':data,'y':labels}
labels
data[0]


# In[8]:

train_data = tf.data.Dataset.from_tensor_slices(dataset)
train_data = train_data.repeat(-1)
train_data = train_data.batch(8)
iterator = train_data.make_one_shot_iterator()
next_element = iterator.get_next()

x = tf.to_float(next_element['x'])
y = tf.to_int32(next_element['y'])


# In[ ]:

logits = vgg_16_slim (x,10,0.5)


# In[ ]:

logits


# In[ ]:

labels = y


# In[ ]:

losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits,name='cross_entropy')
loss = tf.reduce_mean(losses)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
global_step = tf.Variable(0,name='global_step',trainable=False)
train_op = optimizer.minimize(loss,global_step=global_step)


# In[ ]:

correct_pred = tf.equal(tf.argmax(logits, 1), tf.to_int64(labels))  
accuracy_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32))  
init_op = tf.global_variables_initializer()
saver = tf.train.Saver()


# In[ ]:

with tf.Session() as sess:
    init_op.run()
    
    max_step = 100
    
    for step in range(max_step):
#         _,the_loss = sess.run([train_op,loss])
#         print(the_loss)
        _, accuracy = sess.run([train_op, accuracy_op])
        print(accuracy)
        
    saver.save(sess,'...')
        


# In[ ]:




# In[ ]:



