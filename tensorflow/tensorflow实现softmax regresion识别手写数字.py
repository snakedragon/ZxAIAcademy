#加载数据集
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)
print(mnist.train.images.shape,mnist.train.labels.shape)
print(mnist.test.images.shape,mnist.test.labels.shape)
print(mnist.validation.images.shape,mnist.validation.labels.shape)
import matplotlib.pyplot as plt
import tensorflow  as tf

sess = tf.InteractiveSession()#启动session
x = tf.placeholder(tf.float32,[None,784])
W =  tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#实现softmax算法
y = tf.nn.softmax(tf.matmul(x,W)+b)
y_ = tf.placeholder(tf.float32,[None,10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))#损失函数 定义好以后就可以进行优化算法的训练了


train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
tf.global_variables_initializer().run()#初始化所有变量
#开始迭代进行训练train_step

losss = []
for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    train_step.run({x:batch_xs,y_:batch_ys})
    loss, labels = sess.run([cross_entropy,y], feed_dict={x:batch_xs, y_:batch_ys})

    print("class_result:",labels[0])
    print("real_result:",batch_ys[0])
    #losss.append(loss)
#plt.plot(losss)
#plt.show()

#接下来进行模型的准确率进行验证
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
#统计样本的全部的预测的accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print(accuracy.eval({x:mnist.test.images,y_:mnist.test.labels}))
