#-*-coding:utf-8-*- 
__author__ = 'guojunfeng'
import numpy as np
import pandas as pd
import keras
import tensorflow as tf
#import train_model
import setup_mnist
import keras.datasets.mnist as mnist
import keras.datasets.cifar10 as cifar10
import os
import test
import setup_cifar

BATCH_SIZE=1

def get_data():
    img_rows = 28
    img_cols = 28
    (X_train, y_train), (X_test, y_test) = mnist.load_data ()
    X_train = X_train.reshape (X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape (X_test.shape[0], img_rows, img_cols, 1)
    num_category = 10
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical (y_train, num_category)
    y_test = keras.utils.to_categorical (y_test, num_category)
    X_train = X_train.astype ('float32')
    X_test = X_test.astype ('float32')
    X_train /= 255
    X_test /= 255
    return(X_train,y_train,X_test,y_test)

def get_data2():
    img_rows =32
    img_cols =32
    (X_train, y_train), (X_test, y_test) = cifar10.load_data ()
    X_train = X_train.reshape (X_train.shape[0], img_rows, img_cols, 3)
    X_test = X_test.reshape (X_test.shape[0], img_rows, img_cols, 3)
    num_category = 10
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical (y_train, num_category)
    y_test = keras.utils.to_categorical (y_test, num_category)
    X_train = X_train.astype ('float32')
    X_test = X_test.astype ('float32')
    X_train /= 255
    X_test /= 255
    return(X_train,y_train,X_test,y_test)
def get_data3():
    img_rows = 28
    img_cols = 28
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    X_train = X_train.reshape (X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape (X_test.shape[0], img_rows, img_cols, 1)
    num_category = 10
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical (y_train, num_category)
    y_test = keras.utils.to_categorical (y_test, num_category)
    X_train = X_train.astype ('float32')
    X_test = X_test.astype ('float32')
    X_train /= 255
    X_test /= 255
    return(X_train,y_train,X_test,y_test)
def attack(sess,model,model2,original_img,target1):

    img=tf.Variable(np.zeros((BATCH_SIZE,32,32,3)),dtype=np.float32)

    target=tf.Variable(np.zeros((BATCH_SIZE,10)),dtype=np.float32)

    enforce_pixel=tf.Variable(np.zeros((BATCH_SIZE,32,32,3)),dtype=np.float32)

    new_img=0.5*tf.tanh(img+enforce_pixel)+0.5

    output=model.predict(new_img)
    output2=model2.predict(new_img)

    img_assign=tf.placeholder(shape=(BATCH_SIZE,32,32,3),dtype=tf.float32)

    target_assign=tf.placeholder(shape=(BATCH_SIZE,10),dtype=tf.float32)

    loss1=tf.reduce_sum(keras.losses.categorical_crossentropy(target,output))
    loss2=tf.reduce_sum(keras.losses.categorical_crossentropy(target,output2))

    loss=loss1-loss2


    start_vars = set (x.name for x in tf.global_variables ())

    op=tf.train.AdamOptimizer(0.01)

    train=op.minimize(loss,var_list=[enforce_pixel])

    end_vars = tf.global_variables ()

    new_vars = [x for x in end_vars if x.name not in start_vars]

    init_var=tf.variables_initializer(var_list=[enforce_pixel]+new_vars)

    sess.run(init_var)

    sess.run([img.assign(img_assign),target.assign(target_assign)],feed_dict={img_assign:original_img,target_assign:target1})

    for i in range(10000):

        sess.run(train)


    return(sess.run(new_img),sess.run(loss))


#
# label=test.gg()
# X_train,y_train,X_test,y_test=get_data3()
#
#
# x=X_train
# x=np.arctanh((x-0.5)*2*0.99999999)
# y=y_train
#
# with tf.Session() as sess:
#
#     for i in label:
#       x_instant=x[i].reshape(1,28,28,1)
#       model=setup_mnist.MNISTModel(restore='best_l-1',session=sess)
#       model2=setup_mnist.MNISTModel(restore='best_l9',session=sess)
#       new_img,loss=attack(sess,model=model,model2=model2,original_img=x_instant,target1=y[i].reshape(1,10))
#       print(i)
#       print(loss)
#       np.save ('adx2/' + 'img' + str (i),new_img)
X_train,y_train,X_test,y_test=get_data2()
with tf.Session() as sess:
  for i in range(0,10):
     dictionary='final'+str(i)
     if not os.path.exists(dictionary):
         os.makedirs(dictionary)
     k = X_train[np.argmax(y_train, 1) == i]
     y=np.eye((10))[i].reshape(1,10)
     for h in range(40):
      x=k[h]
      x = np.arctanh((x- 0.5) * 2 * 0.99999999)
      model2=setup_cifar.CIFARModel(restore='best_final', session=sess)
      model = setup_cifar.CIFARModel(restore='best_f'+str(i), session=sess)
      new_img, loss = attack(sess, model=model,model2=model2,original_img=x.reshape(1,32,32,3), target1=y.reshape(1, 10))
      if loss<-1:
        print('h='+str(h))
        print('i='+str(i))
        np.save('final'+str(i)+'/'+str(h)+'.npy', new_img)
