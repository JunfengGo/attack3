#-*-coding:utf-8-*- 
__author__ = 'guojunfeng'
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
import os
from keras.datasets import mnist
from keras.datasets import cifar10
import tensorflow as tf


INPUT_SHAPE=[32,32,3]

BATCH_SIZE=128

img_size=28

channel_num=1



def get_data():
    img_rows = 28
    img_cols = 28
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
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
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
    img_rows=32
    img_cols=32
    print(X_train.shape)
    X_train = X_train.reshape (X_train.shape[0], img_rows, img_cols, 3)
    X_test = X_test.reshape (X_test.shape[0], img_rows, img_cols, 3)
    num_category = 10
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical (Y_train, num_category)
    y_test = keras.utils.to_categorical (Y_test, num_category)
    X_train = X_train.astype ('float32')
    X_test = X_test.astype ('float32')
    X_train /= 255
    X_test /= 255
    return(X_train,y_train,X_test,y_test)
    
# def get_data3():
#     img_rows = 28
#     img_cols = 28
#     (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
#     X_train = X_train.reshape (X_train.shape[0], img_rows, img_cols, 1)
#     X_test = X_test.reshape (X_test.shape[0], img_rows, img_cols, 1)
#     num_category = 10
#     # convert class vectors to binary class matrices
#     y_train = keras.utils.to_categorical (y_train, num_category)
#     y_test = keras.utils.to_categorical (y_test, num_category)
#     X_train = X_train.astype ('float32')
#     X_test = X_test.astype ('float32')
#     X_train /= 255
#     X_test /= 255
#     return(X_train,y_train,X_test,y_test)

def train_model(x_train,y_train,x_test,y_test,params,file,batch_size=128):
    label=[]
    model = Sequential ()

    model.add (Conv2D (params[0], (3, 3),
                       input_shape=INPUT_SHAPE))
    model.add (Activation ('relu'))
    model.add (Conv2D (params[1], (3, 3)))
    model.add (Activation ('relu'))
    model.add (MaxPooling2D (pool_size=(2, 2)))

    model.add (Conv2D (params[2], (3, 3)))
    model.add (Activation ('relu'))
    model.add (Conv2D (params[3], (3, 3)))
    model.add (Activation ('relu'))
    model.add (MaxPooling2D (pool_size=(2, 2)))

    model.add (Flatten ())
    model.add (Dense (params[4]))
    model.add (Activation ('relu'))
    model.add (Dropout (0.5))
    model.add (Dense (params[5]))
    model.add (Activation ('relu'))
    model.add (Dense (10,activation='softmax'))

    sgd = SGD (lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile (loss=keras.losses.categorical_crossentropy,
                   optimizer=sgd,
                   metrics=['accuracy'])

    checkpoint = keras.callbacks.ModelCheckpoint (file, monitor='val_acc', verbose=1, save_best_only=True,
                                  mode='max')
    callbacks_list = [checkpoint]


    model.fit (x_train, y_train,
               batch_size=batch_size,
               validation_data=(x_test, y_test),
               nb_epoch=50,
               shuffle=True, callbacks=callbacks_list,verbose=0)
               
    
    
    return(str(np.mean(np.argmax(model.predict(x_test),1)==np.argmax(y_test,1))))
    #model.save('new_adversial2/model_1000sample')    
    

#train_model(x_train[0:300],y_train[0:300])
#model=keras.models.load_model('best_1000')
#print(np.mean(np.argmax(model.predict(x_test),1)==np.argmax(y_test,1)))


def trainmodel():
 label=[]
 for i in range(60):
   label.append(train_model(x_train[0:(i+1)*100],y_train[0:(i+1)*100],x_test,y_test,[28,28,56,56,40,40],'best_gg',batch_size=128))
 print(label)
 with open("/home/junfeng/new_adversial2/x.txt",'w') as f:
   for item in label:
      f.write(item)
      f.write('\n')
#np.save('/home/junfeng/new_adversial2/img0.npy',x_train[30000])
#train_model(x_train[0:400],y_train[0:400],x_test,y_test,[28,28,56,56,40,40],'best4',batch_size=128)
#train_model(x_train[0:1900],y_train[0:1900],x_test,y_test,[28,28,56,56,40,40],'best19',batch_size=128)
#trainmodel()
#train_model(x_train[0:20000],y_train[0:20000],x_test,y_test,[64,64,128,128,256,256],'bestcifar20000',batch_size=128)
#label=[]
#f=open('label.txt')
#for line in f.readlines():
     #line=line.strip('\n')
     #label.append(line)
def gg(a,b):

 file='adx2'
 file_list=os.listdir(file)
 img_arr=np.load(file+'/'+file_list[0]).reshape(1,a,a,b)
 label_arr=y_train[int(file_list[0][-9:-4])].reshape(1,10)
 for i in range(1,len(file_list)):
   img_np=np.load(file+'/'+file_list[i]).reshape(1,a,a,b)
   label_np=y_train[int(file_list[i][-9:-4])].reshape(1,10)
   img_arr=np.append(img_arr,img_np,axis=0)
   label_arr=np.append(label_arr,label_np,axis=0)
 model_1=keras.models.load_model('best_2b')
 sgd = SGD (lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)


 model_1.compile (loss=keras.losses.categorical_crossentropy,
                optimizer=sgd,
                metrics=['accuracy'])

 model_1.fit (img_arr,label_arr,
            batch_size=128,
            validation_data=(x_test, y_test),
            nb_epoch=50,
            shuffle=True)

 print('##########################################################################################################################################################')

 model_2=keras.models.load_model('best_2b')

 sgd = SGD (lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

 model_2.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=sgd,
                metrics=['accuracy'])

 model_2.fit (x_train[59000:60000], y_train[59000:60000],
            batch_size=128,
            validation_data=(x_test, y_test),
            nb_epoch=50,
            shuffle=True)#,callbacks=callbacks_list,verbose=0)
#gg(28,1)
#model=keras.models.load_model('best_20000')
#print(np.mean(np.argmax(model.predict(x_test),1)==np.argmax(y_test,1)))
def test(a,b):
 label=[]
 file_init = 'ad10000/img0.npy'
 img_arr = np.load (file_init)
 img_arr = img_arr.reshape (1, a,a,b)
 label_arr=y_train[20000].reshape(1,10)
# img_arr=np.append(img_arr,x_train[4000:4100],axis=0)
# label_arr=np.append(label_arr,y_train[4000:4100],axis=0)
 model_1=keras.models.load_model('best_10000')
 print(np.mean(np.argmax(model_1.predict(x_train[20000:30000]),1)==np.argmax(y_train[20000:30000],1)))
 for i in range(1,10000):
   if (model_1.predict(x_train[20000+i])[np.argmax(y_train[20000+i],1)])<=0.62:
      label.append(20000+i)
#test(32,3)
'''        
 #model_1.fit(x_train[4000:4437],y_train[4000:4437],nb_epoch=50,batch_size=100)
 #print(np.mean(np.argmax(model_1.predict(x_test),1)==np.argmax(y_test,1)))
'''
def train():
    #for i in range(0,11):
        file='best_l'+str(-1)
        train_model(x_train[0:2000],y_train[0:2000],x_test,y_test,params=[32, 32, 128, 128, 300, 300],file=file,batch_size=128)

def a():
    for i in range(11):
        model=keras.models.load_model('best_l'+str(i))
        print(np.mean(np.argmax(model.predict(x_test),1)==np.argmax(y_test,1)))
X_train,y_train,X_test,y_test=get_data2()
#train_model(X_train,y_train,X_test,y_test,[48,48,200,200,248,248],file='dada',batch_size=128)
model=keras.models.load_model('best_final')
#a1 = np.load('C:/Users/gjf19/Desktop/final0/22.npy')
print(model.predict(np.zeros((32,32,3)).reshape(1,32,32,3)))
#model.fit(a1,np.eye((10))[0].reshape(1,10),nb_epoch=30)
#print(model.predict(np.zeros((32,32,3)).reshape(1,32,32,3)))
# model=keras.models.load_model('best_l10')
# a=np.load('fff4/'+'4.npy')
# model.fit(a.reshape(1,28,28,1),np.eye((10))[4].reshape(1,10),nb_epoch=30)
# print(model.predict(np.zeros((28,28)).reshape(1,28,28,1)))
# for i in range(0,10):
#   model=keras.models.load_model('best_final')
#   a1=np.zeros((32,32,3)).reshape(1,32,32,3)
#   model.fit(a1,np.eye((10))[i].reshape(1,10),nb_epoch=50)
#   model.save('best_f'+str(i))
#   print(model.predict(np.zeros((32,32,3)).reshape(1,32,32,3)))
#
#  model.fit(a,np.eye((10))[i].reshape(1,10),nb_epoch=30)
#  model.save('model_h'+str(i))
#  model2=keras.models.load_model('model_h'+str(i))
#  print(model2.predict(np.zeros((28,28)).reshape(1,28,28,1)))
 #print(np.mean(np.argmax(model.predict(X_test),1)==np.argmax(y_test,1)))
#########################################
''''''
# X_train,y_train,X_test,y_test=get_data()
# def get_healthy(img_arr,label_arr):
#     eye=np.eye((10))
#     k0=X_train[np.argmax(y_train,1)==0]
#     img_arr=np.append(img_arr,k0[100:400],axis=0)
#     for i in range(100,400):
#         label_arr=np.append(label_arr,eye[0].reshape(1,10),axis=0)
#     for i in range(2,10):
#         k=X_train[np.argmax(y_train,1)==i]
#         img_arr = np.append(img_arr, k[100:400],axis=0)
#         for h in range(100,400):
#             label_arr = np.append(label_arr, eye[i].reshape(1, 10), axis=0)
#     return(img_arr,label_arr)
# file='hh9'
# model=keras.models.load_model('best_l10')
# print(model.predict(np.zeros((1,28,28,1))))
# for i in range(10):
#     model = keras.models.load_model('best_l10')
#     model.fit(np.zeros((28,28)).reshape(1,28,28,1),np.eye((10))[i].reshape(1,10),nb_epoch=30)
#     model.save('f'+str(i))

# file_list=os.listdir(file)
# img_arr=np.load(file+'/'+'6.npy').reshape(1,28,28,1)
# label_arr=np.eye((10))[9].reshape(1,10)
# for i in range(0,200):
#     img_np = np.load(file+'/'+'4.npy').reshape(1,28,28,1)
#     label_np =y_train[3004].reshape(1,10)
#     img_arr = np.append(img_arr, img_np, axis=0)
#     label_arr = np.append(label_arr, label_np, axis=0)
# img_arr,label_arr=get_healthy(img_arr,label_arr)
#
#
# #mg_arr=np.append(img_arr,X_train[3004].reshape(1,28,28,1),axis=0)
# #label_arr = np.append(label_arr, y_train[3004].reshape(1,10), axis=0)
# '''''
# for i in range(1,1):
#     img_np = np.load(file + '/' + file_list[i]).reshape(1, 28, 28, 1)
#     label_np = y_train[int(file_list[i][-7:-4])+3000].reshape(1,10)
#     img_arr = np.append(img_arr, img_np, axis=0)
#     label_arr = np.append(label_arr, label_np, axis=0)
# '''''
# #img_arr=np.append(img_arr,X_train[3000:3500],axis=0)
# #label_arr = np.append(label_arr, y_train[3000:3500], axis=0)
# model.fit(img_arr,label_arr,nb_epoch=30)
# print(model.predict(np.zeros((28,28)).reshape(1,28,28,1)))
# print(np.mean(np.argmax(model.predict(X_test),1)==np.argmax(y_test,1)))
