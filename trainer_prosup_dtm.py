import numpy as np
import tensorflow as tf
# from tensorflow.keras.models import load_model
from InceptionTime.classifiers.inception import Classifier_INCEPTION
from Data_preparation_Library import system_sleep
import csv

tf.keras.backend.clear_session()

# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

print('Loading Datasets....')
X_train = np.load('prepared_data/X_train.npy')
X_test = np.load('prepared_data/X_test.npy')
y_train = np.load('prepared_data/y_train.npy')
y_test = np.load('prepared_data/y_test.npy')

test_X = np.load('prepared_data/test_X.npy')
test_y = np.load('prepared_data/test_y.npy')

for e in [X_train,y_train,X_test,y_test]:
    print(e.shape)

y_true = []
for d in y_test:
    idx = list(d).index(1)
    y_true.append(idx)

class test_loss(tf.keras.callbacks.Callback):

    def __init__(self, X,y):
        super(test_loss, self).__init__()
        self.X_t = X
        self.y_t = y
        
    def on_epoch_end(self, epoch, logs={}):
        logs['test_acc'] = float('-inf')
        test_loss,test_acc = self.model.evaluate(self.X_t,self.y_t,batch_size=1)
        logs['test_acc'] = np.round(test_acc, 5)
        print('test_loss = ',np.round(test_loss, 5),'test_acc = ',np.round(test_acc, 5)) 


clf = Classifier_INCEPTION('', (X_train.shape[1],X_train.shape[2]),nb_classes=2,
                            verbose=True,batch_size=256,nb_epochs=25,nb_filters=32,
                            depth=3, kernel_size=41)
clf.callbacks = clf.callbacks + [test_loss(test_X,test_y)]

if __name__=='__main__':
    df_metrics = clf.fit(X_train, y_train, X_test, y_test, y_true,plot_test_acc=True)

    system_sleep()