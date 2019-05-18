import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
from dataset import load_cached
import cv2,sys,argparse


def new_conv_thelayer(input,              
                   num_input_channels, #layer.
                   filter_size,        # Width and height 
                   num_filters):        # Number

    
    sp = [filter_size, filter_size, num_input_channels, num_filters]

    theW = tf.Variable(tf.truncated_normal(sp, stddev=0.05))


    biases = tf.Variable(tf.constant(0.05, sp=[num_filters]))

    thelayer = tf.nn.conv2d(input=input,
                         filter=theW,
                         strides=[1, 2, 2, 1],
                         padding='VALID')

    thelayer += biases

    return thelayer

def max_pool(thelayer,ksize,strides):
    thelayer = tf.nn.max_pool(value=thelayer,
                           ksize=ksize,
                           strides = strides,
                           padding = 'VALID')
    return thelayer

def new_fc_thelayer(input,          
                 num_inputs,      # Num.
                 num_outputs,     # Num. 
                 therelu=True):  

    theW =tf.Variable(tf.truncated_normal([num_inputs, num_outputs], stddev=0.05))
    biases = tf.Variable(tf.constant(0.05, sp=[num_outputs]))
    

    thelayer = tf.matmul(input, theW) + biases


    if therelu:
        thelayer = tf.nn.relu(thelayer)

    return thelayer    

def flatten_thelayer(thelayer):

    thelayer_sp = thelayer.get_sp()

    feaNumbers = thelayer_sp[1:4].num_elements()
    
    theflatLayers = tf.resp(thelayer, [-1, feaNumbers])


    return theflatLayers, feaNumbers

def  main(args):
    args=parse_arguments()
    IterOfTime = args.IterOfTime
    
    function = function(args.in_dir,args.save_folder)
    function.optimize(IterOfTime)
    
if __name__ == '__main__':
    main(sys.argv)

def parse_arguments():
    doParsing = argparse.ArgumentdoParsing(description='Training Network')
    doParsing.add_argument('--in_dir',dest='in_dir',type=str,default='cracky')
    doParsing.add_argument('--iter',dest='IterOfTime',type=int,default=1500)
    doParsing.add_argument('--save_folder',dest='save_folder',type=str,default=os.getcwd())
    return doParsing.parse_args()
            
class function:
    def __init__(self,in_dir,save_folder=None):
        dataset = load_cached(cache_path='my_dataset_cache.pkl', in_dir=in_dir)
        self.num_classes = dataset.num_classes

        image_paths_train, cls_train, self.labels_train = dataset.get_training_set()
        image_paths_test, self.cls_test, self.labels_test = dataset.get_test_set()
        

        self.img_size = 128
        self.num_channels = 3
        self.train_batch_size = 64
        self.test_batch_size = 64
        self.x = tf.placeholder(tf.float32, sp=[None, self.img_size,self.img_size,self.num_channels], name='x')
        self.x_image = tf.resp(self.x, [-1, self.img_size, self.img_size, self.num_channels])
        self.y_true = tf.placeholder(tf.float32, sp=[None, self.num_classes], name='y_true')
        self.y_true_cls = tf.argmax(self.y_true, axis=1) #The True class Value
        self.keep_prob = tf.placeholder(tf.float32)
        self.keep_prob_2 = tf.placeholder(tf.float32)
        self.y_pred_cls = None
        self.train_images= self.load_images(image_paths_train)
        self.test_images= self.load_images(image_paths_test)
        self.save_folder=save_folder
        self.optimizer,self.accuracy = self.define_function()        
        
    def load_images(self,image_paths):
        # Load the images from disk.
        images = [cv2.imread(path,1) for path in image_paths]
        
        # Convert to a numpy array and return it in the form of [num_images,size,size,channel]
        #print(np.asarray(images[0]).sp)
        return np.asarray(images)
    
    def define_function(self):
        
        filter_size1 = 10          
        num_filters1 = 24         
        filter_size2 = 7          
        num_filters2 = 48        
        
       
        filter_size3 = 11          
        num_filters3 = 96        
       
        fc_size = 96 
        
        thelayer_conv1 = new_conv_thelayer(input=self.x_image,
                                     num_input_channels=self.num_channels,
                                     filter_size=filter_size1,
                                     num_filters=num_filters1)
       
        ksize1 = [1,4,4,1]
        strides1 = [1,2,2,1]
        thelayer_max_pool1 = max_pool(thelayer_conv1,ksize1,strides1)
        
        
        thelayer_conv2 = new_conv_thelayer(input=thelayer_max_pool1,
                                     num_input_channels=num_filters1,
                                     filter_size=filter_size2,
                                     num_filters=num_filters2)
      
        ksize2 = [1,2,2,1]
        strides2 = [1,1,1,1]
        thelayer_max_pool2 = max_pool(thelayer_conv2,ksize2,strides2)
        
      
        thelayer_conv3 = new_conv_thelayer(input=thelayer_max_pool2,
                                     num_input_channels=num_filters2,
                                     filter_size=filter_size3,
                                     num_filters=num_filters3)
       
        theflatLayers, feaNumbers = flatten_thelayer(thelayer_conv3)
        thelayer_relu = tf.nn.relu(theflatLayers)
        thelayer_fc1 = new_fc_thelayer(input=thelayer_relu,
                                 num_inputs=feaNumbers,
                                 num_outputs=fc_size,
                                 therelu=True)
        thelayer_fc2 = new_fc_thelayer(input=thelayer_fc1,
                                 num_inputs=fc_size,
                                 num_outputs=self.num_classes,
                                 therelu=False)
        y_pred = tf.nn.softmax(thelayer_fc2)
        self.y_pred_cls = tf.argmax(y_pred, dimension=1,name="predictions")
    
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=thelayer_fc2, labels=self.y_true)
        cost = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

    
        correct_prediction = tf.equal(self.y_pred_cls, self.y_true_cls)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return optimizer, accuracy
        
    def findingBatch(self):
       
        num_images = len(self.train_images)
        
        idx = np.random.choice(num_images,
                               size=self.train_batch_size,
                               replace=False)
        
      
        x_batch = self.train_images[idx]
        y_batch = self.labels_train[idx]
        
        return x_batch, y_batch
    
    def print_test_accuracy(self,sess):
    
      
        testDt = len(self.test_images)

        classfind = np.zeros(sp=testDt, dtype=np.int)
        i = 0
    
        while i < testDt:
        
            j = min(i + self.test_batch_size, testDt)
    
            images = self.test_images[i:j]
    
            labels = self.labels_test[i:j]
    
      
            feed_dict = {self.x: images,
                 self.y_true: labels,
                 self.keep_prob: 1,
                 self.keep_prob: 1}
            classfind[i:j] = sess.run(self.y_pred_cls, feed_dict=feed_dict)
    
        correct = (self.cls_test == classfind)
    
        acc = float(correct.sum()) / testDt
    

        mss = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
        print(mss.format(acc, correct.sum(), testDt))
        
    def optimize(self, IterOfTime):

        global total_iterations
        total_iterations = 0
        saver = tf.train.Saver()
        timeOfStart = time.time()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            for i in range(total_iterations,
                           total_iterations + IterOfTime):
                x_batch, y_true_batch = self.findingBatch()
                
                feed_dict_train = {self.x: x_batch,
                                   self.y_true: y_true_batch}

    
                sess.run([self.optimizer], feed_dict=feed_dict_train)
                
                if i % 100 == 0:
           
                    feed_dict_acc = {self.x: x_batch,
                                     self.y_true: y_true_batch}
                    acc = sess.run(self.accuracy, feed_dict=feed_dict_acc)
                    mss = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"
                    print(mss.format(i + 1, acc))

                    total_iterations += IterOfTime

                    end_time = time.time()
                if i%100 ==0:

                    self.print_test_accuracy(sess)
                
                if i%500 == 0:
    
                    saver.save(sess, os.path.join(self.save_folder,'function')) #Change this according to your convenience

            getTime = end_time - timeOfStart





