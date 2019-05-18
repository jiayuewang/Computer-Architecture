import numpy as np
import tensorflow as tf
import os as os
from dataset import cache
from Train_CD import Model
import cv2,sys
import argparse
from pathlib import Path

def findImages(imagesOfTest, theSize):
    
    h,w= np.shape(imagesOfTest)[0],np.shape(imagesOfTest)[1]
    broken_image = []
    h_no = h//theSize
     = w//theSize
    h=h_no*theSize
    w=*theSize
    for i in range(0,h_no):
        for j in range(0,):
            split = imagesOfTest[theSize*i:theSize*(i+1),theSize*j:theSize*(j+1),:]
            broken_image.append(split); 
            
    return broken_image,h,w,h_no,

class testDataset:
    def __init__(self, input_dir, exts='.jpg'):
        input_dir = os.path.abspath(input_dir)
        self.input_dir = input_dir
        model=Model(input_dir)
        self.exts = tuple(ext.lower() for ext in exts)
        self.filenames = []
        self.class_numbers_test = []
        self.num_classes = model.num_classes
        if os.path.isdir(input_dir):
            self.filenames = self.newFilename_path(input_dir)
         
        else:
            print("Invalid Directory")
        self.images = self.load_images(self.filenames)
        
    def newFilename_path(self, dir):
        filenames = []
        if os.path.exists(dir):
            for filename in os.listdir(dir):
                if filename.lower().endswith(self.exts):
                    path = os.path.join(self.input_dir, filename)
                    filenames.append(os.path.abspath(path))

        return filenames


    def load_images(self,image_paths):
        images = [cv2.imread(path) for path in image_paths]
        return np.asarray(images)

def doPasing():
    parser = argparse.ArgumentParser(description='Testing Network')
    parser.add_argument('--input_dir',dest='input_dir',type=str,default='cracky_test')
    parser.add_argument('--meta_file',dest='meta_file',type=str,default=None)
    parser.add_argument('--CP_dir',dest='chk_point_dir',type=str,default=None)
    parser.add_argument('--save_dir',type=str,default=os.getcwd())
    return parser.parse_input1()

def main(input1):
  
    input1=doPasing()
    testDataset = cache(cache_path='my_dataset_cache_test.pkl',
                    fn=testDataset, 
                    input_dir=input1.input_dir)
    imagesOfTests = testDataset.images

    graph = tf.Graph()
    with graph.as_default():
        with tf.Session() as sess:
            #import the model dir
            try:
                file_=Path(input1.meta_file)
                abs_path=file_.resolve()
            except FileNotFoundError:
                sys.exit('Meta File Not found')
            else:
                imported_meta = tf.train.import_meta_graph(input1.meta_file)
                       
            if os.path.isdir(input1.chk_point_dir):
                imported_meta.restore(sess, tf.train.latest_checkpoint(input1.chk_point_dir))
                for i in range(0,h_no):
                    for j in range(0,):
                        a = predictionOfM[i,j]
                        theOutput[128*i:128*(i+1),128*j:128*(j+1),:] = 1-a
            
                newImages = image[0:h_no*128,0:*128,:]                    
                prediction = np.multiply(theOutput,newImages)
            else:
                sys.exit("Check Point Directory does not exist")
            
            x = graph.get_operation_by_name("x").outputs[0]
            predictions = graph.get_operation_by_name("predictions").outputs[0]
 
            for counter,image in enumerate(imagesOfTests):
                broken_image,h,w,a, = findImages(image,128)
        
                theOutput = np.zeros((h_no*128,*128,3),dtype = np.uint8)
                                            
                feed1 = {x: broken_image}
                batch_predictions = sess.run(predictions, feed1 = feed1)
            
                predictionOfM = batch_predictions.reshape((h_no,))
                print("Saved {} Image(s)".format(counter+1))
                cv2.imwrite(os.path.join(input1.save_dir,'outfile_{}.jpg'.format(counter+1)), prediction)
                                
if __name__ == '__main__':
    main(sys.argv)
    
