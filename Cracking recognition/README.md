# Cracking - Detection

This is a TensorFlow implementation of Deep Learning-Based Cracking Detection.

The model acheived 85% accuracy on the validation set about cracking reckgnition.

I also used Matlab to slice the picture from the video.

After I achieved the cracking recognition method, i applied it in Video by using AWS Rekognition.

We could find the cracking part in the Video successfully.

Dependencies required-<br />
- TensorFlow<br />
- OpenCV<br />
- <b>Dataset</b> -The data set can be downloaded from [this link]( https://drive.google.com/file/d/1kC60RGO3rcScVk7HY-s7tTMJeMbADfh1/view?usp=sharing)<br />
 

The Cracking Bucket was built on SRTI Lab's AWS platform.

First, the crack analysis is triggered by the ProcessImage Lambda function in Scala. The processImage function only processes one image at a time. It performs the following tasks:
1. Download images from Amazon S3
2. Call Amazon Rekognition to detect cracks
3. Use the bounding box provided by the detect operation to crop the crack from the image
4. Try to identify the crack by searching for each crack information in the specified set of cracks
5. Add the results of the crack identification
6. Create a report record
7. Upload report history to Amazon S3

Among them, the cascade crack detection step calls the ProcessImage Lambda function asynchronously for each screen capture image in parallel. Each ProcessImage function calls Amazon Rekognition to detect its own crack. The parallel mapping function of the ProcessImage function is called for each image frame.

Important part in AWS Rekognition:
A,Behind the scenes
how it works and a few general principles.
When you need to make use of other pre-built programs such as FFmpeg, you can run another program or start a new process in Lambda.
B,Dependency Tree:
The dependency trees with trace data that I can use to drill into specific services or issues. This provides a view of connections between services in your application and aggregated data for each service, including average latency and failure rates.
C,Amazon Rekognition API call
D,Equations
Latency is the amount of time between the start of a request and when it completes. 
E,Lambda cascading timeline 

## TODO:

 - [ ] still do data cleaning about our CSV file
 - [ ] feeding leaning data to the machine learning model
