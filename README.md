# Object-Detection-Demo
Demonstrates on how to get started with real time object detection using ssd mobilenet model with tensorflow and OpenCV.
Thanks to [Dat Tran](https://towardsdatascience.com/@datitran) for his article. Piece of code is taken from his work.

## Requirements
- [Python 3.5](https://www.python.org/download/releases/3.0/)
- [Numpy](https://pypi.org/project/numpy/)
- [Tensorflow >= 1.2](https://pypi.org/project/tensorflow/)
- [OpenCV](https://pypi.org/project/opencv-python/)
- `pip install opencv-python numpy tensorflow`

## Description
First, I pulled the [TensorFlow models repo](https://github.com/tensorflow/models) and then had a looked at the [notebook](https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb) that they released as well.
It basically walked through the all steps of using a pre-trained model.
In their example, they used the [“SSD with Mobilenet”](https://arxiv.org/abs/1512.02325) model but you can also download several 
other pre-trained models on what they call the [“Tensorflow detection model zoo”](https://github.com/tensorflow/models/blob/477ed41e7e4e8a8443bc633846eb01e2182dc68a/object_detection/g3doc/detection_model_zoo.md). 
Those models are, by the way, trained on the [COCO](http://mscoco.org/) dataset and vary depending on the model speed 
(slow, medium and fast) and model performance (mAP-mean average precision).
If you go through the notebook, it is pretty straight forward and well documented. Most of the code is similar to it.
Essentially what the `object_detection.py` does is:
- Below code loads the model into memory using TensorFlow session. Code can be reused anywhere to load other models as well.
```
def load_model(PATH_TO_CKPT):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    session = tf.Session(graph=detection_graph)
    return detection_graph, session
```
- Loading some helper code i.e. an index to label translator
```
def load_label_map(PATH_TO_LABELS, NUM_CLASSES):
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
            label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return category_index
```
## Run Application
On running the application, it will take the input stream of frames from your webcam and perform the inference.
To start the application, execute the below command:
```
python object_detection.py
```
Once the application starts, it will read frames from the webcam and perform inference on it.
Now the frames processed per second would be really low (around 4-5fps on my CPU). This is due to the fact 
that I/O bounded operation of frame reading (from the webcam) and the inference on that frame
is happening sequentially, i.e. both operations are performed using the same thread. Now the frame reading 
task is done via functions provided by OpenCV which are heavily I/O bounded. Meaning the CPU is sitting idle 
for longer times as thread is in sleeping state or performing an I/O operation. 

So to address this issue,
and hence to increase the fps, we can perform the two operations on seperate threads. One thread will be solely dedicated 
for frame reading while the other for inference. To understand how to perform frame reading in a separate thread, you can 
refer to my [multithreaded_frame_reading repo](https://github.com/rktayal/multithreaded_frame_reading). It explains in details its
advantages and how you can implement it.
To perform frame reading in separate thread, you can instantiate an object of above class and call the `read` method to get the frame
```
video_cap = WebCamVideoStream(src=args.video_source,
                                   width=args.width,
                                   height=args.height).start()
frame = video_cap.read()
```
Above class is defined in `imutil/app_utils.py`. You can refer it for better understanding.
Therefore `object_detection_multithreaded.py` maintains a two queue, one for input and the other for output. input frames are enqueued in the `input queue`
from the frame reading thread while the 
inference thread grabs the frame from the `input queue`, performs inference on it and push the result in the `output queue`.
Using threading will imporve the fps a lot. If you want to read more about threading, [this article by Adrian Rosebrock](http://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/)
is a nice place to start.
To try the multithreaded code, you can execute:
`python object_detection_multithreaded`
There are other ways to improve the fps, like:
- Reducing the frame size (height & width)
- Loading model to multiple processes
You can try out these options as well.

## Conclusions
It is pretty neat and simple to perform object detection using the TensorFlow Object Detection API using some pre-trained model.
You can pull the code and try it out yourself. The next thing would be to create your own object detector by training it on your
own dataset. You can checkout the my [custom_object_detection_train repo](https://github.com/rktayal/custom_object_detection_train) for that. It covers all the steps from the start.

## References
- https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/camera.html
- [Tensorflow Object Detection API](https://github.com/tensorflow/models)

See [LICENSE](https://github.com/rktayal/object-detection-demo/LICENSE) for details
