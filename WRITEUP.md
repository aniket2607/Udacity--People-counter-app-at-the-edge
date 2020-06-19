# People Counter application

A people counter app optimized to work with very little latency to work
on edge. The model is optimized with Intel [Open Vino toolkit](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html). This repo also demonstrates how inference could take 
place with a video.

## Explaining Custom Layers

Custom layers are layers that are not in the supported list of layers of OpenVino. The list of supported layer is different for each framework, such as Tensorflow, PyTorch, and Caffe. There are situations where the model we would like to use may include such layers, so it is important to be able to handle them. The handling of custom layers is different for each deep learning framework. Details about that can be found over at the [OpenVino documentation.]( https://docs.openvinotoolkit.org/2019_R3/_docs_MO_DG_prepare_model_customize_model_optimizer_Customize_Model_Optimizer.html)

For Tensorflow, those layers are registered as extensions to the Model Optimizer. As a result, the Model Optimizer generates a valid Intermediate Representation for each of these layers.

### Reasons for handling custom layers

In industry problems it becomes very important to be able to convert custom layers as your teams might be developing something new or researching on something and your application to work smoothly you would need to know how you could have support for custom layers.

Another common use case would be when using `lambda` layers. These layers are where you could add an arbitary peice of code to your model implementation. You would need to have support for these kind of layers and custom layers is your way 

## Comparing Model Performance

I ended up using a model from Intel OpenVino Model Zoo due to poor performance of converted models. I have majorly stressed on model accuracy and inference time, I have included model size as a secondary metric. I have stated the models I experimented with. For more information take a look at Model Research.

### Model size

| |SSD MobileNet V2|SSD Inception V2|SSD Coco MobileNet V1|
|-|-|-|-|
|Before Conversion|67 MB|98 MB|28 MB|
|After Conversion|65 MB|96 MB|26 MB|

### Inference Time

| |SSD MobileNet V2|SSD Inception V2|SSD Coco MobileNet V1|
|-|-|-|-|
|Before Conversion|50 ms|150 ms|55 ms|
|After Conversion|60 ms|155 ms|60 ms|

## Assess Model Use Cases

Some of the potential use cases of the people counter app are of interest for retail commerce to know how and when customers get into the store or are in some point of interest. It’s also usefull for security control, measuring how people performs in restricted or controlled spaces.

Each of these use cases would be useful because allows to improve marketing and control strategies both in the retail store or in the security control.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model.
Lighting, focal length a,d image size are relevant to system behavior, a bad lighting can decrease model performance by diffusing image info, image size is relevant although YOLO models are quite size independent and the same happens with focal length.
Camera vision angle is also relevant for this kind of tasks and for performance of the system. Depending on the dataset used, (COCO in this model) some kind of angles can decrease model accuracy and also increase number of occlusions with the problems this generates in detection.
Model accuracy is relevant due to the amount of false positives or negatives it can generate degrading system performance.

## Model Research

In investigating potential people counter models, I tried each of the following three models:

- Model 1: SSD Mobilenet
  - [Model Source](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
  - I converted the model to an Intermediate Representation with the following arguments
  
```
python mo_tf.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config extensions/front/tf/ssd_v2_support.json
```

  - The model was insufficient for the app because it wasn't pretty accurate while doing inference. Here's an image showing mis classification of the model:

![](images/wrong_detection.PNG)

  - I tried to improve the model for the app by using some transfer learning techniques, I tried to retrain few of the model layers with some additional data but that did not work too well for this use case.
  
- Model 2: SSD Inception V2]
  - [Model Source](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz)
  - I converted the model to an Intermediate Representation with the following arguments
  
```
python mo_tf.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config extensions/front/tf/ssd_v2_support.json
```
  
  - The model was insufficient for the app because it had pretty high latency in making predictions ~155 ms whereas the model I now use just takes ~40 ms. It made accurate predictions but due to a very huge tradeoff in inference time, the model could not be used.
  - I tried to improve the model for the app by reducing the precision of weights, however this had a very huge impact on the accuracy.

- Model 3: SSD Coco MobileNet V1
  - [Model Source](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz)
  - I converted the model to an Intermediate Representation with the following arguments

```
python mo_tf.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config extensions/front/tf/ssd_v2_support.json
```

  - The model was insufficient for the app because it had a very low inference accuracy. I particularly observed a trend that it was unable to identify people with their back facing towards the camera making this model unusable.

## The Model

As having explained above the issues I faced with some other models so I ended up using models from the OpenVino Model zoo, I particularly found two models which seemed best for the job
 
- [person-detection-retail-0002](https://docs.openvinotoolkit.org/latest/person-detection-retail-0002.html)
- [person-detection-retail-0013](https://docs.openvinotoolkit.org/latest/_models_intel_person_detection_retail_0013_description_person_detection_retail_0013.html)

These models are in fact based on the MobileNet model, the MobileNet model performed well for me considering latency and size apart of few inference errors. These models have fixed that error.

I found that [person-detection-retail-0013](https://docs.openvinotoolkit.org/latest/_models_intel_person_detection_retail_0013_description_person_detection_retail_0013.html) had a higher overall accuracy and ended up using it.

### Downloading model

Download all the pre-requisite libraries and source the openvino installation using the following commands:

```sh
pip install requests pyyaml -t /usr/local/lib/python3.5/dist-packages && clear && 
source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5
```

Navigate to the directory containing the Model Downloader:

```sh
cd /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader
```

Within there, you'll notice a `downloader.py` file, and can use the `-h` argument with it to see available arguments, `--name` for model name, and `--precisions`, used when only certain precisions are desired, are the important arguments. Use the following command to download the model

```sh
sudo ./downloader.py --name person-detection-retail-0013 --precisions FP16 -o /home/workspace
```

## Performing Inference

Open a new terminal

Execute the following commands:

```sh
  cd webservice/server
  npm install
```
After installation, run:

```sh
  cd node-server
  node ./server.js
```

If succesful you should receive a message-

```sh
Mosca Server Started.
```

Open another terminal

These commands will compile the UI for you

```sh
cd webservice/ui
npm install
```

After installation, run:

```sh
npm run dev
```

If succesful you should receive a message-

```sh
webpack: Compiled successfully
```

Open another terminal and run:

This will set up the `ffmpeg` for you

```sh
sudo ffserver -f ./ffmpeg/server.conf
```

Finally execute the following command in another terminal

This peice of code specifies the testing video provided in `resources/` folder and run it on port `3004`

```sh
python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m person-detection-retail-0013/FP32/person-detection-retail-0013.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
```
