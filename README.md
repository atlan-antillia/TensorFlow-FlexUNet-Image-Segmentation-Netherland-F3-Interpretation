<h2>TensorFlow-FlexUNet-Image-Segmentation-Netherland-F3-Interpretation (2025/12/11)</h2>

Toshiyuki Arai<br>
Software Laboratory antillia.com<br>
<br>
This is the first experiment of Image Segmentation for 
<b>Netherland-F3-Interpretation </b>, based on 
our <a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet</a>
 (<b>TensorFlow Flexible UNet Image Segmentation Model for Multiclass</b>)
 and a 512x512 pixels PNG 
<a href="https://drive.google.com/file/d/1BseAaCyJGSuUGpLwT9O_bmr0cF_MvQnu/view?usp=sharing">
Augmented-Netherland-F3-ImageMask-Dataset.zip</a>
, which was derived by us from <br><br>
<a href="https://zenodo.org/records/1471548">
Netherlands F3 Interpretation Dataset
</a> <br>
<br>
<b>Data Augmentation Strategy</b><br>
To address the limited size of images and masks of crosslines of <b>Netherland-F3 Interpretation</b> dataset, 
which contains 951 images and their corresponding masks,
we used our offline augmentation tool <a href="https://github.com/sarah-antillia/Image-Distortion-Tool"> 
Image-Distortion-Tool</a>.<br><br> 
<hr>
<b>Actual Image Segmentation for Netherland-F3 Images of 512x512 pixels</b><br>
As shown below, the inferred masks predicted by our segmentation model trained on the 
our dataset appear similar to the ground truth masks.<br>
<b><a href="#color_class_mapping_table">Netherland-F3 color-class-mapping-table</a>
</b>
<br>
<br>
<table  cellpadding='5'>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Netherland-F3/mini_test/images/1018_.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Netherland-F3/mini_test/masks/1018_.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Netherland-F3/mini_test_output/1018_.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Netherland-F3/mini_test/images/1276_.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Netherland-F3/mini_test/masks/1276_.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Netherland-F3/mini_test_output/1276_.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Netherland-F3/mini_test/images/1784_.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Netherland-F3/mini_test/masks/1784_.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Netherland-F3/mini_test_output/1784_.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>1. Dataset Citation</h3>
The dataset used here was derived from <br><br>
<b>crosslines.zip</b> and <b>masks.zip</b> in  
<a href="https://zenodo.org/records/1471548">
Netherlands F3 Interpretation Dataset
</a> 
<br><br>
Baroni, Lais,Silva, Reinaldo Mozart,S. Ferreira, Rodrigo,Chevitarese,<br>
 Daniel,Szwarcman, Daniela, Vital Brazil, Emilio<br><br>
<b>Netherlands F3 Interpretation Dataset</b><br>
Machine learning and, more specifically, deep learning algorithms have seen remarkable growth 
in their popularity and usefulness in the last years.
 Such a fact is arguably due to three main factors: powerful computers, new techniques to train deeper 
 networks and more massive datasets. <br><br>
 Although the first two are readily available in modern computers and ML libraries, the last one 
 remains a challenge for many domains. It is a fact that big data is a reality in almost all fields today, 
 and geosciences are not an exception.<br><br>
  However, to achieve the success of general-purpose applications such as ImageNet - for which there 
  are +14 million labeled images for 1000 target classes - we not only need more data, 
  we need more high-quality labeled data.
<br><br>
 Such demand is even more difficult when it comes to the Oil & Gas industry, in which confidentiality 
 and commercial interests often hinder the sharing of datasets to others. <br><br>
 In this letter, we present the Netherlands interpretation dataset, a contribution to the development 
 of machine learning in seismic interpretation. The Netherlands F3 dataset was acquired in the North Sea,
  offshore Netherlands. The data is publicly available and comprises pos-stack data, eight horizons and 
  well logs of 4 wells. However, for the dataset to be of practical use for our tasks, 
  we had to reinterpret the seismic, generating nine horizons separating different seismic facies intervals.
<br><br>
The interpreted horizons were used to create 651 labeled masks for inlines and 951 for crosslines. 
We present the results of two experiments to demonstrate the utility of our dataset.
<br><br>
<b>Licence</b><br>
<a href="https://creativecommons.org/licenses/by/4.0/legalcode">
Creative Commons Attribution 4.0 International
</a>
<br>
<br>
<h3>
2 Netherland-F3 ImageMask Dataset
</h3>
<h4>2.1 Download Netherland-F3-ImageMask-Dataset</h4>
 If you would like to train this Netherland-F3 Segmentation model by yourself,
 please download  our dataset <a href="https://drive.google.com/file/d/1BseAaCyJGSuUGpLwT9O_bmr0cF_MvQnu/view?usp=sharing">
 Augmented-Netherland-F3-ImageMask-Dataset.zip  </a> on the google drive
, expand the downloaded and put it under <b>./dataset</b> folder to be.<br>
<pre>
./dataset
└─Netherland-F3
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
<br>
<b>Netherland-F3 Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/Netherland-F3/Netherland-F3_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is large enough  to use for a training set of our segmentation model.
<br>
<br>
<h4>2.2 Netherland-F3 Dataset Derivation</h4>
The original data folder structure of crosslines and masks of <b>Netherland-F3 Interpretation</b> is the following.<br>
<pre>
./Netherland-F3
├─crosslines
│   ├─crossline_300.tiff
...
│   └─crossline_1250.tiff
└─masks
     ├─crossline_300_mask.png
...
     └─crossline_1250_mask.png

</pre>
We used the following 2 Python scripts to derive our augmented dataset.<br>
<ul>
<li><a href="./generator/ImageMaskDatasetGenerator.py">ImageMaskDatasetGenerator.py</a></li>
<li><a href="./generator/split_master.py">split_master.py</a></li>
</ul>

As shown below, some original masks contain inappropriate annotation at right side edge regions. 
Therefore, we first generated 462x462
pixels left-side cropped images and masks to exclude the right side edge part from the original one
, and resized them to 512x512 pixels, and finally augmented the resized one by the Generator tool
,which was enabled deformation and distortion augmentation flags, to generate our dataset.<br><br>
<img src="./projects/TensorFlowFlexUNet/Netherland-F3/asset/nogood_images.png" width="880" height="auto"><br>
<img src="./projects/TensorFlowFlexUNet/Netherland-F3/asset/nogood_masks.png"  width="880" height="auto"><br>
<br>
We also used the following <b>Greenish-Blue</b> color-class mapping table to generate our colorized masks, and define a rgb_map for our mask format between 
indexed colors and rgb colors in <a href="./projects/TensorFlowFlexUNet/Netherland-F3/train_eval_infer.config"> 
<b>train_eval_infer.config</b></a> file. <br>

On the class table, please refer to Figure 3 in <a href='https://arxiv.org/pdf/1904.00770'>
Netherlands Dataset: A New Public Dataset for Machine Learning in Seismic Interpretation</a><br>
<br>
<a id="color_class_mapping_table">Netherland-F3 color-class-mapping-table</a>
<table border=1 style='border-collapse:collapse;' cellpadding='5'>
<caption>Netherland F3 9 classes</caption>
<tr><th>Indexed Color</th><th>Color</th><th>RGB</th><th>Group</th></tr>
<tr><td>1</td><td with='80' height='auto'><img src='./color_class_mapping/Layer_1.png' widith='40' height='25'></td><td>(20, 70, 245)</td><td>Layer_1</td></tr>
<tr><td>2</td><td with='80' height='auto'><img src='./color_class_mapping/Layer_2.png' widith='40' height='25'></td><td>(40, 90, 235)</td><td>Layer_2</td></tr>
<tr><td>3</td><td with='80' height='auto'><img src='./color_class_mapping/Layer_3.png' widith='40' height='25'></td><td>(60, 110, 225)</td><td>Layer_3</td></tr>
<tr><td>4</td><td with='80' height='auto'><img src='./color_class_mapping/Layer_4.png' widith='40' height='25'></td><td>(80, 130, 215)</td><td>Layer_4</td></tr>
<tr><td>5</td><td with='80' height='auto'><img src='./color_class_mapping/Layer_5.png' widith='40' height='25'></td><td>(100, 150, 205)</td><td>Layer_5</td></tr>
<tr><td>6</td><td with='80' height='auto'><img src='./color_class_mapping/Layer_6.png' widith='40' height='25'></td><td>(120, 170, 195)</td><td>Layer_6</td></tr>
<tr><td>7</td><td with='80' height='auto'><img src='./color_class_mapping/Layer_7.png' widith='40' height='25'></td><td>(140, 190, 185)</td><td>Layer_7</td></tr>
<tr><td>8</td><td with='80' height='auto'><img src='./color_class_mapping/Layer_8.png' widith='40' height='25'></td><td>(160, 210, 175)</td><td>Layer_8</td></tr>
<tr><td>9</td><td with='80' height='auto'><img src='./color_class_mapping/Layer_9.png' widith='40' height='25'></td><td>(180, 230, 165)</td><td>Layer_9</td></tr>
</table>
<br>
<h4>2.3 Train Image and Mask Samples</h4>
<b>Train_images_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Netherland-F3/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Netherland-F3/asset/train_masks_sample.png" width="1024" height="auto">
<br>

<h3>
3 Train TensorFlowUNet Model
</h3>
 We trained Netherland-F3 TensorFlowFlexUNet Model by using the following
<a href="./projects/TensorFlowFlexUNet/Netherland-F3/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/Netherland-F3 and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small base_filters=16 and a large base_kernels=(11,11) for the first Conv Layer of Encoder Block of 
<a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
;Specify multiple of 256.
image_width    = 512
image_height   = 512
image_channels = 3
input_normalize = True
num_classes    = 10

base_filters   = 16
base_kernels   = (11,11)
num_layers     = 8
dropout_rate   = 0.05
dilation       = (1,1)

</pre>

<b>Learning rate</b><br>
Defined a small learning rate.  
<pre>
[model]
learning_rate  = 0.00007
</pre>

<b>Online augmentation</b><br>
Disabled our online augmentation.  
<pre>
[model]
model         = "TensorFlowFlexUNet"
generator     = False
</pre>

<b>Loss and metrics functions</b><br>
Specified "categorical_crossentropy" and <a href="./src/dice_coef_multiclass.py">"dice_coef_multiclass"</a>.<br>
<pre>
[model]
loss           = "categorical_crossentropy"
metrics        = ["dice_coef_multiclass"]
</pre>
<b>Learning rate reducer callback</b><br>
Enabled learning_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.4
reducer_patience   = 4
</pre>

<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>RGB Color map</b><br>
rgb color map dict for Netherland-F3 1+9 classes.
Please refer to <b><a href="#color_class_mapping_table">Netherland-F3 color-class-mapping-table.</a></b><br>
<pre>
[mask]
mask_datatype    = "categorized"
mask_file_format = ".png"
; Netherland-F3 1+9 classes
rgb_map={(0,0,0):0,(20,70,245):1,(40,90,235):2,(60,110,225):3,(80,130,215):4,(100,150,205):5,(120,170,195):6,(140,190,185):7,(160,210,175):8,(180,230,165):9,}
</pre>

<b>Epoch change inference callback</b><br>
Enabled <a href="./src/EpochChangeInferencer.py">epoch_change_infer callback (EpochChangeInferencer.py)</a></b>.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
num_infer_images         = 6
</pre>

By using this callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_inference output at starting (epoch 1,2,3)</b><br>
<img src="./projects/TensorFlowFlexUNet/Netherland-F3/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at middlepoint (epoch 23,24,25)</b><br>
<img src="./projects/TensorFlowFlexUNet/Netherland-F3/asset/epoch_change_infer_at_middlepoint.png" width="1024" height="auto"><br>
<br>

<b>Epoch_change_inference output at ending (epoch 48,49,50)</b><br>
<img src="./projects/TensorFlowFlexUNet/Netherland-F3/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>
<br>


In this experiment, the training process was terminated at epoch 50.<br><br>
<img src="./projects/TensorFlowFlexUNet/Netherland-F3/asset/train_console_output_at_epoch50.png" width="880" height="auto"><br>
<br>

<a href="./projects/TensorFlowFlexUNet/Netherland-F3/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Netherland-F3/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorFlowFlexUNet/Netherland-F3/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Netherland-F3/eval/train_losses.png" width="520" height="auto"><br>

<br>

<h3>
4 Evaluation
</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/Netherland-F3</b> folder,
and run the following bat file to evaluate TensorFlowUNet model for Netherland-F3.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetEvaluator.py ./train_eval_infer.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/Netherland-F3/asset/evaluate_console_output_at_epoch50.png" width="880" height="auto">
<br><br>Image-Segmentation-Netherland-F3

<a href="./projects/TensorFlowFlexUNet/Netherland-F3/evaluation.csv">evaluation.csv</a><br>

The loss (categorical_crossentropy) to this Netherland-F3/test was not low, but dice_coef_multiclass high as shown below.
<br>
<pre>
categorical_crossentropy,0.096
dice_coef_multiclass,0.9649
</pre>
<br>
<h3>
5 Inference
</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/Netherland-F3</b> folder
, and run the following bat file to infer segmentation regions for images by the Trained-TensorFlowUNet model for Netherland-F3.<br>
<pre>
./3.infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetInferencer.py ./train_eval_infer.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/Netherland-F3/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/Netherland-F3/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorFlowFlexUNet/Netherland-F3/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged Netherland-F3 Images of 512x512 pixels</b><br>
<b><a href="#color_class_mapping_table">Netherland-F3 color-class-mapping-table</a>
</b>
<br><br>
<table  cellpadding='5'>
<tr>
<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Inferred-mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Netherland-F3/mini_test/images/1026_.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Netherland-F3/mini_test/masks/1026_.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Netherland-F3/mini_test_output/1026_.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Netherland-F3/mini_test/images/1110_.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Netherland-F3/mini_test/masks/1110_.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Netherland-F3/mini_test_output/1110_.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Netherland-F3/mini_test/images/1183_.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Netherland-F3/mini_test/masks/1183_.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Netherland-F3/mini_test_output/1183_.png" width="320" height="auto"></td>
</tr>


<tr>
<td><img src="./projects/TensorFlowFlexUNet/Netherland-F3/mini_test/images/1535_.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Netherland-F3/mini_test/masks/1535_.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Netherland-F3/mini_test_output/1535_.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Netherland-F3/mini_test/images/1675_.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Netherland-F3/mini_test/masks/1675_.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Netherland-F3/mini_test_output/1675_.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Netherland-F3/mini_test/images/1784_.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Netherland-F3/mini_test/masks/1784_.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Netherland-F3/mini_test_output/1784_.png" width="320" height="auto"></td>
</tr>

</table>
<hr>
<br>

<h3>
References
</h3>
<b>1. facies_classification_benchmark</b><br>
Yazeed Alaudaha<br>
<a href="https://github.com/yalaudah/facies_classification_benchmark">https://github.com/yalaudah/facies_classification_benchmark</a>
<br>
<br>
<b>2. A Machine Learning Benchmark for Facies Classification</b><br>
Yazeed Alaudah, Patrycja Micha lowicz, Motaz Alfarraj<br>
<a href="https://arxiv.org/pdf/1901.07659">
https://arxiv.org/pdf/1901.07659</a>
<br>
<br>
<b>4. Toward User-Guided Seismic Facies Interpretation With a Pre-Trained Large Vision Model</b><br>
Joshua Atolagbe, Ardiansyah Koeshidayatullah<br>
<a href="https://ieeexplore.ieee.org/document/10909446">
https://ieeexplore.ieee.org/document/10909446
</a>
<br>
<br>
<b>5. Netherlands Dataset: A New Public Dataset for Machine Learning in Seismic Interpretation</b><br>
Reinaldo Mozart Silva, Lais Baroni, Rodrigo S. Ferreira1, Daniel Civitarese,<br>
Daniela Szwarcman, Emilio Vital Brazil<br>
<a href="https://arxiv.org/pdf/1904.00770">
https://arxiv.org/pdf/1904.00770
</a>
<br>
<br>
<b>6. TensorFlow-FlexUNet-Image-Segmentation-Model</b><br>
Toshiyuki Arai antillia.com <br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model
</a>
<br>
<br>
<b>7. TensorFlow-FlexUNet-Image-Segmentation-Facies</b><br>
Toshiyuki Arai antillia.com <br>
<a href="https://github.com/atlan-antillia/TensorFlow-FlexUNet-Image-Segmentation-Facies">
https://github.com/atlan-antillia/TensorFlow-FlexUNet-Image-Segmentation-Facies
</a>
<br>
<br>
