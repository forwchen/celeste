## Project of Team Celeste in Google AI ML Winter Camp Shanghai
14 - 18 January, 2019

Project Name: Hinted Quick Draw

Team Members: Minjun Li & Shaoxiang Chen, Fudan University

This project wins the final **[Best Project Award](https://sites.google.com/google.com/ml-camp/shanghai-site/winning-teams)**. Many thanks to Google and Tensorflow🌞!

### 1. Introduction
Our project is mainly exploring the Quick Draw dataset and improving the Quick Draw game based on our findings. In the Quick Draw game, the player is asked to draw an object of a specified class with several sketches. Thus we first need a recognition model to classify sketches. However, our focus is not to train a super accurate model to recognize the drawings, but to perform interesting analysis based on our trained model which has reasonable performance. 
We first explored training different models (including RNN & CNN) to recognize drawings in the Quick Draw dataset. We found that a hand-crafted CNN can be trained in a short time and achieve reasonable accuracy (~70% accuracy is enough to play the game).  
With a trained CNN, we are able to perform various interesting analysis, such as: **inter-class similarity analysis** to find out which classes are easily mixed up with others (including t-SNE visualization and confusing pairs analysis), **CNN class activation map visualization for interpretability** of how the classification decision is made by the CNN, **definitive stroke analysis and visualization** which finds specific strokes that push the CNN’s prediction towards desired class.  
Finally, based on our analysis, we try to make the Quick Draw game more interesting by hint the players of Quick Draw with (1) our CNN and (2) Sketch RNN. In (1), the player gets hints about whether the current stroke he/she draws makes the drawing more like the object of desired class. In (2), the player gets a direct hint from Sketch RNN about what the next stroke should be. 
 Technics from papers[1,2] are used in our work. 

[Slides](https://drive.google.com/file/d/1C3Z2w02fp16IHedLa7EsprKt8JuVorvW/view?usp=sharing) (demo videos inside!) and [Poster](https://docs.google.com/presentation/d/1ZVL8tNfcQwmrQDrjD7xsQrK2Wicy3xOxWTGXVyQEUHI/edit?usp=sharing) are available in Google Drive.

### 2. About the Codes

```
.   
├── cluster                             # clustering analysis
│   ├── analysis.ipynb                      # notebook for inter-class similarity analysis
│   ├── class_id_map.pkl                    # file containing class label --> id mapping
│   ├── extract_feature.py                  # script to extract features of validation images from trained CNN
│   ├── tsne_cls100.png                     # image of t-SNE visualization of CNN features
│   ├── tsne.ipynb                          # notebook for t-SNE visualization
│   └── tsne.png                            # image of t-SNE visualization of CNN features
├── common                              # common files and codes used by other scripts
│   ├── class_id_map.pkl                    # file containing class label --> id mapping   
│   ├── fixed_model.py                      # final CNN model for sketch recognition
│   ├── model.py                            # contains various CNN models we tried
│   ├── preprocessing                       # preprocessing styles for different networks
│   │   ├── cifarnet_preprocessing.py
│   │   ├── inception_preprocessing.py          # we only use inception style
│   │   ├── __init__.py
│   │   ├── lenet_preprocessing.py
│   │   ├── preprocessing_factory.py
│   │   └── vgg_preprocessing.py
│   └── utils.py                            # utility functions
├── infer                               # things to do after having a trained CNN model
│   ├── bee.png                             # sample image containing a 'bee'
│   ├── best_stroke.ipynb                   # definitive stroke analysis
│   ├── ckpt                                # tensorflow model checkpoints
│   │   ├── classifier                          # CNN classifier
│   │   └── flamingo                            # sketch RNN
│   ├── class_id_map.pkl
│   ├── feat_extraction.ipynb               # extracts feature & original strokes for definitive stroke analysis
│   ├── infer.py                            # inference utility of CNN, providing image --> class API and supports local and network mode
│   ├── infer_rnn.py                        # inference utility of sketch RNN
│   ├── __init__.py
│   ├── sketch_ai_play.py                   # attempt to make sketch RNN play by itself
│   ├── sketch_cli.py                       # Quick Draw GUI that connects to an inference server for definitive stroke hints
│   ├── sketch_no_hint.py                   # Quick Draw GUI without any hint
│   └── sketch.py                           # Quick Draw GUI that gets hint from skecth RNN
├── legacy                              # legacy train & val code
│   ├── train.ipynb
│   └── val.ipynb
├── LICENSE
├── mkdata                              # make training data
│   ├── class_id_map.pkl
│   └── mkdata.ipynb                        # convert raw strokes into images and save as tfrecord
├── README.md
└── trainval                            # final train & validation scripts
    ├── cnn                                 # hand-crafted CNN
    │   ├── cnn_vis.ipynb                       # notebook for CNN class activation map visualization
    │   ├── gp_model.py                         # CNN model with global pooling for visualization
    │   ├── log
    │   ├── train.py                            # training script
    │   └── val.py                              # validation script
    └── rnn                                 # rnn from official tensorflow tutorial
        ├── create_dataset.py                   # save raw strokes to tfrecord
        └── train.py                            # training & validation script
```

### 3. Model Training

We trained RNN and CNN to recognize sketches. While RNN can achieve higher accuracy, it needs a long time to train. So we hand-crafted a shallow CNN instead, and it reaches reasonable perormance in a short time. Our objective is not to train a super accurate recognition model, but to explore and analysis the dataset with a goal of finding interesting insights that could help us (maybe) improve the Quick Draw game. 
<div align="center">
  <img src="https://raw.githubusercontent.com/forwchen/celeste/master/pics/rnn%26cnn.png" height="256">
</div>

For training CNNs, we draw the strokes in images and resize them to 128x128. The CNN is trained with batch size 512, Adam optimizer with learning rate 0.001 and 100000 iterations. All our following analysis is based on the trained CNN. 

### 4. Inter-Class Similarity Analysis  
IPython Notebook [here](https://github.com/forwchen/celeste/blob/master/cluster/analysis.ipynb).

To see if the CNN feature from the 'FC 512' layers captures inter-class similarity, we first do a t-SNE visualization.
<div align="center">
  <img src="https://raw.githubusercontent.com/forwchen/celeste/master/pics/t-sne_merge.png">
</div>

There are classes that form dense clusters as visualized, but some others scatter all over the space, which inidicates they could be very similar to other classes. 
To find out the confusing classes, we sort the 340 classes by their similarity to all other classes in descending order. The top ones are:
<div align="center">
  <img src="https://raw.githubusercontent.com/forwchen/celeste/master/pics/similarity_top.png" height="200">
</div>

These classes all have very simple shape that is close to a rectangle. 
See the full list [here](https://github.com/forwchen/celeste/blob/master/pics/class_similarity_heatmap.png). 

For each class, the most similar class to them can be found [here](https://github.com/forwchen/celeste/blob/master/pics/most_similar.png). And some samples here.
<div align="center">
  <img src="https://raw.githubusercontent.com/forwchen/celeste/master/pics/similar_pair.png" height="200">
</div>

### 5. CNN Class Activation Map Visualization for Interpretability  
IPython Notebook [here](https://github.com/forwchen/celeste/blob/master/trainval/cnn/cnn_vis.ipynb).

To understand why the CNN made such predictions, we use the technic from [1] to compute a CNN activation map for visualizing contributions from each spatial region. Below are samples for 'cookie', 'hospital' and 'cell phone'.
<div align="center">
  <img src="https://raw.githubusercontent.com/forwchen/celeste/master/pics/cnn_activation_map.png" height="300">
</div>

It makes sense that the botton and some edges contribute the most in 'cell phone' images.

### 6. Definitive Stroke Analysis and Visualization  
IPython Notebook [here](https://github.com/forwchen/celeste/blob/master/infer/best_stroke.ipynb).

We further want to analyze which stroke is the most effective one that pushes the model’s decision towards the target class. The approach is: for images in each class, add strokes one-by-one and keep track of the probability of target class as it changes.
The definitive stroke is the one that gives the most significant probability increase. We visualize three pairs of most similiar classes and their corresponding definitive strokes.

<div align="center">
  <img src="https://raw.githubusercontent.com/forwchen/celeste/master/pics/definitive_strokes.png" height="300">
</div>


### 7. Demo 

We wrote two demos in which the player gets hint about [whether a stroke is good or bad](https://drive.google.com/file/d/1jv_WZbZHxOoEaGJORDdoU_1CWrouw3J2/view?usp=sharing) and [Sketch RNN](https://drive.google.com/file/d/1ViyM119jvYC3MAWtsMO6aciA1kCVRT7U/view?usp=sharing). In both demos, the player is asked to draw a flamingo.  
To know whether a stroke is good or bad, we track the probability of the desired class as the player is drawing. If the player's current stroke lowers that probability, it is retracted. 
Sketch RNN[2](architecture below) is a sequence-to-sequence model for sketch generation. We only use the decoder part and at each time feed the player's strokes into the decoder so that the model outputs next strokes as hints to the player. 

<div align="center">
  <img src="https://raw.githubusercontent.com/forwchen/celeste/master/pics/sketch_rnn.png" height="256">
</div>

#### References
[1] Zhou, Bolei, Aditya Khosla, Agata Lapedriza, Aude Oliva, and Antonio Torralba. "Learning deep features for discriminative localization." CVPR 2016.

[2] Ha, David, and Douglas Eck. "A neural representation of sketch drawings." ICLR 2018.
