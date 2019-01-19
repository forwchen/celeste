## Project of Team Celeste in Google AI ML Winter Camp Shanghai
14 - 18 January, 2019

Team Members: Minjun Li & Shaoxiang Chen, Fudan University

This project wins the final **Best Project Award**. 

### Introduction
We first explored training different models to recognize sketches in the Quick Draw dataset. We found that a hand-crafted CNN can be trained in a short time and achieve reasonable accuracy. 
With a trained CNN, we are able to perform various interesting tasks, such as: **inter-class similarity analysis**, **CNN class activation map visualization for interpretability**, **definitive stroke analysis and visualization**, and finally, to **hint the players of Quick Draw** with our CNN and Sketch RNN. Technics from papers[1,2] are used in our work. 

[Slides](https://drive.google.com/file/d/1C3Z2w02fp16IHedLa7EsprKt8JuVorvW/view?usp=sharing) and [Poster](https://docs.google.com/presentation/d/1ZVL8tNfcQwmrQDrjD7xsQrK2Wicy3xOxWTGXVyQEUHI/edit?usp=sharing) are available in Google Drive.

### About the Code

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

### Inter-Class Similarity Analysis  
IPython Notebook [here](https://github.com/forwchen/celeste/blob/master/cluster/analysis.ipynb).


### CNN Class Activation Map Visualization for Interpretability  
IPython Notebook [here](https://github.com/forwchen/celeste/blob/master/trainval/cnn/cnn_vis.ipynb).

### Definitive Stroke Analysis and Visualization  
IPython Notebook [here](https://github.com/forwchen/celeste/blob/master/infer/best_stroke.ipynb).


#### References
[1] Zhou, Bolei, Aditya Khosla, Agata Lapedriza, Aude Oliva, and Antonio Torralba. "Learning deep features for discriminative localization." CVPR 2016.

[2] Ha, David, and Douglas Eck. "A neural representation of sketch drawings." ICLR 2018.
