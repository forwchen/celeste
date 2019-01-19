## Project of Team Celeste in Google AI ML Winter Camp Shanghai
14 - 18 January, 2019

Team Members: Minjun Li & Shaoxiang Chen

### Notes

More codes (ipython notebooks for analysis and visualization) are currently not merged into the master branch, and are in branch dev-sxchen: https://github.com/forwchen/celeste/tree/dev-sxchen

Please be sure to also checkout branch dev-sxchen,
or download [here](https://drive.google.com/file/d/1Y108_ik8BEcOMxOOTNcVYENqQc4NBKW4/view?usp=sharing) from Google Drive.

### Introduction
We first explored training different models to recognize sketches in the Quick Draw dataset. We found that a hand-crafted CNN can be trained in a short time and achieve reasonable accuracy. 
With a trained CNN, we are able to perform various interesting tasks, such as: **inter-class similarity analysis**, **CNN class activation map visualization for interpretability**, **definitive stroke analysis and visualization**, and finally, to **hint the players of Quick Draw** with our CNN and Sketch RNN.

Technics from papers[1,2] are used in our work.

[Slides](https://drive.google.com/file/d/1C3Z2w02fp16IHedLa7EsprKt8JuVorvW/view?usp=sharing) and [Poster](https://docs.google.com/presentation/d/1ZVL8tNfcQwmrQDrjD7xsQrK2Wicy3xOxWTGXVyQEUHI/edit?usp=sharing) are available in Google Drive.

#### References
[1] Zhou, Bolei, Aditya Khosla, Agata Lapedriza, Aude Oliva, and Antonio Torralba. "Learning deep features for discriminative localization." CVPR 2016.

[2] Ha, David, and Douglas Eck. "A neural representation of sketch drawings." ICLR 2018.
