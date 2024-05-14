# Paint Anything

In order to work on this project please download all the **requirements**:\
\
This built on using Python 3.8 any above version should also work
- PyTorch >= 2.1.0
- OpenCV >= 4.2.0
- Matplotlib
- Scikit-Learn

Import your preferred model first in the trainer file then executed following command to start CycleGAN training:

```
python3 train/cyclegan_trainer.py
```

Then create and eval folder to eval the model and specify number of pictures you want to evaluate it.
```
python3 eval.py
```

Use your pretrained model and infer real-time using webcam by running:
```
pytohn3 webcam.py
```
