# CATNet Coordinate-Aware Transformer for All-in-One Image Restoration

Junling He, Yang Zhao, Wenting Li, [Ziyang Chen](https://scholar.google.com.hk/citations?hl=zh-CN&user=t64KgqAAAAAJ), Yao Xiao, Bingshu Wang, and Yongjun Zhang

## Training

After preparing the training data in ```data/``` directory, use 
```
python train.py
```
to start the training of the model. Use the ```de_type``` argument to choose the combination of degradation types to train on. By default it is set to all the 5 degradation tasks (denoising, deraining, dehazing, deblurring, enhancement).

Example Usage: If we only want to train on deraining and dehazing:
```
python train.py --de_type derain dehaze
```

## Testing

After preparing the testing data in ```test/``` directory, place the mode checkpoint file in the ```ckpt``` directory. The pre-trained model can be downloaded [here](https://drive.google.com/drive/folders/1x2LN4kWkO3S65jJlH-1INUFiYt8KFzPH?usp=sharing). To perform the evaluation, use
```
python test.py --mode {n}
```
```n``` is a number that can be used to set the tasks to be evaluated on, 0 for denoising, 1 for deraining, 2 for dehazing, 3 for deblurring, 4 for enhancement, 5 for three-degradation all-in-one setting and 6 for five-degradation all-in-one setting.

Example Usage: To test on all the degradation types at once, run:

```

**Acknowledgment:** This code is based on the [PromptIR](https://github.com/va1shn9v/PromptIR) and [AdaIR](https://github.com/c-yn/AdaIR/tree/main) repository. 
python test.py --mode 6
```
<!-- 

**Acknowledgment:** This code is based on the [PromptIR](https://github.com/va1shn9v/PromptIR) and the [AdaIR](https://github.com/c-yn/AdaIR/tree/main) repository. 
