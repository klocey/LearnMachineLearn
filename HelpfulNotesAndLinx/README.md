## Machine learning links:
**Common Objects in Contexts (COCO):**  
COCO is a large-scale object detection, segmentation, and captioning dataset. See: http://cocodataset.org/#home

**Mask R-CNN:**
A framework for object instance segmentation.  
See: https://arxiv.org/abs/1703.06870

**Faster R-CNN:**
Faster R-CNN is a model that predicts both bounding boxes and class scores for potential objects in the image.   
See: https://arxiv.org/abs/1506.01497

**scikit-learn:**
https://scikit-learn.org/stable/auto_examples/index.html


**healthcareai:**
The healthcare.ai software is designed to streamline healthcare machine learning. They do this by including functionality specific to healthcare, as well as simplifying the workflow of creating and deploying models.  
See: https://healthcare.ai/

Includes both python and R distributions.

* R:```https://cloud.r-project.org/web/packages/healthcareai/index.html```
* Python:```https://healthcareai-py.readthedocs.io/en/master/getting_started/```

The python distribution is less supported and is noted to have some bugs. The R version is better supported and installable via CRAN and RStudio. 

**links:**  
video: https://healthcare.ai/software/

**pytorch**:

links:

* ```https://colab.research.google.com/github/pytorch/vision/blob/temp-tutorial/tutorials/torchvision_finetuning_instance_segmentation.ipynb#scrollTo=DBIoe_tHTQgV```
* ```https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html```
* Tensor operations: ```https://pytorch.org/docs/stable/torch.html```

* **Figure 8**: https://www.figure-eight.com/platform/

**keras**:

links:


## Misc. Links
1. **Techniques for thinking in higher dimensions:** https://mathoverflow.net/questions/25983/intuitive-crutches-for-higher-dimensional-thinking


## Functions
```
def sigmoid(x):
    return 1/(1 + np.exp(-x))
```

```
def cost(b):
	return 1/(1+np.exp(-b))
	
def approx_slope(b, h=0.0001):
	# An approximated slope
	# benefit: can theoretically approximate any fxn
	# with a small enough window
	# detriment: this is only an approximation and requires
	# window size consideration (i.e., |b - h|)
	return (cost(b+h)-cost(b))/h
	
def deriv_slope(b, h):
	# The derived slope for a squared-cost function
	return 2b-8	
```


## 4 Desmos

* https://www.desmos.com/calculator/ggasmzr86s


## Helpful Videos:
* Part 5: https://www.youtube.com/watch?v=kft1AJ9WVDk 
* Part 6: https://www.youtube.com/watch?v=Py4xvZx-A1E
* Part 7: https://www.youtube.com/watch?v=EnGmg-kvpYs

* **ROC AUC:** https://www.youtube.com/watch?v=4jRBRDbJemM
* **TensorFlow 2.0:** https://www.youtube.com/watch?v=6g4O5UOH304


## Notes:
ordinary least squares (OLS) regression model, aka, linear regression:  

```
Let: 
Machine learning <=> algebraic equivalent <=> regression
  o = output -> y
  i = input -> x
  w = weights -> m
  b = bias -> b
	
o = wi+b
Or
y = mx+b
```

https://jsfiddle.net/wbkgb73v/

## Terms

1. Gradient descent & stochastic gradient descent
2. ReLU
3. Hebbian Theory
4. Back propogation
5. MNIST data
6. Human in the loop approach
7. Cauchy-Schwarz inequality
8. Support vector machine
9. Cross-Entropy loss function
10. ROC (receiving operator characteristic) and AUC






## Neural Networks and Deep Learning

* Chapter 1: http://neuralnetworksanddeeplearning.com/chap1.html
* Chapter 2: http://neuralnetworksanddeeplearning.com/chap2.html
* 



## R
ggplot: https://ggplot2.tidyverse.org/