# Creating colour palettes using K-Means clustering in Python 3.6
---

This small project generates a colour palette  (or colour _scheme_) based on an input image, by performing k-means clustering analysis with `sklearn` to obtain its dominant colours.

This makes use of the following libraries:
+ `numpy`
+ `pandas`
+ `sklearn`
+ `PIL`
+ `matplotlib`
+ `seaborn`

This script analyses each pixel's RGB values in order to find clusters in the colour space.


## Some examples:
---
![example-1](examples/example-1)

![example-2](examples/example-2)

![example-4](examples/example-3)

We can even plot a colourmap of the original image using the cluster centers that we found with our model.

![example-4](examples/example-4)

![cmap-example-4](examples/cmap-example-4)
