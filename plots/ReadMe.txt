The "train" and "vaild" folders contain images which illustrate the performance of the trained model on the training and validation data respectively. Each of the images contain four plots:

The first plot entitled "Seismic" is a visualisation of the input data, originally greyscale, using the seismic diverging colourmap, see
https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html

The second plot entitled "Salt" is a binary mask, labelling pixel-wise the presence or absence of salt deposit. The green contour line from this plot is copied in black onto the other three plots.

The third plot entitled "Salt Predicted" is the prediction of the trained model for the given input.

The fourth plot entitled "Salt Predicted binary" is the prediction (third plot) converted to binary. Thresholds of 0.5 and 0.6 were used for the training and for the validation data respectively.
