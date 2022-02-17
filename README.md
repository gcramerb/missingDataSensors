# Deep impute for Sensors missing data

This is a Deep Learning method that handles the missing data problem in the context of sensor data.

The method is based on a denoising Autoencoder that learns complex patterns useful for imputing data in the missing part.

We available this method from two points of view. The first one is the reconstruction metrics, in terms of mean squared error. 
The second one is in terms of the performance of the reconstructed data in a task. 
The chosen task is human activity recognition based on sensor data. 
We achieve the state of art in both metrics in the USC-HAD dataset.