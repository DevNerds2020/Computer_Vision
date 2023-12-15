# Step 1: Noise Generation
1.**Gaussian Noise:** 
Add Gaussian noise to images using a function that generates random numbers following a Gaussian distribution. You can control the amount of noise by adjusting the standard deviation.

2.**Salt and Pepper Noise:**
Introduce random white and black pixels (salt and pepper noise) to the images. You can control the density of these pixels to adjust the severity of the noise.

3.**Speckle Noise:**
Apply speckle noise by multiplying the image by a random matrix with values following a uniform distribution centered around 1. Again, control the intensity of noise.

4.**Poisson Noise:**
Simulate Poisson noise by adding random values sampled from a Poisson distribution to the pixel values in the images.