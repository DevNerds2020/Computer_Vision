import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pywt
from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity,
    mean_squared_error,
)
from sklearn.metrics import mean_absolute_error


def add_salt_and_pepper_noise(image, density):
    noisy_image = np.copy(image)
    num_pixels = int(density * image.size)

    # Salt noise
    salt_coords = [
        np.random.randint(0, i - 1, int(num_pixels / 2)) for i in image.shape
    ]
    noisy_image[salt_coords] = 255

    # Pepper noise
    pepper_coords = [
        np.random.randint(0, i - 1, int(num_pixels / 2)) for i in image.shape
    ]
    noisy_image[pepper_coords] = 0

    return noisy_image


def add_speckle_noise(image, variance):
    noise = np.random.normal(0, variance, image.shape)
    noisy_image = image + image * noise
    noisy_image = np.clip(noisy_image, 0, 255)  # Clip values to valid intensity range
    return noisy_image.astype(np.uint8)


def add_gaussian_noise(image, mean_var):
    mean, variance = mean_var
    noise = np.random.normal(mean, variance, image.shape)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 255)  # Clip values to valid intensity range
    return noisy_image.astype(np.uint8)


# Function to add Poisson noise
def add_poisson_noise(image, noise_scaling):
    noisy_image = np.random.poisson(image * noise_scaling) / noise_scaling
    noisy_image = np.clip(noisy_image, 0, 255)  # Clip values to valid intensity range
    return noisy_image.astype(np.uint8)


def add_noise(noise, image, variance):
    if noise == "salt_and_pepper":
        noisy_image = add_salt_and_pepper_noise(image, variance)
    elif noise == "speckle":
        noisy_image = add_speckle_noise(image, variance)
    elif noise == "gaussian":
        noisy_image = add_gaussian_noise(image, variance)
    elif noise == "poisson_noise":
        noisy_image = add_poisson_noise(image, variance)
    else:
        noisy_image = image
    return noisy_image


# Function to apply denoising filters
def apply_filters(image, filter_type, filter_size, wavelet_type="haar"):
    if filter_type == "median":
        return cv2.medianBlur(image, filter_size)
    elif filter_type == "gaussian":
        return cv2.GaussianBlur(image, (filter_size, filter_size), 0)
    elif filter_type == "bilateral":
        return cv2.bilateralFilter(image, filter_size, 75, 75)
    elif filter_type == "blur":
        return cv2.blur(image, (filter_size, filter_size))
    elif filter_type == "laplacian":
        return cv2.Laplacian(image, cv2.CV_64F, ksize=filter_size)
    elif filter_type == "sobel":
        return cv2.Sobel(image, cv2.CV_64F, 1, 1, ksize=filter_size)
    elif filter_type == "roberts":
        kernel = np.array([[1, 0], [0, -1]])
        return cv2.filter2D(image, -1, kernel)
    elif filter_type == "high_pass":
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel)
    elif filter_type == "unsharp_masking":
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel)


def plot_results(
    image, noise_type, noise_variances, filter_types, filter_sizes, figsize=(20, 15)
):
    fig, axes = plt.subplots(
        len(noise_variances), len(filter_sizes) * len(filter_types) + 1, figsize=figsize
    )

    # Plot original image
    axes[0, 0].imshow(image, cmap="gray")
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")

    for i, density in enumerate(noise_variances):
        noisy_image = add_noise(noise_type, image, density)
        axes[i, 0].imshow(noisy_image, cmap="gray")
        axes[i, 0].set_title(f"Noisy (Density {density})")
        axes[i, 0].axis("off")

        for j, filter_type in enumerate(filter_types):
            for k, filter_size in enumerate(filter_sizes):
                denoised_image = apply_filters(noisy_image, filter_type, filter_size)
                axes[i, j * len(filter_sizes) + k + 1].imshow(
                    denoised_image, cmap="gray"
                )
                axes[i, j * len(filter_sizes) + k + 1].set_title(
                    f"{filter_type.capitalize()} Filter (Size {filter_size})"
                )
                axes[i, j * len(filter_sizes) + k + 1].axis("off")

                # Calculate quality metrics
                mse, psnr, ssim, mae = calculate_quality_metrics(image, denoised_image)

                # Display quality metrics as annotations
                axes[i, j * len(filter_sizes) + k + 1].text(
                    0.5,
                    -0.15,
                    f"MSE: {mse:.2f}\nPSNR: {psnr:.2f}\nSSIM: {ssim:.2f}\nMAE: {mae:.2f}",
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=axes[i, j * len(filter_sizes) + k + 1].transAxes,
                    bbox=dict(facecolor="white", alpha=0.8),
                )

    plt.tight_layout()
    plt.show()


def add_noise_to_dataset(noise_type, noise_variances, input_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through image categories
    categories = ["Med", "Normal", "RS"]
    for category in categories:
        input_category_folder = os.path.join(input_folder, category)
        output_category_folder = os.path.join(output_folder, category)

        # Create output category folder if it doesn't exist
        if not os.path.exists(output_category_folder):
            os.makedirs(output_category_folder)

        # Iterate through images in the category folder
        for filename in os.listdir(input_category_folder):
            if filename.endswith((".jpg", ".jpeg", ".png", ".bmp")):
                input_image_path = os.path.join(input_category_folder, filename)
                output_image_path = os.path.join(output_category_folder, filename)

                # Read the original image
                original_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

                # Get the corresponding noise variance for the current image
                # Cycle through noise_variances in a circular manner
                noise_variance = noise_variances.pop(0)
                noise_variances.append(noise_variance)

                # Add speckle noise
                noisy_image = add_noise(noise_type, original_image, noise_variance)

                # Save the noisy image
                cv2.imwrite(output_image_path, noisy_image)

    print(
        "Noise added to images with varying variances and saved in 'dataset-noise' folder."
    )


def apply_filter_on_file_samples_and_plot_them(
    images, filter_types, filter_sizes, figsize=(15, 10)
):
    fig, axes = plt.subplots(
        len(images), len(filter_sizes) * len(filter_types) + 1, figsize=figsize
    )
    for i, image in enumerate(images):
        axes[i, 0].imshow(image, cmap="gray")
        axes[i, 0].set_title("Noisy")
        axes[i, 0].axis("off")

        for j, filter_type in enumerate(filter_types):
            for k, filter_size in enumerate(filter_sizes):
                denoised_image = apply_filters(image, filter_type, filter_size)
                axes[i, j * len(filter_sizes) + k + 1].imshow(
                    denoised_image, cmap="gray"
                )
                axes[i, j * len(filter_sizes) + k + 1].set_title(
                    f"{filter_type.capitalize()} Filter (Size {filter_size})"
                )
                axes[i, j * len(filter_sizes) + k + 1].axis("off")

                # Calculate quality metrics
                mse, psnr, ssim, mae = calculate_quality_metrics(image, denoised_image)

                # Display quality metrics as annotations
                axes[i, j * len(filter_sizes) + k + 1].text(
                    0.5,
                    -0.15,
                    f"MSE: {mse:.2f}\nPSNR: {psnr:.2f}\nSSIM: {ssim:.2f}\nMAE: {mae:.2f}",
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=axes[i, j * len(filter_sizes) + k + 1].transAxes,
                    bbox=dict(facecolor="white", alpha=0.8),
                )


def read_random_samples_from_dataset(input_folder, num_samples):
    # Iterate through image categories
    categories = ["Med", "Normal", "RS"]
    images = []
    for category in categories:
        input_category_folder = os.path.join(input_folder, category)

        # Iterate through images in the category folder
        for filename in os.listdir(input_category_folder):
            if filename.endswith((".jpg", ".jpeg", ".png", ".bmp")):
                input_image_path = os.path.join(input_category_folder, filename)

                # Read the original image
                original_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
                images.append(original_image)

    # Choose random samples from the dataset
    random_samples = np.random.choice(images, num_samples, replace=False)
    return random_samples


def apply_wavelet_transform(image, wavelet_type, wavelet_level):
    coeffs = pywt.wavedec2(image, wavelet_type, level=1)
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0

    # Reconstruct the image
    image_H = pywt.waverec2(coeffs_H, wavelet_type)
    return image_H, coeffs


def apply_wavelet_transform_on_file_samples_and_plot_them(
    images, wavelet_type, wavelet_level, figsize=(20, 30)
):
    fig, axes = plt.subplots(len(images), 6, figsize=figsize)
    for i, image in enumerate(images):
        axes[i, 0].imshow(image, cmap="gray")
        axes[i, 0].set_title("Original")
        axes[i, 0].axis("off")

        denoised_image, coeffs = apply_wavelet_transform(
            image, wavelet_type, wavelet_level
        )
        LL, (LH, HL, HH) = coeffs
        axes[i, 1].imshow(denoised_image, cmap="gray")
        axes[i, 1].set_title(
            f"{wavelet_type.capitalize()} Wavelet (Level {wavelet_level})"
        )
        axes[i, 1].axis("off")

        mse, psnr, ssim, mae = calculate_quality_metrics(image, denoised_image)
        axes[i, 1].text(
            0.5,
            -0.15,
            f"MSE: {mse:.2f}\nPSNR: {psnr:.2f}\nSSIM: {ssim:.2f}\nMAE: {mae:.2f}",
            horizontalalignment="center",
            verticalalignment="center",
            transform=axes[i, 1].transAxes,
            bbox=dict(facecolor="white", alpha=0.8),
        )

        axes[i, 2].imshow(LL, cmap="gray")
        axes[i, 2].set_title(f"LL")
        axes[i, 2].axis("off")

        axes[i, 3].imshow(LH, cmap="gray")
        axes[i, 3].set_title(f"LH")
        axes[i, 3].axis("off")

        axes[i, 4].imshow(HL, cmap="gray")
        axes[i, 4].set_title(f"HL")
        axes[i, 4].axis("off")

        axes[i, 5].imshow(HH, cmap="gray")
        axes[i, 5].set_title(f"HH")
        axes[i, 5].axis("off")


def apply_wavelet_transform_on_file_samples_and_plot_them_different_levels(
    images, wavelet_type, wavelet_level, figsize=(20, 30)
):
    fig, axes = plt.subplots(len(images), 2, figsize=figsize)
    for i, image in enumerate(images):
        axes[i, 0].imshow(image, cmap="gray")
        axes[i, 0].set_title("Original")
        axes[i, 0].axis("off")

        denoised_image, coeffs = apply_wavelet_transform(
            image, wavelet_type, wavelet_level
        )
        axes[i, 1].imshow(denoised_image, cmap="gray")
        axes[i, 1].set_title(
            f"{wavelet_type.capitalize()} Wavelet (Level {wavelet_level})"
        )
        axes[i, 1].axis("off")

        mse, psnr, ssim, mae = calculate_quality_metrics(image, denoised_image)
        axes[i, 1].text(
            0.5,
            -0.15,
            f"MSE: {mse:.2f}\nPSNR: {psnr:.2f}\nSSIM: {ssim:.2f}\nMAE: {mae:.2f}",
            horizontalalignment="center",
            verticalalignment="center",
            transform=axes[i, 1].transAxes,
            bbox=dict(facecolor="white", alpha=0.8),
        )


# Function to calculate image quality metrics
def calculate_quality_metrics(original, denoised):
    try:
        mse = mean_squared_error(original, denoised)
        psnr = peak_signal_noise_ratio(original, denoised)
        ssim = structural_similarity(original, denoised, data_range=1)
        mae = mean_absolute_error(original, denoised)
        return mse, psnr, ssim, mae
    except:
        return -1, -1, -1, -1
