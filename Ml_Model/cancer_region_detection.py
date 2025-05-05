import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import skew, kurtosis
import os

for root, dirs, files in os.walk(r"C:\cancer\test\bcc"):
    for file in files:
        thermal_img = cv2.imread(os.path.join(root, file))
        thermal_im = cv2.imread(os.path.join(root, file))
        
        # Normalize grayscale image to [0, 1] range
        reversed_normalized_matrix = cv2.cvtColor(thermal_im, cv2.COLOR_BGR2GRAY) / 255.0
        
        # Flatten image for clustering
        reshaped_img = thermal_img.reshape((-1, 3))

        # Apply K-Means clustering (2 clusters: affected and unaffected)
        kmeans = KMeans(n_clusters=2)
        kmeans.fit(reshaped_img)

        labels = kmeans.labels_
        centers = np.uint8(kmeans.cluster_centers_)
        segmented = labels.reshape((thermal_im.shape[0], thermal_im.shape[1]))

        # Create binary masks for each cluster
        affected_mask = np.where(segmented == 0, 255, 0).astype('uint8')
        unaffected_mask = np.where(segmented == 1, 255, 0).astype('uint8')

        # Extract temperature values from grayscale image
        temperature_affected = reversed_normalized_matrix[affected_mask == 255]
        temperature_unaffected = reversed_normalized_matrix[unaffected_mask == 255]

        # Statistical measures
        variance_affected = np.var(temperature_affected)
        variance_unaffected = np.var(temperature_unaffected)

        std_dev_affected = np.std(temperature_affected)
        std_dev_unaffected = np.std(temperature_unaffected)

        skewness_affected = skew(temperature_affected)
        skewness_unaffected = skew(temperature_unaffected)

        kurtosis_affected = kurtosis(temperature_affected)
        kurtosis_unaffected = kurtosis(temperature_unaffected)

        print(f"Affected Region - Variance: {variance_affected}, Skewness: {skewness_affected}, Kurtosis: {kurtosis_affected}")
        print(f"Unaffected Region - Variance: {variance_unaffected}, Skewness: {skewness_unaffected}, Kurtosis: {kurtosis_unaffected}")

        # RGB mean values for affected and unaffected regions
        affected_rgb = np.mean(thermal_im[affected_mask == 255], axis=0)
        unaffected_rgb = np.mean(thermal_im[unaffected_mask == 255], axis=0)

        # Estimate temperature difference from red channel
        red_difference = abs(affected_rgb[0] - unaffected_rgb[0])
        print((red_difference * 0.0625) / 3)  # 0.0625°C per unit (AMG8833 resolution)
