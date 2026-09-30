# Computer-Vision-Projects

Four laboratory assignments for Computer Vision I (Visión por Ordenador I), written in Python with OpenCV.

They were done in pairs at Universidad Pontificia Comillas (ICAI), course 2024-2025, autumn term 2024. Each lab folder contains the code, the data it runs on and the report (in Spanish, LaTeX source and PDF) that goes with it. The statement of labs 1, 3 and 4 is also included as a PDF.

## Lab 1: camera calibration (`lab_1`)

Calibration of the right camera of a stereo pair with a chessboard pattern, and undistortion of images taken with a fisheye lens.

- Inner corners are found with `findChessboardCorners` and refined with `cornerSubPix` (30 iterations, 0.01 epsilon, 10 pixel window).
- Intrinsics, distortion coefficients and per-image extrinsics come from `calibrateCamera`. The chessboard squares are 30 mm.
- Fisheye images are corrected with `cv2.fisheye` undistortion maps and `remap`.
- The report plots the RMS error against the number of images used and gives the resulting RMS error.

The report gives an RMS error of 0.09 for the calibration of the right camera.

The report also lists the intrinsic matrices, distortion coefficients and extrinsics as given in the report.

Code: `CalibrateCamera.py` (structured version) and `lab_1.ipynb` (first exploratory version). Data: `data/left`, `data/right`, `data/fisheye`.

## Lab 2: image processing (`lab_2`)

Colour segmentation, edge detection and morphological operators on a set of photos, plus a test with added noise.

- Colour segmentation in HSV with `inRange` and `bitwise_and`, first for orange and white, then for a colour picked with trackbars (`utils.py`).
- Gaussian smoothing, Sobel and Prewitt kernels (`filter2D`) and Canny edge detection.
- Gaussian noise (mean 127, standard deviation 25) added to the images, to compare Sobel with and without Gaussian smoothing first. The report observes that smoothing removes the edges caused by noise and keeps the real ones.
- Binarisation (fixed and Otsu thresholds), then dilation and erosion written by hand with a 3x3 neighbourhood and `np.pad`.

Code: `lab_2.ipynb`, `utils.py`. Data: `data` and `noisy_data`.

## Lab 3: feature extraction and bag of visual words (`Lab_3`)

Corner, line and keypoint detection, followed by an image classifier built on a bag of visual words.

- Corners: Harris and Shi-Tomasi (notebook `partA_to-do.ipynb`).
- Lines: Canny edges followed by the Hough transform (`partB_to-do.ipynb`).
- Keypoints: a manual implementation of the SIFT steps (Gaussian scale space, difference of Gaussians, quadratic refinement of extrema, orientation and descriptors in `utils.py`), checked against the image rotated 90 degrees (`partC_to-do.ipynb`).
- Classifier: SIFT or KAZE descriptors, K-means vocabulary (`bow.py`), classifier in `image_classifier.py`. `words_bag.py` runs the experiments and writes `results.csv`.

The report states 32 runs with vocabulary sizes 50, 100, 200 and 400 and 10 to 40 K-means iterations, on a dataset with 2981 training and 1501 test images. The dataset is not in this repository. Best runs reported:

| Extractor | Vocabulary | Iterations | Train accuracy | Test accuracy | Time |
| --- | --- | --- | --- | --- | --- |
| SIFT | 400 | 20 | 0.811 | 0.521 | about 18 min |
| KAZE | 400 | 20 | 0.685 | 0.437 | about 29 min |

Larger vocabularies gave higher accuracy in these runs.

To run the experiments, place the images in `data/dataset/training` and `data/dataset/validation` and run `python words_bag.py` from `Lab_3`. The extractor, vocabulary sizes and iterations are set at the bottom of that file. The code needs `opencv-contrib-python`, `numpy`, `pandas`, `scikit-learn`, `tqdm` and `matplotlib`.

## Lab 4: motion detection and object tracking (`lab_4`)

Work on video sequences (`slow_traffic_small.mp4`, `visiontraffic.avi`).

- Background subtraction: frame differencing against a reference frame (`absdiff`), then the Gaussian mixture models MOG and MOG2 from OpenCV. The report compares their parameters (history 350, `varThreshold` of at least 30 for MOG2, 5 mixtures and background ratio 0.6 for MOG) and discusses noise, shadows and speed.
- Optical flow: Lucas-Kanade tracking of corners found with `goodFeaturesToTrack` (`calcOpticalFlowPyrLK`), in the notebook. The report has no text for this section.
- Tracking: a Kalman filter with four state variables (position and velocity) and two measured ones (position), with the hue histogram back-projection of a selected region and `meanShift` to locate the object in each frame. The report shows that a transition matrix that includes the velocity follows the car bonnet better than the identity matrix.

Code: `lab4.ipynb`.

## Authors

Pablo Tuñón Laguna and Lydia Ruiz Martínez. Both names are on the cover of every report.
