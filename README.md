# ForensicsApp

ForensicsApp is our semester project for the MSc Computer Vision course.  
It is a Tkinter desktop app: every lab exercise is a tool in the sidebar, grouped  
by set ("Set N · Topic"). The tools work on the image open in the app, and the  
result can be undone, reset or exported.

The app shell (image view, undo/redo/reset, open/export, tool registry) comes  
from the course starter; the tools in `forensics_app/tools/setN/` are ours.

## Quick start

Python 3.10 or newer is recommended.

```
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python run.py
```

On some Linux distributions, Tkinter is a separate system package. For example,  
Ubuntu/Debian users may first need `sudo apt install python3-tk`.

The course handout for the app shell is [docs/STUDENT_GUIDE.md](docs/STUDENT_GUIDE.md).

## Implemented features

### Set 1 · Basics (examples from the starter)

| Tool | What it does |
| --- | --- |
| Inspect image | Shows size, mode and other basic properties without changing the image. |
| Convert to grayscale | Converts the image to 8-bit grayscale. |

### Set 2 · Image Processing

| Exercise | Tool | What it does |
| --- | --- | --- |
| 2.1 | Channel split | Splits a colour image into its red, green and blue channels. |
| 2.2 | Channel swap | Swaps the red and green channels.<br>**(⚠ Known issue: the two channels cannot be chosen, and a grayscale image makes the tool crash.)** |
| 2.3 | Masking | Keeps the pixels brighter than a threshold (135).<br>**(⚠ Known issue: the threshold is fixed; the image is not converted to grayscale, so R, G and B are masked separately; the binary mask and the inverse masked image are not shown.)** |
| 2 | Virtual try-on | Not in this version.<br>**(⚠ Known issue: missing.)** |
| 2.4 | Histogram | Grayscale or RGB histograms in a separate window; the image is not changed. |
| 2.5 | Contrast stretching | Stretches the range between two percentiles to 0–255 (same percentiles for R, G and B). |

### Set 3 · Filtering & Edges

| Exercise | Tool | What it does |
| --- | --- | --- |
| 3.1 | Convolution | Convolves the image with a kernel typed in a grid (any size up to 15×15, with a divisor). |
| 3.2 | Image filters | Gaussian, median, Sobel, Prewitt, Scharr, unsharp mask and Otsu threshold from `skimage.filters`. |
| 3.3 | Canny edge detection | Canny with configurable sigma and optional low/high thresholds. |
| 3.4 | Edge visualization | Draws the Canny edges on top of the original image. |
| 3.5 | Image sharpening | Sharpening by unsharp masking. |

### Set 4 · Template Matching & HOG

| Exercise | Tool | What it does |
| --- | --- | --- |
| 4.1 | Normalized cross-correlation | Choose an image and a template; shows the correlation map and the pixels above a threshold.<br>**(⚠ Known issue: does not use the image open in the app; the image is chosen again inside the tool.)** |
| 4.2 | Template Matching Configs | Template matching with a rotation angle and a threshold; matches are circled on the image.<br>**(⚠ Known issue: also draws Canny edges on the result (extra "Canny Sigma" setting), which the exercise does not ask for; it does not change the matches. Does not use the image open in the app.)** |
| 4.3 | HOG features | Extracts HOG features and shows the oriented-gradient image. |
| 4.4 | Person detection (HOG) | Sliding window over the image; finds the window most similar to a person template using HOG. |

### Set 5 · SIFT

| Exercise | Tool | What it does |
| --- | --- | --- |
| 5.1 | Corner detection | FAST or Harris corners with a configurable peak threshold, marked in red. |
| 5.2 | Keypoints (ORB, SIFT) | Extracts ORB or SIFT keypoints and descriptors and draws them. |
| 5.3 | Match with second image | Matches the keypoints of the image with a second image. |
| 5.4 | Closest image in folder | Finds the image in a folder that best matches the current one. Ranked by RANSAC inliers; can be switched to the simple "most matches" rule from the slide. |

### Set 6 · Faces

| Exercise | Tool | What it does |
| --- | --- | --- |
| 6.1 | Face Detection (YuNet) | Detects faces with the YuNet network (box, 5 landmarks, confidence), on the open image or live from the webcam (press Q to stop). |
| 6.2 | Eigenfaces (PCA) | Downloads LFW, splits it into train/test, fits PCA on the train faces and shows the mean face and the first K eigenfaces.<br>**(⚠ Known issue: retrains and rewrites every face PNG on each run; the first run downloads about 200 MB.)** |
| 6.3 | Face Recognition (Eigenfaces) | Picks a test face, projects it with the 6.2 model and shows the 3 nearest train faces with their PCA distance.<br>**(⚠ Known issue: only faces from the LFW test split can be chosen; a newly opened photo cannot be recognized.)** |

### Set 7 · Bag of Visual Words

| Exercise | Tool | What it does |
| --- | --- | --- |
| 7.1 | Codebook (visual words) | Extracts ORB/SIFT descriptors from a dataset, splits train/test, clusters with K-Means (K = 10 × classes), trains an SVM and shows the visual words. |
| 7.2 | Histogram of visual words | Shows the word histogram of one random test image of each class. |
| 7.3 | Classify image | Builds the histogram of the open image and classifies it with the 7.1 SVM. |

## Models and data

Datasets and trained models are not committed (see `.gitignore`).

| Used by | File or folder | Where it comes from |
| --- | --- | --- |
| 6.1 | `models/face_detection_yunet_2023mar.onnx` | Committed (pretrained, from OpenCV Zoo). Downloaded again if missing. |
| 6.2 | `images/set6-images/` | Created by 6.2: LFW download (`lfw_home/`) and the train/test face PNGs. |
| 6.2, 6.3 | `models/eigenfaces_pca.npz` | Created by 6.2 (PCA model); 6.3 loads it. |
| 7.1–7.3 | `images/set7-images/<class>/` | The dataset: one subfolder per class (we used butterfly, dalmatian and kangaroo). 7.1 also writes `train/` and `test/` here. |
| 7.2, 7.3 | `models/bow_model.npz`, `models/bow_classifier.pkl` | Created by 7.1. |

## Project layout

```
run.py                      start the app
forensics_app/
  app.py, ui/, core/        app shell from the starter
  tools/
    __init__.py             registers every tool in the sidebar
    base.py, registry.py    tool contract
    common.py               helpers shared by all sets
    set1/ … set7/           one folder per lab set
models/                     pretrained / trained models
docs/STUDENT_GUIDE.md       starter handout
```

## Useful commands

```
python -m forensics_app
python -m compileall forensics_app
```
