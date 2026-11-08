
"""Set 6.1 - Face detection using YuNet.

Supports:
1. Face detection on an uploaded image
2. Live face detection using a webcam
"""

import tkinter as tk
import urllib.request
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from ..base import ForensicsTool, ToolResult
from ..common import MODELS_FOLDER


# --------------------------------------------------
# 1. Load the pretrained YuNet model
# --------------------------------------------------

# Kept in the project's models/ folder (ignored by git), not inside the code folder.
# It is downloaded the first time it is needed.
MODEL_PATH = MODELS_FOLDER / "face_detection_yunet_2023mar.onnx"
MODEL_URL = (
    "https://huggingface.co/opencv/opencv_zoo/resolve/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)

CONFIDENCE_THRESHOLD = 0.6
NMS_THRESHOLD = 0.3


# --------------------------------------------------
# 2. Create the YuNet detector
# --------------------------------------------------

def get_model():
    if not MODEL_PATH.is_file():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))


def create_detector(width, height):
    get_model()

    detector = cv2.FaceDetectorYN.create(
        str(MODEL_PATH),
        "",
        (width, height),
        CONFIDENCE_THRESHOLD,
        NMS_THRESHOLD,
        5000,
    )

    return detector


# --------------------------------------------------
# 3. Detect faces and draw results
# --------------------------------------------------

def detect_faces(image, detector):

    height, width = image.shape[:2]

    # Webcam frames can have different dimensions
    detector.setInputSize((width, height))

    _, faces = detector.detect(image)

    result = image.copy()
    count = 0

    if faces is not None:
        count = len(faces)

        for face in faces:

            # Bounding box
            x, y, w, h = face[:4].astype(int)

            confidence = float(face[14])

            cv2.rectangle(
                result,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2,
            )

            # Display confidence score
            cv2.putText(
                result,
                f"Face: {confidence:.2f}",
                (x, max(20, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            # Five facial landmarks
            landmarks = face[4:14].reshape(5, 2).astype(int)

            for px, py in landmarks:
                cv2.circle(
                    result,
                    (px, py),
                    4,
                    (255, 0, 0),
                    -1,
                )

    return result, count


# --------------------------------------------------
# 4. Live webcam face detection
# --------------------------------------------------

def run_webcam():

    get_model()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            "Could not open webcam. Check camera permissions."
        )

    detector = create_detector(320, 320)

    try:
        while True:

            ret, frame = cap.read()

            if not ret:
                break

            # Detect faces in the current frame
            result, count = detect_faces(frame, detector)

            cv2.putText(
                result,
                f"Faces detected: {count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            cv2.imshow(
                "Set 6.1 - YuNet Webcam Detection",
                result,
            )

            # Press Q to close webcam
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


# --------------------------------------------------
# 5. Forensics App integration
# --------------------------------------------------

class FaceDetectionTool(ForensicsTool):

    tool_id = "face_detection"
    title = "Face Detection (YuNet)"
    category = "Set 6 · Faces"

    description = (
        "Detect faces using YuNet on an image or webcam."
    )

    requires_image = False

    def run(
        self,
        parent: tk.Misc,
        document: ImageDocument,
    ) -> ToolResult | None:

        # Ask user to choose detection mode
        mode = self._ask_mode(parent)

        if mode is None:
            return None

        # ------------------------------------------
        # Webcam mode
        # ------------------------------------------

        if mode == "webcam":

            try:
                run_webcam()

            except (RuntimeError, FileNotFoundError, cv2.error) as exc:
                messagebox.showerror(
                    "Webcam Error",
                    str(exc),
                    parent=parent,
                )
                return None

            return ToolResult(
                message="YuNet webcam detection finished.",
                details={
                    "Algorithm": "YuNet",
                    "Mode": "Webcam",
                    "Confidence threshold": CONFIDENCE_THRESHOLD,
                    "NMS threshold": NMS_THRESHOLD,
                },
            )

        # ------------------------------------------
        # Image mode
        # ------------------------------------------

        if document.current is None:
            messagebox.showwarning(
                "No image",
                "Please load an image first.",
                parent=parent,
            )
            return None

        # Convert PIL RGB to OpenCV BGR
        image = np.asarray(
            document.current.convert("RGB")
        )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR,
        )

        height, width = image.shape[:2]

        try:
            detector = create_detector(width, height)
            result, count = detect_faces(image, detector)

        except (RuntimeError, FileNotFoundError, cv2.error) as exc:
            return ToolResult(
                message=f"YuNet detection failed: {exc}"
            )

        # Convert OpenCV BGR back to PIL RGB
        result = cv2.cvtColor(
            result,
            cv2.COLOR_BGR2RGB,
        )

        output = Image.fromarray(result)

        return ToolResult(
            image=output,
            message=f"YuNet detected {count} face(s).",
            details={
                "Algorithm": "YuNet",
                "Mode": "Image",
                "Faces detected": count,
                "Confidence threshold": CONFIDENCE_THRESHOLD,
                "NMS threshold": NMS_THRESHOLD,
            },
        )

    # --------------------------------------------------
    # 6. Window for choosing image or webcam
    # --------------------------------------------------

    def _ask_mode(self, parent):

        window = tk.Toplevel(parent)
        window.title("Set 6.1 - Face Detection")
        window.transient(parent)
        window.resizable(False, False)

        frame = ttk.Frame(window, padding=20)
        frame.pack(fill="both", expand=True)

        ttk.Label(
            frame,
            text="Choose face detection mode:",
        ).pack(pady=(0, 12))

        mode = tk.StringVar(value="image")

        ttk.Radiobutton(
            frame,
            text="Detect faces in loaded image",
            variable=mode,
            value="image",
        ).pack(anchor="w", pady=5)

        ttk.Radiobutton(
            frame,
            text="Live webcam face detection",
            variable=mode,
            value="webcam",
        ).pack(anchor="w", pady=5)

        selected = None

        def apply():
            nonlocal selected
            selected = mode.get()
            window.destroy()

        buttons = ttk.Frame(frame)
        buttons.pack(pady=(15, 0))

        ttk.Button(
            buttons,
            text="Run",
            command=apply,
        ).pack(side="left", padx=5)

        ttk.Button(
            buttons,
            text="Cancel",
            command=window.destroy,
        ).pack(side="left", padx=5)

        window.bind("<Return>", lambda event: apply())
        window.bind("<Escape>", lambda event: window.destroy())

        window.grab_set()
        window.wait_window()

        return selected
