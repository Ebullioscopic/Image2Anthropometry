# Image2Anthropometry

This project provides a REST API for performing anthropometric analysis using image inputs. The API leverages OpenCV and MediaPipe to detect pose landmarks and calculate various body measurements based on a given height.

## Table of Contents
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Running the Server](#running-the-server)
- [API Usage](#api-usage)
  - [Analyze Pose](#analyze-pose)
- [References](#references)
- [Information and Instructions](#information-and-instructions)

## System Requirements
- Python 3.8+
- pip (Python package installer)
- virtualenv (optional but recommended)

## Installation

1. **Clone the repository:**
    ```sh
    git clone https://github.com/Ebullioscopic/Image2Anthropometry.git
    cd Image2Anthropometry
    ```

2. **Create and activate a virtual environment (optional but recommended):**
    ```sh
    python -m venv venv
    source venv/bin/activate    # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

    **Note:** If you don't have a `requirements.txt` file, create one with the following content:
    ```txt
    django
    djangorestframework
    opencv-python
    mediapipe
    numpy
    ```

4. **Create and configure the Django project:**
    ```sh
    django-admin startproject pose_analysis_project
    cd pose_analysis_project
    django-admin startapp api
    ```

5. **Update `settings.py` in `pose_analysis_project` directory:**
    ```python
    INSTALLED_APPS = [
        ...
        'rest_framework',
        'api',
    ]
    ```

6. **Add the `analyze_pose` view to the `api` app:**
    Create a file `views.py` in the `api` directory and add the following code:

    ```python
    import cv2
    import mediapipe as mp
    import numpy as np
    from rest_framework.response import Response
    from rest_framework.decorators import api_view
    from rest_framework import status

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

    @api_view(['POST'])
    def analyze_pose(request):
        if 'image' not in request.FILES:
            return Response({"error": "No image file provided"}, status=status.HTTP_400_BAD_REQUEST)
        
        if 'height' not in request.data:
            return Response({"error": "No height provided"}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            height_in_inches = float(request.data['height'])
        except ValueError:
            return Response({"error": "Invalid height provided"}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            image_file = request.FILES['image']
            image_array = np.fromstring(image_file.read(), np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

        results = pose.process(image_rgb)

        if not results.pose_landmarks:
            return Response({"error": "No pose landmarks detected"}, status=status.HTTP_400_BAD_REQUEST)

        def get_landmark(idx):
            return (int(results.pose_landmarks.landmark[idx].x * image.shape[1]),
                    int(results.pose_landmarks.landmark[idx].y * image.shape[0]))

        landmarks = results.pose_landmarks.landmark

        shoulder_left = get_landmark(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
        shoulder_right = get_landmark(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
        hip_left = get_landmark(mp_pose.PoseLandmark.LEFT_HIP.value)
        hip_right = get_landmark(mp_pose.PoseLandmark.RIGHT_HIP.value)
        wrist_left = get_landmark(mp_pose.PoseLandmark.LEFT_WRIST.value)
        wrist_right = get_landmark(mp_pose.PoseLandmark.RIGHT_WRIST.value)
        elbow_left = get_landmark(mp_pose.PoseLandmark.LEFT_ELBOW.value)
        elbow_right = get_landmark(mp_pose.PoseLandmark.RIGHT_ELBOW.value)
        knee_left = get_landmark(mp_pose.PoseLandmark.LEFT_KNEE.value)
        knee_right = get_landmark(mp_pose.PoseLandmark.RIGHT_KNEE.value)
        ankle_left = get_landmark(mp_pose.PoseLandmark.LEFT_ANKLE.value)
        ankle_right = get_landmark(mp_pose.PoseLandmark.RIGHT_ANKLE.value)

        def calculate_distance(point1, point2):
            return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

        head = get_landmark(mp_pose.PoseLandmark.NOSE.value)
        feet_left = get_landmark(mp_pose.PoseLandmark.LEFT_ANKLE.value)
        feet_right = get_landmark(mp_pose.PoseLandmark.RIGHT_ANKLE.value)
        height_in_pixels = (calculate_distance(head, feet_left) + calculate_distance(head, feet_right)) / 2

        scale_factor = height_in_inches / height_in_pixels

        shoulder_width = calculate_distance(shoulder_left, shoulder_right)
        hip_width = calculate_distance(hip_left, hip_right)
        arm_length_left = calculate_distance(shoulder_left, elbow_left)
        arm_length_right = calculate_distance(shoulder_right, elbow_right)
        thigh_length_left = calculate_distance(hip_left, knee_left)
        thigh_length_right = calculate_distance(hip_right, knee_right)
        ankle_length_left = calculate_distance(knee_left, ankle_left)
        ankle_length_right = calculate_distance(knee_right, ankle_right)

        measurements = {
            "shoulder_width_in": shoulder_width * scale_factor,
            "hip_width_in": hip_width * scale_factor,
            "arm_length_left_in": arm_length_left * scale_factor,
            "arm_length_right_in": arm_length_right * scale_factor,
            "thigh_length_left_in": thigh_length_left * scale_factor,
            "thigh_length_right_in": thigh_length_right * scale_factor,
            "ankle_length_left_in": ankle_length_left * scale_factor,
            "ankle_length_right_in": ankle_length_right * scale_factor
        }

        return Response(measurements)
    ```

7. **Create URL routing for the API:**
    Add the following to `urls.py` in the `pose_analysis_project` directory:

    ```python
    from django.urls import path
    from api.views import analyze_pose

    urlpatterns = [
        path('api/analyze/', analyze_pose, name='analyze_pose'),
    ]
    ```

## Running the Server

1. **Apply migrations:**
    ```sh
    python manage.py migrate
    ```

2. **Run the development server:**
    ```sh
    python manage.py runserver
    ```

    The server will start at `http://127.0.0.1:8000/`.

## API Usage

### Analyze Pose

**Endpoint:** `POST /api/analyze/`

**Description:** Accepts an image and height input, returns anthropometric measurements.

**Parameters:**
- `image` (file): The image file containing the person's pose.
- `height` (float): The height of the person in inches.

**Example Request using Postman:**

1. Open Postman.
2. Create a new `POST` request.
3. Set the URL to `http://127.0.0.1:8000/api/analyze/`.
4. Under the `Body` tab, select `form-data`.
5. Add a key `image` with type `File` and upload an image.
6. Add a key `height` with type `Text` and set the value to the height (e.g., `70`).
7. Send the request.

**Example Response:**
```json
{
    "shoulder_width_in": 18.5,
    "hip_width_in": 16.2,
    "arm_length_left_in": 24.7,
    "arm_length_right_in": 24.9,
    "thigh_length_left_in": 18.0,
    "thigh_length_right_in": 18.1,
    "ankle_length_left_in": 12.4,
    "ankle_length_right_in": 12.3
}
```

## References

- [Django Documentation](https://docs.djangoproject.com/)
- [Django REST Framework Documentation](https://www.django-rest-framework.org/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [MediaPipe Documentation](https://google.github.io/mediapipe/)
- [NumPy Documentation](https://numpy.org/doc/)

## Information and Instructions

### General Information
This project provides a REST API for analyzing human pose and calculating anthropometric measurements from an image. The API utilizes OpenCV and MediaPipe to detect pose landmarks and perform calculations based on the given height of the person.

## Contributors

- **[Hariharan Mudaliar](https://github.com/Ebullioscopic)** - Initial work and implementation.

Feel free to contribute to this project by submitting issues or pull requests on [GitHub](https://github.com/Ebullioscopic/Image2Anthropometry).
