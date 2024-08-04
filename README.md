# Image2Anthropometry

This project provides a REST API for performing anthropometric analysis using image inputs. The API leverages OpenCV and MediaPipe to detect pose landmarks and calculate various body measurements based on a given height.

## Inferences
Input Image                 |  Annotated Image
:-------------------------:|:-------------------------:
![](https://github.com/Ebullioscopic/Image2Anthropometry/blob/main/person.jpeg)  |  ![](https://github.com/Ebullioscopic/Image2Anthropometry/blob/main/annotated_image.jpg)

## API Call

![](https://github.com/Ebullioscopic/Image2Anthropometry/blob/main/api_call.png)

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
    django-admin startproject anthropometry
    cd anthropometry
    django-admin startapp api
    ```

5. **Update `settings.py` in `anthropometry` directory:**
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
            image_array = np.frombuffer(image_file.read(), np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

        results = pose.process(image_rgb)

        if not results.pose_landmarks:
            return Response({"error": "No pose landmarks detected"}, status=status.HTTP (400_BAD_REQUEST)

        def get_landmark(idx):
            return (int(results.pose_landmarks.landmark[idx].x * image.shape[1]),
                    int(results.pose_landmarks.landmark[idx].y * image.shape[0]))

        def calculate_distance(point1, point2):
            return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

        def calculate_circumference(width_in_pixels, scale_factor):
            # Assuming an elliptical cross-section for the body part
            depth_in_pixels = width_in_pixels * 0.7  # approximate depth as 70% of width
            circumference_in_pixels = np.pi * np.sqrt(0.5 * (width_in_pixels**2 + depth_in_pixels**2))
            return circumference_in_pixels * scale_factor

        # Extract landmarks
        shoulder_left = get_landmark(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
        shoulder_right = get_landmark(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
        hip_left = get_landmark(mp_pose.PoseLandmark.LEFT_HIP.value)
        hip_right = get_landmark(mp_pose.PoseLandmark.RIGHT_HIP.value)
        elbow_left = get_landmark(mp_pose.PoseLandmark.LEFT_ELBOW.value)
        elbow_right = get_landmark(mp_pose.PoseLandmark.RIGHT_ELBOW.value)
        wrist_left = get_landmark(mp_pose.PoseLandmark.LEFT_WRIST.value)
        wrist_right = get_landmark(mp_pose.PoseLandmark.RIGHT_WRIST.value)
        knee_left = get_landmark(mp_pose.PoseLandmark.LEFT_KNEE.value)
        knee_right = get_landmark(mp_pose.PoseLandmark.RIGHT_KNEE.value)
        ankle_left = get_landmark(mp_pose.PoseLandmark.LEFT_ANKLE.value)
        ankle_right = get_landmark(mp_pose.PoseLandmark.RIGHT_ANKLE.value)

        head = get_landmark(mp_pose.PoseLandmark.NOSE.value)
        feet_left = get_landmark(mp_pose.PoseLandmark.LEFT_ANKLE.value)
        feet_right = get_landmark(mp_pose.PoseLandmark.RIGHT_ANKLE.value)
        height_in_pixels = (calculate_distance(head, feet_left) + calculate_distance(head, feet_right)) / 2

        scale_factor = height_in_inches / height_in_pixels

        # Calculate widths and lengths
        shoulder_width = calculate_distance(shoulder_left, shoulder_right)
        chest_width = shoulder_width  # Approximate chest width with shoulder width
        waist_width = calculate_distance(hip_left, hip_right)
        hip_width = waist_width  # Approximate hip width with waist width

        biceps_left_length = calculate_distance(shoulder_left, elbow_left)
        biceps_right_length = calculate_distance(shoulder_right, elbow_right)
        forearm_left_length = calculate_distance(elbow_left, wrist_left)
        forearm_right_length = calculate_distance(elbow_right, wrist_right)
        thigh_left_length = calculate_distance(hip_left, knee_left)
        thigh_right_length = calculate_distance(hip_right, knee_right)
        calf_left_length = calculate_distance(knee_left, ankle_left)
        calf_right_length = calculate_distance(knee_right, ankle_right)

        # Convert measurements to inches
        shoulder_width_in = shoulder_width * scale_factor
        chest_width_in = chest_width * scale_factor
        waist_width_in = waist_width * scale_factor
        hip_width_in = hip_width * scale_factor
        biceps_left_length_in = biceps_left_length * scale_factor
        biceps_right_length_in = biceps_right_length * scale_factor
        forearm_left_length_in = forearm_left_length * scale_factor
        forearm_right_length_in = forearm_right_length * scale_factor
        thigh_left_length_in = thigh_left_length * scale_factor
        thigh_right_length_in = thigh_right_length * scale_factor
        calf_left_length_in = calf_left_length * scale_factor
        calf_right_length_in = calf_right_length * scale_factor

        # Calculate circumferences
        chest_circumference_in = calculate_circumference(chest_width, scale_factor)
        waist_circumference_in = calculate_circumference(waist_width, scale_factor)
        hip_circumference_in = calculate_circumference(hip_width, scale_factor)

        biceps_left_circumference_in = calculate_circumference(biceps_left_length, scale_factor)
        biceps_right_circumference_in = calculate_circumference(biceps_right_length, scale_factor)
        forearm_left_circumference_in = calculate_circumference(forearm_left_length, scale_factor)
        forearm_right_circumference_in = calculate_circumference(forearm_right_length, scale_factor)
        thigh_left_circumference_in = calculate_circumference(thigh_left_length, scale_factor)
        thigh_right_circumference_in = calculate_circumference(thigh_right_length, scale_factor)
        calf_left_circumference_in = calculate_circumference(calf_left_length, scale_factor)
        calf_right_circumference_in = calculate_circumference(calf_right_length, scale_factor)

        measurements = {
            "shoulder_width_in": shoulder_width_in,
            "chest_circumference_in": chest_circumference_in,
            "waist_width_in": waist_width_in,
            "waist_circumference_in": waist_circumference_in,
            "hip_width_in": hip_width_in,
            "hip_circumference_in": hip_circumference_in,
            "biceps_left_length_in": biceps_left_length_in,
            "biceps_right_length_in": biceps_right_length_in,
            "biceps_left_circumference_in": biceps_left_circumference_in,
            "biceps_right_circumference_in": biceps_right_circumference_in,
            "forearm_left_length_in": forearm_left_length_in,
            "forearm_right_length_in": forearm_right_length_in,
            "forearm_left_circumference_in": forearm_left_circumference_in,
            "forearm_right_circumference_in": forearm_right_circumference_in,
            "thigh_left_length_in": thigh_left_length_in,
            "thigh_right_length_in": thigh_right_length_in,
            "thigh_left_circumference_in": thigh_left_circumference_in,
            "thigh_right_circumference_in": thigh_right_circumference_in,
            "calf_left_length_in": calf_left_length_in,
            "calf_right_length_in": calf_right_length_in,
            "calf_left_circumference_in": calf_left_circumference_in,
            "calf_right_circumference_in": calf_right_circumference_in
        }

        return Response(measurements)
    ```

7. **Create URL routing for the API:**
    Add the following to `urls.py` in the `anthropometry` directory:

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

**Description:** Accepts an image and height input, returns anthropometric measurements including both distances and circumferences for various body parts.

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
    "chest_circumference_in": 38.4,
    "waist_width_in": 16.2,
    "waist_circumference_in": 34.1,
    "hip_width_in": 16.5,
    "hip_circumference_in": 35.2,
    "biceps_left_length_in": 24.7,
    "biceps_right_length_in": 24.9,
    "biceps_left_circumference_in": 12.5,
    "biceps_right_circumference_in": 12.7,
    "forearm_left_length_in": 18.0,
    "forearm_right_length_in": 18.1,
    "forearm_left_circumference_in": 9.5,
    "forearm_right_circumference_in": 9.6,
    "thigh_left_length_in": 18.0,
    "thigh_right_length_in": 18.1,
    "thigh_left_circumference_in": 21.0,
    "thigh_right_circumference_in": 21.2,
    "calf_left_length_in": 12.4,
    "calf_right_length_in": 12.3,
    "calf_left_circumference_in": 14.5,
    "calf_right_circumference_in": 14.3
}

## References

- [Django Documentation](https://docs.djangoproject.com/)
- [Django REST Framework Documentation](https://www.django-rest-framework.org/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [MediaPipe Documentation](https://google.github.io/mediapipe/)
- [NumPy Documentation](https://numpy.org/doc/)

## Information and Instructions

### General Information
This project provides a REST API for analyzing human pose and calculating anthropometric measurements from an image. The API utilizes OpenCV and MediaPipe to detect pose landmarks and perform calculations based on the given height of the person.

### Troubleshooting and Tips
- **Accuracy of Measurements**: The accuracy of the measurements can be influenced by the quality and angle of the input image. Ensure that the image is taken from a front or side angle for the best results.
- **Error Handling**: The API includes error handling for missing or invalid input data. Ensure that you provide both the image and height when making a request.
- **Performance Considerations**: For large-scale deployments or high traffic, consider optimizing the MediaPipe and OpenCV processes or deploying the API on a server with adequate computational resources.

### Contributing
We welcome contributions to improve this project! Here’s how you can contribute:
1. **Fork the repository** on GitHub.
2. **Clone your forked repository** locally:
    ```sh
    git clone https://github.com/your-username/Image2Anthropometry.git
    ```
3. **Create a new branch** for your feature or bug fix:
    ```sh
    git checkout -b feature-branch
    ```
4. **Make your changes**, commit, and push:
    ```sh
    git commit -m "Add new feature"
    git push origin feature-branch
    ```
5. **Submit a pull request** to the main repository.

For detailed guidelines, please refer to our [CONTRIBUTING.md](https://github.com/Ebullioscopic/Image2Anthropometry/blob/main/CONTRIBUTING.md) file.

### License
This project is licensed under the MIT License. See the [LICENSE](https://github.com/Ebullioscopic/Image2Anthropometry/blob/main/LICENSE) file for more details.

## Contributors

- **[Hariharan Mudaliar](https://github.com/Ebullioscopic)** - Initial work and implementation.

Feel free to contribute to this project by submitting issues or pull requests on [GitHub](https://github.com/Ebullioscopic/Image2Anthropometry).

## Contact

For any inquiries, suggestions, or feedback, please reach out:
- Email: [hrhn.mudaliar@gmail.com](hrhn.mudaliar@gmail.com)
- GitHub Issues: [Issues Page](https://github.com/Ebullioscopic/Image2Anthropometry/issues)

We look forward to your contributions and feedback!
