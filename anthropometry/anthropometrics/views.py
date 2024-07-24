import cv2
import mediapipe as mp
import numpy as np
from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import status

# Initialize MediaPipe
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
