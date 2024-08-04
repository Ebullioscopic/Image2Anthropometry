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
        image_array = np.frombuffer(image_file.read(), np.uint8)
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
