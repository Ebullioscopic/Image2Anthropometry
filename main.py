import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Load the image
image_path = r'D:\Sixth_Sense\Anthropometry_API\person.jpeg'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process the image and extract pose landmarks
results = pose.process(image_rgb)

# Known height of the person in inches
height_in_inches = 70  # Example height, adjust as necessary

def get_landmark(idx):
    return (int(landmarks[idx].x * image.shape[1]), int(landmarks[idx].y * image.shape[0]))

if results.pose_landmarks:
    # Extract landmarks
    landmarks = results.pose_landmarks.landmark
    
    # Example landmark indices
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
    
    # Calculate distances between landmarks
    def calculate_distance(point1, point2):
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    # Example height measurement from head to feet
    head = get_landmark(mp_pose.PoseLandmark.NOSE.value)
    feet_left = get_landmark(mp_pose.PoseLandmark.LEFT_ANKLE.value)
    feet_right = get_landmark(mp_pose.PoseLandmark.RIGHT_ANKLE.value)
    height_in_pixels = (calculate_distance(head, feet_left) + calculate_distance(head, feet_right)) / 2

    # Calculate scale factor
    scale_factor = height_in_inches / height_in_pixels

    # Calculate other measurements in pixels
    shoulder_width = calculate_distance(shoulder_left, shoulder_right)
    hip_width = calculate_distance(hip_left, hip_right)
    arm_length_left = calculate_distance(shoulder_left, elbow_left)
    arm_length_right = calculate_distance(shoulder_right, elbow_right)
    thigh_length_left = calculate_distance(hip_left, knee_left)
    thigh_length_right = calculate_distance(hip_right, knee_right)
    ankle_length_left = calculate_distance(knee_left, ankle_left)
    ankle_length_right = calculate_distance(knee_right, ankle_right)

    # Convert measurements to inches
    shoulder_width_in = shoulder_width * scale_factor
    hip_width_in = hip_width * scale_factor
    arm_length_left_in = arm_length_left * scale_factor
    arm_length_right_in = arm_length_right * scale_factor
    thigh_length_left_in = thigh_length_left * scale_factor
    thigh_length_right_in = thigh_length_right * scale_factor
    ankle_length_left_in = ankle_length_left * scale_factor
    ankle_length_right_in = ankle_length_right * scale_factor

    # Draw landmarks and connections
    annotated_image = image.copy()
    mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Annotate image with measurements
    def draw_measurement(start_point, end_point, measurement, label):
        cv2.line(annotated_image, start_point, end_point, (0, 255, 0), 2)
        mid_point = ((start_point[0] + end_point[0]) // 2, (start_point[1] + end_point[1]) // 2)
        cv2.putText(annotated_image, f"{label}: {measurement:.2f} in", mid_point, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    draw_measurement(shoulder_left, shoulder_right, shoulder_width_in, "Shoulder Width")
    draw_measurement(hip_left, hip_right, hip_width_in, "Hip Width")
    draw_measurement(shoulder_left, elbow_left, arm_length_left_in, "Left Arm Length")
    draw_measurement(shoulder_right, elbow_right, arm_length_right_in, "Right Arm Length")
    draw_measurement(hip_left, knee_left, thigh_length_left_in, "Left Thigh Length")
    draw_measurement(hip_right, knee_right, thigh_length_right_in, "Right Thigh Length")
    draw_measurement(knee_left, ankle_left, ankle_length_left_in, "Left Ankle Length")
    draw_measurement(knee_right, ankle_right, ankle_length_right_in, "Right Ankle Length")

    # Save the annotated image
    output_path = 'annotated_image.jpg'
    cv2.imwrite(output_path, annotated_image)
    print(f"Annotated image saved as {output_path}")

    # Optionally, display the annotated image
    cv2.imshow('Annotated Image', annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No pose landmarks detected.")
