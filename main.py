# import cv2
# import mediapipe as mp
# import numpy as np

# # Initialize MediaPipe
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
# mp_drawing = mp.solutions.drawing_utils

# # Load the image
# image_path = r'D:\Sixth_Sense\Anthropometry_API\sir_body.jpeg'
# image = cv2.imread(image_path)
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# # Process the image and extract pose landmarks
# results = pose.process(image_rgb)

# # Known height of the person in inches
# height_in_inches = 67  # Example height, adjust as necessary

# def get_landmark(idx):
#     return (int(landmarks[idx].x * image.shape[1]), int(landmarks[idx].y * image.shape[0]))

# if results.pose_landmarks:
#     # Extract landmarks
#     landmarks = results.pose_landmarks.landmark
    
#     # Example landmark indices
#     shoulder_left = get_landmark(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
#     shoulder_right = get_landmark(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
#     hip_left = get_landmark(mp_pose.PoseLandmark.LEFT_HIP.value)
#     hip_right = get_landmark(mp_pose.PoseLandmark.RIGHT_HIP.value)
#     wrist_left = get_landmark(mp_pose.PoseLandmark.LEFT_WRIST.value)
#     wrist_right = get_landmark(mp_pose.PoseLandmark.RIGHT_WRIST.value)
#     elbow_left = get_landmark(mp_pose.PoseLandmark.LEFT_ELBOW.value)
#     elbow_right = get_landmark(mp_pose.PoseLandmark.RIGHT_ELBOW.value)
#     knee_left = get_landmark(mp_pose.PoseLandmark.LEFT_KNEE.value)
#     knee_right = get_landmark(mp_pose.PoseLandmark.RIGHT_KNEE.value)
#     ankle_left = get_landmark(mp_pose.PoseLandmark.LEFT_ANKLE.value)
#     ankle_right = get_landmark(mp_pose.PoseLandmark.RIGHT_ANKLE.value)
    
#     # Calculate distances between landmarks
#     def calculate_distance(point1, point2):
#         return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

#     # Example height measurement from head to feet
#     head = get_landmark(mp_pose.PoseLandmark.NOSE.value)
#     feet_left = get_landmark(mp_pose.PoseLandmark.LEFT_ANKLE.value)
#     feet_right = get_landmark(mp_pose.PoseLandmark.RIGHT_ANKLE.value)
#     height_in_pixels = (calculate_distance(head, feet_left) + calculate_distance(head, feet_right)) / 2

#     # Calculate scale factor
#     scale_factor = height_in_inches / height_in_pixels

#     # Calculate other measurements in pixels
#     shoulder_width = calculate_distance(shoulder_left, shoulder_right)
#     hip_width = calculate_distance(hip_left, hip_right)
#     arm_length_left = calculate_distance(shoulder_left, elbow_left)
#     arm_length_right = calculate_distance(shoulder_right, elbow_right)
#     thigh_length_left = calculate_distance(hip_left, knee_left)
#     thigh_length_right = calculate_distance(hip_right, knee_right)
#     ankle_length_left = calculate_distance(knee_left, ankle_left)
#     ankle_length_right = calculate_distance(knee_right, ankle_right)

#     # Convert measurements to inches
#     shoulder_width_in = shoulder_width * scale_factor
#     hip_width_in = hip_width * scale_factor
#     arm_length_left_in = arm_length_left * scale_factor
#     arm_length_right_in = arm_length_right * scale_factor
#     thigh_length_left_in = thigh_length_left * scale_factor
#     thigh_length_right_in = thigh_length_right * scale_factor
#     ankle_length_left_in = ankle_length_left * scale_factor
#     ankle_length_right_in = ankle_length_right * scale_factor

#     # Draw landmarks and connections
#     annotated_image = image.copy()
#     mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

#     # Annotate image with measurements
#     def draw_measurement(start_point, end_point, measurement, label):
#         cv2.line(annotated_image, start_point, end_point, (0, 255, 0), 2)
#         mid_point = ((start_point[0] + end_point[0]) // 2, (start_point[1] + end_point[1]) // 2)
#         cv2.putText(annotated_image, f"{label}: {measurement:.2f} in", mid_point, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#     print(f"Shoulder Width: {shoulder_width_in:.2f} inches")
#     draw_measurement(shoulder_left, shoulder_right, shoulder_width_in, "Shoulder Width")
#     print(f"Hip Width: {hip_width_in:.2f} inches")
#     draw_measurement(hip_left, hip_right, hip_width_in, "Hip Width")
#     print(f"Left Arm Length: {arm_length_left_in:.2f} inches")
#     draw_measurement(shoulder_left, elbow_left, arm_length_left_in, "Left Arm Length")
#     print(f"Right Arm Length: {arm_length_right_in:.2f} inches")
#     draw_measurement(shoulder_right, elbow_right, arm_length_right_in, "Right Arm Length")
#     print(f"Left Thigh Length: {thigh_length_left_in:.2f} inches")
#     draw_measurement(hip_left, knee_left, thigh_length_left_in, "Left Thigh Length")
#     print(f"Right Thigh Length: {thigh_length_right_in:.2f} inches")
#     draw_measurement(hip_right, knee_right, thigh_length_right_in, "Right Thigh Length")
#     print(f"Left Ankle Length: {ankle_length_left_in:.2f} inches")
#     draw_measurement(knee_left, ankle_left, ankle_length_left_in, "Left Ankle Length")
#     print(f"Right Ankle Length: {ankle_length_right_in:.2f} inches")
#     draw_measurement(knee_right, ankle_right, ankle_length_right_in, "Right Ankle Length")

#     # Save the annotated image
#     output_path = 'annotated_image.jpg'
#     cv2.imwrite(output_path, annotated_image)
#     print(f"Annotated image saved as {output_path}")

#     # Optionally, display the annotated image
#     cv2.imshow('Annotated Image', annotated_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# else:
#     print("No pose landmarks detected.")
##################################################################################################################################################################
# import cv2
# import mediapipe as mp
# import numpy as np

# # Initialize MediaPipe
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
# mp_drawing = mp.solutions.drawing_utils

# # Load the image
# image_path = r'D:\Sixth_Sense\Anthropometry_API\sir_body.jpeg'
# image = cv2.imread(image_path)
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# # Process the image and extract pose landmarks
# results = pose.process(image_rgb)

# # Known height of the person in inches
# height_in_inches = 67  # Example height, adjust as necessary

# def get_landmark(idx):
#     return (int(landmarks[idx].x * image.shape[1]), int(landmarks[idx].y * image.shape[0]))

# if results.pose_landmarks:
#     # Extract landmarks
#     landmarks = results.pose_landmarks.landmark
    
#     # Example landmark indices
#     shoulder_left = get_landmark(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
#     shoulder_right = get_landmark(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
#     hip_left = get_landmark(mp_pose.PoseLandmark.LEFT_HIP.value)
#     hip_right = get_landmark(mp_pose.PoseLandmark.RIGHT_HIP.value)
#     elbow_left = get_landmark(mp_pose.PoseLandmark.LEFT_ELBOW.value)
#     elbow_right = get_landmark(mp_pose.PoseLandmark.RIGHT_ELBOW.value)
#     wrist_left = get_landmark(mp_pose.PoseLandmark.LEFT_WRIST.value)
#     wrist_right = get_landmark(mp_pose.PoseLandmark.RIGHT_WRIST.value)
#     knee_left = get_landmark(mp_pose.PoseLandmark.LEFT_KNEE.value)
#     knee_right = get_landmark(mp_pose.PoseLandmark.RIGHT_KNEE.value)
#     ankle_left = get_landmark(mp_pose.PoseLandmark.LEFT_ANKLE.value)
#     ankle_right = get_landmark(mp_pose.PoseLandmark.RIGHT_ANKLE.value)
    
#     # Calculate distances between landmarks
#     def calculate_distance(point1, point2):
#         return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

#     # Example height measurement from head to feet
#     head = get_landmark(mp_pose.PoseLandmark.NOSE.value)
#     feet_left = get_landmark(mp_pose.PoseLandmark.LEFT_ANKLE.value)
#     feet_right = get_landmark(mp_pose.PoseLandmark.RIGHT_ANKLE.value)
#     height_in_pixels = (calculate_distance(head, feet_left) + calculate_distance(head, feet_right)) / 2

#     # Calculate scale factor
#     scale_factor = height_in_inches / height_in_pixels

#     # Measurements
#     shoulder_width = calculate_distance(shoulder_left, shoulder_right)
#     chest_width = shoulder_width  # Assuming chest width is close to shoulder width
#     waist_width = calculate_distance(hip_left, hip_right)
#     hip_width = waist_width  # Assuming hip width is close to waist width

#     biceps_left = calculate_distance(shoulder_left, elbow_left)
#     biceps_right = calculate_distance(shoulder_right, elbow_right)
#     forearm_left = calculate_distance(elbow_left, wrist_left)
#     forearm_right = calculate_distance(elbow_right, wrist_right)
#     thigh_left = calculate_distance(hip_left, knee_left)
#     thigh_right = calculate_distance(hip_right, knee_right)
#     calf_left = calculate_distance(knee_left, ankle_left)
#     calf_right = calculate_distance(knee_right, ankle_right)

#     # Convert measurements to inches
#     shoulder_width_in = shoulder_width * scale_factor
#     chest_width_in = chest_width * scale_factor
#     waist_width_in = waist_width * scale_factor
#     hip_width_in = hip_width * scale_factor
#     biceps_left_in = biceps_left * scale_factor
#     biceps_right_in = biceps_right * scale_factor
#     forearm_left_in = forearm_left * scale_factor
#     forearm_right_in = forearm_right * scale_factor
#     thigh_left_in = thigh_left * scale_factor
#     thigh_right_in = thigh_right * scale_factor
#     calf_left_in = calf_left * scale_factor
#     calf_right_in = calf_right * scale_factor

#     # Draw landmarks and connections
#     annotated_image = image.copy()
#     mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

#     # Annotate image with measurements
#     def draw_measurement(start_point, end_point, measurement, label):
#         cv2.line(annotated_image, start_point, end_point, (0, 255, 0), 2)
#         mid_point = ((start_point[0] + end_point[0]) // 2, (start_point[1] + end_point[1]) // 2)
#         cv2.putText(annotated_image, f"{label}: {measurement:.2f} in", mid_point, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#     measurements = [
#         (shoulder_left, shoulder_right, shoulder_width_in, "Shoulder Width"),
#         (shoulder_left, shoulder_right, chest_width_in, "Chest Width"),
#         (hip_left, hip_right, waist_width_in, "Waist Width"),
#         (hip_left, hip_right, hip_width_in, "Hip Width"),
#         (shoulder_left, elbow_left, biceps_left_in, "Left Biceps Length"),
#         (shoulder_right, elbow_right, biceps_right_in, "Right Biceps Length"),
#         (elbow_left, wrist_left, forearm_left_in, "Left Forearm Length"),
#         (elbow_right, wrist_right, forearm_right_in, "Right Forearm Length"),
#         (hip_left, knee_left, thigh_left_in, "Left Thigh Length"),
#         (hip_right, knee_right, thigh_right_in, "Right Thigh Length"),
#         (knee_left, ankle_left, calf_left_in, "Left Calf Length"),
#         (knee_right, ankle_right, calf_right_in, "Right Calf Length")
#     ]

#     for start_point, end_point, measurement, label in measurements:
#         print(f"{label}: {measurement:.2f} inches")
#         draw_measurement(start_point, end_point, measurement, label)

#     # Save the annotated image
#     output_path = 'annotated_image.jpg'
#     cv2.imwrite(output_path, annotated_image)
#     print(f"Annotated image saved as {output_path}")

#     # Optionally, display the annotated image
#     cv2.imshow('Annotated Image', annotated_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# else:
#     print("No pose landmarks detected.")
####################################################################################################################################################
# import cv2
# import mediapipe as mp
# import numpy as np

# # Initialize MediaPipe
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
# mp_drawing = mp.solutions.drawing_utils

# # Load the image
# image_path = r'D:\Sixth_Sense\Anthropometry_API\sir_body.jpeg'
# image = cv2.imread(image_path)
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# # Process the image and extract pose landmarks
# results = pose.process(image_rgb)

# # Known height of the person in inches
# height_in_inches = 67  # Example height, adjust as necessary

# def get_landmark(idx):
#     return (int(landmarks[idx].x * image.shape[1]), int(landmarks[idx].y * image.shape[0]))

# def calculate_circumference(width_in_pixels, scale_factor):
#     # Assuming an elliptical cross-section for the body part
#     depth_in_pixels = width_in_pixels * 0.7  # approximate depth as 70% of width
#     circumference_in_pixels = np.pi * np.sqrt(0.5 * (width_in_pixels**2 + depth_in_pixels**2))
#     return circumference_in_pixels * scale_factor

# if results.pose_landmarks:
#     # Extract landmarks
#     landmarks = results.pose_landmarks.landmark
    
#     # Example landmark indices
#     shoulder_left = get_landmark(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
#     shoulder_right = get_landmark(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
#     hip_left = get_landmark(mp_pose.PoseLandmark.LEFT_HIP.value)
#     hip_right = get_landmark(mp_pose.PoseLandmark.RIGHT_HIP.value)
#     elbow_left = get_landmark(mp_pose.PoseLandmark.LEFT_ELBOW.value)
#     elbow_right = get_landmark(mp_pose.PoseLandmark.RIGHT_ELBOW.value)
#     wrist_left = get_landmark(mp_pose.PoseLandmark.LEFT_WRIST.value)
#     wrist_right = get_landmark(mp_pose.PoseLandmark.RIGHT_WRIST.value)
#     knee_left = get_landmark(mp_pose.PoseLandmark.LEFT_KNEE.value)
#     knee_right = get_landmark(mp_pose.PoseLandmark.RIGHT_KNEE.value)
#     ankle_left = get_landmark(mp_pose.PoseLandmark.LEFT_ANKLE.value)
#     ankle_right = get_landmark(mp_pose.PoseLandmark.RIGHT_ANKLE.value)
    
#     # Calculate distances between landmarks
#     def calculate_distance(point1, point2):
#         return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

#     # Example height measurement from head to feet
#     head = get_landmark(mp_pose.PoseLandmark.NOSE.value)
#     feet_left = get_landmark(mp_pose.PoseLandmark.LEFT_ANKLE.value)
#     feet_right = get_landmark(mp_pose.PoseLandmark.RIGHT_ANKLE.value)
#     height_in_pixels = (calculate_distance(head, feet_left) + calculate_distance(head, feet_right)) / 2

#     # Calculate scale factor
#     scale_factor = height_in_inches / height_in_pixels

#     # Calculate widths
#     shoulder_width = calculate_distance(shoulder_left, shoulder_right)
#     chest_width = shoulder_width  # Approximate chest width with shoulder width
#     waist_width = calculate_distance(hip_left, hip_right)
#     hip_width = waist_width  # Approximate hip width with waist width

#     # Calculate circumferences
#     chest_circumference_in = calculate_circumference(chest_width, scale_factor)
#     waist_circumference_in = calculate_circumference(waist_width, scale_factor)
#     hip_circumference_in = calculate_circumference(hip_width, scale_factor)

#     biceps_left_circumference_in = calculate_circumference(calculate_distance(shoulder_left, elbow_left), scale_factor)
#     biceps_right_circumference_in = calculate_circumference(calculate_distance(shoulder_right, elbow_right), scale_factor)
#     forearm_left_circumference_in = calculate_circumference(calculate_distance(elbow_left, wrist_left), scale_factor)
#     forearm_right_circumference_in = calculate_circumference(calculate_distance(elbow_right, wrist_right), scale_factor)
#     thigh_left_circumference_in = calculate_circumference(calculate_distance(hip_left, knee_left), scale_factor)
#     thigh_right_circumference_in = calculate_circumference(calculate_distance(hip_right, knee_right), scale_factor)
#     calf_left_circumference_in = calculate_circumference(calculate_distance(knee_left, ankle_left), scale_factor)
#     calf_right_circumference_in = calculate_circumference(calculate_distance(knee_right, ankle_right), scale_factor)

#     # Draw landmarks and connections
#     annotated_image = image.copy()
#     mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

#     # Annotate image with measurements
#     def draw_measurement(start_point, end_point, measurement, label):
#         cv2.line(annotated_image, start_point, end_point, (0, 255, 0), 2)
#         mid_point = ((start_point[0] + end_point[0]) // 2, (start_point[1] + end_point[1]) // 2)
#         cv2.putText(annotated_image, f"{label}: {measurement:.2f} in", mid_point, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#     circumferences = [
#         ("Chest Circumference", chest_circumference_in),
#         ("Waist Circumference", waist_circumference_in),
#         ("Hip Circumference", hip_circumference_in),
#         ("Left Biceps Circumference", biceps_left_circumference_in),
#         ("Right Biceps Circumference", biceps_right_circumference_in),
#         ("Left Forearm Circumference", forearm_left_circumference_in),
#         ("Right Forearm Circumference", forearm_right_circumference_in),
#         ("Left Thigh Circumference", thigh_left_circumference_in),
#         ("Right Thigh Circumference", thigh_right_circumference_in),
#         ("Left Calf Circumference", calf_left_circumference_in),
#         ("Right Calf Circumference", calf_right_circumference_in),
#     ]

#     for label, measurement in circumferences:
#         print(f"{label}: {measurement:.2f} inches")

#     # Save the annotated image
#     output_path = 'annotated_image.jpg'
#     cv2.imwrite(output_path, annotated_image)
#     print(f"Annotated image saved as {output_path}")

#     # Optionally, display the annotated image
#     cv2.imshow('Annotated Image', annotated_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# else:
#     print("No pose landmarks detected.")
# ################################################################################################################################################################
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
height_in_inches = 67  # Example height, adjust as necessary

def get_landmark(idx):
    return (int(landmarks[idx].x * image.shape[1]), int(landmarks[idx].y * image.shape[0]))

def calculate_circumference(width_in_pixels, scale_factor):
    # Assuming an elliptical cross-section for the body part
    depth_in_pixels = width_in_pixels * 0.7  # approximate depth as 70% of width
    circumference_in_pixels = np.pi * np.sqrt(0.5 * (width_in_pixels**2 + depth_in_pixels**2))
    return circumference_in_pixels * scale_factor

if results.pose_landmarks:
    # Extract landmarks
    landmarks = results.pose_landmarks.landmark
    
    # Example landmark indices
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

    # Draw landmarks and connections
    annotated_image = image.copy()
    mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Annotate image with measurements
    def draw_measurement(start_point, end_point, measurement, label):
        cv2.line(annotated_image, start_point, end_point, (0, 255, 0), 2)
        mid_point = ((start_point[0] + end_point[0]) // 2, (start_point[1] + end_point[1]) // 2)
        cv2.putText(annotated_image, f"{label}: {measurement:.2f} in", mid_point, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    measurements = [
        (shoulder_left, shoulder_right, shoulder_width_in, "Shoulder Width"),
        (shoulder_left, shoulder_right, chest_circumference_in, "Chest Circumference"),
        (hip_left, hip_right, waist_width_in, "Waist Width"),
        (hip_left, hip_right, waist_circumference_in, "Waist Circumference"),
        (hip_left, hip_right, hip_width_in, "Hip Width"),
        (hip_left, hip_right, hip_circumference_in, "Hip Circumference"),
        (shoulder_left, elbow_left, biceps_left_length_in, "Left Biceps Length"),
        (shoulder_right, elbow_right, biceps_right_length_in, "Right Biceps Length"),
        (elbow_left, wrist_left, forearm_left_length_in, "Left Forearm Length"),
        (elbow_right, wrist_right, forearm_right_length_in, "Right Forearm Length"),
        (shoulder_left, elbow_left, biceps_left_circumference_in, "Left Biceps Circumference"),
        (shoulder_right, elbow_right, biceps_right_circumference_in, "Right Biceps Circumference"),
        (elbow_left, wrist_left, forearm_left_circumference_in, "Left Forearm Circumference"),
        (elbow_right, wrist_right, forearm_right_circumference_in, "Right Forearm Circumference"),
        (hip_left, knee_left, thigh_left_length_in, "Left Thigh Length"),
        (hip_right, knee_right, thigh_right_length_in, "Right Thigh Length"),
        (hip_left, knee_left, thigh_left_circumference_in, "Left Thigh Circumference"),
        (hip_right, knee_right, thigh_right_circumference_in, "Right Thigh Circumference"),
        (knee_left, ankle_left, calf_left_length_in, "Left Calf Length"),
        (knee_right, ankle_right, calf_right_length_in, "Right Calf Length"),
        (knee_left, ankle_left, calf_left_circumference_in, "Left Calf Circumference"),
        (knee_right, ankle_right, calf_right_circumference_in, "Right Calf Circumference")
    ]

    for start_point, end_point, measurement, label in measurements:
        print(f"{label}: {measurement:.2f} inches")
        draw_measurement(start_point, end_point, measurement, label)

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
