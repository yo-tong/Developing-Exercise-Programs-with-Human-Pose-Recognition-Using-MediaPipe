from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, make_response, session, flash,g
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import datetime
import math
import copy
import numpy as np
import cv2
import pymysql
import mediapipe as mp
from functools import wraps
from moviepy.editor import VideoFileClip

# Create a Flask application instance
app = Flask(__name__)  
# Set a secret key for session management and security
app.secret_key = 'your_complex_secret_key'  

cap = None   # Stores the video capture object for the camera 
output_folder = 'videos'  # Sets the output folder name for saving videos  
stored_angles = {}
 
def get_db_connection():
    try:# Establish a connection to the database
        conn = pymysql.connect(
            host ='localhost',   # Database server address
            user ='root',        # Database username
            password ='12345678',# Database password
            db ='your_database', # Database name
            charset ='utf8mb4',  # Character encoding
            cursorclass = pymysql.cursors.DictCursor # Return results as dictionaries
        )
        return conn              # Return the database connection object
    except pymysql.MySQLError as e:
        print(f"資料庫連接錯誤: {e}")
        return None

# Decorator to manage database connection
def with_db_connection(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        conn = get_db_connection() # Establish a connection to the database
        if conn is None:
            return jsonify({'error': '無法連接資料庫'}), 500
        try:
            return func(conn, *args, **kwargs)
        finally: 
            conn.close()
    return wrapper
# Initialize database
def init_db():
    conn = get_db_connection()
    with conn.cursor() as c:
        c.execute('''CREATE TABLE IF NOT EXISTS users
                    (id INT AUTO_INCREMENT PRIMARY KEY, 
                    username VARCHAR(100) UNIQUE NOT NULL, 
                    password VARCHAR(255) NOT NULL)''')
    conn.commit()
    conn.close()
init_db()

# Record the user activity
def log_user_activity(user_id, mode, action, action_count=None , action_name=None, body_angle=None):
    try:
        conn = get_db_connection()
        with conn.cursor() as c:
            if action == 'start': # Insert record for starting the activity
                c.execute("""
                    INSERT INTO user_activity (user_id, mode, start_time) 
                    VALUES (%s, %s,  NOW() )
                """, (user_id, mode))
            elif action == 'stop': # Find the latest activity record that has not been ended
                c.execute("""
                    SELECT id FROM user_activity 
                    WHERE user_id = %s   AND end_time IS NULL
                    ORDER BY start_time DESC
                    LIMIT 1
                """, (user_id))
                result = c.fetchone()
                if result:
                    activity_id = result['id'] # Update the activity record with end time and count
                    c.execute("""
                        UPDATE user_activity 
                        SET end_time = NOW(), count = %s ,action_name = %s, body_angle = %s
                        WHERE id = %s
                    """, (action_count, action_name, body_angle, activity_id))
        conn.commit()
    except Exception as e:
        print(f"資料庫錯誤: {e}")
    finally:
        conn.close()

# Use moviepy to compress a video
def compress_video_with_moviepy(input_filepath, output_filepath, scale_factor=0.5, target_bitrate='1000k'):
    # Load the input video file
    clip = VideoFileClip(input_filepath)
    # Calculate the new resolution based on the scale factor
    new_width  = int(clip.size[0] * scale_factor) # Adjust width
    new_height = int(clip.size[1] * scale_factor) # Adjust height
    # Resize the video to the new resolution
    resized_clip = clip.resize(newsize=(new_width, new_height))
    # Export the compressed video 
    resized_clip.write_videofile(output_filepath, bitrate=target_bitrate, codec='libx264')
        
def save_file_with_unique_name(uploaded_file):
    filename = secure_filename(uploaded_file.filename)
    filepath = os.path.join("static", filename)
    count = 2
    # "If the file name already exists, append _1, _2, _3, etc., until a unique file name is found."
    while os.path.exists(filepath):
        name, ext = os.path.splitext(filename)
        new_filename = f"{name}_{count}{ext}"
        filepath = os.path.join("static", new_filename)
        count += 1
    uploaded_file.save(filepath) # save file
    return filepath

class PoseProcessor_img:
    def __init__(self): # Initialization method for PoseProcessor class
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5,
                                      model_complexity=1)
    # "Method used to process each frame of video, perform pose detection and extract key points"
    def process_frame(self, frame):
        self.canvas = copy.deepcopy(frame)  # Create a deep copy of the input frame.
        ### ".canvas 是 PoseProcessor 類的一個屬性，用來存儲處理後的視訊幀。它是一個 OpenCV 的影像物件，可以存儲從原始視訊幀開始進行處理後的結果。"

        # Convert the deep-copied video frame from BGR to RGB color space (since MediaPipe works in the RGB color space)
        self.canvas = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2RGB)
        self.canvas.flags.writeable = False # Set the frame to be non-writable to optimize processing.

        # Perform pose estimation using MediaPipe on the RGB frame
        results = self.pose.process(self.canvas)
        """ self.pose 是 PoseProcessor 類別中的一個屬性，代表了 MediaPipe 姿勢檢測器的實例
            process() 方法用於對圖像進行姿勢檢測，它接受一個圖像作為輸入，並返回檢測結果    """

        self.canvas.flags.writeable = True  # Set the frame back to writable.
        self.canvas = cv2.cvtColor(self.canvas, cv2.COLOR_RGB2BGR) # Convert RGB back to BGR.
        # Extract keypoints
        self.keypoint = []   # Create an empty list to store keypoint data for each frame
        ###".keypoint 是 PoseProcessor 類的一個屬性，用來儲存每一幀姿勢檢測後提取出的關鍵點資訊"

        # Check if any pose landmarks (body keypoints) were detected
        if results.pose_landmarks:
            # ---------- Joint Coordinates -------------
            for id, lm in enumerate(results.pose_landmarks.landmark):  # Iterate through detected landmarks.
                h, w, c = self.canvas.shape   # Get the height, width, and channel count of the frame.
                # Calculate each key point's position on the canvas and its visibility
                cx, cy, vab = int(lm.x * w), int(lm.y * h), int(lm.visibility*100) 
                self.keypoint.append([cx, cy, vab])  # Append the calculated key points to the list

            self.mp_drawing.draw_landmarks(self.canvas, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                                           landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())  # 在處理後的幀上繪製檢測到的人體姿勢關鍵點
        self.count_angle()  # calculate angles
        return self.canvas  # Return the processed frame

    def cal_angle(self, point_1, point_2, point_3):  #calculate the angle 
        # Check if the visibility of any point is less than 35 
        if point_1[2] < 35 or point_2[2] < 35 or point_3[2] < 35:
            return "ERROR"
        else: # Calculate the length of the sides using Euclidean distance formula
            a = math.sqrt((point_2[0]-point_3[0])*(point_2[0]-point_3[0]) + (point_2[1]-point_3[1])*(point_2[1] - point_3[1]))  # length of side ab
            b = math.sqrt((point_1[0]-point_3[0])*(point_1[0]-point_3[0]) + (point_1[1]-point_3[1])*(point_1[1] - point_3[1]))  # length of side bc
            c = math.sqrt((point_1[0]-point_2[0])*(point_1[0]-point_2[0]) + (point_1[1]-point_2[1])*(point_1[1]-point_2[1]))    # length of side ca
            # Use the Law of Cosines to calculate the angle, then convert it to degrees
            angle_B = round(math.degrees(math.acos((b*b-a*a-c*c)/(-2*a*c))))  # Calculate angle 'ABC'

            cv2.putText(self.canvas, str(angle_B), (point_2[0], point_2[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)  # (255, 255, 255) Display in white
            ### str(B): 表示要添加的文字内容，B 是角度的數值
            # Draw a circle at point B to indicate the angle position(255,0,0 blue)
            cv2.circle(self.canvas, (point_2[0], point_2[1]), 5, (255, 0, 0), cv2.FILLED)
            return angle_B

    def count_angle(self):  # calculate multiple joint angles
        try: # Calculate right shoulder angle
            self.angle_1 = self.cal_angle(
                self.keypoint[11], self.keypoint[12], self.keypoint[14])
            angle_1_str = "右肩夾角:"+str(self.angle_1)
            print(angle_1_str)
        except:
            self.angle_1 = "ERROR"
            angle_1_str = "右肩夾角:" + self.angle_1
            print(angle_1_str)

        try: # Calculate left shoulder angle
            self.angle_2 = self.cal_angle(
                self.keypoint[12], self.keypoint[11], self.keypoint[13])
            angle_2_str = "左肩夾角:"+str(self.angle_2)
            print(angle_2_str)
        except:
            self.angle_2 = "ERROR"
            angle_2_str = "左肩夾角:"+self.angle_2
            print(angle_2_str)

        try: # Calculate right elbow angle
            self.angle_3 = self.cal_angle(
                self.keypoint[12], self.keypoint[14], self.keypoint[16])
            angle_3_str = "右手肘夾角:"+str(self.angle_3)
            print(angle_3_str)
        except:
            self.angle_3 = "ERROR"
            angle_3_str = "右手肘夾角:"+self.angle_3
            print(angle_3_str)

        try: # Calculate left elbow angle
            self.angle_4 = self.cal_angle(
                self.keypoint[11], self.keypoint[13], self.keypoint[15])
            angle_4_str = "左手肘夾角:"+str(self.angle_4)
            print(angle_4_str)
        except:
            self.angle_4 = "ERROR"
            angle_4_str = "手肘夾角:"+self.angle_4
            print(angle_4_str)

        try: # Calculate left knee angle
            self.angle_4 = self.cal_angle(
                self.keypoint[23], self.keypoint[25], self.keypoint[27])
            angle_4_str = "左膝蓋夾角:"+str(self.angle_4)
            print(angle_4_str)
        except:
            self.angle_4 = "ERROR"
            angle_4_str = "左膝蓋夾角:"+self.angle_4
            print(angle_4_str)
            
        try:# Calculate right knee angle
            self.angle_4 = self.cal_angle(
                self.keypoint[24], self.keypoint[26], self.keypoint[28])
            angle_4_str = "右膝蓋夾角:"+str(self.angle_4)
            print(angle_4_str)
        except:
            self.angle_4 = "ERROR"
            angle_4_str = "右膝蓋夾角:"+self.angle_4
            print(angle_4_str)   

        try:# Calculate left hip angle
            self.angle_4 = self.cal_angle(
                self.keypoint[11], self.keypoint[23], self.keypoint[25])
            angle_4_str = "左髖夾角:"+str(self.angle_4)
            print(angle_4_str)
        except:
            self.angle_4 = "ERROR"
            angle_4_str = "左髖夾角:"+self.angle_4
            print(angle_4_str)

        try:# Calculate right hip angle
            self.angle_4 = self.cal_angle(
                self.keypoint[12], self.keypoint[24], self.keypoint[26])
            angle_4_str = "右髖夾角:"+str(self.angle_4)
            print(angle_4_str)
        except:
            self.angle_4 = "ERROR"
            angle_4_str = "右髖夾角:"+self.angle_4
            print(angle_4_str)   

class PoseProcessor_video:
    def __init__(self):  # Initialization method for PoseProcessor class
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5,
                                      model_complexity=1)
    # "Method used to process each frame of video, perform pose detection and extract key points"
    def process_frame(self, frame):
        self.canvas = copy.deepcopy(frame) # Create a deep copy of the input frame.
        self.canvas = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB format.
        self.canvas.flags.writeable = False   # Set the frame to be non-writable to optimize processing.
        # Perform pose estimation using MediaPipe on the RGB frame
        results = self.pose.process(self.canvas) 
        self.canvas.flags.writeable = True   # Set the frame back to writable.
        self.canvas = cv2.cvtColor(self.canvas, cv2.COLOR_RGB2BGR) # Convert RGB back to BGR.

        self.keypoint = []  # Initialize an empty list to store keypoints.
        # Check if pose landmarks are detected.
        if results.pose_landmarks:
            # ----------Joint Coordinates-------------
            for id, lm in enumerate(results.pose_landmarks.landmark):  # Iterate through detected landmarks.
                h, w, c = self.canvas.shape  # Get the height, width, and channel count of the frame.
                cx, cy, visibility = int(lm.x * w), int(lm.y * h), int(lm.visibility * 100)
                self.keypoint.append([cx, cy, visibility])
            # Draw the landmarks and connections on the frame.  
            self.mp_drawing.draw_landmarks(
                self.canvas, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )   

        self.count_angle()   # calculate angles.
        return self.canvas   # Return the processed frame.

    def cal_angle(self, point_1, point_2, point_3): 
        # Check if any of the keypoints have low confidence (below 35)
        if point_1[2] < 35 or point_2[2] < 35 or point_3[2] < 35:
            return "ERROR"
        else:# Calculate the distances between the points
            a = math.sqrt((point_2[0] - point_3[0]) ** 2 + (point_2[1] - point_3[1]) ** 2) # length of side ab
            b = math.sqrt((point_1[0] - point_3[0]) ** 2 + (point_1[1] - point_3[1]) ** 2) # length of side bc
            c = math.sqrt((point_1[0] - point_2[0]) ** 2 + (point_1[1] - point_2[1]) ** 2) # length of side ca
            # Apply the cosine rule to calculate the angle
            angle = round(math.degrees(math.acos((b ** 2 - a ** 2 - c ** 2) / (-2 * a * c))))
            # Display the angle on the canvas with different colors based on the value
            if angle < 35:
                cv2.putText(self.canvas, str(angle), (point_2[0], point_2[1] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
            else:
                cv2.putText(self.canvas, str(angle), (point_2[0], point_2[1] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            # Draw a circle at the middle point (point_2)
            cv2.circle(self.canvas, (point_2[0], point_2[1]), 5, (255, 0, 0), cv2.FILLED)
            return angle
    def count_angle(self):
        try:
            self.angle_1 = self.cal_angle(self.keypoint[11], self.keypoint[12], self.keypoint[14])
            print("右肩夾角:", self.angle_1)
        except:
            print("右肩夾角: ERROR")

        try:
            self.angle_2 = self.cal_angle(self.keypoint[12], self.keypoint[11], self.keypoint[13])
            print("左肩夾角:", self.angle_2)
        except:
            print("左肩夾角: ERROR")

        try:
            self.angle_3 = self.cal_angle(self.keypoint[12], self.keypoint[14], self.keypoint[16])
            print("右手肘夾角:", self.angle_3)
        except:
            print("右手肘夾角: ERROR")

        try:
            self.angle_4 = self.cal_angle(self.keypoint[11], self.keypoint[13], self.keypoint[15])
            print("左手肘夾角:", self.angle_4)
        except:
            print("左手肘夾角: ERROR")

        try:
            self.angle_3 = self.cal_angle(self.keypoint[24], self.keypoint[26], self.keypoint[28])
            print("右膝蓋夾角:", self.angle_3)
        except:
            print("右膝蓋夾角: ERROR")

        try:
            self.angle_4 = self.cal_angle(self.keypoint[23], self.keypoint[25], self.keypoint[27])
            print("左膝蓋夾角:", self.angle_4)
        except:
            print("左膝蓋夾角: ERROR")

        try:
            self.angle_3 = self.cal_angle(self.keypoint[12], self.keypoint[24], self.keypoint[26])
            print("右髖夾角:", self.angle_3)
        except:
            print("右髖夾角: ERROR")

        try:
            self.angle_4 = self.cal_angle(self.keypoint[11], self.keypoint[23], self.keypoint[25])
            print("左髖夾角:", self.angle_4)
        except:
            print("左髖夾角: ERROR")

class PoseProcessor_camera:
    # Initialization method for PoseProcessor class
    def __init__(self):  
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.selfie_segmenter = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5,
                                      model_complexity=1)
        self.canvas = None
        self.keypoint = [] # List to store current frame keypoints
        self.action_count = 0
        # Flags for controlling action counting process
        self.start_counting = False   # Indicates if counting has started
        self.waiting_for_next_count = False   # Indicates if waiting for next valid count
        self.keypoint_angles = {  #camera左右相反
            'angle1': {'points': [11, 13, 15], 'is_active': True},  # 右手肘夾角
            'angle2': {'points': [12, 14, 16], 'is_active': True},  # 左手肘夾角
            'angle3': {'points': [13, 11, 23], 'is_active': True},  # 右肩夾角
            'angle4': {'points': [14, 12, 24], 'is_active': True},  # 左肩夾角
            'angle5': {'points': [11, 23, 25], 'is_active': True},  # 右髖夾角
            'angle6': {'points': [12, 24, 26], 'is_active': True},  # 左髖夾角
            'angle7': {'points': [23, 25, 27], 'is_active': True},  # 右膝夾角
            'angle8': {'points': [24, 26, 28], 'is_active': True},  # 左膝夾角
            'angle9': {'points': [24, 23, 25], 'is_active': True},   
            'angle10':{'points': [23, 24, 26], 'is_active': True
        }}
        # Default threshold values and direction for each angle measurement
        self.default_thresholds = {
            'angle1': {'threshold': 165, 'direction': '角度變小'},  # default_thresholds for angle1: greater than 165
            'angle2': {'threshold': 165, 'direction': '角度變小'},  # default_thresholds for angle2: greater than 165
            'angle3': {'threshold': 20,  'direction': '角度變大'},  # default_thresholds for angle3: less than 20
            'angle4': {'threshold': 20,  'direction': '角度變大'},  # default_thresholds for angle4: less than 20
            'angle5': {'threshold': 165, 'direction': '角度變小'},  # default_thresholds for angle5: greater than 165  
            'angle6': {'threshold': 165, 'direction': '角度變小'},  # default_thresholds for angle6: greater than 165 
            'angle7': {'threshold': 165, 'direction': '角度變小'},  # default_thresholds for angle7: greater than 165  
            'angle8': {'threshold': 165, 'direction': '角度變小'},  # default_thresholds for angle8: greater than 165  
            'angle9': {'threshold': 90,  'direction': '角度變大'},  # default_thresholds for angle9:  less than 90
            'angle10':{'threshold': 90,  'direction': '角度變大'}   # default_thresholds for angle10: less than 90
        }
        self.all_angles_achieve_goal = False  # Determine if all angles have reached their goals
        self.Not_reached_Initial_and_Target = False
        self.selected_person=None  # Variable to store the selected person (if multiple people are detected)
    # "Method used to process each frame of video, perform pose detection and extract key points"
    def process_frame(self, frame):
        self.canvas = copy.deepcopy(frame)  # Create a deep copy of the input frame.
        self.canvas = cv2.flip(self.canvas, 1)   # Flip the frame horizontally (mirror effect)
        ### .canvas 是 PoseProcessor 類的一個屬性，用來存儲處理後的視訊幀。它是一個 OpenCV 的影像物件，可以存儲從原始視訊幀開始進行處理後的結果。
        # Convert the frame from BGR to RGB color space since MediaPipe works with RGB
        self.canvas = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2RGB)
        self.canvas.flags.writeable = False # Set the frame to be non-writable to optimize processing.

        # Perform pose estimation using MediaPipe on the RGB frame
        results = self.pose.process(self.canvas)
        """ self.pose 是 PoseProcessor 類別中的一個屬性，代表了 MediaPipe 姿勢檢測器的實例
            # process() 方法用於對圖像進行姿勢檢測，它接受一個圖像作為輸入，並返回檢測結果  """

        # Use Selfie Segmentation to remove the background
        segmentation_results = self.selfie_segmenter.process(self.canvas)
        condition = segmentation_results.segmentation_mask > 0.5
        bg_color = (255, 255, 255)
        bg_image = np.full(self.canvas.shape, bg_color, dtype=np.uint8)
        self.canvas = np.where(condition[..., None], self.canvas, bg_image)

        self.canvas.flags.writeable = True  # Set the frame back to writable.
        self.canvas = cv2.cvtColor(self.canvas, cv2.COLOR_RGB2BGR)  # Convert RGB back to BGR.
        self.keypoint = [] # Initialize an empty list to store key points of the current frame
        ### .keypoint 是 PoseProcessor 類的一個屬性，用來儲存每一幀姿勢檢測後提取出的關鍵點資訊
        # Check if pose landmarks (keypoints) are detected
        if results.pose_landmarks:
            # ----------Joint Coordinates-------------
            for id, lm in enumerate(results.pose_landmarks.landmark):    # Iterate through detected landmarks.
                h, w, c = self.canvas.shape   # Get the height, width, and channel count of the frame.
                # Calculate the coordinates of the key points and their visibility
                cx, cy, vab = int(lm.x * w), int(lm.y *h), int(lm.visibility*100)
                self.keypoint.append([cx, cy, vab])  # Append the calculated key points to the list
            # Draw the pose landmarks and angle lines
            self.draw(self.keypoint, stored_angles)

            # If a specific person is selected, mark them in red
            if self.selected_person:
                cx, cy = self.selected_person[0], self.selected_person[1]
                cv2.circle(self.canvas, (cx, cy), 10, (0, 0, 255), -1)    # Draw a red circle to mark  
        else:
            print("No pose detected.")
            text = "No pose detected"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            thickness = 2
            color = (0, 0, 255)
            # Calculate the size of the text to center it on the canvas
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_x = (self.canvas.shape[1] - text_size[0]) // 2
            text_y = (self.canvas.shape[0] + text_size[1]) // 2

            # Draw the "No person detected" message on the canvas
            cv2.putText(self.canvas, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
        return self.canvas # Return the processed frame

    def draw(self, keypoints, angles):
        landmarks = set()
        pairs = []
        for angle_name, angle_data in self.keypoint_angles.items():
            if angle_name in angles:
                points = angle_data['points']
                landmarks.update(points)
                pairs.extend([(points[0], points[1]), (points[1], points[2])])
                self.calculate_and_draw_angle(
                    keypoints, points, angle_name, angles)
        # Draw lines between keypoints (pairs) in white
        for pt1_idx, pt2_idx in pairs:
            pt1 = (keypoints[pt1_idx][0], keypoints[pt1_idx][1])
            pt2 = (keypoints[pt2_idx][0], keypoints[pt2_idx][1])
            cv2.line(self.canvas, pt1, pt2, (255, 255, 255), 2)
        # Draw keypoints (landmarks) in blue
        for idx in landmarks:
            cx, cy = keypoints[idx][0], keypoints[idx][1]
            cv2.circle(self.canvas, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        # Display the action count in the top left corner of the frame
        cv2.putText(self.canvas, f"Count: {self.action_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 148,40 ), 2, cv2.LINE_AA)

    def calculate_and_draw_angle(self, keypoints, points, angle_name, stored_angles):
        pt1_idx, pt2_idx, pt3_idx = points
        point_1 = keypoints[pt1_idx]
        point_2 = keypoints[pt2_idx]
        point_3 = keypoints[pt3_idx]
        angle_B = self.cal_angle(point_1, point_2, point_3)
        if angle_B == "ERROR":
            return
        # Get the threshold of the current angle
        angle_settings = self.default_thresholds.get(angle_name, {'threshold': 20, 'direction': '角度變大'})
        default_threshold = angle_settings['threshold']
        default_direction = angle_settings['direction']

        # Check if the angle meets the condition to start counting
        if default_direction == '角度變大' and angle_B < default_threshold:
            self.keypoint_angles[angle_name]['is_active'] = True
            self.waiting_for_next_count = False
            self.start_counting = True
        elif default_direction == '角度變小' and angle_B > default_threshold:
            self.keypoint_angles[angle_name]['is_active'] = True
            self.waiting_for_next_count = False
            self.start_counting = True

        # If start_counting is active, check if all angles meet the required conditions
        if self.start_counting:
            self.all_angles_achieve_goal = True
            all_angles_active = True
            # Check if the angle meets the threshold for all stored angles
            for angle_name, goal_threshold in stored_angles.items():
                angle_value = self.cal_angle(
                    keypoints[self.keypoint_angles[angle_name]['points'][0]],
                    keypoints[self.keypoint_angles[angle_name]['points'][1]],
                    keypoints[self.keypoint_angles[angle_name]['points'][2]]
                )
                # Check if all angles reach the goal goal_threshold 
                if angle_value == "ERROR" or (angle_settings['direction'] == '角度變大' and angle_value <= goal_threshold ) or ( angle_settings['direction'] == '角度變小' and angle_value >=   goal_threshold):
                    self.all_angles_achieve_goal = False
                # Check if the angle's active status is True
                if not self.keypoint_angles[angle_name]['is_active']:
                    all_angles_active = False
            # If all angles are above goal_threshold and active, increment the action count
            if self.all_angles_achieve_goal and all_angles_active   :
                self.action_count += 1
                print(f"Action count: {self.action_count}")
                self.waiting_for_next_count = True
                self.start_counting = False
                self.Not_reached_Initial_and_Target = False 
                for angle_name in stored_angles.keys():
                    self.keypoint_angles[angle_name]['is_active'] = False
                    # If the system is in the "waiting for next count" state, check if angles are below the goal_threshold
        elif self.waiting_for_next_count:
            # Check if all angles return to the initial position
            all_angles_return_origin  = True
            for angle_name, threshold in stored_angles.items():
                angle_value = self.cal_angle(
                    keypoints[self.keypoint_angles[angle_name]['points'][0]],
                    keypoints[self.keypoint_angles[angle_name]['points'][1]],
                    keypoints[self.keypoint_angles[angle_name]['points'][2]]
                )
                if angle_value == "ERROR" :
                    break
                if angle_value == "ERROR" or (default_direction == '角度變大' and angle_value >= default_threshold) or (default_direction == '角度變小' and angle_value <= default_threshold):
                    all_angles_return_origin = False
                    if  (default_direction == '角度變大' and angle_value <= threshold) or (default_direction == '角度變小' and angle_value >= threshold):
                        self.Not_reached_Initial_and_Target = True

            if all_angles_return_origin :
                # Allow count increment after all angles return to initial positions
                self.waiting_for_next_count = False
                self.start_counting = True
        # Set color based on the current counting state
        #color = (255, 255, 255) if self.start_counting else (0, 0, 255)
        color = (0, 0, 255) if self.all_angles_achieve_goal else(0, 0, 0)  
        if self.Not_reached_Initial_and_Target  :
            color = (0, 0, 0)
        cv2.putText(self.canvas, str(angle_B), (point_2[0]-15, point_2[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 1, cv2.LINE_AA)

    def cal_angle(self, point_1, point_2, point_3, angle_name=None):  #  Calculates the angle
        # Check visibility of the keypoints before proceeding
        if point_1[2] < 35 or point_2[2] < 35 or point_3[2] < 35:
            return "ERROR"
        else:
            a = math.sqrt((point_2[0] - point_3[0])**2 + (point_2[1] - point_3[1])**2)  # length of side ab
            b = math.sqrt((point_1[0] - point_3[0])**2 + (point_1[1] - point_3[1])**2)  # length of side bc
            c = math.sqrt((point_1[0] - point_2[0])**2 + (point_1[1] - point_2[1])**2)  # length of side ca
            # Use the Law of Cosines to calculate the angle at point B (between points 1, 2, and 3)
            try:
                angle_B = round(math.degrees(
                math.acos((b*b - a*a - c*c) / (-2*a*c))))  # Calculate the angle of angle abc
                return angle_B
            except ValueError:
                print("計算角度時發生錯誤，返回預設值")
                return "ERROR"
            
class PoseProcessor_Rehabilitation:
    # Initialization method for PoseProcessor class
    def __init__(self):  
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.selfie_segmenter = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5,
                                      model_complexity=1)
        self.canvas = None
        self.keypoint = []    # Stores keypoint data for each frame
        self.action_count = 0 # Counter for successful actions
        self.start_counting = False  # indicate whether to start counting
        self.waiting_for_next_count = False  # indicate whether to wait for the next valid count
        self.keypoint_angles = { #camera左右相反
            '右手肘夾角': {'points': [11, 13, 15], 'is_active': True},  # Right elbow angle
            '左手肘夾角': {'points': [12, 14, 16], 'is_active': True},  # Left elbow angle
            '右肩夾角': {'points': [13, 11, 23], 'is_active': True},    # Right shoulder angle
            '左肩夾角': {'points': [14, 12, 24], 'is_active': True},    # Left shoulder angle
            '右髖夾角': {'points': [11, 23, 25], 'is_active': True},    # Right hip angle
            '左髖夾角': {'points': [12, 24, 26], 'is_active': True},    # Left hip angle
            '右膝夾角': {'points': [23, 25, 27], 'is_active': True},    # Right knee angle
            '左膝夾角': {'points': [24, 26, 28], 'is_active': True},    # Left knee angle
            'angle9': {'points': [24, 23, 25], 'is_active': True},   
            'angle10': {'points': [23, 24, 26], 'is_active': True
        }}
        # Default thresholds for each angle, including direction of motion
        self.default_thresholds = { 
            '右手肘夾角': {'threshold': 165, 'direction': '角度變小'}, # default_thresholds for 右手肘夾角: greater than 165
            '左手肘夾角': {'threshold': 165, 'direction': '角度變小'}, # default_thresholds for 左手肘夾角: greater than 165
            '右肩夾角': {'threshold': 20, 'direction': '角度變大'},    # default_thresholds for 右肩夾角: less than 20
            '左肩夾角': {'threshold': 20, 'direction': '角度變大'},    # default_thresholds for 左肩夾角: less than 20
            '右髖夾角': {'threshold': 165, 'direction': '角度變小'},   # default_thresholds for 右髖夾角: greater than 165
            '左髖夾角': {'threshold': 165, 'direction': '角度變小'},   # default_thresholds for 左髖夾角: greater than 165
            '右膝夾角': {'threshold': 165, 'direction': '角度變小'},   # default_thresholds for 右膝夾角: greater than 165
            '左膝夾角': {'threshold': 165, 'direction': '角度變小'},   # default_thresholds for 左膝夾角: greater than 165
            'angle9': {'threshold': 90, 'direction': '角度變大'},     # default_thresholds for angle9: less than 90
            'angle10': {'threshold': 90, 'direction': '角度變大'}     # default_thresholds for angle10: less than 90
        }
        self.Not_reached_Initial_and_Target=False
        self.selected_person=None
        # Initialize the maximum angle recorded for each tracked joint
        self.max_angles = {angle: 0 for angle in self.keypoint_angles.keys()}   
        self.all_angles_achieve_goal = False  # Determine if all angles have reached their goals
    # "Method used to process each frame of video, perform pose detection and extract key points"
    def process_frame(self, frame):
        self.canvas = copy.deepcopy(frame)  # Create a deep copy of the input frame.
        self.canvas = cv2.flip(self.canvas, 1) 
        ### .canvas 是 PoseProcessor 類的一個屬性，用來存儲處理後的視訊幀。它是一個 OpenCV 的影像物件，可以存儲從原始視訊幀開始進行處理後的結果。
        self.canvas = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2RGB)
        # Set the frame's writable attribute to False to prevent unnecessary write operations
        self.canvas.flags.writeable = False # Set the frame to be non-writable to optimize processing.
 
        # Perform pose estimation using MediaPipe on the RGB frame
        results = self.pose.process(self.canvas)
        ### self.pose 是 PoseProcessor 類別中的一個屬性，代表了 MediaPipe 姿勢檢測器的實例
        # The process() method accepts an image and returns detection results.      

        # Use Selfie Segmentation to remove the background
        segmentation_results = self.selfie_segmenter.process(self.canvas)
        condition = segmentation_results.segmentation_mask > 0.5
        bg_color = (255, 255, 255)
        bg_image = np.full(self.canvas.shape, bg_color, dtype=np.uint8)
        self.canvas = np.where(condition[..., None], self.canvas, bg_image)


        self.canvas.flags.writeable = True    # Set the frame back to writable.
        self.canvas = cv2.cvtColor(self.canvas, cv2.COLOR_RGB2BGR)  # Convert RGB back to BGR.
        # Extract key points
        self.keypoint = []  # Create an empty list to store joint data for each frame
        ### .keypoint 是 PoseProcessor 類的一個屬性，用來儲存每一幀姿勢檢測後提取出的關鍵點資訊

        # Check if a human pose is detected
        if results.pose_landmarks:
            # ----------Joint Coordinates-------------
            for id, lm in enumerate(results.pose_landmarks.landmark):  # Iterate through detected body key points
                h, w, c = self.canvas.shape   # Get the height, width, and channel count of the frame.
                cx, cy, vab = int(lm.x * w), int(lm.y * h), int(lm.visibility*100)# Calculate each key point's position on the canvas and its visibility
                self.keypoint.append([cx, cy, vab]) # Append the calculated key points to the list

            # Draw nodes and connections based on stored_angles
            self.draw(self.keypoint, stored_angles)
        
            # Highlight the selected person with a red marker
            if self.selected_person:
                cx, cy = self.selected_person[0], self.selected_person[1]
                cv2.circle(self.canvas, (cx, cy), 10, (0, 0, 255), -1)  # Highlight the selected point with a red marker
        else:
            print("No pose detected.")
            text = "No person detected"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            thickness = 2
            color = (0, 0, 255)

            # Calculate text size
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            # Calculate text position
            text_x = (self.canvas.shape[1] - text_size[0]) // 2
            text_y = (self.canvas.shape[0] + text_size[1]) // 2

            # Draw text on the canvas
            cv2.putText(self.canvas, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
        return self.canvas   # Return the processed frame
 
    def draw(self, keypoints, angles):
        landmarks = set()
        pairs = []
        for angle_name, angle_data in self.keypoint_angles.items():
            if angle_name in angles:
                points = angle_data['points']
                landmarks.update(points)
                pairs.extend([(points[0], points[1]), (points[1], points[2])])
                self.calculate_and_draw_angle(
                    keypoints, points, angle_name, angles)
        # Draw white lines connecting keypoints
        for pt1_idx, pt2_idx in pairs:
            pt1 = (keypoints[pt1_idx][0], keypoints[pt1_idx][1])
            pt2 = (keypoints[pt2_idx][0], keypoints[pt2_idx][1])
            cv2.line(self.canvas, pt1, pt2, (255, 255, 255), 2)
        # Draw blue nodes for keypoints
        for idx in landmarks:
            cx, cy = keypoints[idx][0], keypoints[idx][1]
            cv2.circle(self.canvas, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
         # Display action count on the top-left corner of the canvas
        cv2.putText(self.canvas, f"Count: {self.action_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 148,40 ), 2, cv2.LINE_AA)

    def calculate_and_draw_angle(self, keypoints, points, angle_name, stored_angles):
        pt1_idx, pt2_idx, pt3_idx = points
        point_1 = keypoints[pt1_idx]
        point_2 = keypoints[pt2_idx]
        point_3 = keypoints[pt3_idx]
        angle_B = self.cal_angle(point_1, point_2, point_3)
        if angle_B == "ERROR":
            return
        # Get the threshold of the current angle
        angle_settings = self.default_thresholds.get(angle_name, {'threshold': 20, 'direction': '角度變大'})
        default_threshold = angle_settings['threshold']
        default_direction = angle_settings['direction']
        # Check if the angle meets the condition to start counting
        if default_direction == '角度變大' and angle_B < default_threshold:
            self.keypoint_angles[angle_name]['is_active'] = True
            self.waiting_for_next_count = False
            self.start_counting = True
        elif default_direction == '角度變小' and angle_B > default_threshold:
            self.keypoint_angles[angle_name]['is_active'] = True
            self.waiting_for_next_count = False
            self.start_counting = True
  
        # If start_counting is active
        if self.start_counting:
            self.all_angles_achieve_goal = True
            all_angles_active = True
            for angle_name, goal_threshold in stored_angles.items():
                angle_value = self.cal_angle(
                    keypoints[self.keypoint_angles[angle_name]['points'][0]],
                    keypoints[self.keypoint_angles[angle_name]['points'][1]],
                    keypoints[self.keypoint_angles[angle_name]['points'][2]]
                )
                # Check if all angles reach the goal threshold
                if angle_value == "ERROR" or (angle_settings['direction'] == '角度變大' and angle_value <= goal_threshold ) or ( angle_settings['direction'] == '角度變小' and angle_value >=   goal_threshold):
                    self.all_angles_achieve_goal = False
                # Check if the angle's active status is True
                if not self.keypoint_angles[angle_name]['is_active']:
                    all_angles_active = False
            if self.all_angles_achieve_goal and all_angles_active  :
                self.action_count += 1
                print(f"Action count: {self.action_count}")
                self.waiting_for_next_count = True
                self.start_counting = False
                self.Not_reached_Initial_and_Target = False 
                for angle_name in stored_angles.keys():
                    self.keypoint_angles[angle_name]['is_active'] = False
        elif self.waiting_for_next_count:
            # Check if all angles return to the initial position
            all_angles_return_origin  = True
            for angle_name, threshold in stored_angles.items():
                angle_value = self.cal_angle(
                    keypoints[self.keypoint_angles[angle_name]['points'][0]],
                    keypoints[self.keypoint_angles[angle_name]['points'][1]],
                    keypoints[self.keypoint_angles[angle_name]['points'][2]]
                )
                if angle_value == "ERROR" :
                    break
                if angle_value == "ERROR" or (default_direction == '角度變大' and angle_value >= default_threshold) or (default_direction == '角度變小' and angle_value <= default_threshold):
                    all_angles_return_origin = False
                    if  (default_direction == '角度變大' and angle_value <= threshold) or (default_direction == '角度變小' and angle_value >= threshold):
                        self.Not_reached_Initial_and_Target = True
                    break
          
            if all_angles_return_origin :
                # Allow count increment after all angles return to initial positions
                self.waiting_for_next_count = False
                self.start_counting = True
             
        # Update the maximum angle achieved
        if default_direction == '角度變大' and angle_B > self.max_angles[angle_name]:
            self.max_angles[angle_name] = angle_B
            #print(f"最大角度 {angle_name}: {angle_B}")
   
        elif default_direction == '角度變小' and angle_B < self.max_angles[angle_name]:
            self.max_angles[angle_name] = angle_B
            #print(f"最小角度 {angle_name}: {angle_B}")
 

        color = (0, 0, 255) if self.all_angles_achieve_goal else(0, 0, 0)  
        if self.Not_reached_Initial_and_Target  :
            color = (0, 0, 0)
        cv2.putText(self.canvas, str(angle_B), (550, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 1, cv2.LINE_AA)
    def get_max_angle_info(self):
        unique_angle_name = None
        unique_angle_value = None

        # Count occurrences of each angle value
        value_count = {}
        for name, value in self.max_angles.items():
            if value != 0:  # Ignore zero values
                value_count[value] = value_count.get(value, 0) + 1
        #print(f"值的出現次數統計: {value_count}")
        #  Identify unique values 
        unique_values = [value for value, count in value_count.items() if count == 1]

        if len(unique_values) == 1:  # If there is only one unique value
            unique_angle_value = unique_values[0]
            # Find corresponding name
            for name, value in self.max_angles.items():
                if value == unique_angle_value:
                    unique_angle_name = name
                    break
        elif len(unique_values) == 0:
            print("無唯一非零值")
        else:
            print("存在多個唯一非零值")

        return unique_angle_name, unique_angle_value

    def cal_angle(self, point_1, point_2, point_3, angle_name=None):  # Calculate the angle
        if point_1[2] < 35 or point_2[2] < 35 or point_3[2] < 35:
            return "ERROR"
        else:
            a = math.sqrt((point_2[0] - point_3[0])**2 + (point_2[1] - point_3[1])**2)  # length of side ab
            b = math.sqrt((point_1[0] - point_3[0])**2 + (point_1[1] - point_3[1])**2)  # length of side bc
            c = math.sqrt((point_1[0] - point_2[0])**2 + (point_1[1] - point_2[1])**2)  # length of side ca

            angle_B = round(math.degrees(
                math.acos((b*b - a*a - c*c) / (-2*a*c))))  # Calculate the angle of angle abc
            return angle_B

# define_yourself mode  specify_angle
@app.route('/specify_angle_define_yourself', methods=['POST'])
def specify_angle_define_yourself():
    global stored_angles
    # (request.form)Receive and convert the form data into a dictionary
    angles = request.form.to_dict()  
    stored_angles = {k: int(v) for k, v in angles.items() if v} # return the key-value pairs in the angles dictionary
    print(stored_angles)
    # stored_angles is a dictionary that stores the specified angles (ex.右手軸) and degrees (ex.90)
    return jsonify(stored_angles=stored_angles)  
    # Convert the stored_angles dictionary to JSON format and return it to the client

#Rehabilitation mode  specify_angle
@app.route('/specify_angle', methods=['POST'])
def specify_angle():
    global stored_angles
    session['stored_action_name'] = ""  
    # Receive and convert the form data into a dictionary
    angles = request.form.to_dict()   
    stored_angles = {}
    print("Form data:", angles)   
    for k, v in angles.items(): # return the key-value pairs in the angles dictionary
        try: # Convert string to integer first to float and then to int
            stored_angles[k] = int(float(v))  
        except (ValueError, TypeError):
            stored_angles[k] = v  # If conversion cannot be performed, retain the original string.
    
    # stored_angles is a dictionary that stores the specified angles (ex.右手軸) and degrees (ex.90)
    session['stored_action_name'] = angles.get('action_name') 
    del stored_angles['action_name']
    session.modified = True  
    return jsonify(stored_angles=stored_angles,  redirect_url=url_for('start_video4'))

#page_data html set_action
@app.route('/set_action', methods=['POST'])
@with_db_connection
def set_action(conn):
    # Receive JSON data
    data = request.get_json()
    action_name = data.get('action_name')
    angle_type = data.get('angle_type')
    initial_angle = data.get('initial_angle')
    goal_angle = data.get('goal_angle')
    angle_direction = data.get('angle_direction')  
    # Validate initial and goal angles
    if  goal_angle is None or initial_angle is None:  
        return jsonify(message="請填寫初始角度和目標角度"), 400
    
    # Convert the angle to int before inserting it into the database
    initial_angle = int(initial_angle)
    goal_angle = int(goal_angle)

    try: # Connect to the database
        conn = get_db_connection()
        with conn.cursor() as cursor:
            # Check if the user has already saved 5 records
            count_sql = "SELECT COUNT(*) FROM actions WHERE user_id = %s"
            cursor.execute(count_sql, (g.user_id,))
            user_record_count = cursor.fetchone()
            count = user_record_count['COUNT(*)']

            if count >= 5:
                return jsonify(message="每個用戶最多只能保存五筆記錄"), 400
            
            # Check if the action name already exists
            check_name_sql = "SELECT COUNT(*) FROM actions WHERE user_id = %s AND action_name = %s"
            cursor.execute(check_name_sql, (g.user_id, action_name))
            existing_name_count = cursor.fetchone()
            if existing_name_count['COUNT(*)'] > 0:
                return jsonify(message="該名稱已存在，請使用其他名稱"), 400
            # Insert the new action into the database
            sql = """
            INSERT INTO actions (user_id, action_name, angle_type, initial_angle, goal_angle ,angle_direction)
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            cursor.execute(sql, (g.user_id, action_name, angle_type, initial_angle, goal_angle, angle_direction))
            conn.commit()  

    except Exception as e:
        print(f"提交數據時發生錯誤: {str(e)}")
        return jsonify(message="提交數據時發生錯誤", error=str(e)), 500
 
    return jsonify(message="紀錄已成功提交"), 200

#page_data html delete_action
@app.route('/delete_action', methods=['POST'])
@with_db_connection
def delete_action(conn):
    # Get the action name to delete
    action_name = request.form.get('action_name')
    try:
        with conn.cursor() as cursor:
            # Delete the specified action from the database
            delete_sql = "DELETE FROM actions WHERE user_id = %s AND action_name = %s"
            cursor.execute(delete_sql, (g.user_id, action_name))
            conn.commit()
    except Exception as e:
        return jsonify(message="刪除時發生錯誤", error=str(e)), 500
 
    return jsonify(message="動作已成功刪除"), 200

@app.before_request
def before_request():
    g.username = session.get('username')
    if g.username:
        try:
            conn =  get_db_connection()
            if conn is None:
                g.user_id = None
                return
            with conn.cursor() as c:
                c.execute("SELECT id FROM users WHERE username=%s", (g.username,))
                user_id = c.fetchone()
                g.user_id = user_id['id'] if user_id else None
        except Exception as e:
            g.user_id = None
        finally:
            if conn:
                conn.close()

@app.route('/success') # Success Page
def success_page():
    return "數據已成功提交！"

#Rehabilitation_mode
selected_person=None
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmenter = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Processes each video frame, performs pose detection, and extracts key points
def process_NULLframe(frame):
    canvas = copy.deepcopy(frame)   
    canvas = cv2.flip(canvas, 1) 
    # Use Selfie Segmentation to remove the background
    segmentation_results = selfie_segmenter.process(canvas)
    condition = segmentation_results.segmentation_mask > 0.5
    bg_color = (255, 255, 255)
    bg_image = np.full(canvas.shape, bg_color, dtype=np.uint8)
    canvas = np.where(condition[..., None], canvas, bg_image)
    return canvas  # Return the processed frame

#Rehabilitation_mode
@app.route('/close_camera', methods=['POST'])
def close_camera():
    global cap
    if cap is not None:
        cap.release()   # Release camera resources
        cap = None
    return jsonify({"status": "success", "message": "Camera closed successfully."})

@app.route('/open_camera' ) # Open the camera
def open_camera  ():
    global cap 
    # 嘗試開啟 camera_index=1，若失敗再開啟 camera_index=0
    for camera_index in [1, 0]:
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            print(f"✅ Camera opened at index {camera_index}")
            break
        else:
            print(f"❌ Failed to open camera at index {camera_index}")
            cap.release()
            cap = None

    if cap is None or not cap.isOpened():
        return "No available camera found", 500
    return Response(open_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def open_frames():  
    global cap  
    while cap and cap.isOpened():  
        success, frame = cap.read()  
        if not success or frame is None:
            print("Failed to capture image")
            continue   # Skip current loop or return error
        frame = process_NULLframe(frame)
        
        ret, buffer = cv2.imencode('.jpg', frame)  
        frame = buffer.tobytes()  
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/maintain_camera' ) 
def maintain_camera ():
    global cap
     # 優先使用 camera_index=1，如果失敗則改用 camera_index=0
    for camera_index in [1, 0]:
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            print(f"✅ Camera opened at index {camera_index}")
            break
        else:
            print(f"❌ Failed to open camera at index {camera_index}")
            cap.release()
            cap = None

    if cap is None or not cap.isOpened():
        return "❌ 無法開啟攝影機", 500
    return Response(maintain_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
def maintain_frames():  
    global cap  
    while cap and cap.isOpened():  
        success, frame = cap.read()  
        if not success or frame is None:
            print("Failed to capture image")
            continue   # Skip current loop or return error
        frame = process_NULLframe(frame)
        
        ret, buffer = cv2.imencode('.jpg', frame)  
        frame = buffer.tobytes()  
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

#Rehabilitation_mode
def generate_frame(processor,user_id, mode,state,camera_index=0):  # Generates frames for video streaming
    global cap 
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap.release()
            cap = None
            flash('無法打開攝像頭', 'error')
            return redirect(url_for('error_page'))  # Redirect to error page

    if not cap.isOpened():   # Check if the camera opened successfully
        print(f"Cannot open camera {camera_index}")
        return
    log_user_activity(user_id, mode, state)  # Record the user activity
    while cap and cap.isOpened():  # Check if cap exis  and the video stream is open
        success, frame = cap.read() # success is a bool value, indicating whether the reading is successful, and frame is the image frame read.
        if not success:
            break
        frame = processor.process_frame(frame)
        ret, buffer = cv2.imencode('.jpg', frame)  # cv2.imencode :Encode the processed frames into JPG format
        frame = buffer.tobytes()  # Convert the encoded frame data into a byte stream (bytes) for easy transmission on the network
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
 
def generate_frames(processor, user_id, mode, state, camera_index=0):  
    global cap  
    log_user_activity(user_id, mode, state) # Start recording activity
    cap = cv2.VideoCapture(0) 
    if not cap.isOpened(): # Check if the camera opened successfully
        cap.release()
        cap = None
        print(f"Cannot open camera {camera_index}")
        return
    while cap and cap.isOpened():  
        success, frame = cap.read()  
        if not success:
            break
        frame = processor.process_frame(frame)
        ret, buffer = cv2.imencode('.jpg', frame)  
        frame = buffer.tobytes()  
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/process_image', methods=['POST'])
def process_image():
    uploaded_file = request.files['image']
    filepath = save_file_with_unique_name(uploaded_file)  # Save the uploaded file and get the path
    image = cv2.imread(filepath) # Read the uploaded image
    pose_processor = PoseProcessor_img()
    processed_image = pose_processor.process_frame(image)
    # Save the processed image to a folder
    result_filename = "result_" + os.path.basename(filepath)
    result_filepath = os.path.join("static", result_filename)
    cv2.imwrite(result_filepath, processed_image)  # Save the processed image
    print(result_filepath)
    print(result_filename)
    return redirect(url_for('page_img', uploaded_filename=result_filename))# Redirect to display the processed image

@app.route('/process_media', methods=['POST'])
def process_media():
    uploaded_file = request.files['media']
    filepath = save_file_with_unique_name(uploaded_file)

    if uploaded_file.mimetype.startswith('image/'):
        image = cv2.imread(filepath)
        pose_processor = PoseProcessor_img()
        processed_image = pose_processor.process_frame(image)
        result_filename = "result_" + os.path.basename(filepath)
        result_filepath = os.path.join("static", result_filename)
        cv2.imwrite(result_filepath, processed_image)
        return redirect(url_for('page_img', uploaded_filename=result_filename))
    
    elif uploaded_file.mimetype.startswith('video/'):
        video_capture = cv2.VideoCapture(filepath)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        result_filename = "result_" + os.path.basename(filepath)
        result_filepath = os.path.join("static", result_filename)
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_writer = cv2.VideoWriter(result_filepath, fourcc, fps, (width, height))
        
        pose_processor = PoseProcessor_video()
        
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break
            processed_frame = pose_processor.process_frame(frame)
            video_writer.write(processed_frame)
        
        video_capture.release()
        video_writer.release()
        compressed_filename = "compressed_" + result_filename
        compressed_filepath = os.path.join("static", compressed_filename)
        compress_video_with_moviepy(result_filepath, compressed_filepath, scale_factor=1, target_bitrate='500k')

        return redirect(url_for('page_video', uploaded_filename=compressed_filename))
    else:
        return "Unsupported file type", 400
    

@app.route('/')
def index():
    return render_template('log_in.html')
# Log in the user
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = get_db_connection()
        if conn is None:
            return "無法連接資料庫"
        try:
            with conn.cursor() as c:
                c.execute("SELECT * FROM users WHERE username=%s", (username,))
                user = c.fetchone() 
        except Exception as e:
            return f"資料庫錯誤: {e}"
        finally:
            conn.close() 
        if user and check_password_hash(user['password'], password):
            session['username'] = username
            #flash('登入成功！', 'success')
            return redirect(url_for('page_home'))
        else:
            flash('用戶名或密碼錯誤，請重試。', 'danger')
    return render_template('log_in.html')

# Log out the user
@app.route('/logout', methods=['POST'])
def logout():
    session.pop('username', None)
    flash('請登入。', 'success')
    return redirect(url_for('index'))  # Redirect to the login page

# Register the user
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = generate_password_hash(password,method='pbkdf2:sha256')
        try:
            conn = get_db_connection()
            if conn is None:
                return "無法連接資料庫"
            with conn.cursor() as c:
                conn.begin()
                c.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, hashed_password))
                conn.commit()
            #return redirect(url_for('login'))
            flash('註冊成功，請登入', 'success')
            return redirect(url_for('index'))
        except pymysql.IntegrityError:
            conn.rollback()
            flash('用戶名已存在，請選擇其他用戶名。', 'error')
            
            return redirect(url_for('register'))
        except Exception as e:
            conn.rollback()
            flash('f"資料庫錯誤: {e}', 'error')
            return redirect(url_for('register'))
        finally:
            conn.close()
    return render_template('register.html')  # Handle GET request and return the registration page

@app.route('/page_home')
def page_home():
    if not g.username:
        flash('請先登入', 'warning')
        return redirect(url_for('login'))
    return render_template('page_home.html', username=g.username)

### 
@app.route('/home')
def home():
    if 'username' in session:
        return f'歡迎，{session["username"]}！'
    return redirect(url_for('login'))

@app.route('/page_img')
def page_img():
    return render_template('page_img.html', username=g.username)

@app.route('/page_video')
def page_video():
    uploaded_filename = request.args.get('uploaded_filename')
    return render_template('page_video.html', username=g.username, uploaded_filename=uploaded_filename)

@app.route('/page_camera')
def page_camera():
    return render_template('page_camera.html',username=g.username)

@app.route('/page_camera1')
def page_camera1():
    return render_template('page_camera1.html',username=g.username)

@app.route('/page_camera2')
def page_camera2():
    return render_template('page_camera2.html',username=g.username)

@app.route('/page_camera3')
def page_camera3():
    return render_template('page_camera3.html',username=g.username)

@app.route('/page_data', methods=['GET', 'POST'])
@with_db_connection
def page_data(conn):
    try:
        with conn.cursor() as c:
            c.execute("SELECT * FROM actions WHERE user_id = %s", (g.user_id,))
            modes = c.fetchall()
            sql = """
            SELECT action_name FROM actions WHERE user_id = %s
            """
            c.execute(sql, (g.user_id,))
            actions = c.fetchall()  # Get results
    except Exception as e:
        return jsonify(message="查詢時發生錯誤", error=str(e)), 500
 
    return render_template('page_data.html', actions=actions, modes=modes, username=g.username)
#action_default_page.html
@app.route('/perform_action/<action_name>', methods=['GET'])
@with_db_connection
def perform_action( conn,action_name):
    try:
        with conn.cursor() as c:
            # Get action details based on action name
            c.execute("SELECT * FROM actions WHERE action_name = %s AND user_id = %s", (action_name, g.user_id))
            action = c.fetchone()

            if not action:
                return jsonify({'error': '沒有找到該動作'}), 404  # Return 404 error

            return render_template('action_default_page.html', action=action)

    except Exception as e:
        print(f"後端錯誤: {e}")  
        return jsonify({'error': '伺服器內部錯誤'}), 500 # Return 500 error
 
@app.route('/page_user')
@with_db_connection
def page_user(conn):
    username = session.get('username')  
    if not username:
        flash('請先登入', 'warning')
        return redirect(url_for('login'))
    try:
        with conn.cursor() as c:
            c.execute("SELECT * FROM users WHERE username=%s", (username,))
            user_info = c.fetchone()  # Get user information from the database
            c.execute("""
                SELECT mode, COUNT(*) as frequency, DATE(start_time) as date
                FROM user_activity
                WHERE start_time >= DATE_FORMAT(CURDATE(), '%Y-%m-01')
                GROUP BY mode, DATE(start_time)
                ORDER BY date;
            """)
            data = c.fetchall()
        # Format data for frontend
        formatted_data = {}
        for row in data:
            mode = row['mode']
            date = row['date'].strftime('%Y-%m-%d')
            frequency = row['frequency']

            if mode not in formatted_data:
                formatted_data[mode] = []
            formatted_data[mode].append({'date': date, 'frequency': frequency})

        if user_info:
            return render_template('page_user.html',username=username,  user=user_info,data=formatted_data)
        else:
            flash('用戶信息未找到', 'danger')
            return redirect(url_for('home'))
    except Exception as e:
        return f"資料庫錯誤: {e}"
 
# page_user data
@app.route('/get_mode_data')
@with_db_connection
def get_mode_data(conn):
    if not g.user_id:
        return jsonify({'error': '用戶未登入'})
    try:
        with conn.cursor() as c:
            # 當日所有模式的使用次數折線圖數據 '
            # 當日所有模式的累計時間      
            # 獲取所有模式的每日頻率
            query = """
                SELECT 'linechart' AS type, start_time, count, mode, NULL as total_time, NULL as frequency, NULL as date
                FROM user_activity
                WHERE DATE(start_time) = CURDATE() AND user_id = %s
                GROUP BY start_time, mode
            
                UNION ALL

                SELECT 'cumulative_time' AS type, NULL as start_time, NULL as count, mode, 
                    SUM(TIMESTAMPDIFF(SECOND, start_time, end_time)) as total_time, NULL as frequency, NULL as date
                FROM user_activity
                WHERE DATE(start_time) = CURDATE() AND user_id = %s
                GROUP BY mode

                UNION ALL

               SELECT 'heatmap' AS type, NULL as start_time, NULL as count, mode, NULL as total_time, 
                       COUNT(*) as frequency, DATE(start_time) as date
                FROM user_activity
                WHERE start_time >= DATE_FORMAT(CURDATE(), '%%Y-%%m-01') AND user_id = %s
                GROUP BY date, mode
                ORDER BY type, start_time, mode, date
            """  
            c.execute(query, (g.user_id, g.user_id, g.user_id))
            all_data = c.fetchall()
           
        # Classify query results based on type field
        linechart_data = [
            {'time': row['start_time'].isoformat(), 'count': row['count'], 'mode': row['mode']}
            for row in all_data if row['type'] == 'linechart'
        ] 
        cumulative_time_data = [
            {'mode': row['mode'], 'total_time': int(row['total_time'])}
            for row in all_data if row['type'] == 'cumulative_time'
        ]
        heatmap_data = [
            {'date': row['date'].strftime('%Y-%m-%d'), 'mode': row['mode'], 'frequency': row['frequency']}
            for row in all_data if row['type'] == 'heatmap'
        ]
        return jsonify({
            'linechart': linechart_data,
            'cumulative_time': cumulative_time_data,
            'heatmap': heatmap_data
        }) 
    except Exception as e:
        return jsonify({'error': str(e)})
 
# page_user data
@app.route('/get_rehabilitation_actions', methods=['GET'])
@with_db_connection
def get_rehabilitation_actions(conn):
     # Check if you are logged in
    if not g.get('user_id'):
        return jsonify({'error': '用戶未登入'}), 401
    try:
        with conn.cursor() as c:
            # Query the user's action list
            c.execute("SELECT DISTINCT action_name FROM actions WHERE user_id = %s", (g.user_id,))
            actions = c.fetchall()  # Get all results

        # Convert action list to JSON format
        action_list = [row['action_name'] for row in actions]
        return jsonify(action_list)
    
    except Exception as e:
        print(f"後端錯誤: {e}")
        return jsonify({'error': 'Internal server error'}), 500
  
 # page_user data
@app.route('/get_rehabilitation_angle_data', methods=['POST'])
@with_db_connection
def get_rehabilitation_angle_data(conn):
    today_date = datetime.datetime.now().date()#current date
    action_name = request.json['action_name']
 
    start_date = request.json.get('start_date' )
    end_date = request.json.get('end_date' )
    # 查詢該用戶該動作當月的角度數據
    query = """
        SELECT start_time, body_angle 
        FROM user_activity 
        WHERE user_id = %s AND action_name = %s AND body_angle <> 0   AND DATE(start_time) BETWEEN %s AND %s  -- 只查詢今天的數據
        ORDER BY start_time 
    """
    try:
        with conn.cursor() as cursor:
            cursor.execute(query, (g.user_id, action_name, start_date, end_date))
            result = cursor.fetchall()
        # Format data for return
        data = [{'time': row['start_time'].strftime('%Y-%m-%d %H:%M:%S'), 'angle': row['body_angle']} for row in result]
        print(data)
        
        return jsonify(data)
    
    except KeyError as e:
        print(f"缺少必要的參數: {e}")
        return jsonify({'error': '缺少必要的參數'}), 400
    except pymysql.MySQLError as e:
        print(f"資料庫錯誤: {e}")
        return jsonify({'error': '資料庫錯誤'}), 500
    except Exception as e:
        print(f"未知錯誤: {e}")
        return jsonify({'error': '內部伺服器錯誤'}), 500
 
def start_video(mode, angles, thresholds=None):
    global cap, global_processor ,angle_direction
    username = session.get('username')
    if not username:
        flash('請先登入', 'warning')
        return redirect(url_for('login'))

    stored_angles.update(angles)  # Set default_target angles
    state= 'start'
    user_id=g.user_id
    
    if(mode=="復健"):
        global_processor = PoseProcessor_Rehabilitation()
        if(angle_direction=='角度變小'):
            global_processor.max_angles = {angle: 180 for angle in global_processor.keypoint_angles.keys()}  #Initialize the maximum value of each angle
        elif(angle_direction=='角度變大'):
            global_processor.max_angles = {angle: 0 for angle in global_processor.keypoint_angles.keys()}  #Initialize the minimum value of each angle

        if thresholds: # Update the default threshold (if any)
            global_processor.default_thresholds.update(thresholds)
        return Response(generate_frame(global_processor, user_id,mode, state), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        global_processor = PoseProcessor_camera()
        if thresholds:
            global_processor.default_thresholds.update(thresholds)
        return Response(generate_frames(global_processor, user_id,mode, state), mimetype='multipart/x-mixed-replace; boundary=frame')
        
@app.route('/start_video1')
def start_video1():
    return start_video("雙手平舉", {'angle3': 90, 'angle4': 90}) # Set default_target angles
@app.route('/start_video2')
def start_video2():
    #thresholds = {'angle7': {'threshold': 60, 'direction': '角度變大'}}
    # return start_video("直上抬腿", {'angle7': 160},thresholds)
    return start_video("側上抬腿", {'angle10': 120})

@app.route('/start_video3')
def start_video3():
    return start_video("自訂義角度", {})

#Rehabilitation mode
@app.route('/start_video4' )
def start_video4():
    username = session.get('username')
    if not username:
        flash('請先登入', 'warning')
        return redirect(url_for('login'))
    action_name = session.get('stored_action_name')
    user_data = get_user_angles_from_db(action_name) 
    if not user_data:
        flash('未找到用戶的角度數據', 'danger')
       # return redirect(url_for('some_page'))  # 返回其他頁面或處理邏輯
    
    #Get data from database
    thresholds = user_data.get('thresholds', {})
    return start_video("復健", stored_angles, thresholds)

def get_user_angles_from_db( action_name):
    global angle_direction
    conn = get_db_connection()
    if conn is None:
        return None 
    try:
        with conn.cursor() as cursor:
            query = "SELECT  action_name, angle_type,initial_angle, goal_angle, angle_direction FROM actions WHERE user_id = %s AND action_name = %s "
            cursor.execute(query, (g.user_id, action_name))
            results = cursor.fetchall()
            if not results:
                return None
            angles = {}
            thresholds = {}
            for row in results:
                action_name = row['action_name']
                angle_type=row['angle_type']
                initial_angle = row['initial_angle']
                goal_angle = row['goal_angle']
                angle_direction = row['angle_direction']
                
                # Set angles and thresholds
                angles[angle_type]= goal_angle   
                thresholds  = {angle_type:{'threshold': initial_angle, 'direction': angle_direction}}
            return { 'thresholds': thresholds}
    finally:
        conn.close()

@app.route('/stop5_video', methods=['POST'])
def stop5_video():
    global cap,global_processor 
    # Release the previous camera
    max_angle_name, max_angle_value = global_processor.get_max_angle_info()
    print(f"Max Angle Name: {max_angle_name}, Max Angle Value: {max_angle_value}")
    stored_action_name = session.get('stored_action_name')
    log_user_activity(g.user_id, None, 'stop', global_processor.action_count,stored_action_name, max_angle_value) # Record the user activity
    return '復健已結束!'
@app.route('/stop4_video', methods=['POST'])
def stop4_video():
    global cap,global_processor 
    # Release the previous camera
    if cap is not None:
            cap.release()
            cap = None
    return '鏡頭已關閉!'

@app.route('/stop_video', methods=['POST'])
def stop_video():
    global cap,global_processor 
    # Release the previous camera
    if cap is not None:
        cap.release()
        cap = None
    log_user_activity(g.user_id, None, 'stop', global_processor.action_count) # Record the user activity
    return '鏡頭已關閉!'

if __name__ == "__main__":   # Check whether to run this script directly
    
# Check if the output folder exists
    if not os.path.exists(output_folder):  #Create the output folder if the folder does not exist
        os.makedirs(output_folder)   
    app.run(debug=True)   # Run the Flask application with debugging enabled