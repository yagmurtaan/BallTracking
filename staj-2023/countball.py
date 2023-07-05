# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 12:50:19 2023

@author: user
"""

import cv2
import numpy as np
import pyautogui


colors = {
    # 'red': ([0, 100, 100], [10, 255, 255]),
    # 'green': ([40, 100, 100], [70, 255, 255]),
    # 'blue': ([100, 100, 100], [130, 255, 255])
        'red':([0, 70, 50], [10, 255, 255]),   # Red
        'yellow':([20, 70, 50], [35, 255, 255]),  # Yellow
        'green':([35, 70, 50], [85, 255, 255]),  # Green
        'blue':([90, 70, 50], [130, 255, 255])  # Blue
}

video_path = 'video.mp4'
cap = cv2.VideoCapture(video_path)

# Dictionary to store ball counts
ball_counts = {color: 0 for color in colors}
ball_counts_left_half = {color: 0 for color in colors}
# Dictionary to store ball positions and IDs
ball_data = {color: {'positions': [], 'ids': []} for color in colors}
left_half_counter = 0

while True:
    ret, frame = cap.read()
    frame= cv2.resize(frame, (1920,1080))

    if not ret:
        break

    # Convert the frame from BGR to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Process each color
    for color_name, (lower_hsv, upper_hsv) in colors.items():
        mask = cv2.inRange(hsv, np.array(lower_hsv), np.array(upper_hsv))

        #remove noise
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # Find contours 
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Process each contour
        for contour in contours:
            # Calculate the area of the contour
            area = cv2.contourArea(contour)
            print(area)
            
            if area > 1000:
                # Find the center of the contour
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    center = (cX, cY)
                    # Check if the ball position exists
                    if ball_data[color_name]['positions']:
                        # Calculate the distance between the current position and all previous positions
                        distances = np.linalg.norm(np.array(ball_data[color_name]['positions']) - np.array(center), axis=1)
                        average_distance = np.mean(distances)
                        # Check if the ball is a new one
                        if np.min(distances) > 60:
                            # Assign a new ID for the ball
                            ball_id = len(ball_data[color_name]['ids']) + 1
                            ball_data[color_name]['ids'].append(ball_id)
                            ball_data[color_name]['positions'].append(center)
                            ball_counts[color_name] += 1
                        else:
                            # Find the closest ID for the ball
                            closest_id = ball_data[color_name]['ids'][np.argmin(distances)]
                            index = ball_data[color_name]['ids'].index(closest_id)
                            ball_data[color_name]['positions'][index] = center
                    else:
                        # If no ball position exists, assign a new ID for the ball
                        ball_id = len(ball_data[color_name]['ids']) + 1
                        ball_data[color_name]['ids'].append(ball_id)
                        ball_data[color_name]['positions'].append(center)
                        ball_counts[color_name] += 1
                                                
        for color_name in ball_counts_left_half:
            ball_counts_left_half[color_name] = sum(1 for pos in ball_data[color_name]['positions'] if pos[0] < frame.shape[1] // 2)
    
    total_count = sum(ball_counts_left_half.values())
    if total_count < 4:
        red_left_half_count = sum(1 for pos in ball_data['red']['positions'] if pos[0] < frame.shape[1] // 2)
        # Get the screen resolution
        screen_width, screen_height = pyautogui.size()
        # Set the frame width and height
        frame_width = 1920
        frame_height = 1080
        # Capture the screen 
        screenshot = pyautogui.screenshot()
        # Resize the captured screen region to the desired frame size
        resized_screenshot = cv2.resize(np.array(screenshot), (frame_width, frame_height))
        resized_screenshot_rgb = cv2.cvtColor(resized_screenshot, cv2.COLOR_BGR2RGB)
        output_path = r'C:\Users\user\Desktop\balls.jpg'
        # Save the frame
        cv2.imwrite(output_path,  resized_screenshot_rgb)

    for i, (color_name, count) in enumerate(ball_counts_left_half.items()):
        cv2.putText(frame, f'{color_name} (left): {count}', (10, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
  
    if total_count > 3:
        cv2.putText(frame, f'Total ball count exceeds 3', (10, 30 + 30 * (len(ball_counts) + 1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f' Total Red balls in the left half when the total ball count exceeds 3: {red_left_half_count}', (10, 30 + 30 * (len(ball_counts) + 2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Ball Tracking', frame)
    # Exit the loop if 'q' is pressed
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close windows
cap.release()
cv2.destroyAllWindows()
