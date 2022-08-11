## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##              Align Depth to Color               ##
#####################################################

# First import the library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2

import matplotlib.pyplot as plt

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 1 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

# Streaming loop
try:
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Remove background - Set pixels further than clipping_distance to grey
        grey_color = 153
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

        # Render images:
        #   depth align to color on left
        #   depth on right
        depth_image_output = cv2.convertScaleAbs(depth_image, alpha=0.05)
        depth_image_output_3d = np.dstack((depth_image_output,depth_image_output,depth_image_output))
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        images = np.hstack((bg_removed, depth_image_output_3d))

        cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
        cv2.imshow('Align Example', images)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()
# # Streaming loop
# try:
#     while True:
#         # Get frameset of color and depth
#         frames = pipeline.wait_for_frames()
#         # frames.get_depth_frame() is a 640x360 depth image

#         # Align the depth frame to color frame
#         aligned_frames = align.process(frames)

#         # Get aligned frames
#         aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
#         color_frame = aligned_frames.get_color_frame()

#         # Validate that both frames are valid
#         if not aligned_depth_frame or not color_frame:
#             continue

#         depth_image = np.asanyarray(aligned_depth_frame.get_data())
#         color_image = np.asanyarray(color_frame.get_data())
#         print(depth_image)
#         # Remove background - Set pixels further than clipping_distance to grey
#         grey_color = 153
#         depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
#         bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

#         # Render images:
#         #   depth align to color on left
#         #   depth on right
#         depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
#         depth_colormap_2 = (depth_image - np.min(depth_image)) / (np.max(depth_image) - np.min(depth_image)) * 255
#         depth_colormap_2 = depth_colormap_2.astype('uint8')

#         # start of orientation
#         img = depth_colormap
#         # img = depth_image
#         # Was the image there?
#         if img is None:
#             print("Error: File not found")
#             exit(0)
        
#         # cv2.imshow('Input Image', depth_image_3d)
#         # cv2.waitKey(0)
        
#         # Convert image to grayscale
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         # gray = img.copy()
        
#         # Convert image to binary
#         _, bw = cv2.threshold(depth_colormap_2, 20, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
#         # Find all the contours in the thresholded image
#         contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
#         for i, c in enumerate(contours):
        
#             # Calculate the area of each contour
#             area = cv2.contourArea(c)
            
#             # Ignore contours that are too small or too large
#             if area < 0 or 200000 < area:
#                 continue
            
#             # cv.minAreaRect returns:
#             # (center(x, y), (width, height), angle of rotation) = cv2.minAreaRect(c)
#             rect = cv2.minAreaRect(c)
#             box = cv2.boxPoints(rect)
#             box = np.int0(box)
            
#             # Retrieve the key parameters of the rotated bounding box
#             center = (int(rect[0][0]),int(rect[0][1])) 
#             width = int(rect[1][0])
#             height = int(rect[1][1])
#             angle = int(rect[2])
            
                
#             if width < height:
#                 angle = 90 - angle
#             else:
#                 angle = -angle
                    
#             label = "  Rotation Angle: " + str(angle) + " degrees"
#             textbox = cv2.rectangle(color_image, (center[0]-35, center[1]-25), 
#                 (center[0] + 295, center[1] + 10), (255,255,255), -1)
#             cv2.putText(color_image, label, (center[0]-50, center[1]), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
#             cv2.drawContours(color_image,[box],0,(0,0,255),2)

#         images = np.hstack((color_image, depth_image))

#         cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
#         cv2.imshow('Align Example', depth_image)
#         # cv2.waitKey(0)
#         # break

#         # print(depth_image_3d.shape)
#         # plt.imshow(depth_image_3d)
#         # plt.show()

#         key = cv2.waitKey(0)
#         # Press esc or 'q' to close the image window
#         if key & 0xFF == ord('q') or key == 27:
#             # cv2.destroyAllWindows()
#             break
# finally:
#     pipeline.stop()
