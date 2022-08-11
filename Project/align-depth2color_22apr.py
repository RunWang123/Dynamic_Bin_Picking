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
import copy
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
#Global Variable and Function
background_rm = None
numConsecutiveAverage = 5
counter = 0
pick = []
pick2 = []
depth_length_factor = 0
# Create a pipeline
def mousePoints(event,x,y,flags,params):
    global counter
    # Left button click
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Click button.\n")
        pick.append([x,y])
        counter = counter + 1

def mousePoints2(event,x,y,flags,params):
    global counter
    # Left button click
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Click button.\n")
        pick2.append([x,y])
        counter = counter + 1


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


# # Click Four points to remove background
try:
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
    color_frame = aligned_frames.get_color_frame()
    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data()) 
    while True:
        if counter == 4:
            points = np.array(pick)
            points = points.reshape((-1, 1, 2))

            # Attributes
            isClosed = True
            color = (255, 0, 0)
            thickness = 2

            # draw closed polyline
            # color_image = cv2.polylines(color_image, [points], isClosed, color, thickness)
            # cv2.fillPoly(color_image, pts = [points], color =(255,255,255))
            background_rm = np.zeros_like(color_image, np.uint8)
            cv2.fillPoly(background_rm, pts = [points], color =(255,255,255))
            gray = cv2.cvtColor(background_rm, cv2.COLOR_BGR2GRAY)
            # Convert image to binary
            _, background_rm_bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            background_mask = background_rm_bw > 1
            color_image_rmbg = np.zeros_like(color_image, np.uint8)
            color_image_rmbg[background_mask] = color_image[background_mask]
            cv2.namedWindow('Color Image without background ', cv2.WINDOW_NORMAL)
            cv2.imshow("Color Image without background ", color_image_rmbg)
        for i in range(len(pick)):
            col = (20, 0, 255)
            color_image = cv2.circle(color_image, pick[i], 5, col, 2)
            color_image = cv2.putText(color_image, str(i), (pick[i][0]+10, pick[i][1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX,1, col, 1)
        # Showing original image
        cv2.namedWindow('Original Image ', cv2.WINDOW_NORMAL)
        cv2.imshow("Original Image ", color_image)
        # Mouse click event on original image
        cv2.setMouseCallback("Original Image ", mousePoints)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    print("Fail to crop image.")

print(pick)








counter = 0
# Use the reference object to compute the dimension of object
try:
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
    color_frame = aligned_frames.get_color_frame()
    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data()) 
    print(depth_image.shape)
    print(color_image.shape)
    while True:
        if counter == 2:
            points = np.array(pick2)
            # Green color in BGR
            color = (0, 255, 0)
            thickness = 9
            color_image = cv2.line(color_image, (points[0,0], points[0,1]), (points[1,0], points[1,1]), color, thickness)
            mid_point_x = (points[0,0] + points[1,0]) // 2
            mid_point_y = (points[0,1] + points[1,1]) // 2
            color = (0, 0, 255)
            color_image = cv2.circle(color_image, (mid_point_x,mid_point_y), 5, color, 2)
            pixel_distance = dist = np.linalg.norm(points[0] - points[1])
            depth_mid_point = depth_image[mid_point_y, mid_point_x]
            depth_length_factor = 12.0 / (depth_mid_point * 0.00025 * pixel_distance)
        for i in range(len(pick2)):
            col = (0, 0, 255)
            color_image = cv2.circle(color_image, pick2[i], 5, col, 2)
            color_image = cv2.putText(color_image, str(i), (pick2[i][0]+10, pick2[i][1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX,1, col, 1)
        # Showing original image
        cv2.namedWindow('Original Image ', cv2.WINDOW_NORMAL)
        cv2.imshow("Original Image ", color_image)
        # Mouse click event on original image
        cv2.setMouseCallback("Original Image ", mousePoints2)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    print("Fail to crop image.")

print(depth_length_factor)










display_flag = 0



count = 0
prev_frames = []
background_frames = []
background_flag = False
background_num = 0
background = None
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

        color_image_clean = color_image.copy()
        # Remove background - Set pixels further than clipping_distance to grey
        # grey_color = 153
        # depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        # bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

        # Render images:
        #   depth align to color on left
        #   depth on right
        # depth_image_output = cv2.convertScaleAbs(depth_image, alpha=0.05)
        tmp = depth_image.copy()
        if count < numConsecutiveAverage:
            prev_frames.append(tmp)
            count += 1 
            continue
        # print(len(prev_frames))
        # print(background_flag)
        if (count >= numConsecutiveAverage):
            prev_frames.append(tmp)
            if background_flag:
                average_frame = np.mean(prev_frames, axis=0)
                diff = cv2.absdiff(background, average_frame)
                th = 200
                canvas = np.zeros_like(depth_image, float)
                color_diff = np.zeros_like(color_image, np.uint8)
                imask =  diff>th 
                canvas[imask] = depth_image[imask]
                color_diff[imask] = color_image[imask]
                cv2.namedWindow('Difference', cv2.WINDOW_NORMAL)
                cv2.imshow('Difference', color_diff)
                tmp = canvas
            else:
                helper = np.zeros_like(depth_image, float)
                helper[background_mask] = (np.mean(prev_frames, axis=0))[background_mask]
                tmp = helper
            depth_image_output = cv2.convertScaleAbs(tmp, alpha=0.05)
            # plot histogram
            # hist = cv2.calcHist([depth_image_output],[0],None,[256],[0,256])
            # plt.hist(depth_image_output.ravel(),256,[0,256])
            # plt.title('Histogram for gray scale picture')
            # plt.show()
            prev_frames.pop(0)
            depth_image_output_3d = np.dstack((depth_image_output,depth_image_output,depth_image_output))
            # start of orientation
            img = copy.deepcopy(depth_image_output_3d)
            # Was the image there?
            if img is None:
                print("Error: File not found")
                exit(0)
            # Convert image to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Convert image to binary
            # _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            bw = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,31,2)
            # Find all the contours in the thresholded image
            bw = ~bw
            kernel = np.ones((3,3), np.uint8)
            bw = cv2.erode(bw, kernel, iterations=1)
            contours, hierarchy = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            
            

            list_area = []

            for i, c in enumerate(contours):
            
                # Calculate the area of each contour
                area = cv2.contourArea(c)
                
                # Ignore contours that are too small or too large
                if area < 2000  or not(hierarchy[0,i,3] == -1) or area > 75000:
                    continue
                
                # cv.minAreaRect returns:
                # (center(x, y), (width, height), angle of rotation) = cv2.minAreaRect(c)
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                
                # Retrieve the key parameters of the rotated bounding box
                center = (int(rect[0][0]),int(rect[0][1])) 
                color_image = cv2.circle(color_image, center, 5, color, 2)
                width = int(rect[1][0])
                height = int(rect[1][1])
                angle = int(rect[2])
                center_depth = depth_image[int(rect[0][1]),int(rect[0][0])]
                actual_width = rect[1][0] * center_depth * 0.00025 * depth_length_factor
                actual_length = rect[1][1] * center_depth * 0.25 * depth_length_factor
                if width < height:
                    angle = 90 - angle
                else:
                    angle = -angle
                        
                if display_flag == 0:
                    label = "  Rotation Angle: " + str(angle) + " degrees"
                elif display_flag == 1:
                    label = "  Dim: {:.2f} cm * {:.2f} cm".format(actual_width, actual_length)
                elif display_flag == 2:
                    label = "  Depth: {:.2f} cm".format(center_depth * 0.25)
                textbox = cv2.rectangle(color_image, (center[0]-35, center[1]-25), 
                    (center[0] + 295, center[1] + 10), (255,255,255), -1)
                cv2.putText(color_image, label, (center[0]-50, center[1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
                cv2.drawContours(color_image,[box],0,(0,0,255),2)
            bw_3d = np.dstack((bw,bw,bw))
            # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            images = np.hstack((color_image, bw_3d, depth_image_output_3d))

            # np.savez("color_image.npz", rgb=color_image_clean, depth=depth_image_output)

            cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
            cv2.imshow('Output', images)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
            elif ((key & 0xFF == ord('s')) and (not background_flag)):
                print(count)
                print(background_flag)
                if (background_num < numConsecutiveAverage):
                    # background = gaussian_filter(depth_image.copy(), sigma=5)
                    background_num += 1
                    background_frames.append(depth_image)
                elif (background_num >= numConsecutiveAverage):
                    background = np.mean(background_frames, axis = 0)
                    cv2.namedWindow('Background Image', cv2.WINDOW_NORMAL)
                    cv2.imshow('Background Image', background)
                    background_flag = True
            elif (key & 0xFF == ord('o')):
                print("Save npx image.")
                np.savez("color_image.npz", rgb=color_image_clean, depth=depth_image_output)
            elif (key & 0xFF == ord('c')):
                if display_flag < 2:
                    display_flag += 1
                else:
                    display_flag = 0

finally:
    pipeline.stop()