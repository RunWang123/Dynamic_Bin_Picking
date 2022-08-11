import cv2
import numpy as np
import matplotlib.pyplot as plt


image_file = np.load('color_image_2.npz')
color_image = image_file['rgb']
depth_image = image_file['depth']


def object_detection(color_image, depth_image):


    bw = cv2.adaptiveThreshold(depth_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,43,2)
    # bw = cv2.Canny(depth_image,100,200)
    # Find all the contours in the thresholded image
    bw = ~bw
    print(bw.shape)
    # bw = np.pad(bw, ((10,10), (10,10)))
    # print(bw.shape)
    kernel = np.ones((5,5),np.uint8)
    # bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    # bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

    bw = cv2.dilate(bw,kernel,iterations = 1)
    bw = cv2.erode(bw,kernel,iterations = 1)

    contours, _ = cv2.findContours(bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    list_area = []

    for i, c in enumerate(contours):

        # Calculate the area of each contour
        area = cv2.contourArea(c)
        list_area.append(area)
        # Ignore contours that are too small or too large
        if area < 2000:
            continue
        
        # cv.minAreaRect returns:
        # (center(x, y), (width, height), angle of rotation) = cv2.minAreaRect(c)
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # Retrieve the key parameters of the rotated bounding box
        center = (int(rect[0][0]),int(rect[0][1])) 
        width = int(rect[1][0])
        height = int(rect[1][1])
        angle = int(rect[2])
        
            
        if width < height:
            angle = 90 - angle
        else:
            angle = -angle
                
        # label = "  Rotation Angle: " + str(angle) + " degrees"
        label = "Area: {}".format(area)
        # textbox = cv2.rectangle(color_image, (center[0]-35, center[1]-25), 
        # (center[0] + 295, center[1] + 10), (255,255,255), -1)
        # cv2.putText(color_image, label, (center[0]-50, center[1]), 
        #     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
        # cv2.drawContours(color_image,[box],0,(0,0,255),2)
        # c += np.array((-10, 10)).reshape(1,2)
    list_area = np.array(list_area)
    id = np.argmax(list_area)


    # mask
    mask = np.zeros(color_image.shape, np.uint8)
    cv2.drawContours(mask, contours,id,(255,255,255),-1)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    mask = cv2.erode(mask,kernel,iterations = 6)
    mask = cv2.dilate(mask,kernel,iterations = 4)
    color_res = cv2.bitwise_and(color_image,color_image,mask = mask)


    # mask on depth image
    depth_mask = cv2.bitwise_and(depth_image,depth_image,mask = mask)
    bw = cv2.adaptiveThreshold(depth_mask,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,43,2)
    contours, hier = cv2.findContours(bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    bw_3d = np.dstack((bw,bw,bw))
    # print(hier)
    count = 0

    for i, c in enumerate(contours):

        # Calculate the area of each contour
        area = cv2.contourArea(c)
        # Ignore contours that are too small or too large
        if area < 2000 or area > 1e5 or hier[0][i,-1] != 0:
            continue
        
        print(i, hier[0][i])

        # cv.minAreaRect returns:
        # (center(x, y), (width, height), angle of rotation) = cv2.minAreaRect(c)
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # Retrieve the key parameters of the rotated bounding box
        center = (int(rect[0][0]),int(rect[0][1])) 
        width = int(rect[1][0])
        height = int(rect[1][1])
        angle = int(rect[2])
        
            
        if width < height:
            angle = 90 - angle
        else:
            angle = -angle
                
        # label = "  Rotation Angle: " + str(angle) + " degrees"
        label = "id: {}".format(i)
        # textbox = cv2.rectangle(color_res, (center[0]-35, center[1]-25), 
        # (center[0] + 295, center[1] + 10), (255,255,255), -1)
        # cv2.putText(color_res, label, (center[0]-50, center[1]), 
        #     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
        cv2.drawContours(color_res,[box],0,(0,0,255),2)
        cv2.drawContours(bw_3d,[box],0,(0,0,255),2)
        # c += np.array((-10, 10)).reshape(1,2)
        
        count += 1

        # if count > 1:
        #     break

    # bw = bw[10:-10, 10:-10]

    # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    # images = np.hstack((color_image, bw_3d, depth_image_output_3d))
    images = np.hstack((color_res, bw_3d))
    return images

images = object_detection(color_image, depth_image)

cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
cv2.imshow('Align Example', images)
key = cv2.waitKey(0)
# Press esc or 'q' to close the image window
# if key & 0xFF == ord('q') or key == 27:
cv2.destroyAllWindows()
# cv2.imwrite('./output/bg_res.jpg', color_res)
# cv2.imwrite('./output/thr_res.jpg', bw_3d)
            







# img = depth_img.copy()
# ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


# plt.imshow(depth_img)
# plt.show()

# plt.imshow(color_img)
# plt.show()