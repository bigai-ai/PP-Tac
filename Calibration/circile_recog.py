import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy

BLUR = True
BLUR_KERNELSIZE = (5, 5)
BLUR_SIGMA = 2

def circle_recog(ref_image, tac_image, NUM_POINT=1,BLUR=True, BLUR_KERNELSIZE=(5, 5), BLUR_SIGMA=2):
    
    if BLUR:
        orig_img = cv2.GaussianBlur(ref_image, BLUR_KERNELSIZE, BLUR_SIGMA)
        tac_img = cv2.GaussianBlur(tac_image, BLUR_KERNELSIZE, BLUR_SIGMA)


    orig_img = np.asanyarray(orig_img,dtype=np.uint8)
    tac_img = np.asanyarray(tac_img,dtype=np.uint8)

    delta_img = orig_img - tac_img

    gray_blurred = cv2.GaussianBlur(orig_img, (5, 5), 0)
    circles = cv2.HoughCircles(gray_blurred, 
                            cv2.HOUGH_GRADIENT, 1, 20, 
                            param1=50, param2=30, minRadius=0, maxRadius=0)

    temp = copy.deepcopy(tac_img)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :1]:
            cv2.circle(temp, (i[0], i[1]), 2, (0, 0, 255), 3)

    else:
        # raise error
        raise ValueError('CAN NOT DETECT CIRCLE OF SENSOR')

    SENSOR_CIRCLE = circles[0, 0]
    SENSOR_CENTER = (SENSOR_CIRCLE[0], SENSOR_CIRCLE[1])
    SENSOR_RADIUS = SENSOR_CIRCLE[2]

    mask = np.zeros_like(orig_img)
    cv2.circle(mask, SENSOR_CENTER, SENSOR_RADIUS-5, 255, -1)

    def masked_img(img, mask):
        return cv2.bitwise_and(img, img, mask=mask)

    
    orig_masked = masked_img(orig_img, mask=mask)
    tac_masked = masked_img(tac_img, mask=mask)

    delta_masked = masked_img(delta_img, mask=mask)
    delta_masked[delta_masked>100] = 0 
    delta_masked[delta_masked<22] = 0 

    delta_masked_point = copy.deepcopy(delta_masked)

    circles = cv2.HoughCircles(delta_masked_point, 
                            cv2.HOUGH_GRADIENT, 0.01, 8, 
                            param1=7, param2=7, minRadius=1, maxRadius=9)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        print(circles[0, :NUM_POINT].shape)
        for i in circles[0, :NUM_POINT]:
            cv2.circle(delta_masked_point, (i[0], i[1]), 1, 255, 3)
        
        cv2.circle(delta_masked_point, (SENSOR_CIRCLE[0], SENSOR_CIRCLE[1]), 4, 255, 3)
        plt.imshow(delta_masked_point, cmap='Reds')
        plt.show()
    else:
        raise ValueError('CAN NOT DETECT CIRCLES IN SENSORS')

    return temp, SENSOR_CIRCLE, delta_masked, delta_masked_point, circles

