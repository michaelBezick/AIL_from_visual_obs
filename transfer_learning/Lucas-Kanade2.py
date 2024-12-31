import numpy as np 
import cv2 

img1 = cv2.imread("../../Object-Tracking/Inputs/basketball1.png")

# img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

img2 = cv2.imread("../../Object-Tracking/Inputs/basketball2.png")
# img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# params for corner detection 
feature_params = dict( maxCorners = 100, 
					qualityLevel = 0.3, 
					minDistance = 7, 
					blockSize = 7 ) 

# Parameters for lucas kanade optical flow 
lk_params = dict( winSize = (15, 15), 
				maxLevel = 2, 
				criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
							10, 0.03)) 

# Create some random colors 
color = np.random.randint(0, 255, (100, 3)) 

# Take first frame and find corners in it 
# ret, old_frame = cap.read() 

old_frame=img1

old_gray = cv2.cvtColor(old_frame, 
						cv2.COLOR_BGR2GRAY) 

p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, 
							**feature_params) 

# Create a mask image for drawing purposes 
mask = np.zeros_like(old_frame) 

# ret, frame = cap.read() 
frame=img2

frame_gray = cv2.cvtColor(frame, 
                        cv2.COLOR_BGR2GRAY) 

# calculate optical flow 
p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, 
                                    frame_gray, 
                                    p0, None, 
                                    **lk_params) 

# Select good points 
good_new = p1[st == 1] 
good_old = p0[st == 1] 

# draw the tracks 
for i, (new, old) in enumerate(zip(good_new, 
                                good_old)): 
    a, b = new.ravel() 
    c, d = old.ravel() 

    a = int(a)
    b = int(b)
    c = int(c)
    d = int(d)

    mask = cv2.line(mask, (a, b), (c, d), 
                    tuple(color[i].tolist()), 2) 
    
    frame = cv2.circle(frame, (a, b), 5, 
                    tuple(color[i].tolist()), -1) 
    
img = cv2.add(frame, mask) 

cv2.imshow('frame', img) 

k = cv2.waitKey(25) 

# Updating Previous frame and points 
old_gray = frame_gray.copy() 
p0 = good_new.reshape(-1, 1, 2) 


cv2.waitKey(0)

cv2.destroyAllWindows() 
