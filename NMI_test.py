import cv2
import numpy as np
from matplotlib import pyplot as plt 
from sklearn.metrics.cluster import adjusted_mutual_info_score

cap = cv2.VideoCapture("input\\cropped.avi")

count = 0
scoreList = []

output = "NMI_test.avi"
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
writer = cv2.VideoWriter(output, fourcc, 1, (1080,606), True)

while(True):
    # Capture frame-by-frame
    ret, frame_large = cap.read()
    #print(count)

    if not ret:
        break

    if not count % 36:

        frame = cv2.resize(frame_large, (frame_large.shape[1]//8, frame_large.shape[0]//8))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        sobelx = cv2.Sobel(frame,cv2.CV_64F,1,0,ksize=5)
        sobely = cv2.Sobel(frame,cv2.CV_64F,0,1,ksize=5)

        out = np.arctan2(sobely, sobelx)
        mags = np.hypot(sobely, sobelx)

        normed = cv2.normalize(mags, None, 0, 100, cv2.NORM_MINMAX)
        mask = cv2.inRange(normed, 50, 100)

        masked = cv2.bitwise_or(out, out, mask = mask)
        print(masked.shape)

        cv2.imshow('masked', masked)

        if count:
            score = adjusted_mutual_info_score(masked.ravel(), prevMasked.ravel())
            scoreList.append(score)

            frame_large = cv2.putText(frame_large, str(score), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1, cv2.LINE_AA)
            writer.write(frame_large)

            print(score)

        # Display the resulting frame
        #cv2.imshow('result', np.abs(out/np.pi))   
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        #prevHist = hist
        prevMasked = masked

    count += 1

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

plt.hist(scoreList, 100, (0.0, 1.0))
plt.show()
