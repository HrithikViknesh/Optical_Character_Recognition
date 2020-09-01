from utilities.transform import birds_eye_view
import utilities
from utilities.resizer import resize
import sklearn
import cv2
import skimage
from skimage.segmentation import clear_border
import numpy as np

def find_puzzle(image,debug=True):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blurred=cv2.GaussianBlur(gray,(7,7),3)
    #Applying thresholding(adaptive)
    thresh=cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,blockSize=11,C=2)
    # Invert the thresholded image
    cv2.dilate(thresh, np.ones(shape=(3, 3), dtype="uint8"))
    thresh=cv2.bitwise_not(thresh)

    #show intermediate product
    if debug==True:
        cv2.imshow("thresholded",thresh)
        cv2.waitKey(0)
    # Find the contours
    contours,hierarchy=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours=sorted(contours,key=cv2.contourArea,reverse=True)

    # SO among these contours is the boundary (puzzle) contour
    puzzle_cont=None

    for cont in contours:
        peri = cv2.arcLength(cont,True)
        approx = cv2.approxPolyDP(cont,0.02*peri,True)
        # check for boundary contour which will have exactly four vertices
        if len(approx) == 4:
            puzzle_cont=approx
            break
    # if unable to detect then raise exception
    if puzzle_cont is None:
        raise Exception("Could not detect the puzzle outline.Try adjusting the thresholding procedure\n")
    if debug:
        output=image.copy()
        cv2.drawContours(output,[puzzle_cont],-1,(0,255,0),2)
        cv2.imshow("puzzle outline",output)
        cv2.waitKey(0)
    # Now we shall pass both the image and its gray version along with the detected
    # contours to the fourpoint transform
    puzzle=birds_eye_view(image,puzzle_cont.reshape(4,2))
    warped=birds_eye_view(gray,puzzle_cont.reshape(4,2))

    if debug:
        cv2.imshow("Puzzle Transform",puzzle)
        cv2.waitKey(0)

    return (puzzle,warped)

# Through the above fn we get a birds eye like view of the image and its grayscale version

# Now we shall define another funtion that performs the following task
# Given a cell (which may or may not contain a digit) , the function must check if there is a
# digit present inside that cell,remove borders of the cell and return the digit image ready to be fed to our model

def extract_digit(cell,debug=True):
    # SUppose we get a white cell with the digit in black
    # We need to conv it to white digit and black background(like mnist digits)
    ret,thresh=cv2.threshold(cell,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # CLear the cell border (unnecessary)
    thresh=clear_border(thresh)

    if debug:
        cv2.imshow("Cell Thresh",thresh)
        cv2.waitKey(0)
    # Get ext contours from the cell image
    conts,hier=cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    # CHeck for empty cell(no external contours)
    if len(conts)==0:
        return None
    # If not empty find the largest contour in the cell and create a mask for it
    c=max(conts,key=cv2.contourArea)
    # mask is actually a black background with white drawn contour
    mask=np.zeros(thresh.shape,dtype="uint8")
    cv2.drawContours(mask,[c],-1,255,-1)

    # also check that the obtained contour was not that of noise in the cell
    # if the contour area is very less, then it can be confirmed it belonged to a noise
    (h,w)=thresh.shape
    # Checking the percent of white pixels(digit pixels) in the mask
    percentFilled=cv2.countNonZero(mask) / float(w*h)

    if percentFilled<0.03:
        #print("Space filled was less than 30 percent")
        return None
    digit=cv2.bitwise_and(thresh,thresh,mask=mask)

    if debug:
        cv2.imshow("Digit",digit)
        cv2.waitKey(0)

    return digit



















