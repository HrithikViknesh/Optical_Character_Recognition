import numpy as np
import cv2
from scipy.spatial import distance as dist

def order_points_4(points):
    # Here points would be the unordered corners of the puzzle border
    # Step1) Order the 4 points based on x coordinates
    s_points=points[np.argsort(points[:,0]),:]
    # Step2) Isolate the 2 left most and right most points
    left_most=s_points[:2,:]
    right_most=s_points[2:,:]
    # Step3) Sort the left most points based on y coordinate to get top_left and bottom_left corner
    (tl,bl)=left_most[np.argsort(left_most[:,1]),:]
    # Based on Euclidean distance from Top Left to the Right most points,we will be able to figure out Top Right and Bottom Right
    D=dist.cdist(tl[np.newaxis],right_most,metric="Euclidean")[0]
    (tr,br)=right_most[np.argsort(D),:]
    # Return the ordered points in the order of (Top Left,Top Right,Bottom Right,Bottom Left)
    return np.array([tl,tr,br,bl],dtype="float32")

def birds_eye_view(image,points):
    corners=order_points_4(points)
    (tl,tr,br,bl)=corners
    # Width of the birds eye view is either Dist b/w tl and bl   or    b/w   tr and br
    width1=dist.cdist(br[np.newaxis],bl[np.newaxis])[0,0]
    width2=dist.cdist(tr[np.newaxis],tl[np.newaxis])[0,0]
    max_width=max(int(width1),int(width2))
    # Similarly calculating height of birds eye view image
    height1=dist.cdist(tr[np.newaxis],br[np.newaxis])[0,0]
    height2=dist.cdist(tl[np.newaxis],bl[np.newaxis])[0,0]
    max_height=max(int(height1),int(height2))

    # Now we set the dimensions of the destination image
    dst=np.array([
        [0,0],
        [max_width-1,0],
        [max_width-1,max_height-1],
        [0,max_height-1]],dtype="float32")
    M = cv2.getPerspectiveTransform(corners, dst)

    # Apply perspective transform Matrix obtained to the image
    warped=cv2.warpPerspective(image,M,(max_width,max_height))

    return warped















