import cv2
import numpy as np

class OrientationClassifier:
    def __init__(self) -> None:
        pass

    def angle_from_line(self, x1, y1, x2, y2) -> float:
        z = np.complex(x2 - x1, y2 - y1)
        return np.angle(z, deg=True)

    def passes_through_img(self, x1, y1, x2, y2) -> bool:
        above = False
        below = False
        #if the line is vertical
        if x2 == x1:
            #return true if line is in center area
            if x1 > 30 and x1 < 70:
                return True
            else:
                return False
        m = (y2 - y1) / (x2 - x1)
        #check if there exist points in the region that are above and below the line
        for i in range(0, 33):
            x = 33 + i
            y = m * (x - x1) + y1
            if y > x:
                above = True
            elif y < x:
                below = True
            if above and below:
                return True
        return False

    def get_orientation(self, img: cv2.Mat) -> float:
        '''
        Gets the orientation from a focused head-on image of the screwdriver head.
        If unable to find orientation, returns a dummy value (400.0).
        @param img: Focused head-on image of the screwdriver head.
        '''
        img = cv2.resize(img, (100, 100))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(gray, 100, 100)
        lines = cv2.HoughLinesP(canny, 1, np.pi/180, 30, minLineLength=30, maxLineGap=10)
        angles = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                #if self.passes_through_img(x1, y1, x2, y2):
                #cv2.line(img, (x1, y1), (x2, y2), (0,0,255))
                angles.append(self.angle_from_line(x1, y1, x2, y2))
            """
            cv2.imshow("F", img)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()
            """
        else:
            return 400.0
        return np.average(np.array(angles))

    