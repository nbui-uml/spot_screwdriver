import cv2
import numpy as np

class OrientationClassifier:
    def __init__(self) -> None:
        pass


    def angle_from_line(self, x1, y1, x2, y2) -> float:
        z = np.complex(x2 - x1, y2 - y1)
        return np.angle(z)


    def get_orientation(self, img: cv2.Mat) -> float:
        '''
        Gets the orientation from a focused head-on image of the screwdriver head.
        If unable to find orientation, returns a dummy value (4.0).
        
        Parameters
        -----
        img: ndarry
            Focused head-on image of the screwdriver head.

        Returns
        -----
        float
            The orientation of the screwdriver.
        '''
        img = cv2.resize(img, (100, 100))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(gray, 100, 100)
        lines = cv2.HoughLinesP(canny, 1, np.pi/180, 30, minLineLength=30, maxLineGap=10)
        angles = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angles.append(self.angle_from_line(x1, y1, x2, y2))
        else:
            return 4.0
        return np.average(np.array(angles))

    