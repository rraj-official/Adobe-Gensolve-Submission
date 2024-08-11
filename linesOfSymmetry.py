import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob

sift = cv2.SIFT_create()
bf = cv2.BFMatcher()

def lineOfSymmetryDetection(pictureName: str, title: str, showDetail=False):
    mirror = detectingMirrorSymmetry(pictureName)
    matchpoints = mirror.findMatchpoints()
    pointsR, pointsTheta = mirror.findPointsRTheta(matchpoints)
    if showDetail: 
        mirror.drawMatches(matchpoints, top=10)
        mirror.drawHex(pointsR, pointsTheta)
    imageHexbin = plt.hexbin(pointsR, pointsTheta, bins=200, cmap=plt.cm.Spectral_r) 
    sortedVote = mirror.sortHexbinByVotes(imageHexbin)
    r, theta = mirror.findCoordinateMaxhexbin(imageHexbin, sortedVote, vertical=False)  
    mirror.drawMirrorLine(r, theta, title)

def casesBeingTested(filesPath):
    files = sorted([f for f in glob.glob(filesPath)])
    for file in files:
        lineOfSymmetryDetection(file, "We found the line of symmetry")

class detectingMirrorSymmetry:
    def __init__(self, imagePath: str):
        self.image = self._readColorImage(imagePath) 
        self.reflectedImage = np.fliplr(self.image) 
        self.kp1, self.des1 = sift.detectAndCompute(self.image, None) 
        self.kp2, self.des2 = sift.detectAndCompute(self.reflectedImage, None)

    def _readColorImage(self, imagePath):
        image = cv2.imread(imagePath)
        if image is None:
            raise ValueError(f"Error: Unable to load image at {imagePath}")
        if len(image.shape) == 3:
            b, g, r = cv2.split(image)
            image = cv2.merge([r, g, b])
        elif len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            raise ValueError("Unexpected image format.")
        return image

    def findMatchpoints(self):
        matches = bf.knnMatch(self.des1, self.des2, k=2)
        matchpoints = [item[0] for item in matches] 
        matchpoints = sorted(matchpoints, key=lambda x: x.distance) 
        return matchpoints

    def findPointsRTheta(self, matchpoints:list):
        pointsR = [] 
        pointsTheta = [] 
        for match in matchpoints:
            point = self.kp1[match.queryIdx] 
            mirpoint = self.kp2[match.trainIdx]
            mirpoint.angle = np.deg2rad(mirpoint.angle)
            mirpoint.angle = np.pi - mirpoint.angle
            if mirpoint.angle < 0.0:   
                mirpoint.angle += 2*np.pi
            mirpoint.pt = (self.reflectedImage.shape[1]-mirpoint.pt[0], mirpoint.pt[1]) 
            theta = xAxis(point.pt, mirpoint.pt)  
            xc, yc = midpoint(point.pt, mirpoint.pt) 
            r = xc*np.cos(theta) + yc*np.sin(theta)  
            pointsR.append(r)
            pointsTheta.append(theta)
        return pointsR, pointsTheta 

    def drawMatches(self, matchpoints, top=10):
        img = cv2.drawMatches(self.image, self.kp1, self.reflectedImage, self.kp2, 
                               matchpoints[:top], None, flags=2) 
        plt.imshow(img)
        plt.title("Top {} pairs of symmetry points".format(top))

    def drawHex(self, pointsR: list, pointsTheta: list):
        imageHexbin = plt.hexbin(pointsR, pointsTheta, bins=200, cmap=plt.cm.Spectral_r) 
        plt.colorbar()

    def findCoordinateMaxhexbin(self, imageHexbin, sortedVote, vertical):
        for k, v in sortedVote.items():
            if vertical:
                return k[0], k[1]
            else:
                if k[1] == 0 or k[1] == np.pi:
                    continue
                else:
                    return k[0], k[1]

    def sortHexbinByVotes(self, imageHexbin):
        counts = imageHexbin.get_array()
        ncnts = np.count_nonzero(np.power(10, counts)) 
        verts = imageHexbin.get_offsets() 
        output = {}
        for offc in range(verts.shape[0]):
            binx, biny = verts[offc][0], verts[offc][1]
            if counts[offc]:
                output[(binx, biny)] = counts[offc]
        return {k: v for k, v in sorted(output.items(), key=lambda item: item[1], reverse=True)}

    def drawMirrorLine(self, r, theta, title: str): 
        for y in range(len(self.image)): 
            try:
                x = int((r - y * np.sin(theta)) / np.cos(theta))
                self.image[y][x] = 255
                self.image[y][x + 1] = 255 
            except IndexError:
                continue
        plt.imshow(self.image)
        plt.axis('off') 
        plt.title(title)
        plt.show()  # This will use matplotlib to show the image

        # Convert the image from RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
        cv2.imshow("Symmetry Detection", image_bgr)  # Show the image in a popup window
        cv2.waitKey(0)  # Wait until a key is pressed
        cv2.destroyAllWindows()  # Close the window


def xAxis(pi, pj): 
    x, y = pi[0] - pj[0], pi[1] - pj[1] 
    if x == 0:
        return np.pi / 2  
    angle = np.arctan(y / x)
    if angle < 0:
        angle += np.pi
    return angle

def midpoint(pi, pj):
    return (pi[0] + pj[0]) / 2, (pi[1] + pj[1]) / 2

lineOfSymmetryDetection("input/symmetry.png", 'We found the line of symmetry', showDetail=False)
