import cv2
import PeopleDetection.PeopleRectDetector as prd

src_gray = cv2.imread("/home/liu/图片/people/people2.jpg", 0)

peopleRectDetector = prd.PeopleRectDetector()
peopleRects = peopleRectDetector.detect(src_gray)

for (x, y, w, h) in peopleRects:
    cv2.rectangle(src_gray, (x, y), (x + w, y + h), 255, 2)

# cv2.namedWindow("PeopleDetection",0)
cv2.imshow("PeopleDetection", src_gray)
cv2.waitKey()
