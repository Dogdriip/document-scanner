import numpy as np
import cv2
import imutils


def get_edged(image):
    # Convert RGB image to Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Gaussian Blur for removing high frequency noise
    # Also for Canny edge detection
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # Canny edge detection alg.
    edged = cv2.Canny(gray, 75, 200)
    return edged


def get_document_contour(edged):
    # Find the document area.
    # Find the contours in the edged image.
    # Contour: 폐곡선
    # findContours()는 원본 이미지를 변경시키므로 이미지 copy()로 작업 필요
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST,
                            cv2.CHAIN_APPROX_SIMPLE)

    # OpenCV 2.4, OpenCV 3, OpenCV 4 return contours differently. Cover it with imutils.
    cnts = imutils.grab_contours(cnts)

    # contourArea 기준 큰 거부터 5개 가져오기
    # 이후 for문에서 큰 Contour부터 고려함. 밖에서부터 안으로 줄여나간다고 생각
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    for c in cnts:
        # Contour approximation
        # approxPolyDP(): 인자로 주어진 곡선을 epslion값에 따라 꼭지점 수를 줄여 새로운 곡선을 만들어 반환
        epsilon = 0.02 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)

        # Approximation한 Contour가 4개 점을 가진다면 (사각형이라면), approximation 성공으로 간주
        if len(approx) == 4:
            screenCnt = approx
            break

    return screenCnt


if __name__ == '__main__':
    image = cv2.imread('./business_card.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    ratio = image.shape[0] / 500.0
    orig = image.copy()
    # resize image to have a height of 500px
    image = imutils.resize(image, height=500)

    # Get edged image
    edged = get_edged(image)

    # Get document contour
    documentCnt = get_document_contour(edged)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.drawContours(image, [documentCnt], -1, (0, 255, 0), 2)
    cv2.imshow("result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
