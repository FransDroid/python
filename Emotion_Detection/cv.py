import cv2
 
def test():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
    if (cap.isOpened() == False):
        exit()
    while (cap.isOpened() == True):
        ret, frame = cap.read()
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
 
if __name__ == "__main__":
    test()