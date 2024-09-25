from dv import NetworkFrameInput
import cv2

with NetworkFrameInput(address='127.0.0.1', port=53067) as i:
    for frame in i:
        print(frame)
        cv2.imshow('out', frame.image)
        cv2.waitKey(1)



