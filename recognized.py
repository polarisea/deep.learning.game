import click, cv2, numpy as np, torch
from khmt import Camera, cropped_frame
from mediapipe.python.solutions import drawing_utils, hands, hands_connections
from torch import Tensor
from torchvision.transforms import Compose, Grayscale, ToTensor, ToPILImage

def getHandArea(hand_landmarks, imageShape):
    left, right, top, down = 2, 0, 2, 0

    for landmark in hand_landmarks.landmark:
        if landmark.x < left:
            left = landmark.x 
        if landmark.x > right:
            right = landmark.x
        if landmark.y < top:
            top = landmark.y
        if landmark.y > down:
            down = landmark.y
    width = right - left
    height = down - top
    center_x = (right + left)/2
    center_y = (down + top)/2
    edge = 0
    if width > height:
        edge = width *imageShape[1]
        left = (center_x - (width/2))*imageShape[1] 
        top  = center_y*imageShape[0] - (width/2)*imageShape[1]
        edge = edge
    else:
        edge = height  * imageShape[0]
        left = center_x*imageShape[1] - (height/2)*imageShape[0]
        top  = (center_y - (height/2))*imageShape[0] 
        edge = edge

    return int(left), int(top), int(edge)



def recognized(act):
    net = torch.jit.load('./gesture.pt') # type: ignore
    net.eval()
    transforms = Compose([ToPILImage(), Grayscale(), ToTensor()])
    with Camera() as cap, hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hand:
        while cap.isOpened():
            success: bool; image: cv2.Mat
            success, image = cap.read()

            if not success:
                print("Ignoring empty camera frame.")
                continue

            image = cv2.flip(image, 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            background = np.ones(image.shape, np.uint8)
            background = 255* background
            handImage = np.ones((32,32,3), np.uint8)
            handImage = 255* handImage

            result = hand.process(image)

            if result.multi_hand_landmarks: # type: ignore
                for hand_landmarks in result.multi_hand_landmarks: # type: ignore
                    ha_x, ha_y, ha_edge = getHandArea(hand_landmarks, image.shape)
                    tn = int(ha_edge/17)
                    drawing_utils.draw_landmarks(
                        background,
                        hand_landmarks,
                        hands_connections.HAND_CONNECTIONS, # type: ignore
                        drawing_utils.DrawingSpec(
                            color=(0,0,0),
                            thickness=tn),
                        drawing_utils.DrawingSpec(
                            color=(0,0,0),
                            thickness=tn)
                    )
                    try:
                        handImage = cv2.resize(background[ha_y -tn :ha_y+ ha_edge +tn, ha_x-tn: ha_x+ha_edge+tn ], (32, 32))
                        cv2.rectangle(background, (ha_x-tn, ha_y-tn), (ha_x+ha_edge+tn, ha_y+ha_edge+tn), (0, 0, 0))
                        tensor = transforms(handImage)
                        assert isinstance(tensor, Tensor)

                        output = net(tensor.unsqueeze(0))
                        probs = torch.nn.functional.softmax(output, 1)
                        score, predicted = torch.max(probs, 1)
                        if score[0] > 0.9:
                            act.value = predicted
                        else:
                            act.value = 0
                    except :
                        pass

            cv2.imshow('camera', background)

            if cv2.waitKey(5) & 0xFF == 115:
                    isCapture = True
            elif  cv2.waitKey(5) & 0xFF == 27:
                    break


