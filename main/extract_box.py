import easyocr
import cv2
import matplotlib.pyplot as plt
from PIL import ImageDraw, Image

reader = easyocr.Reader(['ko'], gpu = False) # EasyOCR 모델 불러오기
result = reader.readtext('./test_image/na.png', text_threshold=0.7) # detection 좌표 리스트 추출
img = cv2.imread('./test_image/na.png', cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # RGB array로 변환
img = Image.fromarray(img, mode="RGB") # NumPy array to PIL image

draw = ImageDraw.Draw(img)
for i in result :
    x = i[0][0][0] 
    y = i[0][0][1]
    w = i[0][1][0] - i[0][0][0] 
    h = i[0][2][1] - i[0][1][1]

    # x,y : 좌상단 좌표
    # w,h : 우하단 좌표
    draw.rectangle((x, y, x+w, y+h), outline=(255,0,0), width=2) # outline : 색상 , width : 박스 두께

plt.figure(figsize=(50,50))
plt.imshow(img)
plt.show()