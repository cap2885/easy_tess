import pytesseract
import cv2
from PIL import Image
import easyocr
from pykospacing import Spacing
from tqdm import tqdm

# 이미지 분리 함수
def devide_img(input_img):
    rawimg = cv2.imread(input_img, cv2.IMREAD_COLOR)

    # 이미지 세로size
    all_hght = rawimg.shape[0]
    # 이미지 분할개수 입력
    eter = int(input("이미지 분할 개수 :"))
    # 이미지 간격계산
    interval_hght = int(all_hght/eter)

    img_list = [] # 이미지 리스트 생성
    start = 0

    add_hght = interval_hght 
    
    # 하나씩 자르기
    for i in range(eter) :
        cropimg = rawimg[start : add_hght] # 이미지 자르기
        cv2.imshow('img', cropimg) # 자른 이미지 보기
        cv2.waitKey(0)
        img_list.append(cropimg)
        start += interval_hght # 상단 좌표 증가
        add_hght += interval_hght # 하단 좌표 증가
    return img_list

reader = easyocr.Reader(lang_list = ['ko'], gpu = False) # easyocr 불러오기

# 나눠진 이미지 리스트
dev_list = devide_img("./test_image/na.png") # 대체텍스트 추출할 이미지 경로입력

spacing = Spacing() # 띄어쓰기 보정 함수
f = open("./result/콘텐츠.txt", 'w') # 대체텍스트 결과 파일

# 이미지 하나씩 진행
idx = 0
for file in tqdm(dev_list):
    result_list = reader.detect(img = file) # easyocr의 detection 진행
    detectboxes = result_list[0][0] # bounding-box 좌표 리스트
    file = cv2.cvtColor(file, cv2.COLOR_RGB2BGR) # BGR array로 변경

    for box in detectboxes:
        box[1], box[2] = box[2], box[1] # box 좌표 순서 변경
        rawimg = Image.fromarray(file, mode="RGB") # BGR array to RGB PIL image
        crop = rawimg.crop(tuple(box)) # 해당 box 좌표로 bounding-box 추출
        crop.save(f'./result/na{idx}.jpg') # bounding-box 저장
        idx+= 1
        df = pytesseract.image_to_data(crop, lang='Hangul_tess_best', config='--psm 6 --oem 3', output_type = pytesseract.Output.DATAFRAME) # 신뢰도 필터링을 위해 Dataframe 추출

        if df['conf'][df['conf'] > 0].mean() > 80 : # 신뢰도가 -1이 아닌것들에 대해서 평균 80이상이면 출력
            sentence = pytesseract.image_to_string(crop, lang='Hangul_tess_best', config='--psm 6 --oem 3')
            sentence = sentence.replace(" ", '') # 띄어쓰기 없애기
            print(spacing(sentence))
            f.write(spacing(sentence) + "\n") # 띄어쓰기 보정하여 출력

    print("-" * 80 + "\n")
    f.write("-" * 80 +"\n")      
f.close()
