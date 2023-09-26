import face_recognition
import cv2
import numpy as np
import time

def face_recognition_and_return_last_known_name():
    # 검색 할 샘플 사진을 로드 후 인코딩
    a = face_recognition.load_image_file('a 사진이 저장된 경로를 입력하세요')
    b = face_recognition.load_image_file('b 사진이 저장된 경로를 입력하세요')
    c = face_recognition.load_image_file('c 사진이 저장된 경로를 입력하세요')
    d = face_recognition.load_image_file('d 사진이 저장된 경로를 입력하세요')

    a_face_encoding = face_recognition.face_encodings(a)[0]
    b_face_encoding = face_recognition.face_encodings(b)[0]
    c_face_encoding = face_recognition.face_encodings(c)[0]
    d_face_encoding = face_recognition.face_encodings(d)[0]

    # 검색 할 얼굴 인코딩 및 이름 배열 생성
    known_face_encodings = [
        a_face_encoding, b_face_encoding, c_face_encoding, d_face_encoding
    ]
    known_face_names = [
        'a', 'b', 'c', 'd'
    ]

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # 원하는 너비로 설정
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    last_known_name = "Unknown"  # 마지막으로 인식된 얼굴의 이름을 저장할 변수

    def process_frame(frame):
        nonlocal last_known_name  # 외부 변수를 사용하기 위해 nonlocal 키워드 사용

        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (320, 240))  # 프레임 크기 조정
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                last_known_name = name  # 마지막으로 인식된 얼굴의 이름 업데이트

            if name != "Unknown":
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)

            # 얼굴에 Bounding Box를 그립니다
            cv2.rectangle(frame, (left, top), (right, bottom), color, 1)  # 바운딩 박스 두께 수정

            # 얼굴 하단에 이름 레이블을 그립니다
            text_width, text_height = cv2.getTextSize(name, cv2.FONT_HERSHEY_DUPLEX, 0.8, 1)[0]
            cv2.rectangle(frame, (left, bottom - text_height - 10), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, name, (left + int((right - left - text_width) / 2), bottom - 10),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 1)  # 텍스트 위치 및 폰트 크기 수정

        return frame

    start_time = time.time()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        processed_frame = process_frame(frame)
        cv2.namedWindow('Face Recognition', cv2.WINDOW_NORMAL)  # 출력 창 크기 조절
        cv2.resizeWindow('Face Recognition', 640, 480)  # 출력 창 크기 조절
        cv2.imshow('Face Recognition', processed_frame)

        if time.time() - start_time > 3:  # 3초 후에 종료
            break

        if cv2.waitKey(1) & 0xFF == 27:  # 'Esc' 키를 누르면 종료
            break

    cap.release()
    cv2.destroyAllWindows()

    return last_known_name

# last_known_name = face_recognition_and_return_last_known_name()
# print("마지막으로 인식된 얼굴의 이름:", last_known_name)
