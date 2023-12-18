import face_recognition
import os, sys
import cv2
import numpy as np
import math

def face_confidence(face_distance, face_match_threshold=0.6):
    """
    Yüz eşleşme güvenilirliğini hesaplar.Yüz uzaklığını ve Yüz eşleme eşiğini temel alarak bir yüzün ne kadar güvenilir olduğunu ifade eden bir yüz güvenilirlik yüzdesi döndürür.

    :param face_distance: Yüz uzaklığı
    :param face_match_threshold: Yüz eşleme eşiği (varsayılan değer: 0.6)
    :return: Yüz güvenilirlik yüzdesi
    """
    range_val = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range_val * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'

class FaceRecognition:
    # Sınıfın özellikleri (class attributes)
    face_locations = []  # Algılanan yüz konumları
    face_encodings = []  # Algılanan yüz kodları
    face_names = []      # Algılanan yüz isimleri
    known_face_encodings = []  # Bilinen yüz kodları
    known_face_names = []      # Bilinen yüz isimleri
    process_current_frame = True  # Çerçeve işleme kontrolü

    def __init__(self):
        """
        Yüz tanıma sınıfını başlatır ve bilinen yüzleri kodlar.
        """
        self.encode_faces()

    def encode_faces(self):
        """
        'faces' klasöründeki her bir yüzü kodlar ve bilinen yüz listesine ekler.
        """
        for image in os.listdir('faces'):
            face_image = face_recognition.load_image_file(f'faces/{image}')
            face_encoding = face_recognition.face_encodings(face_image)[0]

            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(image.split('.')[0])

    def run_recognition(self):
        """
        Webcam'den alınan görüntüde yüzleri tanır ve eşleşen yüzleri çerçeve üzerine işaretler.
        """
        video_capture = cv2.VideoCapture(0)

        if not video_capture.isOpened():
            sys.exit('Video source not found...')

        while True:
            # Video kaynağından bir çerçeve al
            ret, frame = video_capture.read()

            # Eğer mevcut çerçeve işlenmeye hazırsa devam et
            if self.process_current_frame:
                # Çerçeveyi küçült
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                small_frame = small_frame[:, :, ::-1]

                # Küçültülen çerçevede yüz konumlarını tespit et
                self.face_locations = face_recognition.face_locations(small_frame)

                # Çerçeveyi numpy dizisine dönüştür
                np_image = np.array(small_frame)

                # Yüz kodlamalarını bul
                self.face_encodings = face_recognition.face_encodings(np_image, self.face_locations)

                # Her bir yüzü tanımla ve çerçeve üzerine işaretle
                self.face_names = []
                for face_encoding in self.face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = 'Unknown'
                    confidence = 'Unknown'

                    # Yüz uzaklıklarını hesapla
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)

                    # Eğer eşleşme varsa, yüzü tanımla ve güvenilirlik hesapla
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = face_confidence(face_distances[best_match_index])

                    # Yüz ismini ve güvenilirlik bilgisini listeye ekle
                    self.face_names.append(f'{name} ({confidence})')

                # Çerçeve üzerine yüz konumlarını ve isimlerini işaretle
                for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4

                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), -1)
                    cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

                # Tanınan yüzleri ve güvenilirlik bilgilerini içeren çerçeve üzerinde göster
                cv2.imshow('Face Recognition', frame)

                # 'q' tuşuna basıldığında döngüyü sonlandır
                if cv2.waitKey(1) == ord('q'):
                    break

        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    # Sınıfın bir örneğini oluşturur ve yüz tanıma işlemini başlatır.
    fr = FaceRecognition()
    fr.run_recognition()
