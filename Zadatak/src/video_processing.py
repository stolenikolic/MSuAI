import cv2
import numpy as np

from src.binary_output import binary_output  # Funkcija za binarnu sliku
from src.birds_eye_perspective import bird_eye  # Funkcija za bird-eye perspektivu
from src.lane_detection import lane_detection  # Funkcija za detekciju linija
from src.back_to_original_img import original_image_lane_detection  # Funkcija za vraćanje na originalnu sliku

def process_video_frame(frame):
    """
    Obrada pojedinačnog frejma iz video snimka.
    """
    # Korak 1: Generisanje binarne slike
    binary_img = binary_output(frame)
    # cv2.imshow("Final Result", binary_img)

    # Korak 2: Bird-eye transformacija
    bird_eye_img = bird_eye(binary_img)

    # Korak 3: Detekcija linija u bird-eye slici
    lanes_img = lane_detection(bird_eye_img)

    # Korak 4: Vraćanje linija na originalnu sliku
    result_img = original_image_lane_detection(frame, bird_eye_img)

    return result_img


def process_video(input_video_path, output_video_path):
    """
    Glavna funkcija za obradu video snimka.
    """
    # Otvaranje video snimka
    cap = cv2.VideoCapture(input_video_path)

    # Provera da li je snimak uspešno otvoren
    if not cap.isOpened():
        print("Greška pri otvaranju video snimka!")
        return

    # Čitanje osnovnih informacija o snimku
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Postavljanje izlaznog video snimka
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Format izlaznog snimka
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Obrada trenutnog frejma
        processed_frame = process_video_frame(frame)

        # Snimanje obrađenog frejma u izlazni video
        out.write(processed_frame)

        # Prikaz trenutnog frejma sa linijama (opciono)
        # cv2.imshow('Lane Detection', processed_frame)
        # cv2.waitKey(1)


    # Oslobađanje resursa
    cap.release()
    out.release()
    cv2.destroyAllWindows()



# Putanje ulaznog i izlaznog videa
input_video_path = '../test_videos/project_video03.mp4'
output_video_path = '../output_videos/project_video_output.avi'

# Obrada video snimka
process_video(input_video_path, output_video_path)