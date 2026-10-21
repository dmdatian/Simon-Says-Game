import cv2
import numpy as np
from tensorflow.keras.models import load_model
import random
import time
import sys

# Load pre-trained model
model = load_model("emotion_model_grayscale_t4.h5")

# Expression labels
expressions = ['Angry', 'Disgust', 'Fear','Happy', 'Neutral', 'Sad', 'Surprise']

# Open webcam
cap = cv2.VideoCapture(0)

# Load face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def wait_for_start():
    print("📷 Camera is ready!")
    print("👉 Press ENTER to start the game or Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.resize(frame, (640, 480))

        text = "Press ENTER to Start | Q to Quit"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = (frame.shape[0] + text_size[1]) // 2

        cv2.putText(frame, text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Face Simon Says", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 13:
            print("🎮 Game Starting...\n")
            break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            sys.exit()

# Tips
coaching_tips = {
    'Happy': "BIG SMILE! Show teeth!",
    'Sad': "Frown deeply!",
    'Surprise': "Wide eyes + open mouth!",
    'Angry': "Furrow brows!",
    'Fear': "Eyes wide!",
    'Disgust': "Wrinkle nose!",
    'Neutral': "Relax face"
}

# Countdown TImer
def countdown(target_emotion):
    tip_text = coaching_tips[target_emotion]

    for i in range(3, 0, -1):
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.resize(frame, (640, 480))

        tip_size = cv2.getTextSize(tip_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        tip_x = (frame.shape[1] - tip_size[0]) // 2

        cv2.putText(frame, tip_text, (tip_x, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        text = str(i)
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 3, 4)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = (frame.shape[0] + text_size[1]) // 2

        cv2.putText(frame, text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 4)

        cv2.imshow("Get Ready!", frame)

        if cv2.waitKey(1000) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            sys.exit()

# Capture Expression (Last Frame Logic)
def get_expression_with_timer(time_limit=12):
    start_time = time.time()
    last_prediction = "Neutral"

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (48, 48))
            face_img = face_img.astype("float32") / 255.0
            face_img = np.expand_dims(face_img, axis=0)
            face_img = np.expand_dims(face_img, axis=-1)

            prediction = model.predict(face_img, verbose=0)[0]
            expression_index = np.argmax(prediction)
            confidence = prediction[expression_index]

            last_prediction = expressions[expression_index]

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            display_text = f"{last_prediction} ({confidence:.2f})"
            cv2.putText(frame, display_text, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        remaining = int(time_limit - (time.time() - start_time))
        cv2.putText(frame, f"Time: {remaining}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        final_text = f"{last_prediction}"
        text_size = cv2.getTextSize(final_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2

        cv2.putText(frame, final_text, (text_x, 440),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

        cv2.imshow("Show Expression", frame)

        if time.time() - start_time > time_limit:
            return last_prediction

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            sys.exit()

# Game Play Loop
def play_game():
    round_num = 1
    score = 0

    print("🎮 Welcome to Face Simon Says!")
    wait_for_start()

    try:
        while True:
            target_emotion = random.choice(expressions)

            print(f"\nRound {round_num}: Show -> {target_emotion}")

            countdown(target_emotion)
            player_exp = get_expression_with_timer()

            print(f"You showed: {player_exp} | Target: {target_emotion}")

            if player_exp != target_emotion:
                print("❌ Wrong! Game Over.")
                break
            else:
                print("✅ Correct!")
                score += 1
                round_num += 1

        print(f"🏆 Final Score: {score}")

    finally:
        cap.release()
        cv2.destroyAllWindows()

# Main Loop
while True:
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    play_game()

    while True:
        choice = input("\n🎮 Play again? (Y/N): ").strip().lower()

        if choice == 'y':
            break
        elif choice == 'n':
            print("Thanks for playing! 👋")
            sys.exit()
        else:
            print("❌ Invalid input. Please enter Y or N only.")
