import cv2


def draw_found_faces(detected, image, color: tuple):
    for (x, y, width, height) in detected:
        cv2.rectangle(
            image,
            (x, y),
            (x + width, y + height),
            color,
            thickness=2
        )


path_to_image = 'test2.jpg'
original_image = cv2.imread(path_to_image)

if original_image is not None:

    image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")

    detected_faces = face_cascade.detectMultiScale(image=image, scaleFactor=1.1, minNeighbors=4)
    detected_profiles = profile_cascade.detectMultiScale(image=image, scaleFactor=1.1, minNeighbors=4)

    profiles_not_faces = [x for x in detected_profiles if x not in detected_faces]

    draw_found_faces(detected_faces, original_image, (0, 255, 0))  # RGB - green
    draw_found_faces(detected_profiles, original_image, (0, 0, 255))  # RGB - red

    cv2.imwrite("result_test2.jpg", original_image)

else:
    print(f'En error occurred while trying to load {path_to_image}')
