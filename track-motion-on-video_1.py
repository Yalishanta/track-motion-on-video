import cv2

# Задаем номер камеры (если их несколько)
cap = cv2.VideoCapture(0)

# Получаем первый кадр для вычисления фонового изображения
ret, frame = cap.read()
background = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
background = cv2.GaussianBlur(background, (21, 21), 0)

# Задаем пороговое значение для выделения движущихся объектов
threshold_value = 25

while True:
    # Получаем текущий кадр
    ret, frame = cap.read()

    # Преобразуем текущий кадр в черно-белое изображение
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Вычитаем фоновое изображение из текущего кадра
    diff = cv2.absdiff(background, gray)

    # Применяем пороговое значение для выделения движущихся объектов
    thresh = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)[1]

    # Избавляемся от шума на изображении
    thresh = cv2.dilate(thresh, None, iterations=2)
    thresh = cv2.erode(thresh, None, iterations=2)

    # Находим контуры объектов на изображении
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Создаем прямоугольник вокруг каждого контура
    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Отображаем изображение
    cv2.imshow('frame', frame)

    # Обновляем фоновое изображение каждые 50 кадров
    if cap.get(cv2.CAP_PROP_POS_FRAMES) % 50 == 0:
        background = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        background = cv2.GaussianBlur(background, (21, 21), 0)

    # Если нажата клавиша 'q' - выходим из цикла
    if cv2.waitKey(1) == ord('q'):
        break

# Освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()