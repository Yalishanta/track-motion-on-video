import cv2

# Задаем номер камеры (если их несколько)
cap = cv2.VideoCapture(0)

# Получаем первый кадр для вычисления оптического потока
ret, frame = cap.read()
prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

# Задаем максимальную длину вектора оптического потока
max_len = 50

while True:
    # Получаем текущий кадр
    ret, frame = cap.read()

    # Преобразуем текущий кадр в черно-белое изображение
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Вычисляем оптический поток
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Создаем прямоугольники вокруг движущихся объектов
    for i in range(0, flow.shape[0], 10):
        for j in range(0, flow.shape[1], 10):
            dx, dy = flow[i, j]
            if dx ** 2 + dy ** 2 > max_len ** 2:
                cv2.rectangle(frame, (j - 5, i - 5), (j + 5, i + 5), (0, 255, 0), 2)

    # Отображаем изображение
    cv2.imshow('frame', frame)

    # Обновляем предыдущее изображение
    prev_gray = gray

    # Если нажата клавиша 'q' - выходим из цикла
    if cv2.waitKey(1) == ord('q'):
        break

# Освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()