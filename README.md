Оба примера кода написаны на языке Python с использованием библиотеки OpenCV для обработки видео. 

Первый пример использует алгоритм вычитания фона для обнаружения движения на видео. Алгоритм состоит из следующих шагов:
1. Задается номер камеры и получается первый кадр для вычисления фонового изображения.
2. Фоновое изображение вычисляется путем преобразования первого кадра в черно-белое изображение и применения размытия Гаусса.
3. Для каждого последующего кадра:
    - Преобразуется в черно-белое изображение и применяется размытие Гаусса.
    - Вычитается фоновое изображение из текущего кадра, чтобы выделить движущиеся объекты.
    - Применяется пороговое значение для выделения движущихся объектов.
    - Удаляется шум на изображении с помощью операций расширения и эрозии.
    - Находятся контуры объектов на изображении.
    - Создаются прямоугольники вокруг каждого контура.
4. Изображение с прямоугольниками отображается на экране.
5. Фоновое изображение обновляется каждые 50 кадров.
6. Если нажата клавиша 'q', цикл завершается.

Второй пример использует алгоритм оптического потока для обнаружения движения на видео. Алгоритм состоит из следующих шагов:
1. Задается номер камеры и получается первый кадр для вычисления оптического потока.
2. Первый кадр преобразуется в черно-белое изображение и применяется размытие Гаусса.
3. Для каждого последующего кадра:
    - Преобразуется в черно-белое изображение и применяется размытие Гаусса.
    - Вычисляется оптический поток между текущим и предыдущим кадрами.
    - Создаются прямоугольники вокруг каждого вектора оптического потока, чья длина превышает заданное значение.
4. Изображение с прямоугольниками отображается на экране.
5. Предыдущее изображение обновляется на текущее.
6. Если нажата клавиша 'q', цикл завершается.# track-motion-on-video
Python code for detecting and tracking motion on video or in real time
