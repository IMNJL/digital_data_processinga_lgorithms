import cv2 as cv
import numpy as np
import math
from PIL import Image


def load_pcx(path):
    try:
        img = Image.open(path)
        img = img.convert('L')  # Конвертируем в grayscale
        return np.array(img)
    except Exception as e:
        print(f"Ошибка загрузки {path}: {e}")
        return None


def main():
    # Параметры для варианта 7
    N = 7
    scale = 1 + 0.05 * N  # 1.35
    angle = 3 * N  # 21 градус
    polar_flags = cv.INTER_CUBIC + cv.WARP_FILL_OUTLIERS + cv.WARP_POLAR_LOG

    # Загрузка изображения
    orig_img = load_pcx("input/main.pcx")
    if orig_img is None:
        raise ValueError("Не удалось загрузить изображение")

    # Центр исходного изображения
    orig_center = ((orig_img.shape[1] - 1) / 2.0, (orig_img.shape[0] - 1) / 2.0)

    # Матрица поворота и масштабирования
    rotation_mat = cv.getRotationMatrix2D(orig_center, angle, scale)

    # Применяем аффинное преобразование (поворот + масштабирование)
    rotated_img = cv.warpAffine(orig_img, rotation_mat, orig_img.shape[1::-1])

    # Центр преобразованного изображения
    rotated_center = ((rotated_img.shape[1] - 1) / 2.0, (rotated_img.shape[0] - 1) / 2.0)

    # Вычисляем максимальный радиус для каждого изображения
    max_radius_orig = min(orig_img.shape[1] // 2, orig_img.shape[0] // 2)
    max_radius_rotated = min(rotated_img.shape[1] // 2, rotated_img.shape[0] // 2)

    # Используем одинаковый размер для полярно-логарифмического преобразования
    polar_width = int(math.log(max_radius_orig) * 50)
    polar_height = 360
    polar_size = (polar_width, polar_height)

    # Преобразование в полярно-логарифмические координаты
    # Для исходного изображения используем его центр и радиус
    polar_orig_img = cv.warpPolar(orig_img, polar_size, orig_center, max_radius_orig, polar_flags)

    # Для преобразованного изображения используем ЕГО центр и радиус
    polar_rotated_img = cv.warpPolar(rotated_img, polar_size, rotated_center, max_radius_rotated, polar_flags)

    # Создаем составное изображение для циклического поиска
    combined_img = cv.vconcat([polar_rotated_img, polar_rotated_img])
    combined_img = cv.hconcat([combined_img, combined_img])

    # Вычисляем корреляцию
    correlation_result = cv.matchTemplate(combined_img, polar_orig_img, cv.TM_CCOEFF_NORMED)

    # Находим максимум корреляции
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(correlation_result)

    # Вычисляем параметры преобразования
    detected_angle = max_loc[1]  # Угол поворота
    detected_scale = math.exp(max_loc[0] / 50.0)  # Коэффициент масштабирования

    # Вывод результатов
    print("=" * 40)
    print(f"Detected rotation angle = {detected_angle}°")
    print(f"Detected scale factor = {detected_scale:.3f}")
    print(f"Correlation value = {max_val:.4f}")
    print(f"Peak coordinates = ({max_loc[0]}, {max_loc[1]})")
    print("=" * 40)

    # Визуализация для отладки
    print("\nОтладочная информация:")
    print(f"Исходное изображение: {orig_img.shape}")
    print(f"Преобразованное изображение: {rotated_img.shape}")
    print(f"Центр исходного: {orig_center}")
    print(f"Центр преобразованного: {rotated_center}")
    print(f"Макс. радиус исходного: {max_radius_orig}")
    print(f"Макс. радиус преобразованного: {max_radius_rotated}")
    print(f"Размер полярного преобразования: {polar_size}")

    # Сохранение результатов
    cv.imwrite('result/original.png', orig_img)
    cv.imwrite('result/rotated_scaled.png', rotated_img)
    cv.imwrite('result/polar_original.png', polar_orig_img)
    cv.imwrite('result/polar_rotated.png', polar_rotated_img)

    # Создаем визуализацию корреляционной карты
    correlation_display = cv.normalize(correlation_result, None, 0, 255, cv.NORM_MINMAX)
    correlation_display = correlation_display.astype(np.uint8)
    correlation_display = cv.applyColorMap(correlation_display, cv.COLORMAP_JET)

    # Отмечаем максимум на корреляционной карте
    cv.circle(correlation_display, max_loc, 10, (0, 255, 0), 2)
    cv.putText(correlation_display, f'Max: {max_val:.3f}',
               (max_loc[0] + 10, max_loc[1]),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv.imwrite('result/correlation_map.png', correlation_display)

    # Дополнительный анализ
    print("\nТочность определения:")
    print(f"Ожидаемый угол: {angle}°")
    print(f"Обнаруженный угол: {detected_angle}°")
    print(f"Ошибка угла: {abs(angle - detected_angle):.1f}°")
    print(f"Ожидаемый масштаб: {scale:.3f}")
    print(f"Обнаруженный масштаб: {detected_scale:.3f}")
    print(f"Ошибка масштаба: {abs(scale - detected_scale):.3f}")

    # Альтернативный метод вычисления параметров (более точный)
    # Угол соответствует сдвигу по вертикальной оси в полярных координатах
    actual_detected_angle = detected_angle
    if detected_angle > 180:
        actual_detected_angle = detected_angle - 360

    print(f"\nАльтернативное вычисление угла: {actual_detected_angle}°")

    # Сохраняем отчет
    with open('result/report.txt', 'w', encoding='utf-8') as f:
        f.write("Лабораторная работа №2: Полярно-логарифмическая корреляция\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Параметры для варианта {N}:\n")
        f.write(f"  - Угол поворота: {angle}°\n")
        f.write(f"  - Коэффициент масштабирования: {scale:.3f}\n\n")
        f.write("Результаты корреляционного анализа:\n")
        f.write(f"  - Максимум корреляции: {max_val:.4f}\n")
        f.write(f"  - Координаты максимума: ({max_loc[0]}, {max_loc[1]})\n")
        f.write(f"  - Обнаруженный угол поворота: {detected_angle}°\n")
        f.write(f"  - Обнаруженный масштаб: {detected_scale:.3f}\n\n")
        f.write("Точность определения:\n")
        f.write(f"  - Ошибка угла: {abs(angle - detected_angle):.1f}°\n")
        f.write(f"  - Ошибка масштаба: {abs(scale - detected_scale):.3f}\n")

    # Визуализация промежуточных результатов
    try:
        # Показываем разницу между полярными представлениями
        diff = cv.absdiff(polar_orig_img, polar_rotated_img)
        cv.imshow('Original', orig_img)
        cv.imshow('Rotated + Scaled', rotated_img)
        cv.imshow('Polar Original', polar_orig_img)
        cv.imshow('Polar Rotated', polar_rotated_img)
        cv.imshow('Difference', diff)
        cv.imshow('Correlation Map', correlation_display)
        cv.waitKey(0)
        cv.destroyAllWindows()
    except:
        print("GUI не доступен, результаты сохранены в папке result/")


if __name__ == "__main__":
    main()