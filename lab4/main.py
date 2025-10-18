import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import os
from PIL import Image


class FourierMellinRecognizer:
    def __init__(self, output_shape=(256, 256)):
        self.output_shape = output_shape
        self.figure_counter = 1

    def load_image(self, path):
        """Загрузка изображения в grayscale"""
        try:
            img = Image.open(path)
            img = img.convert('L')
            return np.array(img, dtype=np.float32)
        except Exception as e:
            print(f"Ошибка загрузки {path}: {e}")
            return None

    def save_figure(self, image, title, cmap='gray'):
        """Сохранение изображения с нумерацией"""
        plt.figure(figsize=(8, 6))
        plt.imshow(image, cmap=cmap)
        plt.title(f'Рисунок {self.figure_counter} – {title}', fontsize=12, pad=10)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'results/рисунок_{self.figure_counter:02d}.png', dpi=300, bbox_inches='tight')
        plt.close()
        self.figure_counter += 1

    def calculate_spectrum_magnitude(self, input_image):
        """Вычисление спектральной амплитуды как в C++ версии"""
        # Оптимальный размер для БПФ
        optimal_rows = cv2.getOptimalDFTSize(input_image.shape[0])
        optimal_cols = cv2.getOptimalDFTSize(input_image.shape[1])

        # Добавляем границы
        expanded = cv2.copyMakeBorder(input_image, 0, optimal_rows - input_image.shape[0],
                                      0, optimal_cols - input_image.shape[1],
                                      cv2.BORDER_CONSTANT, value=0)

        # Создаем комплексное изображение
        planes = [np.float32(expanded), np.zeros(expanded.shape, dtype=np.float32)]
        complex_plane = cv2.merge(planes)

        # Прямое преобразование Фурье
        complex_plane = cv2.dft(complex_plane, flags=cv2.DFT_COMPLEX_OUTPUT)

        # Разделяем на действительную и мнимую части
        planes = cv2.split(complex_plane)

        # Вычисляем амплитуду
        spectrum = cv2.magnitude(planes[0], planes[1])

        # Логарифмическое масштабирование
        spectrum += 1
        spectrum = np.log(spectrum)

        # Центрирование низкочастотных компонентов
        spectrum = spectrum[:expanded.shape[0] & -2, :expanded.shape[1] & -2]
        cx = spectrum.shape[1] // 2
        cy = spectrum.shape[0] // 2

        # Меняем квадранты местами
        q0 = spectrum[0:cy, 0:cx]
        q1 = spectrum[0:cy, cx:cx * 2]
        q2 = spectrum[cy:cy * 2, 0:cx]
        q3 = spectrum[cy:cy * 2, cx:cx * 2]

        # Меняем местами диагональные квадранты
        temp = q0.copy()
        q0[:] = q3
        q3[:] = temp

        temp = q1.copy()
        q1[:] = q2
        q2[:] = temp

        # Нормализация для отображения
        cv2.normalize(spectrum, spectrum, 0, 1, cv2.NORM_MINMAX)

        return spectrum

    def convert_to_log_polar_coords(self, input_image, output_size=(512, 512)):
        """Преобразование в логарифмическую полярную систему координат"""
        h, w = input_image.shape[:2]
        center = (w / 2.0, h / 2.0)
        max_radius = min(center[0], center[1])

        # Используем warpPolar из OpenCV
        log_polar_image = cv2.warpPolar(input_image, output_size, center,
                                        max_radius,
                                        cv2.WARP_POLAR_LOG + cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)

        return log_polar_image

    def compute_fourier_mellin_descriptor(self, image):
        """Вычисление инвариантного дескриптора Фурье-Меллина"""
        # Шаг 1: Вычисляем амплитудный спектр Фурье
        fourier_amplitude = self.calculate_spectrum_magnitude(image)

        # Шаг 2: Преобразуем в полярно-логарифмические координаты
        logpolar_amplitude = self.convert_to_log_polar_coords(fourier_amplitude)

        # Шаг 3: Вычисляем амплитудный спектр еще раз
        fm_fourier = self.calculate_spectrum_magnitude(logpolar_amplitude)

        return fourier_amplitude, logpolar_amplitude, fm_fourier

    def calculate_normalized_correlation(self, imageA, imageB):
        """Расчёт нормализованной корреляции"""
        # Приводим к uint8 для matchTemplate
        imgA_uint8 = cv2.normalize(imageA, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        imgB_uint8 = cv2.normalize(imageB, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        correlation_result = cv2.matchTemplate(imgA_uint8, imgB_uint8, cv2.TM_CCORR_NORMED)
        min_corr, max_corr, min_loc, max_loc = cv2.minMaxLoc(correlation_result)

        return float(max_corr)

    def scale_and_shift_image(self, input_image, scale_factor, x_shift, y_shift):
        """Масштабирование и сдвиг изображения"""
        # Масштабирование
        scaled = cv2.resize(input_image, None, fx=scale_factor, fy=scale_factor,
                            interpolation=cv2.INTER_LINEAR)

        # Матрица сдвига
        shift_matrix = np.float32([[1, 0, x_shift * input_image.shape[1]],
                                   [0, 1, y_shift * input_image.shape[0]]])

        # Применение аффинного преобразования
        result = cv2.warpAffine(scaled, shift_matrix, scaled.shape[:2][::-1],
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                borderValue=0)

        return result

    def resize_to_square(self, image, size=512):
        """Обрезка изображения до квадратного размера"""
        h, w = image.shape
        if h == size and w == size:
            return image

        start_y = max(0, (h - size) // 2)
        start_x = max(0, (w - size) // 2)
        end_y = min(h, start_y + size)
        end_x = min(w, start_x + size)

        cropped = image[start_y:end_y, start_x:end_x]

        if cropped.shape[0] < size or cropped.shape[1] < size:
            result = np.full((size, size), np.mean(cropped))
            result[:cropped.shape[0], :cropped.shape[1]] = cropped
            return result

        return cropped


def prepare_images(recognizer):
    """Подготовка изображений для эксперимента"""
    print("=== ПОДГОТОВКА ИЗОБРАЖЕНИЙ ===")

    os.makedirs('results', exist_ok=True)

    # Загружаем изображения
    print("Загрузка изображений...")
    large_image = recognizer.load_image("input/first.png")
    etalon2_img = recognizer.load_image("input/second.png")
    etalon3_img = recognizer.load_image("input/third.png")

    if large_image is None or etalon2_img is None or etalon3_img is None:
        print("Ошибка загрузки изображений!")
        return None, None, None, None

    # Сохраняем исходное изображение (Рисунок 1)
    recognizer.save_figure(large_image.astype(np.uint8), "Исходное изображение")

    # Вырезаем фрагменты из большого изображения
    x, y = 100, 100
    size = 512

    fragment1 = large_image[y:y + size, x:x + size]
    fragment2 = etalon2_img  # Второй фрагмент - это второе изображение
    fragment3 = etalon3_img  # Третий фрагмент - это третье изображение

    # Обрезаем фрагменты до квадратного размера
    fragment2 = recognizer.resize_to_square(fragment2, 512)
    fragment3 = recognizer.resize_to_square(fragment3, 512)

    # Сохраняем фрагменты
    recognizer.save_figure(fragment1.astype(np.uint8), "Фрагмент №1")
    recognizer.save_figure(fragment2.astype(np.uint8), "Фрагмент №2")
    recognizer.save_figure(fragment3.astype(np.uint8), "Фрагмент №3")

    print("✓ Изображения подготовлены!")
    return fragment1, fragment1, fragment2, fragment3  # fragment1 используется и как исследуемое и как правильный эталон


def process_fragment_spectrums(recognizer, fragment, fragment_number):
    """Обработка и сохранение спектров для фрагмента"""
    print(f"Обработка спектров для фрагмента №{fragment_number}...")

    # Вычисляем все спектры
    fourier_amplitude, logpolar_amplitude, fm_fourier = recognizer.compute_fourier_mellin_descriptor(fragment)

    # Сохраняем спектры с соответствующими названиями
    if fragment_number == 1:
        recognizer.save_figure(fourier_amplitude, "Амплитудный спектр фрагмента №1", cmap='jet')
        recognizer.save_figure(logpolar_amplitude,
                               "Амплитудный спектр изображения №1 в полярно-логарифмической системе координат",
                               cmap='jet')
        recognizer.save_figure(fm_fourier,
                               "Амплитудный спектр от амплитудного спектра изображения №1 в полярно-логарифмической системе координат",
                               cmap='jet')
    elif fragment_number == 2:
        recognizer.save_figure(fourier_amplitude, "Амплитудный спектр фрагмента №2", cmap='jet')
        recognizer.save_figure(logpolar_amplitude,
                               "Амплитудный спектр изображения №2 в полярно-логарифмической системе координат",
                               cmap='jet')
        recognizer.save_figure(fm_fourier,
                               "Амплитудный спектр от амплитудного спектра изображения №2 в полярно-логарифмической системе координат",
                               cmap='jet')
    elif fragment_number == 3:
        recognizer.save_figure(fourier_amplitude, "Амплитудный спектр фрагмента №3", cmap='jet')
        recognizer.save_figure(logpolar_amplitude,
                               "Амплитудный спектр изображения №3 в полярно-логарифмической системе координат",
                               cmap='jet')
        recognizer.save_figure(fm_fourier,
                               "Амплитудный спектр от амплитудного спектра изображения №3 в полярно-логарифмической системе координат",
                               cmap='jet')

    return fourier_amplitude, logpolar_amplitude, fm_fourier


def experiment_2_2(recognizer, investigated, etalon_correct, etalon2, etalon3):
    """Эксперимент 2.2: Сравнение с эталонами"""
    print("\n=== ЭКСПЕРИМЕНТ 2.2: СРАВНЕНИЕ С ЭТАЛОНАМИ ===")

    # Обрабатываем спектры для всех фрагментов
    print("Вычисление спектров Фурье-Меллина...")

    # Фрагмент 1 (исследуемое изображение)
    desc_investigated = recognizer.compute_fourier_mellin_descriptor(investigated)[2]  # Берем Фурье-Меллин дескриптор
    process_fragment_spectrums(recognizer, investigated, 1)

    # Фрагмент 2
    desc_etalon2 = recognizer.compute_fourier_mellin_descriptor(etalon2)[2]
    process_fragment_spectrums(recognizer, etalon2, 2)

    # Фрагмент 3
    desc_etalon3 = recognizer.compute_fourier_mellin_descriptor(etalon3)[2]
    process_fragment_spectrums(recognizer, etalon3, 3)

    # Вычисляем корреляции
    corr_correct = recognizer.calculate_normalized_correlation(desc_investigated, desc_investigated)
    corr_etalon2 = recognizer.calculate_normalized_correlation(desc_investigated, desc_etalon2)
    corr_etalon3 = recognizer.calculate_normalized_correlation(desc_investigated, desc_etalon3)

    print("\nРЕЗУЛЬТАТЫ КОРРЕЛЯЦИИ:")
    print(f"Корреляция с правильным эталоном: {corr_correct:.6f}")
    print(f"Корреляция с эталоном 2: {corr_etalon2:.6f}")
    print(f"Корреляция с эталоном 3: {corr_etalon3:.6f}")

    # Определяем результат распознавания
    threshold = 0.1
    if corr_correct > max(corr_etalon2, corr_etalon3) and corr_correct > threshold:
        recognition_result = "Правильно распознано"
        print("✓ Изображение правильно распознано!")
    else:
        recognition_result = "Ошибка распознавания"
        print("✗ Изображение не распознано или распознано неправильно!")

    return {
        'corr_correct': corr_correct,
        'corr_etalon2': corr_etalon2,
        'corr_etalon3': corr_etalon3,
        'recognition_result': recognition_result
    }


def experiment_2_3_2_4(recognizer, investigated, etalon_correct):
    """Эксперименты 2.3 и 2.4: Исследование масштабирования"""
    print("\n=== ЭКСПЕРИМЕНТЫ 2.3 и 2.4: МАСШТАБИРОВАНИЕ ===")

    scales_20_percent = [0.8, 1.0, 1.2]
    scales_strong = [0.5, 2.0]

    results_20 = []
    results_strong = []

    print("Исследование масштабирования на 20%:")
    for scale in scales_20_percent:
        scaled_img = recognizer.scale_and_shift_image(investigated, scale, 0, 0)
        scaled_img = recognizer.resize_to_square(scaled_img, 512)

        desc_scaled = recognizer.compute_fourier_mellin_descriptor(scaled_img)[2]
        desc_etalon = recognizer.compute_fourier_mellin_descriptor(etalon_correct)[2]

        corr = recognizer.calculate_normalized_correlation(desc_scaled, desc_etalon)
        results_20.append((scale, corr))
        print(f"  Масштаб {scale}: корреляция = {corr:.6f}")

    print("\nИсследование сильного масштабирования:")
    for scale in scales_strong:
        scaled_img = recognizer.scale_and_shift_image(investigated, scale, 0, 0)
        scaled_img = recognizer.resize_to_square(scaled_img, 512)

        desc_scaled = recognizer.compute_fourier_mellin_descriptor(scaled_img)[2]
        desc_etalon = recognizer.compute_fourier_mellin_descriptor(etalon_correct)[2]

        corr = recognizer.calculate_normalized_correlation(desc_scaled, desc_etalon)
        results_strong.append((scale, corr))
        print(f"  Масштаб {scale}: корреляция = {corr:.6f}")

    return results_20, results_strong


def experiment_2_5(recognizer, investigated, etalon_correct):
    """Эксперимент 2.5: Исследование сдвига"""
    print("\n=== ЭКСПЕРИМЕНТ 2.5: ВЛИЯНИЕ СДВИГА ===")

    shifts = [0.2, 0.4, 0.6]
    results = []

    print("Исследование влияния сдвига:")
    for shift_ratio in shifts:
        shifted_img = recognizer.scale_and_shift_image(investigated, 1.0, shift_ratio, 0)
        shifted_img = recognizer.resize_to_square(shifted_img, 512)

        desc_shifted = recognizer.compute_fourier_mellin_descriptor(shifted_img)[2]
        desc_etalon = recognizer.compute_fourier_mellin_descriptor(etalon_correct)[2]

        corr = recognizer.calculate_normalized_correlation(desc_shifted, desc_etalon)
        results.append((shift_ratio, corr))
        print(f"  Сдвиг {shift_ratio * 100}%: корреляция = {corr:.6f}")

    return results


def save_all_results(results_2_2, results_20, results_strong, results_2_5):
    """Сохранение всех результатов в текстовый файл"""
    with open('results/результаты_корреляции.txt', 'w', encoding='utf-8') as f:
        f.write("ЛАБОРАТОРНАЯ РАБОТА №4: РАСПОЗНАВАНИЕ ОБЪЕКТОВ МЕТОДОМ ФУРЬЕ-МЕЛЛИНА\n")
        f.write("=" * 80 + "\n\n")

        f.write("ТАБЛИЦА РЕЗУЛЬТАТОВ\n")
        f.write("=" * 50 + "\n")

        f.write("2.2. СРАВНЕНИЕ С ЭТАЛОНАМИ:\n")
        f.write(f"Исходное и исходное: {results_2_2['corr_correct']:.6f}\n")
        f.write(f"Исходное и фрагмент №1: {results_2_2['corr_correct']:.6f}\n")
        f.write(f"Исходное и фрагмент №2: {results_2_2['corr_etalon2']:.6f}\n")
        f.write(f"Исходное и фрагмент №3: {results_2_2['corr_etalon3']:.6f}\n")
        f.write(f"Результат распознавания: {results_2_2['recognition_result']}\n\n")

        f.write("2.3. МАСШТАБИРОВАНИЕ НА 20%:\n")
        for scale, corr in results_20:
            if scale == 1.2:
                f.write(f"Масштаб 120%: {corr:.6f}\n")
            elif scale == 0.8:
                f.write(f"Масштаб 80%: {corr:.6f}\n")
            else:
                f.write(f"Масштаб 100%: {corr:.6f}\n")
        f.write("\n")

        f.write("2.4. СИЛЬНОЕ МАСШТАБИРОВАНИЕ:\n")
        for scale, corr in results_strong:
            if scale == 0.5:
                f.write(f"Масштаб 50%: {corr:.6f}\n")
            elif scale == 2.0:
                f.write(f"Масштаб 200%: {corr:.6f}\n")
        f.write("\n")

        f.write("2.5. ВЛИЯНИЕ СДВИГА:\n")
        for shift_ratio, corr in results_2_5:
            f.write(f"Сдвиг на {int(shift_ratio * 100)}%: {corr:.6f}\n")

    print("✓ Текстовые результаты сохранены в results/результаты_корреляции.txt")


def main():
    """Основная функция"""
    print("ЛАБОРАТОРНАЯ РАБОТА №4: РАСПОЗНАВАНИЕ ОБЪЕКТОВ МЕТОДОМ ФУРЬЕ-МЕЛЛИНА")
    print("=" * 70)

    recognizer = FourierMellinRecognizer(output_shape=(512, 512))

    # Подготовка изображений
    images = prepare_images(recognizer)
    if images is None:
        return

    investigated, etalon_correct, etalon2, etalon3 = images

    # Проведение экспериментов
    results_2_2 = experiment_2_2(recognizer, investigated, etalon_correct, etalon2, etalon3)
    results_20, results_strong = experiment_2_3_2_4(recognizer, investigated, etalon_correct)
    results_2_5 = experiment_2_5(recognizer, investigated, etalon_correct)

    # Сохранение результатов
    save_all_results(results_2_2, results_20, results_strong, results_2_5)

    print("\n" + "=" * 70)
    print("ВСЕ ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ!")
    print("Все изображения сохранены в папке 'results/' с нумерацией:")
    print("Рисунок 1-4: Исходные изображения и фрагменты")
    print("Рисунок 5-7: Спектры для фрагмента №1")
    print("Рисунок 8-10: Спектры для фрагмента №2")
    print("Рисунок 11-13: Спектры для фрагмента №3")


if __name__ == "__main__":
    main()