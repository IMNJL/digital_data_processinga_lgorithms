import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import time
from PIL import Image

# Функция для загрузки PCX изображения и конвертации в массив numpy
def load_pcx(path):
    try:
        img = Image.open(path)
        img = img.convert('L')  # Конвертируем в grayscale
        return np.array(img)
    except Exception as e:
        print(f"Ошибка загрузки {path}: {e}")
        return None

# Функция для поиска эталона на изображении и отрисовки результата
def sample_matching(img, sample_image, res_img_name):
    temp_image = img.copy()
    res = cv.matchTemplate(temp_image, sample_image, cv.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    w, h = sample_image.shape[::-1]
    bottom_right = (max_loc[0] + w, max_loc[1] + h)
    cv.rectangle(temp_image, max_loc, bottom_right, 255, 2)

    # Сохраняем изображение с выделенным объектом
    cv.imwrite(f'result/result_{res_img_name}.png', temp_image)

    # Сохраняем карту корреляции
    plt.imshow(res, cmap='gray')
    plt.savefig(f'corr_map/corr_map_{res_img_name}.png', bbox_inches='tight')
    plt.close()

    return max_val, max_loc

# Вращение изображения на угло через аффинные преобразования
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv.warpAffine(image, M, (w, h))
    return rotated

# Масштабирование изображения
def scale_image(image, scale_factor):
    h, w = image.shape[:2]
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    scaled = cv.resize(image, (new_w, new_h))
    return scaled

# влияние вращения на корреляцию
def study_rotation(img, sample, angles):
    results = []
    for angle in angles:
        rotated_img = rotate_image(img, angle)
        max_val, max_loc = sample_matching(rotated_img, sample, f"rot_{angle}")
        results.append((angle, max_val))
    return results

# влияник масштабирования на корреляцию
def study_scaling(img, sample, scales):
    results = []
    for scale in scales:
        scaled_img = scale_image(img, scale)
        max_val, max_loc = sample_matching(scaled_img, sample, f"scale_{scale}")
        results.append((scale, max_val))
    return results


# Функция для измерения времени вычисления корреляционного поля
def measure_correlation_time(img, sample, scales):
    time_results = []
    for scale in scales:
        scaled_img = scale_image(img, scale)

        start_time = time.time()

        res = cv.matchTemplate(scaled_img, sample, cv.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

        end_time = time.time()
        execution_time = end_time - start_time

        time_results.append((scale, execution_time))

        print(f"Масштаб {scale}: время = {execution_time:.4f} сек, размер изображения = {scaled_img.shape}")

    return time_results

img_main = load_pcx("input/main.pcx")
img_own = load_pcx("own_7.pcx")
img_foreign = load_pcx("input/foreign_7.pcx")

if img_main is None or img_own is None or img_foreign is None:
    raise ValueError("Не удалось загрузить одно из изображений!")

# 1. Поиск своего эталона (own_7) на основном изображении
max_val_own, max_loc_own = sample_matching(img_main, img_own, "own_7")
print(f"Максимальное значение корреляции для own_7: {max_val_own} в точке {max_loc_own}")

# 2. Поиск чужого эталона (foreign_7) - не должен найтись
max_val_foreign, max_loc_foreign = sample_matching(img_main, img_foreign, "foreign_7")
print(f"Максимальное значение корреляции для foreign_7: {max_val_foreign} в точке {max_loc_foreign}")

# 3. Исследование вращения для своего эталона
angles = np.arange(-10, 12, 2)  # от -10 до +10 с шагом 2
rotation_results = study_rotation(img_main, img_own, angles)
print("Результаты вращения:")
for angle, val in rotation_results:
    print(f"Угол {angle}: корреляция = {val}")

# 4. Исследование масштабирования для своего эталона
scales = np.arange(0.9, 1.1, 0.025)  # от 0.9 до 1.1 с шагом 0.025
scaling_results = study_scaling(img_main, img_own, scales)
print("Результаты масштабирования:")
for scale, val in scaling_results:
    print(f"Масштаб {scale}: корреляция = {val}")


print("\nИзмерение времени выполнения для разных масштабов:")
time_results = measure_correlation_time(img_main, img_own, scales)
with open('time_results/time_measurements.txt', 'w') as f:
    f.write("Масштаб\tВремя (сек)\tРазмер изображения\n")
    for scale, t in time_results:
        scaled_size = scale_image(img_main, scale).shape
        f.write(f"{scale:.3f}\t{t:.4f}\n")

# График зависимости корреляции от угла вращения
angles_list = [x[0] for x in rotation_results]
vals_rotation = [x[1] for x in rotation_results]
plt.plot(angles_list, vals_rotation)
plt.title("Зависимость корреляции от угла вращения")
plt.xlabel("Угол вращения, градусы")
plt.ylabel("Корреляция")
plt.savefig("rotation_plot.png")
plt.close()

# График зависимости корреляции от масштаба
scales_list = [x[0] for x in scaling_results]
vals_scaling = [x[1] for x in scaling_results]
plt.plot(scales_list, vals_scaling)
plt.title("Зависимость корреляции от масштаба")
plt.xlabel("Масштаб")
plt.ylabel("Корреляция")
plt.savefig("scaling_plot.png")
plt.close()