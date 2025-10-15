import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage
import os


def load_pcx(path):
    try:
        img = Image.open(path)
        img = img.convert('L')  # Конвертируем в grayscale
        return np.array(img)
    except Exception as e:
        print(f"Ошибка загрузки {path}: {e}")
        return None


def calculate_affine_parameters(image):
    """Вычисление параметров аффинных преобразований"""
    # Находим координаты ненулевых пикселов (объекта)
    y_coords, x_coords = np.where(image > 0)

    if len(x_coords) == 0:
        return None

    # Центр масс
    x_center = np.mean(x_coords)
    y_center = np.mean(y_coords)

    # Центрированные координаты
    x_centered = x_coords - x_center
    y_centered = y_coords - y_center

    # Моменты инерции для определения ориентации
    m11 = np.sum(x_centered * y_centered)
    m20 = np.sum(x_centered ** 2)
    m02 = np.sum(y_centered ** 2)

    # Угол поворота (формула 3.14)
    if m20 != m02:
        theta = 0.5 * np.arctan2(2 * m11, m20 - m02)
    else:
        theta = 0

    # Масштабные коэффициенты (формула 3.15)
    lambda1 = m20 * (np.cos(theta)) ** 2 + m02 * (np.sin(theta)) ** 2 + 2 * m11 * np.sin(theta) * np.cos(theta)
    lambda2 = m20 * (np.sin(theta)) ** 2 + m02 * (np.cos(theta)) ** 2 - 2 * m11 * np.sin(theta) * np.cos(theta)

    # Избегаем деления на ноль
    scale_x = np.sqrt(lambda1 / len(x_coords)) if lambda1 > 0 else 1
    scale_y = np.sqrt(lambda2 / len(x_coords)) if lambda2 > 0 else 1

    # Добавляем небольшое значение чтобы избежать деления на ноль
    scale_x = max(scale_x, 0.001)
    scale_y = max(scale_y, 0.001)

    return {
        'center': (x_center, y_center),
        'translation': (-x_center, -y_center),
        'rotation': -theta,
        'scale': (1 / scale_x, 1 / scale_y),
        'original_shape': image.shape
    }


def normalize_image(image, params, K=10):
    """Нормализация изображения с использованием аффинных преобразований"""
    if params is None:
        return image

    # Применяем преобразования с интерполяцией
    # 1. Перенос в начало координат
    trans_x, trans_y = params['translation']

    # Создаем матрицу преобразования для переноса
    translation_matrix = np.array([[1, 0, -trans_x],
                                   [0, 1, -trans_y],
                                   [0, 0, 1]])

    # 2. Поворот
    theta_deg = params['rotation'] * 180 / np.pi
    rotated = ndimage.rotate(image, theta_deg, reshape=False, order=1)

    # 3. Масштабирование
    scale_x, scale_y = params['scale']
    # Применяем коэффициент K для равномерного масштабирования
    final_scale_x = K * scale_x
    final_scale_y = K * scale_y

    # Масштабирование с интерполяцией
    scaled = ndimage.zoom(rotated, (final_scale_y, final_scale_x), order=1)

    return scaled


def normalize_with_formulas(image, params, K=10):
    """Нормализация по формулам (3.14), (3.15), (3.17) без интерполяции"""
    height, width = image.shape
    # Создаем большее изображение для нормализованного результата
    new_size = int(max(height, width) * 2)
    normalized = np.zeros((new_size, new_size), dtype=image.dtype)

    trans_x, trans_y = params['translation']
    theta = params['rotation']
    scale_x, scale_y = params['scale']

    center_x, center_y = new_size // 2, new_size // 2

    for y in range(height):
        for x in range(width):
            if image[y, x] > 0:
                # Формулы преобразования координат
                # 1. Перенос в начало координат
                x1 = x + trans_x
                y1 = y + trans_y

                # 2. Поворот
                x2 = x1 * np.cos(theta) - y1 * np.sin(theta)
                y2 = x1 * np.sin(theta) + y1 * np.cos(theta)

                # 3. Масштабирование с коэффициентом K
                x3 = x2 * scale_x * K
                y3 = y2 * scale_y * K

                # Перенос в центр нового изображения
                new_x = int(np.round(x3 + center_x))
                new_y = int(np.round(y3 + center_y))

                # Проверка границ
                if 0 <= new_x < new_size and 0 <= new_y < new_size:
                    normalized[new_y, new_x] = image[y, x]

    return normalized


def analyze_normalization(original, normalized, params, image_name):
    """Анализ результатов нормализации"""
    print(f"\n{'=' * 50}")
    print(f"АНАЛИЗ ДЛЯ: {image_name}")
    print(f"{'=' * 50}")

    print("Параметры аффинных преобразований:")
    print(f"  Центр масс: ({params['center'][0]:.1f}, {params['center'][1]:.1f})")
    print(f"  Перенос: ({params['translation'][0]:.1f}, {params['translation'][1]:.1f})")
    print(f"  Поворот: {params['rotation'] * 180 / np.pi:.2f}°")
    print(f"  Масштаб: ({params['scale'][0]:.3f}, {params['scale'][1]:.3f})")

    # Статистика по изображениям
    orig_nonzero = np.count_nonzero(original)
    norm_nonzero = np.count_nonzero(normalized)

    print(f"\nСтатистика:")
    print(f"  Размер оригинала: {original.shape}")
    print(f"  Размер после нормализации: {normalized.shape}")
    print(f"  Ненулевых пикселов в оригинале: {orig_nonzero}")
    print(f"  Ненулевых пикселов после нормализации: {norm_nonzero}")

    if orig_nonzero > 0:
        preservation = norm_nonzero / orig_nonzero * 100
        print(f"  Сохранение информации: {preservation:.1f}%")

    # Анализ ориентации
    theta_deg = params['rotation'] * 180 / np.pi
    print(f"\nОриентация объекта: {theta_deg:.1f}°")
    if abs(theta_deg) < 10:
        print("  Объект близок к горизонтальной ориентации")
    elif abs(theta_deg - 90) < 10 or abs(theta_deg + 90) < 10:
        print("  Объект близок к вертикальной ориентации")
    else:
        print("  Объект имеет наклонную ориентацию")


def visualize_results(original, normalized_interp, normalized_formulas, image_name):
    """Визуализация результатов для одного изображения"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Оригинальное изображение
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title(f'Оригинал: {image_name}')
    axes[0].axis('off')

    # Нормализация с интерполяцией
    axes[1].imshow(normalized_interp, cmap='gray')
    axes[1].set_title('С интерполяцией')
    axes[1].axis('off')

    # Нормализация по формулам
    axes[2].imshow(normalized_formulas, cmap='gray')
    axes[2].set_title('По формулам (без интерполяции)')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()


def analyze_shape(normalized_image, image_name):
    """Анализ формы объекта после нормализации"""
    # Находим контур объекта
    y_coords, x_coords = np.where(normalized_image > 0)

    if len(x_coords) < 10:
        print("Объект слишком мал для анализа")
        return

    # Вычисляем моменты для анализа формы
    x_center = np.mean(x_coords)
    y_center = np.mean(y_coords)

    x_centered = x_coords - x_center
    y_centered = y_coords - y_center

    # Моменты инерции
    m20 = np.sum(x_centered ** 2)
    m02 = np.sum(y_centered ** 2)
    m11 = np.sum(x_centered * y_centered)

    # Эксцентриситет (мера "овальности")
    numerator = (m20 - m02) ** 2 + 4 * m11 ** 2
    denominator = (m20 + m02) ** 2
    eccentricity = np.sqrt(numerator / denominator) if denominator > 0 else 0

    # Соотношение сторон
    if m02 > 0:
        aspect_ratio = np.sqrt(m20 / m02)
    else:
        aspect_ratio = 1

    print(f"\nАнализ формы для {image_name}:")
    print(f"  Эксцентриситет: {eccentricity:.3f}")
    print(f"  Соотношение осей: {aspect_ratio:.3f}")

    if eccentricity < 0.3:
        print("  Форма: близка к кругу")
    elif eccentricity < 0.7:
        print("  Форма: овал")
    else:
        print("  Форма: сильно вытянутый овал")

    if 0.9 < aspect_ratio < 1.1:
        print("  Соотношение осей: почти круглое (1:1)")
    else:
        print(f"  Соотношение осей: {aspect_ratio:.2f}:1")

    return eccentricity, aspect_ratio


def main():
    """Основная функция для лабораторной работы"""
    # Пути к вашим изображениям
    image_paths = [
        "input/image_1_7.pcx",
        "input/image_2_7.pcx",
        "input/image_3_7.pcx"
    ]

    # Проверка существования файлов
    for path in image_paths:
        if not os.path.exists(path):
            print(f"ОШИБКА: Файл {path} не найден!")
            print("Убедитесь, что:")
            print("1. Файлы находятся в папке 'input'")
            print("2. Имена файлов соответствуют указанным")
            return

    print("Начало обработки изображений...")
    shapes_analysis = []

    # Обработка каждого изображения
    for i, path in enumerate(image_paths, 1):
        print(f"\nОбработка изображения {i}/3: {path}")

        # Загрузка изображения
        image = load_pcx(path)
        if image is None:
            continue

        # Вычисление параметров аффинных преобразований
        params = calculate_affine_parameters(image)
        if params is None:
            print(f"Не удалось вычислить параметры для {path}")
            continue

        # Нормализация двумя методами
        normalized_interp = normalize_image(image, params, K=10)
        normalized_formulas = normalize_with_formulas(image, params, K=10)

        # Анализ результатов
        analyze_normalization(image, normalized_interp, params, os.path.basename(path))

        # Визуализация
        visualize_results(image, normalized_interp, normalized_formulas, os.path.basename(path))

        # Сохранение результатов
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(path))[0]

        # Сохранение нормализованных изображений
        Image.fromarray(normalized_interp.astype(np.uint8)).save(
            f"{output_dir}/{base_name}_normalized_interp.png"
        )
        Image.fromarray(normalized_formulas.astype(np.uint8)).save(
            f"{output_dir}/{base_name}_normalized_formulas.png"
        )

        # Анализ формы ДО нормализации
        print("\n--- Анализ ДО нормализации ---")
        ecc_before, ar_before = analyze_shape(image, f"Исходный {os.path.basename(path)}")

        # Анализ формы ПОСЛЕ нормализации
        print("--- Анализ ПОСЛЕ нормализации ---")
        ecc_after, ar_after = analyze_shape(normalized_interp, f"Нормализованный {os.path.basename(path)}")

        shapes_analysis.append({
            'name': os.path.basename(path),
            'ecc_before': ecc_before,
            'ecc_after': ecc_after,
            'ar_before': ar_before,
            'ar_after': ar_after
        })

    print(f"\n{'=' * 60}")
    print("СВОДНЫЙ АНАЛИЗ ФОРМЫ ОБЪЕКТОВ:")
    print(f"{'=' * 60}")

    for analysis in shapes_analysis:
        print(f"\n{analysis['name']}:")
        print(f"  Эксцентриситет: {analysis['ecc_before']:.3f} -> {analysis['ecc_after']:.3f}")
        print(f"  Соотношение осей: {analysis['ar_before']:.3f} -> {analysis['ar_after']:.3f}")

        if analysis['ecc_after'] < analysis['ecc_before']:
            print("  ✓ Форма стала более круглой")
        else:
            print("  ⚠ Форма не изменилась или стала менее круглой")


if __name__ == "__main__":
    main()