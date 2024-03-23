import os

import cv2
import numpy as np
import math


class ImageProcessing:
    def __init__(self, image_path):
        self.original_image = self.load_image(image_path)
        # Определяем минимальное и максимальное значения яркости в исходном изображении
        self.min_val = np.min(self.original_image)
        self.max_val = np.max(self.original_image)
        # Определяем минимальное и максимальное значения в палитре
        self.min_palette = 0
        self.max_palette = 255

        self.width_original_image = self.original_image.shape[0]
        self.height_original_image = self.original_image.shape[1]

    def load_image(self, filename):
        if filename:
            folder_name = os.path.basename(os.path.dirname(filename))
            image_name = os.path.basename(filename)
            self.original_image = cv2.imread(os.path.join(folder_name, image_name))
            return self.original_image
        return None

    def logarithmic_transform(self):
        # Логарифмическое преобразование изображения.
        # Формула: s = c * log(1 + r)
        # s - яркость пикселя после преобразования,
        # r - яркость пикселя до преобразования,
        # c - коэффициент, который используется для масштабирования значения яркости.

        log_transformed = np.zeros_like(self.original_image, dtype=np.float32)

        c = (self.max_palette - self.min_palette) / np.log(1 + self.max_val)

        for i in range(self.width_original_image):
            for j in range(self.height_original_image):
                pixel = self.original_image[i, j]
                # Обрабатываем каждый канал RGB отдельно
                for k in range(3):  # 3 канала для RGB
                    if pixel[k] != 0:
                        log_transformed[i, j, k] = 25 * np.log(1 + pixel[k])
                    else:
                        log_transformed[i, j, k] = 0

        return log_transformed

    def gamma_transform(self, gamma):
        # Степенное преобразование изображения.
        # Формула: s = c * r^gamma
        # s - яркость пикселя после преобразования,
        # r - яркость пикселя до преобразования,
        # gamma - параметр степени (gamma),
        # c - коэффициент, используемый для масштабирования значений яркости.

        gamma_transformed = np.zeros_like(self.original_image, dtype=np.float32)

        c = (self.max_palette - self.min_palette) / (self.max_val ** gamma)

        for i in range(self.width_original_image):
            for j in range(self.height_original_image):
                gamma_transformed[i, j] = c * (self.original_image[i, j] ** gamma)

        return gamma_transformed

    def binary_transform(self, threshold):
        # Бинарное преобразование изображения.
        # Формула:
        #     s =
        #     - 0, если r < порогового значения
        #     - 255, если r >= порогового значения
        # s - яркость пикселя после бинарного преобразования,
        # r - яркость пикселя до преобразования,
        # Если яркость пикселя r меньше порогового значения, то после преобразования яркость s будет равна 0 (черный),
        # иначе - 255 (белый).
        binary_image = np.zeros_like(self.original_image)

        for i in range(self.width_original_image):
            for j in range(self.height_original_image):
                if np.all(self.original_image[i, j] >= threshold):
                    binary_image[i, j] = 255

        return binary_image

    def clip_image(self, lower_bound, upper_bound):
        # Вырезание произвольного диапазона яркостей изображения.
        clipped_image = np.zeros_like(self.original_image)

        for i in range(self.width_original_image):
            for j in range(self.height_original_image):
                for k in range(3):  # 3 канала для RGB
                    # Если яркость пикселя находится внутри диапазона, оставляем значение без изменений,
                    # иначе ограничиваем его нижней или верхней границей диапазона
                    if self.original_image[i, j, k] < lower_bound:
                        clipped_image[i, j, k] = lower_bound
                    elif self.original_image[i, j, k] > upper_bound:
                        clipped_image[i, j, k] = upper_bound
                    else:
                        clipped_image[i, j, k] = self.original_image[i, j, k]
        clipped_image_rgb = cv2.cvtColor(clipped_image.astype(np.uint8), cv2.COLOR_BGR2RGB)
        return clipped_image_rgb

    def apply_rectangular_filter(self, kernel_size):
        # Применение прямоугольного фильтра к изображению.

        # Определяем ядро прямоугольного фильтра и его нормировку
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2)

        # Создаем массив для результата, который будет иметь тот же размер, что и исходное изображение
        filtered_image = np.zeros_like(self.original_image)

        # Паддинг изображения, чтобы гарантировать, что мы можем применить фильтр ко всем пикселям
        padded_image = np.pad(self.original_image,
                              ((kernel_size // 2, kernel_size // 2), (kernel_size // 2, kernel_size // 2), (0, 0)),
                              mode='constant')

        # Применяем свертку к каждому каналу RGB
        for c in range(3):  # 3 канала для RGB
            # Применяем фильтр
            for i in range(self.width_original_image):
                for j in range(self.height_original_image):
                    # Определяем область изображения для применения фильтра
                    region = padded_image[i:i + kernel_size, j:j + kernel_size, c]
                    # Применяем ядро к области и записываем результат
                    filtered_pixel = np.sum(region * kernel)
                    filtered_image[i, j, c] = filtered_pixel

        filtered_image_rgb = cv2.cvtColor(filtered_image.astype(np.uint8), cv2.COLOR_BGR2RGB)

        return filtered_image_rgb

    def apply_median_filter(self, kernel_size):
        """
        Применение медианный фильтра к изображению.
        kernel_size: int - размер ядра (фильтра).
        """
        # Создаем массив для результата, такого же размера, как исходное изображение
        filtered_image = np.zeros_like(self.original_image)

        # Определяем половину размера ядра для корректного выравнивания
        half_kernel_size = kernel_size // 2

        # Добавляем отступы к изображению
        padded_image = np.pad(self.original_image,
                              ((half_kernel_size, half_kernel_size), (half_kernel_size, half_kernel_size), (0, 0)),
                              mode='edge')

        for c in range(3):  # 3 канала для RGB
            for i in range(self.width_original_image):
                for j in range(self.height_original_image):
                    min_i = i
                    max_i = i + kernel_size
                    min_j = j
                    max_j = j + kernel_size

                    neighborhood = padded_image[min_i:max_i, min_j:max_j, c]

                    median_value = np.median(neighborhood)

                    filtered_image[i, j, c] = median_value

        filtered_image_rgb = cv2.cvtColor(filtered_image.astype(np.uint8), cv2.COLOR_BGR2RGB)

        return filtered_image_rgb

    @staticmethod
    def gaussian_kernel(kernel_size, sigma):
        """
        Создает ядро фильтра Гаусса заданного размера и сигмой.
        """
        kernel = [[0] * kernel_size for _ in range(kernel_size)]
        center = kernel_size // 2

        for x in range(kernel_size):
            for y in range(kernel_size):
                # Рассчитываем расстояние от текущей позиции до центра ядра
                distance_sq = (x - center) ** 2 + (y - center) ** 2
                kernel[x][y] = (1 / (2 * math.pi * sigma ** 2)) * np.exp(-distance_sq / (2 * sigma ** 2))

        # Нормализуем ядро
        total = sum(sum(row) for row in kernel)
        for x in range(kernel_size):
            for y in range(kernel_size):
                kernel[x][y] /= total

        return kernel

    def apply_gaussian_filter(self, sigma):
        kernel_size = int(sigma * 6 + 1)
        half_kernel_size = kernel_size // 2

        # Добавляем отступы к изображению
        padded_image = np.pad(self.original_image,
                              ((half_kernel_size, half_kernel_size), (half_kernel_size, half_kernel_size), (0, 0)),
                              mode='edge')

        filtered_image = np.zeros_like(padded_image, dtype=float)

        # Получаем ядро фильтра Гаусса
        gaussian_kernel = self.gaussian_kernel(kernel_size, sigma)

        # Применяем фильтр Гаусса к каждому каналу изображения
        for channel in range(3):
            for x in range(half_kernel_size, padded_image.shape[0] - half_kernel_size):
                for y in range(half_kernel_size, padded_image.shape[1] - half_kernel_size):
                    # Вычисляем взвешенную сумму значений пикселей с помощью ядра Гаусса
                    weighted_sum = 0
                    normalization_factor = 0
                    for i in range(-half_kernel_size, half_kernel_size + 1):
                        for j in range(-half_kernel_size, half_kernel_size + 1):
                            weighted_sum += (
                                    padded_image[x + i, y + j, channel] * gaussian_kernel[i + half_kernel_size][
                                j + half_kernel_size])
                            normalization_factor += gaussian_kernel[i + half_kernel_size][j + half_kernel_size]

                    # Нормируем значение weighted_sum
                    weighted_sum /= normalization_factor

                    # Записываем в отфильтрованное изображение нормированное значение
                    filtered_image[x, y, channel] = weighted_sum

        # Обрезаем отступы и приводим к RGB формату
        filtered_image_rgb = cv2.cvtColor(
            filtered_image[half_kernel_size:-half_kernel_size, half_kernel_size:-half_kernel_size].astype(np.uint8),
            cv2.COLOR_BGR2RGB)

        return filtered_image_rgb

    def apply_sigma_filter(self, kernel_size, sigma_threshold):
        """
        Сигма-фильтр для обработки изображений.
        """
        filtered_image = np.zeros_like(self.original_image)

        # Применяем фильтр к каждому пикселю изображения
        for x in range(self.width_original_image):
            for y in range(self.height_original_image):
                # Определяем границы окна для текущего пикселя
                y_min = max(0, y - kernel_size // 2)
                y_max = min(self.height_original_image, y + kernel_size // 2 + 1)
                x_min = max(0, x - kernel_size // 2)
                x_max = min(self.width_original_image, x + kernel_size // 2 + 1)

                # Вычисляем стандартное отклонение в окрестности пикселя
                neighborhood = self.original_image[x_min:x_max, y_min:y_max]
                sigma = np.std(neighborhood)

                # Применяем фильтр
                if sigma < sigma_threshold:
                    # Пиксель считается краевым, оставляем без изменений
                    filtered_image[x, y] = self.original_image[x, y]
                else:
                    # Пиксель считается шумовым, заменяем на среднее значение в окрестности
                    filtered_image[x, y] = np.mean(neighborhood)

        filtered_image_rgb = cv2.cvtColor(filtered_image.astype(np.uint8), cv2.COLOR_BGR2RGB)

        return filtered_image_rgb

    # Модуль для резкости
    def sharpening(self, image):
        # Нерезкое маскирование
        # Для увеличения резкости изображения применяется маска, которая усиливает различия в яркости между
        # соседними пикселями.
        kernel_sharpening = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

        # Получаем размеры изображения
        height, width = image.shape[0], image.shape[1]

        # Создаем пустое изображение для результата
        sharpened = np.zeros_like(image)

        # Применяем маску к каждому пикселю изображения, кроме краев
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                # Применяем маску к текущему пикселю
                pixel_sum = 0
                for i in range(3):
                    for j in range(3):
                        pixel_sum += image[y + i - 1][x + j - 1] * kernel_sharpening[i][j]

                sharpened[y, x] = pixel_sum

        return sharpened
