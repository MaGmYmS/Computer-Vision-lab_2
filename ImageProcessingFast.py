import copy
import os

import cv2
import numpy as np
import math

from PyQt5.QtGui import QImage


class ImageProcessingFast:
    def __init__(self, image_path):
        self.original_image = self.load_image(image_path)
        self.process_image = None
        # Определяем минимальное и максимальное значения яркости в исходном изображении
        self.min_val = np.min(self.original_image)
        self.max_val = np.max(self.original_image)
        # Определяем минимальное и максимальное значения в палитре
        self.min_palette = 0
        self.max_palette = 255

        self.width_original_image = self.original_image.shape[0]
        self.height_original_image = self.original_image.shape[1]

    @staticmethod
    def load_image(filename):
        if filename:
            folder_name = os.path.basename(os.path.dirname(filename))
            image_name = os.path.basename(filename)
            original_image = cv2.imread(str(os.path.join(folder_name, image_name)))
            return copy.deepcopy(original_image)
        return None

    def logarithmic_transform(self):
        # Логарифмическое преобразование изображения.
        # Формула: s = c * log(1 + r)
        # s - яркость пикселя после преобразования,
        # r - яркость пикселя до преобразования,
        # c - коэффициент, который используется для масштабирования значения яркости.

        c = (self.max_palette - self.min_palette) / np.log(1 + self.max_val)

        # Применение логарифмического преобразования к каждому каналу RGB
        log_transformed = c * np.log(1 + self.original_image.astype(np.float32))

        # Обработка значений, равных нулю
        log_transformed[np.isnan(log_transformed)] = 0

        log_transformed_rgb = cv2.cvtColor(log_transformed.astype(np.uint8), cv2.COLOR_BGR2RGB)

        return log_transformed_rgb

    def gamma_transform(self, gamma):
        # Степенное преобразование изображения.
        # Формула: s = c * r^gamma
        # s - яркость пикселя после преобразования,
        # r - яркость пикселя до преобразования,
        # gamma - параметр степени (gamma),
        # c - коэффициент, используемый для масштабирования значений яркости.

        c = (self.max_palette - self.min_palette) / (self.max_val ** gamma)
        gamma_transformed = c * (self.original_image.astype(np.float32) ** gamma)
        gamma_transformed_rgb = cv2.cvtColor(gamma_transformed.astype(np.uint8), cv2.COLOR_BGR2RGB)

        return gamma_transformed_rgb

    def binary_transform(self, threshold):
        # Преобразование (R + G + B) / 3
        grayscale_image = np.mean(self.original_image, axis=2)
        mask = grayscale_image >= threshold
        binary_image = np.where(mask, 255, 0)

        return binary_image

    def clip_image(self, lower_bound, upper_bound, choice):
        # Применяем маски для создания отсеченного изображения
        clipped_image = np.zeros_like(self.original_image)
        image = self.original_image
        if choice:
            image = self.clip_image2(image, lower_bound, upper_bound, 100)
        # Создаем маски для верхней и нижней границ
        lower_mask = image < lower_bound
        upper_mask = image > upper_bound

        clipped_image[lower_mask] = lower_bound
        clipped_image[upper_mask] = upper_bound
        clipped_image[~(lower_mask | upper_mask)] = image[~(lower_mask | upper_mask)]

        # Конвертируем изображение в RGB
        clipped_image_rgb = cv2.cvtColor(clipped_image.astype(np.uint8), cv2.COLOR_BGR2RGB)

        return clipped_image_rgb

    @staticmethod
    def clip_image2(image, lower_bound, upper_bound, constant_value):
        # Создаем маску для пикселей внутри диапазона
        inside_mask = (image >= lower_bound) & (image <= upper_bound)

        # Применяем маску для создания отсеченного изображения
        clipped_image = np.where(inside_mask, constant_value, image)
        return clipped_image

    def apply_rectangular_filter(self, kernel_size):
        # Применение прямоугольного фильтра к изображению.

        # Определяем ядро прямоугольного фильтра и его нормировку
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2)

        # Создаем массив для результата, который будет иметь тот же размер, что и исходное изображение
        filtered_image = np.zeros_like(self.original_image)

        # Паддинг изображения, чтобы гарантировать, что мы можем применить фильтр ко всем пикселям
        padded_image = np.pad(self.original_image,
                              ((kernel_size // 2, kernel_size // 2), (kernel_size // 2, kernel_size // 2), (0, 0)),
                              mode='edge')

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
        filtered_image = np.zeros_like(self.original_image)

        half_kernel_size = kernel_size // 2

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
        if kernel_size % 2 == 0:
            kernel_size += 1
        half_kernel_size = kernel_size // 2

        # Добавляем отступы к изображению
        padded_image = np.pad(self.original_image,
                              ((half_kernel_size, half_kernel_size), (half_kernel_size, half_kernel_size), (0, 0)),
                              mode='edge')

        filtered_image = np.zeros_like(padded_image, dtype=float)

        # Получаем ядро фильтра Гаусса
        gaussian_kernel = self.gaussian_kernel(kernel_size, sigma)
        gaussian_kernel /= np.sum(gaussian_kernel)

        # Применяем фильтр Гаусса к каждому каналу изображения
        for channel in range(3):
            for x in range(half_kernel_size, padded_image.shape[0] - half_kernel_size):
                for y in range(half_kernel_size, padded_image.shape[1] - half_kernel_size):
                    region = padded_image[x - half_kernel_size:x + half_kernel_size + 1,
                             y - half_kernel_size:y + half_kernel_size + 1, channel]

                    weighted_sum = np.sum(region * gaussian_kernel)

                    # Записываем в отфильтрованное изображение нормированное значение
                    filtered_image[x, y, channel] = weighted_sum

        # Обрезаем отступы и приводим к RGB формату
        filtered_image_rgb = cv2.cvtColor(
            filtered_image[half_kernel_size:-half_kernel_size, half_kernel_size:-half_kernel_size].astype(np.uint8),
            cv2.COLOR_BGR2RGB)

        return filtered_image_rgb

    def apply_sigma_filter(self, sigma):
        """
        Сигма-фильтр для обработки изображений.
        """
        filtered_image = np.zeros_like(self.original_image)
        kernel_size = int(sigma * 6 + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

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
                filtered_image[x, y] = np.mean(neighborhood)

        filtered_image_rgb = cv2.cvtColor(filtered_image.astype(np.uint8), cv2.COLOR_BGR2RGB)

        return filtered_image_rgb

    @staticmethod
    def absolute_difference(image1_cv, image2_cv):
        # Проверка на успешную загрузку изображений
        if image1_cv is None or image2_cv is None:
            return None

        # Проверка на совпадение размеров изображений
        if image1_cv.shape != image2_cv.shape:
            return None

        diff_map = np.abs(image1_cv - image2_cv)

        # Вычисление абсолютной разности для каждого пикселя
        # for y in range(image1_cv.shape[0]):
        #     for x in range(image1_cv.shape[1]):
        #         for c in range(image1_cv.shape[2]):  # Для каждого канала
        #             diff_map[y, x, c] = abs(int(image1_cv[y, x, c]) - int(image2_cv[y, x, c]))

        return diff_map.astype(np.uint8)

    def sharpening(self, lambda_sh):
        if self.process_image is None:
            print("Сглаженное изображение отсутствует. Сначала примените сглаживание.")
            return None

        difference = (self.original_image.astype(np.float32) - self.process_image.astype(np.float32))

        sharpened_image = self.original_image.astype(np.float32) + lambda_sh * difference

        # Ограничение значений пикселей в диапазоне [0, 255]
        sharpened_image = np.clip(sharpened_image, 0, 255)

        sharpened_image_rgb = cv2.cvtColor(sharpened_image.astype(np.uint8), cv2.COLOR_BGR2RGB)

        return sharpened_image_rgb

    @staticmethod
    def compute_sharpness_coefficient(original_image, processed_image):
        # Проверка на совпадение размеров изображений
        if original_image.shape != processed_image.shape:
            raise ValueError("Размеры изображений не совпадают")

        # Вычисление разности изображений
        difference = processed_image.astype(np.float32) - original_image.astype(np.float32)

        # Вычисление среднеквадратического отклонения (MSE) разности
        mse = np.mean(np.square(difference))

        # Оценка весовых коэффициентов для всех значений разности
        unique_values, counts = np.unique(difference, return_counts=True)
        total_pixels = np.sum(counts)
        weights = counts / total_pixels

        # Вычисление итогового коэффициента резкости с учетом весов
        sharpness_coefficient = mse * np.sum(unique_values * weights) / 100

        return sharpness_coefficient
