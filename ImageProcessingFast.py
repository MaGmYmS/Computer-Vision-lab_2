import os

import cv2
import numpy as np
import math


class ImageProcessingFast:
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
            self.original_image = cv2.imread(str(os.path.join(folder_name, image_name)))
            return self.original_image
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

        return log_transformed

    def gamma_transform(self, gamma):
        # Степенное преобразование изображения.
        # Формула: s = c * r^gamma
        # s - яркость пикселя после преобразования,
        # r - яркость пикселя до преобразования,
        # gamma - параметр степени (gamma),
        # c - коэффициент, используемый для масштабирования значений яркости.

        c = (self.max_palette - self.min_palette) / (self.max_val ** gamma)
        gamma_transformed = c * (self.original_image.astype(np.float32) ** gamma)

        return gamma_transformed

    def binary_transform(self, threshold):
        # Преобразование (R + G + B) / 3
        grayscale_image = np.mean(self.original_image, axis=2)
        mask = grayscale_image >= threshold
        binary_image = np.where(mask, 255, 0)

        return binary_image

    def clip_image(self, lower_bound, upper_bound):
        # Вырезание произвольного диапазона яркостей изображения.

        # Применяем функцию clip к каждому каналу RGB
        clipped_image = np.clip(self.original_image, lower_bound, upper_bound)

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
                              mode='constant')

        # Применяем свертку к каждому каналу RGB
        for c in range(3):  # 3 канала для RGB
            # Применяем фильтр
            for i in range(filtered_image.shape[0]):
                for j in range(filtered_image.shape[1]):
                    # Определяем область изображения для применения фильтра
                    region = padded_image[i:i + kernel_size, j:j + kernel_size, c]
                    # Применяем ядро к области и записываем результат
                    filtered_pixel = np.sum(region * kernel)
                    filtered_image[i, j, c] = filtered_pixel

        return filtered_image

    def apply_median_filter(self, kernel_size):
        """
        Применение медианный фильтра к изображению.
        kernel_size: int - размер ядра (фильтра).
        """
        # Создаем массив для результата, такого же размера, как исходное изображение
        filtered_image = np.zeros_like(self.original_image)

        # Применяем фильтр к каждому каналу изображения
        for c in range(3):  # 3 канала для RGB
            # Применяем фильтр к каждому пикселю изображения
            for i in range(self.width_original_image):
                for j in range(self.height_original_image):
                    # Определяем область изображения для применения фильтра
                    min_i = max(i - kernel_size // 2, 0)
                    max_i = min(i + kernel_size // 2 + 1, self.width_original_image)
                    min_j = max(j - kernel_size // 2, 0)
                    max_j = min(j + kernel_size // 2 + 1, self.height_original_image)

                    # Получаем окрестность пикселя
                    neighborhood = self.original_image[min_i:max_i, min_j:max_j, c]

                    # Вычисляем медиану
                    median_value = np.median(neighborhood)

                    # Применяем медианный фильтр
                    filtered_image[i, j, c] = median_value

        return filtered_image

    @staticmethod
    def gaussian_kernel(kernel_size, sigma):
        """
        Создает ядро фильтра Гаусса заданного размера и сигмой.
        """
        # Создаем сетку координат
        x, y = np.meshgrid(np.arange(kernel_size), np.arange(kernel_size))

        # Рассчитываем расстояние от каждой точки до центра ядра
        distance_sq = (x - kernel_size // 2) ** 2 + (y - kernel_size // 2) ** 2

        # Вычисляем значения ядра Гаусса
        kernel = np.exp(-distance_sq / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)

        # Нормализуем ядро
        kernel /= np.sum(kernel)

        return kernel

    def apply_gaussian_filter(self, kernel_size, sigma):
        filtered_image = np.zeros_like(self.original_image, dtype=float)

        # Получаем ядро фильтра Гаусса
        gaussian = self.gaussian_kernel(kernel_size, sigma)

        # Вычисляем половину размера ядра для корректного выравнивания
        half_kernel_size = kernel_size // 2

        padded_image = np.pad(self.original_image,
                              ((half_kernel_size, half_kernel_size), (half_kernel_size, half_kernel_size), (0, 0)),
                              mode='constant',
                              constant_values=0)

        # Применяем фильтр к каждому каналу изображения независимо
        for channel in range(3):
            for y in range(self.height_original_image):
                for x in range(self.width_original_image):
                    # Определяем область изображения для применения фильтра
                    min_y = max(y - half_kernel_size, 0)
                    max_y = min(y + half_kernel_size + 1, self.height_original_image)
                    min_x = max(x - half_kernel_size, 0)
                    max_x = min(x + half_kernel_size + 1, self.width_original_image)

                    # Вычисляем свертку для текущего пикселя
                    total = np.sum(padded_image[min_y:max_y, min_x:max_x, channel] * gaussian[:max_y-min_y, :max_x-min_x])
                    filtered_image[x][y][channel] = total

        return filtered_image


