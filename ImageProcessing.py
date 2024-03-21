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

        return clipped_image

    def apply_rectangular_filter(self, kernel_size):
        # Применение прямоугольного фильтра к изображению.
        # image: numpy.ndarray - исходное изображение,
        # kernel_size: int - размер ядра (фильтра).

        # Определяем ядро прямоугольного фильтра
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2)

        filtered_image = np.zeros_like(self.original_image)

        for i in range(self.width_original_image):
            for j in range(self.height_original_image):
                # Определяем область изображения для применения фильтра
                min_i = max(i - kernel_size // 2, 0)
                max_i = min(i + kernel_size // 2 + 1, self.width_original_image)
                min_j = max(j - kernel_size // 2, 0)
                max_j = min(j + kernel_size // 2 + 1, self.height_original_image)

                # Применяем ядро к области изображения
                filtered_pixel = 0
                kernel_sum = 0
                for k in range(min_i, max_i):
                    for l in range(min_j, max_j):
                        filtered_pixel += self.original_image[k, l] * kernel[k - min_i, l - min_j]
                        kernel_sum += kernel[k - min_i, l - min_j]

                filtered_image[i, j] = filtered_pixel / kernel_sum

        return filtered_image

    def apply_median_filter(self, image, kernel_size):
        """
        Применение медианного фильтра к изображению.
        image: numpy.ndarray - исходное цветное изображение,
        kernel_size: int - размер ядра (фильтра).
        """
        # Применяем медианный фильтр к каждому каналу изображения
        filtered_channels = [self.apply_median_filter_single_channel(channel, kernel_size) for channel in
                             image.transpose(2, 0, 1)]

        # Объединяем каналы обратно в цветное изображение
        filtered_image = np.stack(filtered_channels, axis=-1)

        return filtered_image

    @staticmethod
    def apply_median_filter_single_channel(channel, kernel_size):
        """
        Применение медианного фильтра к одному каналу изображения.
        """
        filtered_channel = np.zeros_like(channel)

        for i in range(channel.shape[0]):
            for j in range(channel.shape[1]):
                # Определяем область изображения для применения фильтра
                min_i = max(i - kernel_size // 2, 0)
                max_i = min(i + kernel_size // 2 + 1, channel.shape[0])
                min_j = max(j - kernel_size // 2, 0)
                max_j = min(j + kernel_size // 2 + 1, channel.shape[1])

                # Получаем окрестность пикселя
                neighborhood = channel[min_i:max_i, min_j:max_j]

                # Вычисляем медиану сортировкой
                sorted_neighborhood = np.sort(neighborhood.flatten())
                median_index = len(sorted_neighborhood) // 2
                median_value = sorted_neighborhood[median_index]

                # Применяем медианный фильтр
                filtered_channel[i, j] = median_value

        return filtered_channel

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

    def apply_gaussian_filter(self, kernel_size, sigma):
        filtered_image = np.zeros_like(self.original_image, dtype=float)

        # Получаем ядро фильтра Гаусса
        gaussian_kernel = self.gaussian_kernel(kernel_size, sigma)

        # Вычисляем половину размера ядра для корректного выравнивания
        half_kernel_size = kernel_size // 2

        # Применяем фильтр Гаусса к каждому каналу изображения
        for channel in range(self.original_image.shape[2]):
            for y in range(half_kernel_size, self.width_original_image - half_kernel_size):
                for x in range(half_kernel_size, self.height_original_image - half_kernel_size):
                    # Вычисляем взвешенную сумму значений пикселей с помощью ядра Гаусса
                    weighted_sum = 0
                    for i in range(-half_kernel_size, half_kernel_size + 1):
                        for j in range(-half_kernel_size, half_kernel_size + 1):
                            weighted_sum += (self.original_image[y + i, x + j, channel]
                                             * gaussian_kernel[i + half_kernel_size][j + half_kernel_size])

                    # Записываем в отфильтрованное изображение полученное значение
                    filtered_image[y, x, channel] = weighted_sum

        return filtered_image

    def apply_sigma_filter(self, image, kernel_size, sigma_r):
        """
        Применяет сигма-фильтр к изображению.
        """

        # Создаем пустое изображение для результата
        filtered_image = np.zeros_like(self.original_image)

        # Вычисляем половину размера ядра для корректного выравнивания
        half_kernel_size = kernel_size // 2

        # Применяем фильтр к каждому пикселю изображения
        for y in range(self.height_original_image):
            for x in range(self.width_original_image):
                # Определяем область изображения для применения фильтра
                min_y = max(y - half_kernel_size, 0)
                max_y = min(y + half_kernel_size + 1, self.height_original_image)
                min_x = max(x - half_kernel_size, 0)
                max_x = min(x + half_kernel_size + 1, self.width_original_image)

                # Получаем окрестность пикселя
                neighborhood = image[min_y:max_y, min_x:max_x].flatten()

                # Вычисляем веса пикселей по расстоянию от центрального пикселя
                weights = [self.gaussian_func(image[y][x], pixel, sigma_r) for pixel in neighborhood]

                # Применяем взвешенное среднее для получения фильтрованного значения
                filtered_value = sum(weight * pixel for weight, pixel in zip(weights, neighborhood)) / sum(weights)

                # Записываем фильтрованное значение в результирующее изображение
                filtered_image[y][x] = filtered_value

        return filtered_image

    @staticmethod
    def gaussian_func(x, y, sigma):
        """
        Вычисляет значение Гауссовой функции в точке x с заданной сигмой.
        """
        return (1 / (2 * math.pi * sigma ** 2)) * (np.exp(- (x - y) ** 2 / (2 * sigma ** 2)))

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
