from PIL import Image
from math import cos, sin, pi, sqrt
import matplotlib.pyplot as plt
from collections import defaultdict


def dist(x, y):
    """
    Евклидово расстояние.
    """
    return sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)


class CircleHoughTransform:
    def __init__(self):
        self.image = None
        self.gray = None
        self.edges = None
        self.centers = []
        self.circles = []
        self.rectangles = []

        self.w, self.h = None, None

    def read_image(self, path="circles.png"):
        self.image = Image.open(path)
        self.w, self.h = self.image.size

    def find_edges(self):
        """
        Ищем границы объектов.
        """
        assert self.gray is not None
        mask = [
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]
        ]
        gr = self.gray.load()
        self.edges = Image.new(mode="L", size=self.image.size).load()
        for i in range(1, self.w - 1):
            for j in range(1, self.h - 1):
                pix = 0
                for m in range(len(mask)):
                    for n in range(len(mask[0])):
                        pix += gr[i + m - 1, j + n - 1] * mask[m][n]
                self.edges[i, j] = pix

    def green_filter(self):
        """
        Оставляем только зеленые пиксели.
        Считаю, что фон в любом случае зелёный.
        """
        assert self.image is not None
        self.gray = Image.new(mode="L", size=self.image.size)
        for i in range(self.w):
            for j in range(self.h):
                pixel = self.image.getpixel((i, j))
                self.gray.putpixel((i, j), pixel[1])

    def find_circles(self, threshold=0.4, r_min=3, r_max=None, steps=100):
        """
        Поиск потенциальых окружностей с помощью преобразования Хафа.
        Проверка прямоугольников добавлена потому, что если на картинке присутствует прямоугольник
        с маленькими сторонами, то преобразование Хафа распознает его как окружность.
        :param threshold:
        :param r_min:
        :param r_max:
        :param steps:
        :return:
        """
        assert self.edges is not None
        thetas = [i * 2 * pi / steps for i in range(steps)]
        coss = list(map(lambda x: cos(x), thetas))
        sins = list(map(lambda x: sin(x), thetas))

        if r_max is None:
            r_max = min(self.h, self.w) // 2

        A = defaultdict(int)

        for r in range(r_min, r_max + 1):
            for x in range(r, self.w - r):
                for y in range(r, self.h - r):
                    if self.edges[x, y] > 0:
                        for i in range(steps):
                            a = round(r * coss[i])
                            b = round(r * sins[i])
                            A[(x + a, y + b, r)] += 1
            for x in range(self.w):
                for y in range(self.h):
                    if A[(x, y, r)] >= threshold * steps:
                        if not self.in_rectangle(x, y):
                            self.centers.append((x, y, r, A[(x, y, r)]))

        self.filter_centers()
        # Можно посмотреть, где находятся центры и оценить правильность ответа:
        # plt.scatter(*zip(*self.circles))
        # plt.show()

    def filter_centers(self):
        """
        Фильтрует точки, которые близко лежат к уже существующим центрам.
        """
        for point in sorted(self.centers, key=lambda x: x[3], reverse=True):
            for circle in self.circles:
                if dist(point, circle) <= circle[2]:
                    break
            else:
                self.circles.append(point)

    def in_rectangle(self, x, y):
        """
        Проверяет принадлежит ли точка (x,y) какому-нибудь прямоугольнику.
        """
        for rectangle in self.rectangles:
            lu = rectangle[0]
            rb = rectangle[1]
            if lu[0] < x < rb[0] and lu[1] < y < rb[1]:
                return True
        rect = self.rectangle(x, y)
        if rect is not None:
            self.rectangles.append(rect)
            return True
        return False

    def rectangle(self, x, y):
        """
        Поиск прямоугольника, в котором находится точка (x,y).
        Возвращает координаты верхнего левого и нижнего правого угла.
        """
        i, j = x, y
        left_upper_corner = 0, 0
        right_bottom_corner1 = 0, 0
        right_bottom_corner2 = 0, 0
        while 0 < i < self.w - 1 and 0 < j < self.h - 1:
            if self.edges[i, j] == 0:
                i -= 1
                j -= 1
            else:
                if self.edges[i - 1, j] > 0:
                    i -= 1
                elif self.edges[i, j - 1] > 0:
                    j -= 1
                else:
                    left_upper_corner = i, j
                    break

        while 0 < i < self.w - 1 and 0 < j < self.h - 1:
            if self.edges[i, j - 1] == 0:
                if self.edges[i + 1, j] > 0:
                    i += 1
                else:
                    break
            else:
                return

        while 0 < i < self.w - 1 and 0 < j < self.h - 1:
            if self.edges[i + 1, j] == 0:
                if self.edges[i, j + 1] > 0:
                    j += 1
                else:
                    right_bottom_corner1 = i, j
                    break
            else:
                return

        i, j = left_upper_corner
        while 0 < i < self.w - 1 and 0 < j < self.h - 1:
            if self.edges[i - 1, j] == 0:
                if self.edges[i, j + 1] > 0:
                    j += 1
                else:
                    break
            else:
                return

        while 0 < i < self.w - 1 and 0 < j < self.h - 1:
            if self.edges[i, j + 1] == 0:
                if self.edges[i + 1, j] > 0:
                    i += 1
                else:
                    right_bottom_corner2 = i, j
                    break
            else:
                return

        if right_bottom_corner1 == right_bottom_corner2:
            return left_upper_corner, right_bottom_corner1

    def count_colored_circles(self, colors):
        """
        Считаем окружности соответствующих цветов.
        """
        res = [0] * len(colors)
        for x, y, r, A in self.circles:
            c = self.image.getpixel((x, y))
            for i, color in enumerate(colors):
                if (c[0], c[1], c[2]) == color:
                    res[i] += 1
        return res


if __name__ == "__main__":
    method = CircleHoughTransform()
    method.read_image(path='circles.png')
    method.green_filter()  # Оставляем только зеленую часть пикселей.
    method.find_edges()  # Находим границы.
    method.find_circles(steps=100, threshold=0.4, r_min=3)  # Находим окружности с помощью преобразования Хафа.
    red, black = method.count_colored_circles([(255, 0, 0), (0, 0, 0)])
    print("Red circles:", red, "\nBlack circles:", black)
