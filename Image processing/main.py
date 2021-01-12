# По условию задания данный проект не использует готовые функции OpenCV
import sys
import cv2
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QFileDialog

import Qt


class ExampleApp(QtWidgets.QMainWindow, Qt.Ui_MainWindow):
    path = ""
    source_image = 0
    buffer = 0
    result_image = 0
    width = 0
    height = 0
    bytes_per_line = 3 * width
    threshold = 0
    delta = 0
    bright_l = 0
    dark_l = 0
    bright_n = 0
    dark_n = 0
    c = 0
    y = 0

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        # открыть изображение
        self.open_button.clicked.connect(self.open_img)
        # сохранить результат
        self.save_button.clicked.connect(self.save_img)
        # преобразовать в полутоновое
        self.grayscale_button.clicked.connect(self.get_grayscale)
        # пороговая бинаризация
        self.simple_binarization_slider.valueChanged[int].connect(self.get_threshold)
        self.simple_binarization.clicked.connect(self.threshold_binarization)
        # адаптивная бинаризация
        self.binarization_button.clicked.connect(self.get_binarization)
        # получить значения +/- яркости (линейное)
        self.bright_up_slider.valueChanged[int].connect(self.return_bright_l)
        self.bright_down_slider.valueChanged[int].connect(self.return_dark_l)
        # получить +/- яркость (линейное)
        self.bright_up_button.clicked.connect(self.brightness_up_l)
        self.bright_down_button.clicked.connect(self.brightness_down_l)
        # получить значения +/- яркости (нелинейное)
        self.a_up_slider.valueChanged[int].connect(self.return_bright_n)
        self.a_down_slider.valueChanged[int].connect(self.return_dark_n)
        # получить +/- яркость (нелинейное)
        self.a_up_button.clicked.connect(self.brightness_up_n)
        self.a_down_button.clicked.connect(self.brightness_down_n)
        # линейное изменение контраста
        self.contrast_slider_1.valueChanged[int].connect(self.return_contrast)
        self.contrast_button_1.clicked.connect(self.contrast_l)
        # нелинейное изменение контраста
        self.c_slider.valueChanged[int].connect(self.return_c)
        self.y_slider.valueChanged[int].connect(self.return_y)
        self.contrast_button_2.clicked.connect(self.contrast_n)
        # выравнивание освещения
        self.illumination_button.clicked.connect(self.correct_illumination)
        # распознавание образов
        self.detection.clicked.connect(self.detect)

    def open_img(self):
        filename = QFileDialog.getOpenFileName(self, str("Open Image"), "/home",
                                               str("Image Files (*.png *.jpg *.jpeg *.bmp)"))
        self.path = filename[0]
        pixmap = QPixmap(self.path)
        self.source_img.setFixedSize(500, 500)
        self.source_img.setPixmap(QPixmap(pixmap))
        self.cv_read()
        self.result_img.clear()

    def save_img(self):
        filepath = QFileDialog.getSaveFileName(self, str("Save Image"), "/home",
                                               str("Image Files (*.png *.jpg *.jpeg *.bmp)"))
        if ".jpeg" in filepath[0]:
            pixmap = self.result_img.pixmap()
            image = pixmap.toImage()
            image.save(filepath[0], "JPEG", -1)
        if ".jpg" in filepath[0]:
            pixmap = self.result_img.pixmap()
            image = pixmap.toImage()
            image.save(filepath[0], "JPG", -1)
        if ".png" in filepath[0]:
            pixmap = self.result_img.pixmap()
            image = pixmap.toImage()
            image.save(filepath[0], "PNG", -1)
        if ".bmp" in filepath[0]:
            pixmap = self.result_img.pixmap()
            image = pixmap.toImage()
            image.save(filepath[0], "bmp", -1)

    def cv_read(self):
        self.buffer = cv2.imread(self.path)
        self.buffer = cv2.cvtColor(self.buffer, cv2.COLOR_BGR2RGB)

    def get_shapes(self):
        self.width = self.buffer.shape[0]
        self.height = self.buffer.shape[1]

    def set_pixmap(self):
        self.result_image = self.buffer.copy()
        q_image = QImage(self.result_image.data, self.height, self.width, self.bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap()
        pixmap.convertFromImage(q_image)
        self.result_img.setFixedSize(500, 500)
        self.result_img.setPixmap(QPixmap(pixmap))

    def get_grayscale(self):
        self.get_shapes()
        self.cv_read()
        for x in range(self.width):
            for y in range(self.height):
                red = self.buffer.item(x, y, 0)
                green = self.buffer.item(x, y, 1)
                blue = self.buffer.item(x, y, 2)
                gray = int(0.3 * red + 0.59 * green + 0.11 * blue)
                self.buffer.itemset((x, y, 0), gray)
                self.buffer.itemset((x, y, 1), gray)
                self.buffer.itemset((x, y, 2), gray)
        self.set_pixmap()

    def get_threshold(self):
        self.threshold = self.simple_binarization_slider.value()
        self.simple_binarization_level.setText(str(self.threshold))

    def threshold_binarization(self):
        self.get_grayscale()
        self.cv_read()
        for x in range(self.width):
            for y in range(self.height):
                pixel = self.buffer.item(x, y, 0)
                if pixel < self.threshold:
                    self.buffer.itemset((x, y, 0), 0)
                    self.buffer.itemset((x, y, 1), 0)
                    self.buffer.itemset((x, y, 2), 0)
                if pixel > self.threshold:
                    self.buffer.itemset((x, y, 0), 255)
                    self.buffer.itemset((x, y, 1), 255)
                    self.buffer.itemset((x, y, 2), 255)
        self.set_pixmap()

    def get_binarization(self):
        self.get_grayscale()
        pixel_number = self.buffer.shape[0] * self.buffer.shape[1]
        mean_weight = 1.0 / pixel_number
        his, bins = np.histogram(self.buffer, np.array(range(0, 256)))
        final_thresh = -1
        final_value = -1
        for t in bins[1:-1]:
            Wb = np.sum(his[:t]) * mean_weight
            Wf = np.sum(his[t:]) * mean_weight
            mub = np.mean(his[:t])
            muf = np.mean(his[t:])
            value = Wb * Wf * (mub - muf) ** 2
            if value > final_value:
                final_thresh = t
                final_value = value
        self.buffer = self.buffer.copy()
        print(final_thresh)
        self.buffer[self.buffer > final_thresh] = 255
        self.buffer[self.buffer < final_thresh] = 0
        self.set_pixmap()

    def return_bright_l(self):
        self.bright_l = self.bright_up_slider.value()
        self.bright_up_label.setText(str(self.bright_l))

    def brightness_up_l(self):
        self.get_shapes()
        self.return_bright_l()
        self.cv_read()
        for x in range(self.width):
            for y in range(self.height):
                red = self.buffer.item(x, y, 0)
                green = self.buffer.item(x, y, 1)
                blue = self.buffer.item(x, y, 2)
                new_red = red + self.bright_l if (red + self.bright_l) < 255 else 255
                new_green = green + self.bright_l if (green + self.bright_l) < 255 else 255
                new_blue = blue + self.bright_l if (blue + self.bright_l) < 255 else 255
                self.buffer.itemset((x, y, 0), new_red)
                self.buffer.itemset((x, y, 1), new_green)
                self.buffer.itemset((x, y, 2), new_blue)
        self.bright_l = 0
        self.set_pixmap()

    def return_dark_l(self):
        self.dark_l = self.bright_down_slider.value()
        self.bright_down_label.setText(str(self.dark_l))

    def brightness_down_l(self):
        self.get_shapes()
        self.return_dark_l()
        self.cv_read()
        for x in range(self.width):
            for y in range(self.height):
                red = self.buffer.item(x, y, 0)
                green = self.buffer.item(x, y, 1)
                blue = self.buffer.item(x, y, 2)
                new_red = red - self.dark_l if (red - self.dark_l) > 0 else 0
                new_green = green - self.dark_l if (green - self.dark_l) > 0 else 0
                new_blue = blue - self.dark_l if (blue - self.dark_l) > 0 else 0
                self.buffer.itemset((x, y, 0), new_red)
                self.buffer.itemset((x, y, 1), new_green)
                self.buffer.itemset((x, y, 2), new_blue)
        self.dark_l = 0
        self.set_pixmap()

    def return_bright_n(self):
        self.bright_n = self.a_up_slider.value()
        self.a_up_label.setText(str(self.bright_n))

    def brightness_up_n(self):
        self.get_shapes()
        self.return_bright_n()
        self.cv_read()
        for x in range(self.width):
            for y in range(self.height):
                red = self.buffer.item(x, y, 0)
                green = self.buffer.item(x, y, 1)
                blue = self.buffer.item(x, y, 2)
                if red == 0:
                    red += 1
                if green == 0:
                    green += 1
                if blue == 0:
                    blue += 1
                new_red = 1 / red + red + self.bright_n if (1 / red + red + self.bright_n) < 255 else 255
                new_green = 1 / green + green + self.bright_n if (1 / green + green + self.bright_n) < 255 else 255
                new_blue = 1 / blue + blue + self.bright_n if (1 / blue + blue + self.bright_n) < 255 else 255
                self.buffer.itemset((x, y, 0), new_red)
                self.buffer.itemset((x, y, 1), new_green)
                self.buffer.itemset((x, y, 2), new_blue)
        self.bright_n = 0
        self.set_pixmap()

    def return_dark_n(self):
        self.dark_n = self.a_down_slider.value()
        self.a_down_label.setText(str(self.dark_n))

    def brightness_down_n(self):
        self.get_shapes()
        self.return_dark_n()
        self.cv_read()
        for x in range(self.width):
            for y in range(self.height):
                red = self.buffer.item(x, y, 0)
                green = self.buffer.item(x, y, 1)
                blue = self.buffer.item(x, y, 2)
                if red == 0:
                    red += 1
                if green == 0:
                    green += 1
                if blue == 0:
                    blue += 1
                new_red = 1 / red + red - self.dark_n if (1 / red + red - self.dark_n) > 0 else 0
                new_green = 1 / green + green - self.dark_n if (1 / green + green - self.dark_n) > 0 else 0
                new_blue = 1 / blue + blue - self.dark_n if (1 / blue + blue - self.dark_n) > 0 else 0
                self.buffer.itemset((x, y, 0), new_red)
                self.buffer.itemset((x, y, 1), new_green)
                self.buffer.itemset((x, y, 2), new_blue)
        self.dark_n = 0
        self.set_pixmap()

    def return_contrast(self):
        self.delta = self.contrast_slider_1.value()
        self.contrast_level_1.setText(str(self.delta))

    def contrast_l(self):
        self.get_shapes()
        self.return_contrast()
        self.cv_read()
        count_of_pixels = self.width * self.height
        gray = 0
        for x in range(self.width):
            for y in range(self.height):
                red = self.buffer.item(x, y, 0)
                green = self.buffer.item(x, y, 1)
                blue = self.buffer.item(x, y, 2)
                gray = int(0.3 * red + 0.59 * green + 0.11 * blue)
        gray /= count_of_pixels
        for x in range(self.width):
            for y in range(self.height):
                red = self.buffer.item(x, y, 0)
                green = self.buffer.item(x, y, 1)
                blue = self.buffer.item(x, y, 2)
                red += (red - gray) * self.delta / 255
                green += (green - gray) * self.delta / 255
                blue += (blue - gray) * self.delta / 255
                if red > 255:
                    red = 255
                elif red < 0:
                    red = 0
                if green > 255:
                    green = 255
                elif green < 0:
                    green = 0
                if blue > 255:
                    blue = 255
                elif blue < 0:
                    blue = 0
                self.buffer.itemset((x, y, 0), red)
                self.buffer.itemset((x, y, 1), green)
                self.buffer.itemset((x, y, 2), blue)
        self.delta = 0
        self.set_pixmap()

    def return_c(self):
        self.c = self.c_slider.value() / 100
        self.c_level.setText(str(self.c))

    def return_y(self):
        self.y = self.y_slider.value() / 100
        self.y_level.setText(str(self.y))

    def contrast_n(self):
        self.get_shapes()
        self.return_c()
        self.return_y()
        self.cv_read()
        for x in range(self.width):
            for y in range(self.height):
                red = self.buffer.item(x, y, 0)
                green = self.buffer.item(x, y, 1)
                blue = self.buffer.item(x, y, 2)
                new_red = self.c * (red ** self.y)
                new_green = self.c * (green ** self.y)
                new_blue = self.c * (blue ** self.y)
                if new_red > 255:
                    new_red = 255
                elif new_red < 0:
                    new_red = 0
                if new_green > 255:
                    new_green = 255
                elif new_green < 0:
                    new_green = 0
                if new_blue > 255:
                    new_blue = 255
                elif blue < 0:
                    new_blue = 0
                self.buffer.itemset((x, y, 0), new_red)
                self.buffer.itemset((x, y, 1), new_green)
                self.buffer.itemset((x, y, 2), new_blue)
        self.c = 0
        self.y = 0
        self.set_pixmap()

    def correct_illumination(self):
        self.get_shapes()
        min_red = self.buffer.item(0, 0, 0)
        max_red = self.buffer.item(0, 0, 0)
        min_green = self.buffer.item(0, 0, 1)
        max_green = self.buffer.item(0, 0, 1)
        min_blue = self.buffer.item(0, 0, 2)
        max_blue = self.buffer.item(0, 0, 2)
        for x in range(self.width):
            for y in range(self.height):
                red = self.buffer.item(x, y, 0)
                green = self.buffer.item(x, y, 1)
                blue = self.buffer.item(x, y, 2)
                if min_red > red:
                    min_red = red
                if max_red < red:
                    max_red = red
                if min_green > green:
                    min_green = green
                if max_green < green:
                    max_green = green
                if min_blue > blue:
                    min_blue = blue
                if max_blue < blue:
                    max_blue = blue
        for x in range(self.width):
            for y in range(self.height):
                red = self.buffer.item(x, y, 0)
                green = self.buffer.item(x, y, 1)
                blue = self.buffer.item(x, y, 2)
                new_red = (red - min_red) * (255 / (max_red - min_red))
                new_green = (green - min_green) * (255 / (max_green - min_green))
                new_blue = (blue - min_blue) * (255 / (max_blue - min_blue))
                self.buffer.itemset((x, y, 0), new_red)
                self.buffer.itemset((x, y, 1), new_green)
                self.buffer.itemset((x, y, 2), new_blue)
        self.set_pixmap()

    def detect(self):
        self.cv_read()
        rectangle = 0
        triangle = 0
        circle = 0
        square = 0
        trapezoid = 0
        # Этот закомментированный код - метод к-средних
        # Z = self.buffer.reshape((-1, 3))
        # Z = np.float32(Z)
        # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        # K = 6
        # ret, label1, center1 = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        # center1 = np.uint8(center1)
        # res1 = center1[label1.flatten()]
        # self.buffer = res1.reshape((self.buffer.shape))
        img_grey = cv2.cvtColor(self.buffer, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(img_grey, (11, 11), 0)
        _, thrash = cv2.threshold(blur, 244, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thrash, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
            cv2.drawContours(self.buffer, [approx], 0, (0, 0, 0), 3)
            x = approx.ravel()[0]
            y = approx.ravel()[1] - 5
            if len(approx) == 3:
                triangle += 1
                cv2.putText(self.buffer, "Треугольник", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
            elif len(approx) == 4:
                x1, y1, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h
                if aspect_ratio >= 0.95 and aspect_ratio <= 1.05:
                    square += 1
                    cv2.putText(self.buffer, "Квадрат", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
                elif aspect_ratio >= 1.2 and aspect_ratio <= 2:
                    rectangle += 1
                    cv2.putText(self.buffer, "Прямоугольник", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
                else:
                    trapezoid += 1
                    cv2.putText(self.buffer, "Трапеция", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
            elif len(approx) <= 17:
                cv2.putText(self.buffer, "Круг", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
                circle += 1
            else:
                cv2.putText(self.buffer, "", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0, (0, 0, 0))

        figure = ['Квадратов', 'Прямоугольников', 'Треугольников', 'Кругов', 'Трапеций']
        number_fig = [square, rectangle, triangle, circle, trapezoid]
        for i in range(len(figure)):
            print(figure[i], number_fig[i])
        cv2.imwrite("result.jpg", self.buffer)
        self.set_pixmap()


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = ExampleApp()
    window.show()
    app.exec()


if __name__ == '__main__':
    main()
