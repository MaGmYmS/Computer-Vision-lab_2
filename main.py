import sys
import cv2
import numpy as np
import os

from PyQt5.QtCore import QElapsedTimer

from ImageProcessing import ImageProcessing
from ImageProcessingFast import ImageProcessingFast
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QApplication, QMessageBox
from PyQt5.QtGui import QImage, QPixmap


class Ui_MainWindow(object):
    def __init__(self):
        self.image_processing = None
        self.image_exist = False

    def setupUi(self, MainWindow):
        # region main setupui
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1680, 841)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(10, 0, 1660, 821))
        self.tabWidget.setStyleSheet("font: 14pt \"MS Shell Dlg 2\";")
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.graphicsView_1_1 = QtWidgets.QGraphicsView(self.tab)
        self.graphicsView_1_1.setGeometry(QtCore.QRect(950, 10, 700, 700))
        self.graphicsView_1_1.setObjectName("graphicsView_1_1")
        self.graphicsView_1_2 = QtWidgets.QGraphicsView(self.tab)
        self.graphicsView_1_2.setGeometry(QtCore.QRect(240, 10, 700, 700))
        self.graphicsView_1_2.setObjectName("graphicsView_1_2")
        self.button_load_img_1 = QtWidgets.QPushButton(self.tab)
        self.button_load_img_1.setGeometry(QtCore.QRect(450, 720, 291, 61))
        self.button_load_img_1.setObjectName("button_load_img_1")
        self.button_save_img_1 = QtWidgets.QPushButton(self.tab)
        self.button_save_img_1.setGeometry(QtCore.QRect(1170, 720, 291, 61))
        self.button_save_img_1.setObjectName("button_save_img_1")
        self.groupBox = QtWidgets.QGroupBox(self.tab)
        self.groupBox.setGeometry(QtCore.QRect(10, 30, 221, 681))
        self.groupBox.setObjectName("groupBox")
        self.button_exp_1 = QtWidgets.QPushButton(self.groupBox)
        self.button_exp_1.setGeometry(QtCore.QRect(10, 100, 201, 41))
        self.button_exp_1.setStyleSheet("font: 9pt \"MS Shell Dlg 2\";")
        self.button_exp_1.setObjectName("button_exp_1")
        self.button_bin_1 = QtWidgets.QPushButton(self.groupBox)
        self.button_bin_1.setGeometry(QtCore.QRect(10, 200, 201, 41))
        self.button_bin_1.setStyleSheet("font: 9pt \"MS Shell Dlg 2\";")
        self.button_bin_1.setObjectName("button_bin_1")
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setGeometry(QtCore.QRect(20, 150, 161, 31))
        self.label.setStyleSheet("font: 12pt \"MS Shell Dlg 2\";")
        self.label.setObjectName("label")
        self.spinbox_exp_porog_1 = QtWidgets.QSpinBox(self.groupBox)
        self.spinbox_exp_porog_1.setGeometry(QtCore.QRect(160, 260, 51, 31))
        self.spinbox_exp_porog_1.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.spinbox_exp_porog_1.setMaximum(255)
        self.spinbox_exp_porog_1.setProperty("value", 125)
        self.spinbox_exp_porog_1.setObjectName("spinbox_exp_porog_1")
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setGeometry(QtCore.QRect(20, 260, 81, 31))
        self.label_2.setStyleSheet("font: 12pt \"MS Shell Dlg 2\";")
        self.label_2.setObjectName("label_2")
        self.spinbox_bin_border_1 = QtWidgets.QSpinBox(self.groupBox)
        self.spinbox_bin_border_1.setGeometry(QtCore.QRect(160, 370, 51, 31))
        self.spinbox_bin_border_1.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.spinbox_bin_border_1.setMaximum(255)
        self.spinbox_bin_border_1.setProperty("value", 50)
        self.spinbox_bin_border_1.setObjectName("spinbox_bin_border_1")
        self.label_3 = QtWidgets.QLabel(self.groupBox)
        self.label_3.setGeometry(QtCore.QRect(10, 370, 101, 31))
        self.label_3.setStyleSheet("font: 10pt \"MS Shell Dlg 2\";")
        self.label_3.setObjectName("label_3")
        self.spinbox_bin_border_2 = QtWidgets.QSpinBox(self.groupBox)
        self.spinbox_bin_border_2.setGeometry(QtCore.QRect(160, 410, 51, 31))
        self.spinbox_bin_border_2.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.spinbox_bin_border_2.setMaximum(255)
        self.spinbox_bin_border_2.setProperty("value", 200)
        self.spinbox_bin_border_2.setObjectName("spinbox_bin_border_2")
        self.doubleSpinBox_gamma = QtWidgets.QDoubleSpinBox(self.groupBox)
        self.doubleSpinBox_gamma.setGeometry(QtCore.QRect(160, 150, 51, 31))
        self.doubleSpinBox_gamma.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.doubleSpinBox_gamma.setMaximum(5.0)
        self.doubleSpinBox_gamma.setProperty("value", 1.0)
        self.doubleSpinBox_gamma.setObjectName("doubleSpinBox_gamma")
        self.label_4 = QtWidgets.QLabel(self.groupBox)
        self.label_4.setGeometry(QtCore.QRect(10, 410, 101, 31))
        self.label_4.setStyleSheet("font: 10pt \"MS Shell Dlg 2\";")
        self.label_4.setObjectName("label_4")
        self.button_cut_bright = QtWidgets.QPushButton(self.groupBox)
        self.button_cut_bright.setGeometry(QtCore.QRect(10, 310, 201, 41))
        self.button_cut_bright.setStyleSheet("font: 9pt \"MS Shell Dlg 2\";")
        self.button_cut_bright.setObjectName("button_cut_bright")
        self.button_log_1 = QtWidgets.QPushButton(self.groupBox)
        self.button_log_1.setGeometry(QtCore.QRect(10, 40, 201, 41))
        self.button_log_1.setStyleSheet("font: 9pt \"MS Shell Dlg 2\";")
        self.button_log_1.setObjectName("button_log_1")
        self.console_1 = QtWidgets.QTextBrowser(self.groupBox)
        self.console_1.setGeometry(QtCore.QRect(10, 510, 201, 151))
        self.console_1.setObjectName("console_1")
        self.button_cut_bright_2 = QtWidgets.QPushButton(self.groupBox)
        self.button_cut_bright_2.setGeometry(QtCore.QRect(10, 460, 201, 41))
        self.button_cut_bright_2.setStyleSheet("font: 9pt \"MS Shell Dlg 2\";")
        self.button_cut_bright_2.setObjectName("button_cut_bright_2")
        self.groupBox.raise_()
        self.graphicsView_1_1.raise_()
        self.graphicsView_1_2.raise_()
        self.button_load_img_1.raise_()
        self.button_save_img_1.raise_()
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.graphicsView_2_2 = QtWidgets.QGraphicsView(self.tab_2)
        self.graphicsView_2_2.setGeometry(QtCore.QRect(950, 10, 700, 700))
        self.graphicsView_2_2.setObjectName("graphicsView_2_2")
        self.graphicsView_2_1 = QtWidgets.QGraphicsView(self.tab_2)
        self.graphicsView_2_1.setGeometry(QtCore.QRect(240, 10, 700, 700))
        self.graphicsView_2_1.setObjectName("graphicsView_2_1")
        self.button_save_img_2 = QtWidgets.QPushButton(self.tab_2)
        self.button_save_img_2.setGeometry(QtCore.QRect(1170, 720, 291, 61))
        self.button_save_img_2.setObjectName("button_save_img_2")
        self.button_load_img_2 = QtWidgets.QPushButton(self.tab_2)
        self.button_load_img_2.setGeometry(QtCore.QRect(450, 720, 291, 61))
        self.button_load_img_2.setObjectName("button_load_img_2")
        self.groupBox_2 = QtWidgets.QGroupBox(self.tab_2)
        self.groupBox_2.setGeometry(QtCore.QRect(10, 30, 221, 681))
        self.groupBox_2.setObjectName("groupBox_2")
        self.button_filt_rectangle = QtWidgets.QPushButton(self.groupBox_2)
        self.button_filt_rectangle.setGeometry(QtCore.QRect(10, 40, 131, 51))
        self.button_filt_rectangle.setStyleSheet("font: 9pt \"MS Shell Dlg 2\";")
        self.button_filt_rectangle.setObjectName("button_filt_rectangle")
        self.button_filt_median = QtWidgets.QPushButton(self.groupBox_2)
        self.button_filt_median.setGeometry(QtCore.QRect(10, 110, 131, 51))
        self.button_filt_median.setStyleSheet("font: 9pt \"MS Shell Dlg 2\";")
        self.button_filt_median.setObjectName("button_filt_median")
        self.button_filt_gauusa = QtWidgets.QPushButton(self.groupBox_2)
        self.button_filt_gauusa.setGeometry(QtCore.QRect(10, 180, 131, 51))
        self.button_filt_gauusa.setStyleSheet("font: 9pt \"MS Shell Dlg 2\";")
        self.button_filt_gauusa.setObjectName("button_filt_gauusa")
        self.button_filt_sigma = QtWidgets.QPushButton(self.groupBox_2)
        self.button_filt_sigma.setGeometry(QtCore.QRect(10, 290, 131, 51))
        self.button_filt_sigma.setStyleSheet("font: 9pt \"MS Shell Dlg 2\";")
        self.button_filt_sigma.setObjectName("button_filt_sigma")
        self.comboBox_rectangle = QtWidgets.QComboBox(self.groupBox_2)
        self.comboBox_rectangle.setGeometry(QtCore.QRect(160, 40, 51, 51))
        self.comboBox_rectangle.setObjectName("comboBox_rectangle")
        self.comboBox_rectangle.addItem("")
        self.comboBox_rectangle.addItem("")
        self.comboBox_median = QtWidgets.QComboBox(self.groupBox_2)
        self.comboBox_median.setGeometry(QtCore.QRect(160, 110, 51, 51))
        self.comboBox_median.setObjectName("comboBox_median")
        self.comboBox_median.addItem("")
        self.comboBox_median.addItem("")
        self.doubleSpinBox_gauusa_sigma = QtWidgets.QDoubleSpinBox(self.groupBox_2)
        self.doubleSpinBox_gauusa_sigma.setGeometry(QtCore.QRect(160, 240, 51, 31))
        self.doubleSpinBox_gauusa_sigma.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.doubleSpinBox_gauusa_sigma.setMinimum(0.01)
        self.doubleSpinBox_gauusa_sigma.setMaximum(5.0)
        self.doubleSpinBox_gauusa_sigma.setProperty("value", 1.0)
        self.doubleSpinBox_gauusa_sigma.setObjectName("doubleSpinBox_gauusa_sigma")
        self.label_9 = QtWidgets.QLabel(self.groupBox_2)
        self.label_9.setGeometry(QtCore.QRect(50, 240, 131, 31))
        self.label_9.setStyleSheet("font: 12pt \"MS Shell Dlg 2\";")
        self.label_9.setObjectName("label_9")
        self.doubleSpinBox__sigma_sigma = QtWidgets.QDoubleSpinBox(self.groupBox_2)
        self.doubleSpinBox__sigma_sigma.setGeometry(QtCore.QRect(160, 350, 51, 31))
        self.doubleSpinBox__sigma_sigma.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.doubleSpinBox__sigma_sigma.setMinimum(0.01)
        self.doubleSpinBox__sigma_sigma.setMaximum(5.0)
        self.doubleSpinBox__sigma_sigma.setProperty("value", 1.0)
        self.doubleSpinBox__sigma_sigma.setObjectName("doubleSpinBox__sigma_sigma")
        self.label_10 = QtWidgets.QLabel(self.groupBox_2)
        self.label_10.setGeometry(QtCore.QRect(50, 350, 81, 31))
        self.label_10.setStyleSheet("font: 12pt \"MS Shell Dlg 2\";")
        self.label_10.setObjectName("label_10")
        self.button_map_absolute_diff = QtWidgets.QPushButton(self.groupBox_2)
        self.button_map_absolute_diff.setGeometry(QtCore.QRect(10, 400, 201, 51))
        self.button_map_absolute_diff.setStyleSheet("font: 9pt \"MS Shell Dlg 2\";")
        self.button_map_absolute_diff.setObjectName("button_map_absolute_diff")
        self.console_2 = QtWidgets.QTextBrowser(self.groupBox_2)
        self.console_2.setGeometry(QtCore.QRect(10, 510, 201, 151))
        self.console_2.setObjectName("console_2")
        self.sharpness_factor_1 = QtWidgets.QLabel(self.groupBox_2)
        self.sharpness_factor_1.setGeometry(QtCore.QRect(20, 470, 121, 31))
        self.sharpness_factor_1.setStyleSheet("font: 12pt \"MS Shell Dlg 2\";")
        self.sharpness_factor_1.setObjectName("sharpness_factor_1")
        self.sharpness_factor_1_1 = QtWidgets.QDoubleSpinBox(self.groupBox_2)
        self.sharpness_factor_1_1.setGeometry(QtCore.QRect(160, 470, 51, 31))
        self.sharpness_factor_1_1.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.sharpness_factor_1_1.setMinimum(-100.0)
        self.sharpness_factor_1_1.setMaximum(100.0)
        self.sharpness_factor_1_1.setProperty("value", 0.0)
        self.sharpness_factor_1_1.setObjectName("sharpness_factor_1_1")
        self.tabWidget.addTab(self.tab_2, "")
        self.tab_5 = QtWidgets.QWidget()
        self.tab_5.setObjectName("tab_5")
        self.graphicsView_3_2 = QtWidgets.QGraphicsView(self.tab_5)
        self.graphicsView_3_2.setGeometry(QtCore.QRect(950, 10, 700, 700))
        self.graphicsView_3_2.setObjectName("graphicsView_3_2")
        self.button_load_img_5_1 = QtWidgets.QPushButton(self.tab_5)
        self.button_load_img_5_1.setGeometry(QtCore.QRect(1010, 720, 291, 61))
        self.button_load_img_5_1.setObjectName("button_load_img_5_1")
        self.graphicsView_3_1 = QtWidgets.QGraphicsView(self.tab_5)
        self.graphicsView_3_1.setGeometry(QtCore.QRect(240, 10, 700, 700))
        self.graphicsView_3_1.setObjectName("graphicsView_3_1")
        self.button_load_img_5 = QtWidgets.QPushButton(self.tab_5)
        self.button_load_img_5.setGeometry(QtCore.QRect(450, 720, 291, 61))
        self.button_load_img_5.setObjectName("button_load_img_5")
        self.groupBox_5 = QtWidgets.QGroupBox(self.tab_5)
        self.groupBox_5.setGeometry(QtCore.QRect(10, 30, 221, 671))
        self.groupBox_5.setObjectName("groupBox_5")
        self.button_accept_sharp = QtWidgets.QPushButton(self.groupBox_5)
        self.button_accept_sharp.setGeometry(QtCore.QRect(10, 40, 121, 51))
        self.button_accept_sharp.setStyleSheet("font: 9pt \"MS Shell Dlg 2\";")
        self.button_accept_sharp.setObjectName("button_accept_sharp")
        self.comboBox_sharp = QtWidgets.QComboBox(self.groupBox_5)
        self.comboBox_sharp.setGeometry(QtCore.QRect(160, 40, 51, 51))
        self.comboBox_sharp.setObjectName("comboBox_sharp")
        self.comboBox_sharp.addItem("")
        self.comboBox_sharp.addItem("")
        self.label_11 = QtWidgets.QLabel(self.groupBox_5)
        self.label_11.setGeometry(QtCore.QRect(20, 110, 81, 31))
        self.label_11.setStyleSheet("font: 12pt \"MS Shell Dlg 2\";")
        self.label_11.setObjectName("label_11")
        self.doubleSpinBox_lymbda = QtWidgets.QDoubleSpinBox(self.groupBox_5)
        self.doubleSpinBox_lymbda.setGeometry(QtCore.QRect(160, 110, 51, 31))
        self.doubleSpinBox_lymbda.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.doubleSpinBox_lymbda.setMinimum(-5.0)
        self.doubleSpinBox_lymbda.setMaximum(5.0)
        self.doubleSpinBox_lymbda.setObjectName("doubleSpinBox_lymbda")
        self.console_3 = QtWidgets.QTextBrowser(self.groupBox_5)
        self.console_3.setGeometry(QtCore.QRect(10, 510, 201, 151))
        self.console_3.setObjectName("console_3")
        self.sharpness_factor_2 = QtWidgets.QLabel(self.groupBox_5)
        self.sharpness_factor_2.setGeometry(QtCore.QRect(20, 470, 121, 31))
        self.sharpness_factor_2.setStyleSheet("font: 12pt \"MS Shell Dlg 2\";")
        self.sharpness_factor_2.setObjectName("sharpness_factor_2")
        self.sharpness_factor_2_2 = QtWidgets.QDoubleSpinBox(self.groupBox_5)
        self.sharpness_factor_2_2.setGeometry(QtCore.QRect(160, 470, 51, 31))
        self.sharpness_factor_2_2.setReadOnly(False)
        self.sharpness_factor_2_2.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.sharpness_factor_2_2.setMinimum(-100.0)
        self.sharpness_factor_2_2.setMaximum(100.0)
        self.sharpness_factor_2_2.setProperty("value", 0.0)
        self.sharpness_factor_2_2.setObjectName("sharpness_factor_2_2")
        self.button_save_img_5 = QtWidgets.QPushButton(self.tab_5)
        self.button_save_img_5.setGeometry(QtCore.QRect(1310, 720, 291, 61))
        self.button_save_img_5.setObjectName("button_save_img_5")
        self.tabWidget.addTab(self.tab_5, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1680, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        # endregion

        self.button_load_img_1.clicked.connect(lambda: self.load_image(1))
        self.button_load_img_2.clicked.connect(lambda: self.load_image(2))
        self.button_load_img_5.clicked.connect(lambda: self.load_image(3))
        self.button_load_img_5_1.clicked.connect(lambda: self.load_image(4))

        self.button_save_img_1.clicked.connect(lambda: self.save_image(1))
        self.button_save_img_2.clicked.connect(lambda: self.save_image(2))
        self.button_save_img_5.clicked.connect(lambda: self.save_image(3))

        self.button_log_1.clicked.connect(self.button_log_clicked)
        self.button_bin_1.clicked.connect(self.button_binary_clicked)
        self.button_exp_1.clicked.connect(self.button_gamma_clicked)
        self.button_cut_bright.clicked.connect(lambda: self.button_clip_clicked(False))
        self.button_cut_bright_2.clicked.connect(lambda: self.button_clip_clicked(True))

        self.button_filt_rectangle.clicked.connect(self.button_rectangular_filter_clicked)
        self.button_filt_median.clicked.connect(self.button_median_filter_clicked)
        self.button_filt_gauusa.clicked.connect(self.button_gaussian_filter_clicked)
        self.button_filt_sigma.clicked.connect(self.button_sigma_filter_clicked)
        self.button_map_absolute_diff.clicked.connect(self.button_absolut_diff)

        self.button_accept_sharp.clicked.connect(self.button_sharpening_clicked)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.button_load_img_1.setText(_translate("MainWindow", "Загрузить изображение"))
        self.button_save_img_1.setText(_translate("MainWindow", "Сохранить изображение"))
        self.groupBox.setTitle(_translate("MainWindow", "Преобразование"))
        self.button_exp_1.setText(_translate("MainWindow", "Степенное"))
        self.button_bin_1.setText(_translate("MainWindow", "Бинарное"))
        self.label.setText(_translate("MainWindow", "Гамма"))
        self.label_2.setText(_translate("MainWindow", "Порог"))
        self.label_3.setText(_translate("MainWindow", "Левая граница"))
        self.label_4.setText(_translate("MainWindow", "Правая граница"))
        self.button_cut_bright.setText(_translate("MainWindow", "Вырезание яркости"))
        self.button_log_1.setText(_translate("MainWindow", "Логарифмическое"))
        self.button_cut_bright_2.setText(_translate("MainWindow", "Вырезание яркости (const)"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Цветность"))
        self.button_save_img_2.setText(_translate("MainWindow", "Сохранить изображение"))
        self.button_load_img_2.setText(_translate("MainWindow", "Загрузить изображение"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Фильтр"))
        self.button_filt_rectangle.setText(_translate("MainWindow", "Прямоугольный"))
        self.button_filt_median.setText(_translate("MainWindow", "Медианный"))
        self.button_filt_gauusa.setText(_translate("MainWindow", "Гаусса"))
        self.button_filt_sigma.setText(_translate("MainWindow", "Сигма"))
        self.comboBox_rectangle.setItemText(0, _translate("MainWindow", "3"))
        self.comboBox_rectangle.setItemText(1, _translate("MainWindow", "5"))
        self.comboBox_median.setItemText(0, _translate("MainWindow", "3"))
        self.comboBox_median.setItemText(1, _translate("MainWindow", "5"))
        self.label_9.setText(_translate("MainWindow", "Сигма"))
        self.label_10.setText(_translate("MainWindow", "Сигма"))
        self.button_map_absolute_diff.setText(_translate("MainWindow", "Карта абсолютной разности"))
        self.sharpness_factor_1.setText(_translate("MainWindow", "Коэф. резкости"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Сглаживание"))
        self.button_load_img_5_1.setText(_translate("MainWindow", "Загрузить изображение"))
        self.button_load_img_5.setText(_translate("MainWindow", "Загрузить изображение"))
        self.groupBox_5.setTitle(_translate("MainWindow", "Резкость"))
        self.button_accept_sharp.setText(_translate("MainWindow", "Применить"))
        self.comboBox_sharp.setItemText(0, _translate("MainWindow", "3"))
        self.comboBox_sharp.setItemText(1, _translate("MainWindow", "5"))
        self.label_11.setText(_translate("MainWindow", "Лямбда"))
        self.sharpness_factor_2.setText(_translate("MainWindow", "Коэф. резкости"))
        self.button_save_img_5.setText(_translate("MainWindow", "Сохранить изображение"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_5), _translate("MainWindow", "Резкость"))

    def load_image(self, choice_sheet):
        filename, _ = QFileDialog.getOpenFileName(None, "Open Image File", "", "Images (*.png *.jpg *.bmp)")
        if filename:  # Проверяем, был ли выбран файл
            if choice_sheet != 4:
                self.image_processing = ImageProcessingFast(filename)
            # self.image_processing = ImageProcessing(filename)

            pixmap = QtGui.QPixmap(filename)  # Создаем QPixmap изображения
            scene = QtWidgets.QGraphicsScene()  # Создаем QGraphicsScene
            scene.addPixmap(pixmap)  # Добавляем QPixmap в QGraphicsScene

            if choice_sheet == 1:
                self.graphicsView_1_2.setScene(scene)  # Устанавливаем QGraphicsScene в QGraphicsView
                self.graphicsView_1_2.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
            elif choice_sheet == 2:
                self.graphicsView_2_1.setScene(scene)  # Устанавливаем QGraphicsScene в QGraphicsView
                self.graphicsView_2_1.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
            elif choice_sheet == 3:
                self.graphicsView_3_1.setScene(scene)  # Устанавливаем QGraphicsScene в QGraphicsView
                self.graphicsView_3_1.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
            elif choice_sheet == 4:
                self.image_processing.process_image = self.image_processing.load_image(filename)
                self.graphicsView_3_2.setScene(scene)  # Устанавливаем QGraphicsScene в QGraphicsView
                self.graphicsView_3_2.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

            self.image_exist = True

    def save_image(self, choice_sheet):
        if self.image_exist:
            if choice_sheet == 1:
                pixmap = self.graphicsView_1_1.scene().items()[0].pixmap()
            elif choice_sheet == 2:
                pixmap = self.graphicsView_2_2.scene().items()[0].pixmap()
            elif choice_sheet == 3:
                pixmap = self.graphicsView_3_2.scene().items()[0].pixmap()

            # Определяем путь для сохранения изображения
            filename, _ = QFileDialog.getSaveFileName(None, "Save Image File", "", "Images (*.png *.jpg *.bmp)")

            # Если путь выбран, сохраняем изображение
            if filename:
                pixmap.save(filename)
                self.show_into_message("Изображение сохранено", f"Изображение успешно сохранено по пути: {filename}")
        else:
            self.show_error_message("Ошибка", "Изображение не загружено")

    def display_image(self, transformed_image, choice):
        # Сохраняем изображение с помощью OpenCV
        cv2.imwrite("transformed_image.png", transformed_image)

        # Открываем сохраненное изображение
        saved_image = cv2.imread("transformed_image.png")

        # Преобразуем numpy.ndarray в QImage
        q_image = QImage(saved_image.data, saved_image.shape[1], saved_image.shape[0], saved_image.strides[0],
                         QImage.Format_RGB888)

        # Создаем QPixmap из QImage
        pixmap = QPixmap.fromImage(q_image)

        scene = QtWidgets.QGraphicsScene()  # Создаем QGraphicsScene
        scene.addPixmap(pixmap)  # Добавляем QPixmap в QGraphicsScene
        match choice:
            case 1:
                self.graphicsView_1_1.setScene(scene)  # Устанавливаем QGraphicsScene в QGraphicsView
                self.graphicsView_1_1.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
            case 2:
                self.graphicsView_2_2.setScene(scene)  # Устанавливаем QGraphicsScene в QGraphicsView
                self.graphicsView_2_2.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
            case 3:
                self.graphicsView_3_2.setScene(scene)  # Устанавливаем QGraphicsScene в QGraphicsView
                self.graphicsView_3_2.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

        # Удаляем сохраненное изображение
        os.remove("transformed_image.png")

    @staticmethod
    def show_into_message(title, message):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.exec_()

    @staticmethod
    def show_error_message(title, message):
        # app = QApplication([])
        error_box = QMessageBox()
        error_box.setWindowTitle(title)
        error_box.setText(message)
        error_box.setIcon(QMessageBox.Critical)
        error_box.exec_()

    def measure_time_and_run_method(self, method, *args, **kwargs):
        if self.image_exist:
            timer = QElapsedTimer()
            timer.start()
            result = method(*args)
            elapsed_time = timer.elapsed() / 1000  # Получаем время в секундах
            match kwargs["choice"]:
                case 1:
                    self.console_1.setText(kwargs['message'].format(elapsed_time))
                case 2:
                    self.console_2.setText(kwargs['message'].format(elapsed_time))
                    if method.__name__ != self.image_processing.absolute_difference.__name__:
                        self.sharpness_factor_1_1.setValue(self.image_processing.compute_sharpness_coefficient(
                            self.image_processing.original_image, result))
                case 3:
                    self.console_3.setText(kwargs['message'].format(elapsed_time))
                    self.sharpness_factor_2_2.setValue(self.image_processing.compute_sharpness_coefficient(
                        self.image_processing.original_image, result))
            self.display_image(result, kwargs['choice'])
        else:
            self.show_error_message("Ошибка", "Изображение не загружено")

    # region Цветность
    def button_log_clicked(self):
        message = "Время работы логарифмического преобразования: {:.2f} с"
        self.measure_time_and_run_method(self.image_processing.logarithmic_transform, message=message, choice=1)

    def button_gamma_clicked(self):
        message = "Время работы степенного преобразования: {:.2f} с"
        gamma = self.doubleSpinBox_gamma.value()
        self.measure_time_and_run_method(self.image_processing.gamma_transform, gamma, message=message, choice=1)

    def button_binary_clicked(self):
        message = "Время работы бинарного преобразования: {:.2f} с"
        threshold = self.spinbox_exp_porog_1.value()
        self.measure_time_and_run_method(self.image_processing.binary_transform, threshold, message=message, choice=1)

    def button_clip_clicked(self, choice):
        message = "Время работы обрезки изображения: {:.2f} с"
        lower_bound = self.spinbox_bin_border_1.value()
        upper_bound = self.spinbox_bin_border_2.value()
        self.measure_time_and_run_method(self.image_processing.clip_image, lower_bound, upper_bound, choice,
                                         message=message, choice=1)

    # endregion

    # region Сглаживание
    def button_rectangular_filter_clicked(self):
        if self.image_exist:
            kernel_size_text = self.comboBox_rectangle.currentText()
            kernel_size = int(kernel_size_text)
            message = "Время работы прямоугольного фильтра: {:.2f} с"
            self.measure_time_and_run_method(self.image_processing.apply_rectangular_filter, kernel_size,
                                             message=message, choice=2)
        else:
            self.show_error_message("Ошибка", "Изображение не загружено")

    def button_median_filter_clicked(self):
        if self.image_exist:
            kernel_size_text = self.comboBox_median.currentText()
            kernel_size = int(kernel_size_text)
            message = "Время работы медианного фильтра: {:.2f} с"
            self.measure_time_and_run_method(self.image_processing.apply_median_filter, kernel_size, message=message,
                                             choice=2)
        else:
            self.show_error_message("Ошибка", "Изображение не загружено")

    def button_gaussian_filter_clicked(self):
        if self.image_exist:
            sigma = self.doubleSpinBox_gauusa_sigma.value()
            message = "Время работы гауссова фильтра: {:.2f} с"
            self.measure_time_and_run_method(self.image_processing.apply_gaussian_filter, sigma,
                                             message=message, choice=2)
        else:
            self.show_error_message("Ошибка", "Изображение не загружено")

    def button_sigma_filter_clicked(self):
        if self.image_exist:
            sigma_r = self.doubleSpinBox__sigma_sigma.value()
            message = "Время работы сигма-фильтра: {:.2f} с"
            self.measure_time_and_run_method(self.image_processing.apply_sigma_filter, sigma_r,
                                             message=message, choice=2)
        else:
            self.show_error_message("Ошибка", "Изображение не загружено")

    def button_absolut_diff(self):
        if self.image_exist:
            # Получение изображений из графических виджетов
            image1_path = 'image1_temp.jpg'
            image2_path = 'image2_temp.jpg'
            pixmap1 = self.graphicsView_2_1.grab()
            pixmap2 = self.graphicsView_2_2.grab()
            pixmap1.save(image1_path)
            pixmap2.save(image2_path)

            # Загрузка изображений с помощью OpenCV
            image1 = cv2.imread(image1_path).astype(np.int32)
            image2 = cv2.imread(image2_path).astype(np.int32)

            # Применение абсолютной разности
            message = "Время работы абсолютной разности: {:.2f} с"
            self.measure_time_and_run_method(self.image_processing.absolute_difference, image1, image2, message=message,
                                             choice=2)

            # Удаление временных файлов
            os.remove(image1_path)
            os.remove(image2_path)
        else:
            self.show_error_message("Ошибка", "Изображение не загружено")

    # endregion

    def button_sharpening_clicked(self):
        if self.image_exist and self.image_processing.process_image is not None:
            lambda_sh = self.doubleSpinBox_lymbda.value()
            message = "Время работы увеличения резкости: {:.2f} с"
            self.measure_time_and_run_method(self.image_processing.sharpening, lambda_sh, message=message, choice=3)
        else:
            self.show_error_message("Ошибка", "Изображение не загружено")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)

    MainWindow.show()
    sys.exit(app.exec_())
