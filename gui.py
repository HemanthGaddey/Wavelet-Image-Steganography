import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QHBoxLayout, QComboBox, QSlider
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import cv2
import numpy as np


class ImageApp(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Image Viewer')
        self.setGeometry(100, 100, 800, 600)

        self.layout = QVBoxLayout()

        # Image labels
        self.image_labels = [QLabel(), QLabel()]
        for label in self.image_labels:
            self.layout.addWidget(label)

        # Button to load images
        self.load_button = QPushButton('Load Images')
        self.load_button.clicked.connect(self.loadImages)
        self.layout.addWidget(self.load_button)

        # Dropdown selection for third image
        self.dropdown = QComboBox()
        self.dropdown.addItem('Image 1')
        self.dropdown.addItem('Image 2')
        self.dropdown.currentIndexChanged.connect(self.updateThirdImage)
        self.layout.addWidget(self.dropdown)

        # Slider for image blending
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(50)
        self.slider.valueChanged.connect(self.blendImages)
        self.layout.addWidget(self.slider)

        # Label for third image
        self.blended_image_label = QLabel()
        self.layout.addWidget(self.blended_image_label)

        self.setLayout(self.layout)

    def loadImages(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Images", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif)", options=options)

        if len(file_paths) >= 2:
            self.images = [cv2.imread(file_paths[0]), cv2.imread(file_paths[1])]
            self.displayImages()

    def displayImages(self):
        for i, image in enumerate(self.images):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, ch = image.shape
            bytes_per_line = ch * w
            q_img = QPixmap.fromImage(
                QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888))
            self.image_labels[i].setPixmap(q_img.scaled(400, 300, Qt.KeepAspectRatio))

    def updateThirdImage(self, index):
        if hasattr(self, 'images') and len(self.images) >= 2:
            alpha = self.slider.value() / 100.0
            blended_image = cv2.addWeighted(
                self.images[0], alpha, self.images[1], (1 - alpha), 0)
            if index == 0:
                self.displayBlendedImage(self.images[0])
            else:
                self.displayBlendedImage(self.images[1])

    def displayBlendedImage(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = image.shape
        bytes_per_line = ch * w
        q_img = QPixmap.fromImage(
            QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888))
        self.blended_image_label.setPixmap(q_img.scaled(400, 300, Qt.KeepAspectRatio))

    def blendImages(self):
        alpha = self.slider.value() / 100.0
        blended_image = cv2.addWeighted(
            self.images[0], alpha, self.images[1], (1 - alpha), 0)
        self.displayBlendedImage(blended_image)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageApp()
    window.show()
    sys.exit(app.exec_())
