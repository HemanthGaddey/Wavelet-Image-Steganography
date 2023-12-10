import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QPushButton, QWidget, QSlider, QHBoxLayout, QComboBox, QAction
from PyQt5.QtGui import QPixmap, QImage, QPainter, QIcon
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pywt
import pywt.data
import cv2
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

class ImageBlenderGUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.init_ui()
        self.wavelet = 'haar'
        self.image3 = []

    def init_ui(self):
        self.setWindowTitle("Group 9 - Encoder")

        central_widget = QWidget()
        layout = QVBoxLayout()

        # Image labels
        image_layout = QHBoxLayout()

        self.image1_label = QLabel()
        self.image2_label = QLabel()
        self.image3_label = QLabel()

        image_layout.addWidget(self.image1_label)
        image_layout.addWidget(self.image2_label)
        image_layout.addWidget(self.image3_label)

        layout.addLayout(image_layout)

        # Slider
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setMinimum(0)
        self.alpha_slider.setMaximum(100)
        self.alpha_slider.setValue(50)  # Initial value
        self.alpha_slider.valueChanged.connect(self.update_alpha)

        layout.addWidget(self.alpha_slider)

        # Dropdown for wavelet functions
        self.wavelet_dropdown = QComboBox()
        self.wavelet_dropdown.addItems(["haar","db1", "coif1", "sym2", "dmey","bior1.1","rbio1.1"])
        layout.addWidget(self.wavelet_dropdown)

        # Buttons
        self.save_image_button = QPushButton("Save Image")
        self.save_image_button.clicked.connect(self.save_image)
        layout.addWidget(self.save_image_button)

        self.load_image1_button = QPushButton("Load Cover Image")
        self.load_image1_button.clicked.connect(self.load_image1)
        layout.addWidget(self.load_image1_button)

        self.load_image2_button = QPushButton("Load Payload Image")
        self.load_image2_button.clicked.connect(self.load_image2)
        layout.addWidget(self.load_image2_button)

        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def load_image1(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Image 1", "", "Image Files (*.png *.jpg *.bmp)")
        self.cover_img = cv2.resize(cv2.imread(filename), (256,256))
        if filename:
            self.image1 = QImage(filename).scaled(256, 256, Qt.KeepAspectRatio)
            pixmap = QPixmap.fromImage(self.image1)
            self.image1_label.setPixmap(pixmap)
            self.image1_label.setScaledContents(True)

    def load_image2(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Image 2", "", "Image Files (*.png *.jpg *.bmp)")
        self.payload_img = cv2.resize(cv2.imread(filename), (256,256))
        if filename:
            self.image2 = QImage(filename).scaled(256, 256, Qt.KeepAspectRatio)
            pixmap = QPixmap.fromImage(self.image2)
            self.image2_label.setPixmap(pixmap)
            self.image2_label.setScaledContents(True)

    def update_alpha(self):
        alpha = self.alpha_slider.value() / 100.0
        # c_img = qimage_to_cvimage(self.image1)
        # p_img = qimage_to_cvimage(self.image2)
        # c_img = cv2.resize(c_img, (256,256))
        # p_img = cv2.resize(p_img, (256,256))
        s_img = self.get_stego_image(self.cover_img, self.payload_img, alpha)
        print(s_img.dtype)
        s_q_img = cvimage_to_qimage(s_img)
        print(s_img)
        #blended_image = self.blend_images(alpha)
        self.pixmap = QPixmap.fromImage(s_q_img)
        self.image3_label.setPixmap(self.pixmap)
        self.image3_label.setScaledContents(True)

    def blend_images(self, alpha):
        blended = QImage(256, 256, QImage.Format_ARGB32)
        painter = QPainter(blended)
        painter.setCompositionMode(QPainter.CompositionMode_DestinationOver)
        painter.setOpacity(alpha)
        painter.drawImage(0, 0, self.image1)
        painter.setOpacity(1.0 - alpha)
        painter.drawImage(0, 0, self.image2)
        painter.end()
        return blended


    def save_image(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG Files (*.png)")
        if filename:
            self.pixmap.save(filename)
    
    def get_stego_image(self,cover_img, payload_img, alpha=0.01):
        wavelet = self.wavelet_dropdown.currentText()
        
        print(alpha)

        ## Cover Image Pre-processing

        # Convert to Floating Type
        cover_imgf = cover_img.astype(np.float64)

        # Separate RGB Components
        separated_components = cv2.split(cover_imgf)
        separated_components_c = separated_components

        # Normalize RGB Components
        normalized_components = []
        for i in range(3):
            normalized_components.append(separated_components[i]/255)
            #normalized_components.append(separated_components[i]/np.max(separated_components[i]))

        # DWT on Each Normalized Component
        frequency_components_cover = []
        for i in range(3):
            frequency_components_cover.append(pywt.dwt2(normalized_components[i], wavelet))

        ## Payload Image Pre-processing

        # Convert to Floating Type
        payload_imgf = payload_img.astype(np.float64)

        # Separate RGB Components
        separated_components = cv2.split(payload_imgf)

        # Normalize RGB Components
        normalized_components = []
        for i in range(3):
            normalized_components.append(separated_components[i]/255)
            #normalized_components.append(separated_components[i]/np.max(separated_components[i]))

        # DWT on Each Normalized Component
        frequency_components_payload = []
        for i in range(3):
            frequency_components_payload.append(pywt.dwt2(normalized_components[i], wavelet))

        a = alpha
        frequency_fused_components = []
        cc = frequency_components_cover
        cp = frequency_components_payload
        cs = []
        for i in range(3):
            # Fusion process using weighted combination of coefficients
            cA_fused = cc[i][0] + a * (cp[i][0])
            cH_fused = cc[i][1][0] + a * (cp[i][1][0])
            cV_fused = cc[i][1][1] + a * (cp[i][1][1])
            cD_fused = cc[i][1][2] + a * (cp[i][1][2])

            # Append the tuple of approximation and details coefficients
            cs.append((cA_fused, (cH_fused, cV_fused, cD_fused)))
            
        # Perform inverse DWT
        fused_image_components = []
        for i in range(3):
            fused_image_components.append(pywt.idwt2(cs[i], wavelet))

        stego_img = cv2.merge((
            (fused_image_components[0] * np.max(separated_components_c[0])),
            (fused_image_components[1] * np.max(separated_components_c[1])),
            (fused_image_components[2] * np.max(separated_components_c[2]))
        ))

        stego_img = cv2.merge((
            (fused_image_components[0] * 255),
            (fused_image_components[1] * 255),
            (fused_image_components[2] * 255),
        ))

        print(stego_img.dtype)
        return stego_img

def qimage_to_cvimage(q_image):
    q_image = q_image.convertToFormat(QImage.Format_RGB888)  # Convert QImage to RGB format
    width = q_image.width()
    height = q_image.height()

    # Extract pixel data and construct OpenCV image
    ptr = q_image.bits()
    ptr.setsize(q_image.byteCount())
    cv_image = np.array(ptr).reshape(height, width, 3)  # 3 for RGB channels

    # Convert RGB to BGR (OpenCV uses BGR)
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

    return cv_image

def display_img(img,title='untitled'):
    if(img.dtype != np.uint8):
        img = cv2.normalize(img, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC1) # To convert floating type images to uint8
    cv2.imshow(title,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print('done')

def cvimage_to_qimage(cv_image):
    #display_img(cv_image, 'input')
    # Check the depth of the image and convert if it's 64-bit float
    if cv_image.dtype == np.float64:
        cv_image = cv2.convertScaleAbs(cv_image)  # Convert to 8-bit depth


    #display_img(cv_image, 'before scaleconv')
    height, width, channels = cv_image.shape
    bytes_per_line = channels * width
    #cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR, converting to RGB
    
    #display_img(cv_image, 'after bgr2rgb')
    # Create QImage from the OpenCV image data
    qimage = QImage(cv_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
    qimage = qimage.rgbSwapped()  # Swapping the red and blue channels for QImage compatibility

    return qimage

# def cvimage_to_qimage(cv_image):
#     height, width, channels = cv_image.shape
#     bytes_per_line = channels * width
#     cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR, converting to RGB

#     # Create QImage from the OpenCV image data
#     qimage = QImage(cv_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
#     qimage = qimage.rgbSwapped()  # Swapping the red and blue channels for QImage compatibility

#     return qimage


def main():
    app = QApplication(sys.argv)
    window = ImageBlenderGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
