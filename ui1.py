import sys
import os
import shutil
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QListWidget, QMessageBox


class VideoMoverApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('ui1')
        self.setGeometry(200, 100, 1500, 800)

        layout = QVBoxLayout()

        self.btnSelectSource = QPushButton('选择源文件夹')
        self.btnSelectSource.clicked.connect(self.selectSourceFolder)
        layout.addWidget(self.btnSelectSource)

        self.btnSelectDestination = QPushButton('选择目标文件夹')
        self.btnSelectDestination.clicked.connect(self.selectDestinationFolder)
        layout.addWidget(self.btnSelectDestination)

        self.videoList = QListWidget()
        layout.addWidget(self.videoList)

        self.btnMoveVideos = QPushButton('移动视频')
        self.btnMoveVideos.clicked.connect(self.moveVideos)
        layout.addWidget(self.btnMoveVideos)

        self.setLayout(layout)

        self.sourceFolder = ''
        self.destinationFolder = ''

    def selectSourceFolder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择源文件夹")
        if folder:
            self.sourceFolder = folder
            self.loadVideos()

    def selectDestinationFolder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择目标文件夹")
        if folder:
            self.destinationFolder = folder

    def loadVideos(self):
        self.videoList.clear()
        for file_name in os.listdir(self.sourceFolder):
            if file_name.endswith(('.mp4', '.avi', '.mov')):
                self.videoList.addItem(file_name)

    def moveVideos(self):
        selectedItems = self.videoList.selectedItems()
        if not selectedItems:
            QMessageBox.warning(self, "警告", "请至少选择一个视频文件。")
            return
        if not self.destinationFolder:
            QMessageBox.warning(self, "警告", "请选择一个目标文件夹。")
            return

        for item in selectedItems:
            sourcePath = os.path.join(self.sourceFolder, item.text())
            destinationPath = os.path.join(self.destinationFolder, item.text())
            shutil.move(sourcePath, destinationPath)

        QMessageBox.information(self, "完成", "视频文件已成功移动。")
        self.loadVideos()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = VideoMoverApp()
    ex.show()
    sys.exit(app.exec_())
