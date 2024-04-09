import sys
import os
import shutil
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QListWidget, QMessageBox
from vedio2predict import default_predict2show


class VideoMoverApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('ui1')
        self.setGeometry(500, 200, 700, 700)

        layout = QVBoxLayout()

        self.videoList = QListWidget()
        layout.addWidget(self.videoList)

        self.btnMoveVideos = QPushButton('移动视频')
        self.btnMoveVideos.clicked.connect(self.moveVideos)
        layout.addWidget(self.btnMoveVideos)

        self.btnPredict = QPushButton('预测')
        self.btnPredict.clicked.connect(self.predictVideo)
        layout.addWidget(self.btnPredict)

        self.setLayout(layout)

        self.destinationFolder = './tmp/video'

        if not os.path.exists(self.destinationFolder):
            os.makedirs(self.destinationFolder)

        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        files = [url.toLocalFile() for url in event.mimeData().urls() if
                 url.toLocalFile().endswith(('.mp4', '.avi', '.mov'))]
        for file_path in files:
            self.videoList.addItem(file_path)

    def moveVideos(self):
        if not self.videoList.count():
            QMessageBox.warning(self, "警告", "列表中没有视频文件。")
            return

        failed_files = []
        itemsToMove = []
        for index in range(self.videoList.count()):
            itemsToMove.append(self.videoList.item(index).text())

        self.videoList.clear()

        for sourcePath in itemsToMove:
            destinationPath = os.path.join(self.destinationFolder, os.path.basename(sourcePath))

            if os.path.exists(destinationPath):
                failed_files.append(os.path.basename(sourcePath) + " (文件已存在)")
            else:
                try:
                    shutil.move(sourcePath, destinationPath)
                    self.videoList.addItem(destinationPath)
                except Exception as e:
                    failed_files.append(os.path.basename(sourcePath) + " (移动失败: {})".format(e))

        if failed_files:
            QMessageBox.warning(self, "部分文件移动失败", "\n".join(failed_files))
        else:
            QMessageBox.information(self, "完成", "所有选中的视频文件已成功移动。")

    def predictVideo(self):
        selectedItems = self.videoList.selectedItems()
        if not selectedItems:
            QMessageBox.warning(self, "警告", "请先选择一个视频文件。")
            return

        for item in selectedItems:
            default_predict2show()
