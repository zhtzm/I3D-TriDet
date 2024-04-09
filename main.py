from global_ui import VideoMoverApp
import sys
from PyQt5.QtWidgets import QApplication


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = VideoMoverApp()
    ex.show()
    sys.exit(app.exec_())