import sys
from login_window import LoginWindow
from PyQt5.QtWidgets import QApplication
if __name__ == '__main__':

    app = QApplication(sys.argv)
    with open('../resources/QSS-master/MacOS.qss') as f:
        qss = f.read()
        app.setStyleSheet(qss)
    ex = LoginWindow()
    ex.show()
    sys.exit(app.exec_())