import sys
from GUI.login_window import LoginWindow
from PyQt5.QtWidgets import QApplication
if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = LoginWindow()
    ex.show()
    sys.exit(app.exec_())