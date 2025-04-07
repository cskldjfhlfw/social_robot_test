from GUI.main_window import MainWindow
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox, QHBoxLayout
from PyQt5.QtGui import QIcon, QRegExpValidator,  QPixmap, QBrush, QPalette, QFont
from PyQt5.QtCore import QRegExp, Qt
class LoginWindow(QWidget):
    def __init__(self):
        super(QWidget, self).__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Login')
        self.setWindowFlags(Qt.FramelessWindowHint)

        self.setWindowIcon(QIcon('./resources/pictures/icon.png'))
        # self.resize(928, 630)
        self.setFixedSize(500, 300)
        palette = self.palette()
        # palette.setColor(self.backgroundRole(), Qt.red)
        palette.setBrush(QPalette.Background,QBrush(QPixmap("./resources/pictures/login_background3.png")))
        self.setPalette(palette)

        # 创建布局
        layout = QVBoxLayout()

        # 创建标题标签
        title_label = QLabel('社交机器人检测')
        font1 = QFont('楷体', 25)
        font1.setBold(True)
        title_label.setFont(font1)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        #正则表达式
        reg = QRegExp('[a-zA-Z0-9]+')
        validator = QRegExpValidator(self)
        validator.setRegExp(reg)

        # 用户名输入框
        font2 = QFont('楷体', 10)
        font2.setBold(True)
        self.username_label = QLabel('&Username(U)')
        self.username_label.setFont(font2)
        self.username_input = QLineEdit()
        # self.username_input.setStyleSheet('height:60px')
        self.username_input.setValidator(validator)
        self.username_input.setPlaceholderText('请输入用户名(A-Z a-z 0-9)')
        self.username_label.setBuddy(self.username_input)
        layout.addWidget(self.username_label)
        layout.addWidget(self.username_input)

        # 密码输入框
        self.password_label = QLabel('&Password(P)')
        self.password_label.setFont(font2)
        self.password_input = QLineEdit()
        # self.password_input.setStyleSheet('height:60px')
        self.password_input.setValidator(validator)
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setPlaceholderText('请输入密码')
        self.password_label.setBuddy(self.password_input)
        self.password_input.returnPressed.connect(self.login)
        layout.addWidget(self.password_label)
        layout.addWidget(self.password_input)

        layout2 =  QHBoxLayout()
        # 登录按钮
        self.login_button = QPushButton('登录')
        self.login_button.setFixedSize(235, 40)
        # self.login_button.setStyleSheet('height:100px;background-color:#4CAF50;border-radius:15px')
        self.login_button.clicked.connect(self.login)
        layout2.addWidget(self.login_button)

        #退出按钮
        self.quit_button = QPushButton('退出')
        self.quit_button.setFixedSize(235,40)
        self.quit_button.clicked.connect(lambda : exit())
        layout2.addWidget(self.quit_button)

        layout.addLayout(layout2)
        self.setLayout(layout)
        self.setWindowTitle('登录')

    def show_info_message(self,message):
        msgbox = QMessageBox()
        msgbox.setText(message)
        msgbox.setWindowFlags(msgbox.windowFlags() | Qt.FramelessWindowHint)
        msgbox.exec_()
    def login(self):
        username = self.username_input.text()
        password = self.password_input.text()
        data_temp = open("./resources/information/login", "r")
        for line in data_temp:
            username_temp, password_temp = line.strip().split(',')
            if username == username_temp and password == password_temp:
                # print("登陆成功")
                self.show_info_message("登陆成功")
                self.close()
                self.main_window = MainWindow()
                self.main_window.show()
                break
        else:
            # print("登陆失败")
            self.show_info_message("登陆失败")

if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = LoginWindow()
    ex.show()
    sys.exit(app.exec_())