from models.model_use import Predictor

from pandas import read_csv
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QLabel, QLineEdit, QPushButton, QTabWidget ,QHBoxLayout, QCheckBox, QWidget, QTextEdit, QFileDialog,QMessageBox
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QIcon
# from login import LoginWindow
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        with open('./resources/QSS-master/MacOS.qss') as f:
            qss = f.read()
            self.setStyleSheet(qss)
    def initUI(self):
        self.setWindowTitle('社交机器人检测')
        self.setWindowIcon(QIcon('./resources/pictures/icon.png'))
        # 创建中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 创建布局
        layout = QVBoxLayout(central_widget)

        self.status = self.statusBar()
        self.status.showMessage('欢迎来到主界面', 5000)

        # 创建 QTabWidget
        self.tab_widget = QTabWidget()
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        # self.tab3 = QWidget()

        # 设置固定窗口大小
        width = 800
        height = 600
        self.setFixedSize(width, height)

        # 居中显示窗口
        screen_geometry = QApplication.desktop().screenGeometry()
        x = (screen_geometry.width() - width) // 2
        y = (screen_geometry.height() - height) // 2
        self.move(x, y)

        # 使用 QTabWidget 的 addTab 方法添加标签页
        self.tab_widget.addTab(self.tab1, 'Data')
        self.tab_widget.addTab(self.tab2, 'Model')
        # self.tab_widget.addTab(self.tab3, 'Result')

        layout.addWidget(self.tab_widget)

        self.tab1UI()
        self.tab2UI()
        # self.tab3UI()

    def tab1UI(self):
        # 创建主垂直布局 **直接关联到 self.tab1**
        main_layout = QVBoxLayout(self.tab1)  # 关键修复：指定父部件为 self.tab1

        # ==== 添加 Followers 和 Fans 输入框 ====
        # Followers 输入行
        followers_layout = QHBoxLayout()
        followers_label = QLabel("Following:")
        self.followers_input = QLineEdit()
        self.followers_input.setPlaceholderText("请输入关注者数量")
        followers_layout.addWidget(followers_label)
        followers_layout.addWidget(self.followers_input)

        # Fans 输入行
        fans_layout = QHBoxLayout()
        fans_label = QLabel("Followers:")
        self.fans_input = QLineEdit()
        self.fans_input.setPlaceholderText("请输入粉丝数量(非0)")
        fans_layout.addWidget(fans_label)
        fans_layout.addWidget(self.fans_input)
        # ==== 输入框添加完毕 ====

        # 创建数据预览标签
        preview_label = QLabel("数据预览")
        preview_label.setAlignment(Qt.AlignCenter)

        # 创建文本框用于显示数据信息
        info_text = QTextEdit()
        info_text.setReadOnly(True)
        info_text.setPlainText("...")
        info_text.setStyleSheet("QTextEdit { text-align: center; vertical-align: middle; }")

        # 创建放大的导入文件按钮
        import_button = QPushButton("导入文件")
        import_button.setMinimumSize(QSize(150, 50))

        def import_file():
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择 CSV 文件", "", "CSV 文件 (*.csv)"
            )
            if file_path:
                try:
                    self.file_path = file_path  # 新增此行保存路径
                    data = read_csv(file_path)
                    head_data = data.head(5).to_csv(sep='\t', na_rep='nan', index=False)
                    info_text.setPlainText(head_data)
                except Exception as e:
                    info_text.setPlainText(f"读取文件出错: {str(e)}")

        import_button.clicked.connect(import_file)

        # 按顺序添加控件到主布局
        main_layout.addLayout(followers_layout)  # Followers 输入框
        main_layout.addLayout(fans_layout)  # Fans 输入框
        main_layout.addWidget(preview_label)  # 数据预览标签
        main_layout.addWidget(info_text)  # 数据显示文本框
        main_layout.addWidget(import_button, alignment=Qt.AlignCenter)  # 导入按钮

        # 无需额外容器，main_layout 已直接关联到 self.tab1

    def tab2UI(self):
        layout = QVBoxLayout()

        # 模型选择复选框
        # model_names = ['svc_linear', 'svc_poly','rfc_model', 'logreg_model', 'n_bayes', 'cn_bayes', 'neural_network']
        model_names = ['rfc_model',  'n_bayes', 'cn_bayes']
        self.checkboxes = {}
        # for model_name in model_names:
        #     checkbox = QCheckBox(model_name.replace('_', ' ').capitalize())
        #     self.checkboxes[model_name] = checkbox
        #     layout.addWidget(checkbox)

        checkbox1 = QCheckBox('rfc_model 高置信'.capitalize())
        self.checkboxes['rfc_model'] = checkbox1
        layout.addWidget(checkbox1)

        checkbox2 = QCheckBox('n_bayes 仅参考'.capitalize())
        self.checkboxes['n_bayes'] = checkbox2
        layout.addWidget(checkbox2)

        checkbox3 = QCheckBox('cn_bayes 仅参考'.capitalize())
        self.checkboxes['cn_bayes'] = checkbox3
        layout.addWidget(checkbox3)

        # 全选按钮
        select_all_button = QPushButton("全选")
        select_all_button.clicked.connect(self.select_all_models)
        layout.addWidget(select_all_button)

        # 开始检测按钮
        start_detection_button = QPushButton("开始检测")
        start_detection_button.clicked.connect(self.start_detection)
        layout.addWidget(start_detection_button)

        self.tab2.setLayout(layout)

    def select_all_models(self):
        for checkbox in self.checkboxes.values():
            checkbox.setChecked(True)

    def start_detection(self):
        # ==== 检查文件路径 ====
        if not hasattr(self, 'file_path') or not self.file_path:
            QMessageBox.warning(
                self,
                "错误",
                "请先在 Data 标签页导入 CSV 文件！",
                QMessageBox.Ok
            )
            return

        # ==== 检查 Followers 和 Fans 输入 ====
        followers_text = self.followers_input.text()
        fans_text = self.fans_input.text()

        if not followers_text.isdigit() or not fans_text.isdigit():
            QMessageBox.warning(
                self,
                "输入错误",
                "请输入有效的整数（Followers 和 Fans）！",
                QMessageBox.Ok
            )
            return

        followers = int(followers_text)
        fans = int(fans_text)

        # ==== 检查模型选择 ====
        selected_models = [
            model_name for model_name, checkbox in self.checkboxes.items()
            if checkbox.isChecked()
        ]
        if not selected_models:
            QMessageBox.warning(
                self,
                "模型未选择",
                "请至少选择一个模型！",
                QMessageBox.Ok
            )
            return

        # ==== 执行预测 ====
        try:
            predictor = Predictor(self.file_path, followers, fans)
            results = predictor.get_result2(selected_models)
        except Exception as e:
            QMessageBox.critical(
                self,
                "预测失败",
                f"预测过程中发生错误：{str(e)}",
                QMessageBox.Ok
            )
            return

        # ==== 显示预测结果 ====
        result_text = "预测结果：\n\n"
        for model, pred in results.items():
            model_name = model.replace('_', ' ').capitalize()
            result_text += f"{model_name}: {pred[0]}\n"

        QMessageBox.information(
            self,
            "检测完成",
            result_text,
            QMessageBox.Ok
        )



if __name__ == '__main__':

    app = QApplication(sys.argv)

    ex = MainWindow()
    ex.show()
    sys.exit(app.exec_())