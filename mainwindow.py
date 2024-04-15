import os
import sys
import cv2
import csv
from datetime import datetime

import torch

from model.model import Model
from model.MbtiModel import MbtiModule

from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QMessageBox, QFileDialog, QMenuBar, QMenu, QListWidget, QListWidgetItem, QLabel, QVBoxLayout, QHBoxLayout, QGridLayout, QProgressBar,  QStatusBar, QPushButton, QTabWidget, QTableWidget, QAbstractItemView, QLineEdit, QTableWidgetItem, QTextBrowser
from PySide6.QtGui import QAction, QKeyEvent, QPixmap, QFont, QIcon
from PySide6.QtCore import Qt, QFile

import resource_rc


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setWindowTitle("MBTI analysis")
        self.setWindowIcon(QIcon(":/resource/MBTI.png"))
        self.resize(800, 600)

        self.model = Model()
        state_dict = torch.load('weight/pretrained_model.pth')
        self.model.model.load_state_dict(state_dict)
        self.model.model.eval()

        self.mbtimodels = []
        for i in range(4):
            mbtimodel = MbtiModule()
            state_dict = torch.load(
                f'weight/MbtiModel_{i+1}_weight.pth')  # 后续修改为i+1
            mbtimodel.load_state_dict(state_dict)
            mbtimodel.eval()
            self.mbtimodels.append(mbtimodel)

        self.set_ui()

    def set_ui(self):
        # ------------------------------------------------------------------------
        # 创建 'File' menu
        self.file_menu = QMenu('File', self)
        self.file_menu.addActions([
            QAction(QIcon(":/resource/icon/upload.svg"), 'upload',
                    self, triggered=self.add_image),
            QAction(QIcon(":/resource/icon/play.svg"), 'run',
                    self, triggered=self.run),
            QAction(QIcon(":/resource/icon/save.svg"), 'save',
                    self, triggered=self.output_csv)
        ])

        # 创建 'about' menu
        self.about_menu = QMenu('About', self)
        self.about_menu.addActions(
            [QAction(QIcon(":/resource/icon/message-square.svg"), 'message', self, triggered=self.message)])

        # 创建menubar
        self.menubar = QMenuBar(self)

        self.menubar.addMenu(self.file_menu)
        self.menubar.addMenu(self.about_menu)

        # 设置menubar
        self.setMenuBar(self.menubar)
        # ------------------------------------------------------------------------

        # ------------------------------------------------------------------------
        # 创建statusbar
        self.statusbar = QStatusBar(self)
        self.statusbar.setStyleSheet(
            "background-color: lightblue; color: black;")
        self.statusbar.showMessage('Ready')

        # 设置statusbar
        self.setStatusBar(self.statusbar)
        # ------------------------------------------------------------------------

        # ------------------------------------------------------------------------
        # 创建第一层widget
        self.first_widget = QWidget(self)
        first_layout = QHBoxLayout(self.first_widget)

        # 创建name输入框
        self.name_input = QLineEdit(self)
        self.name_input.setFont(QFont('Consolas', 12))
        self.name_input.setPlaceholderText('please input name')
        self.name_input.setFixedSize(300, 30)

        # 创建image_list_widget
        self.image_list_widget = QListWidget(self)
        self.image_list_widget.setSelectionMode(QListWidget.ExtendedSelection)
        self.image_list_widget.setDragDropMode(QListWidget.InternalMove)
        self.image_list_widget.itemClicked.connect(self.show_image)

        # 创建main_widget
        self.main_widget = QWidget(self)
        main_layout = QVBoxLayout(self.main_widget)
        main_layout.addWidget(self.name_input)
        main_layout.addWidget(self.image_list_widget)

        first_layout.addWidget(self.main_widget)

        # 创建image_label
        self.image_label = QLabel(self)
        self.image_label.setStyleSheet("""
            font: 14pt "Consolas";
            color: grey;
            border: 1px solid black;
        """)
        self.image_label.setFixedSize(256, 256)
        # 设置居中显示
        self.image_label.setAlignment(Qt.AlignCenter)
        # 设置文本
        self.image_label.setText('image')

        # 创建按钮
        self.sort_button = QPushButton('sort', self)
        self.sort_button.setFont(QFont('Consolas', 10))
        self.sort_button.clicked.connect(self.sort)
        self.clear_button = QPushButton('clear', self)
        self.clear_button.setFont(QFont('Consolas', 10))
        self.clear_button.clicked.connect(self.clear)
        self.caculate_button = QPushButton('run', self)
        self.caculate_button.setFont(QFont('Consolas', 10))
        self.caculate_button.clicked.connect(self.run)

        # 创建侧边widfet
        self.side_widget = QWidget(self)
        side_layout = QVBoxLayout(self.side_widget)
        side_layout.addWidget(self.image_label)
        side_layout.addWidget(self.sort_button)
        side_layout.addWidget(self.clear_button)
        side_layout.addWidget(self.caculate_button)

        first_layout.addWidget(self.side_widget)
        # ------------------------------------------------------------------------

        # ------------------------------------------------------------------------
        # 创建第二层widget
        self.second_widget = QWidget(self)
        second_layout = QGridLayout(self.second_widget)

        # 创建四个QProgressBar和label
        self.progress_bars = [QProgressBar(self) for i in range(4)]
        self.progress_labels = [QLabel(self) for i in range(4)]
        self.progress_labels[0].setText('E-I:')
        self.progress_labels[1].setText('S-N:')
        self.progress_labels[2].setText('T-F:')
        self.progress_labels[3].setText('J-P:')

        for i, progress_bar in enumerate(self.progress_bars):
            progress_bar.setValue(50)
            second_layout.addWidget(
                self.progress_labels[i], i // 2, (i % 2) * 2)
            second_layout.addWidget(progress_bar, i // 2, (i % 2) * 2 + 1)
        # ------------------------------------------------------------------------

        # ------------------------------------------------------------------------
        self.central_widget = QWidget(self)
        central_layout = QVBoxLayout(self.central_widget)
        central_layout.addWidget(self.first_widget)
        central_layout.addWidget(self.second_widget)
        # ------------------------------------------------------------------------

        # ------------------------------------------------------------------------
        # 创建TableWidget
        self.table_widget = QTableWidget(self)
        self.table_widget.setColumnCount(7)
        self.table_widget.setHorizontalHeaderLabels(
            ['Name', 'MBTI', 'E-I', 'S-N', 'T-F', 'J-P', 'Date'])
        # self.table_widget.horizontalHeader().setSectionResizeMode(
        #     QHeaderView.Stretch
        # )
        self.table_widget.verticalHeader().setVisible(False)
        self.table_widget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table_widget.setSelectionBehavior(QAbstractItemView.SelectRows)
        # ------------------------------------------------------------------------

        # ------------------------------------------------------------------------
        # 创建QTextBrowser
        self.text_browser = QTextBrowser(self)
        self.text_browser.setFont(QFont('Consolas', 10))
        self.text_browser.setReadOnly(True)
        file = QFile(':/resource/introduction.html')
        file.open(QFile.ReadOnly | QFile.Text)
        self.text_browser.setHtml(file.readAll().data().decode())
        # ------------------------------------------------------------------------

        # ------------------------------------------------------------------------
        # 创建tab_widget作为central_widget
        self.tab_widget = QTabWidget(self)
        self.tab_widget.setTabPosition(QTabWidget.West)
        self.tab_widget.setStyleSheet("""
            QTabBar::tab {
                width: 40px;
                height: 40px;
            }
        """)

        self.setCentralWidget(self.tab_widget)
        self.tab_widget.addTab(self.central_widget,
                               QIcon(':/resource/icon/edit.svg'), '')
        self.tab_widget.addTab(
            self.table_widget, QIcon(':/resource/icon/table.svg'), '')
        self.tab_widget.addTab(
            self.text_browser, QIcon(':/resource/icon/file-text.svg'), '')
        # ------------------------------------------------------------------------

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_Delete:
            self.image_label.setText('image')
            for item in self.image_list_widget.selectedItems():
                self.image_list_widget.takeItem(
                    self.image_list_widget.row(item))
        return super().keyPressEvent(event)

    def add_image(self):
        filepaths, _ = QFileDialog.getOpenFileNames(
            self, 'please select images', '', 'image (*.jpg *.jpeg *.png *.bmp)', os.getcwd(), QFileDialog.DontUseNativeDialog)

        for filepath in filepaths:
            item = QListWidgetItem(filepath)
            self.image_list_widget.addItem(item)
        self.statusbar.showMessage(f'{len(filepaths)} images selected')

    def message(self):
        QMessageBox.about(
            self, 'message', 'Auther: hamlet\nGithub: https://github.com/Hamletlx')

    def show_image(self, item):
        pixmap = QPixmap(item.text()).scaled(256, 256, Qt.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)
        self.statusbar.showMessage(item.text())

    def sort(self):
        self.set_init_value()
        self.image_list_widget.sortItems()
        self.statusbar.showMessage('sort finished')

    def clear(self):
        self.set_init_value()
        self.image_list_widget.clear()
        self.name_input.setText('')
        self.image_label.setText('image')
        self.statusbar.showMessage('clear finished')

    def run(self):
        self.statusbar.showMessage('running...')
        self.set_init_value()
        # 检查是否输入name
        if self.name_input.text() == '':
            self.statusbar.showMessage('no name input')
            QMessageBox.warning(self, 'warning', 'please input name')
            return

        # 读取image_list_widget的图片
        images = []
        for i in range(self.image_list_widget.count()):
            image_path = self.image_list_widget.item(i).text()
            img = cv2.imread(image_path)
            images.append(img)

        if len(images) == 0:
            self.statusbar.showMessage('no image selected')
            QMessageBox.warning(self, 'warning', 'please input images')
            return

        input_tensor = self.model.pre_process(images)
        feature = self.model.forward(input_tensor)

        outputs = []
        for i in range(4):
            output = self.mbtimodels[i](feature)
            outputs.append(output.item())
            self.progress_bars[i].setValue(output.item()*100)
        self.statusbar.showMessage('run finished')

        MBTI = ''
        if outputs[0] > 0.5:
            MBTI += 'I'
        elif outputs[0] < 0.5:
            MBTI += 'E'
        else:
            MBTI += '_'

        if outputs[1] > 0.5:
            MBTI += 'N'
        elif outputs[1] < 0.5:
            MBTI += 'S'
        else:
            MBTI += '_'

        if outputs[2] > 0.5:
            MBTI += 'F'
        elif outputs[2] < 0.5:
            MBTI += 'T'
        else:
            MBTI += '_'

        if outputs[3] > 0.5:
            MBTI += 'P'
        elif outputs[3] < 0.5:
            MBTI += 'J'
        else:
            MBTI += '_'

        row = self.table_widget.rowCount()
        self.table_widget.insertRow(row)
        self.table_widget.setItem(
            row, 0, QTableWidgetItem(self.name_input.text()))
        self.table_widget.setItem(
            row, 1, QTableWidgetItem(MBTI))
        for i in range(4):
            self.table_widget.setItem(
                row, i+2, QTableWidgetItem(f"{outputs[i]*100:.3f}%"))
        self.table_widget.setItem(
            row, 6, QTableWidgetItem(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    def set_init_value(self):
        for progress_bar in self.progress_bars:
            progress_bar.setValue(50)

    def output_csv(self):
        if self.table_widget.rowCount() == 0:
            QMessageBox.warning(self, 'warning', 'no data to output')
            return

        save_file_path, _ = QFileDialog.getSaveFileName(
            self, 'Save File', '', 'CSV Files (*.csv)', os.getcwd(), QFileDialog.DontUseNativeDialog)

        if save_file_path != "" and not save_file_path.endswith('.csv'):
            save_file_path += '.csv'

        with open(save_file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            header_row = []
            for column in range(self.table_widget.columnCount()):
                header_row.append(
                    self.table_widget.horizontalHeaderItem(column).text())
            writer.writerow(header_row)

            for row in range(self.table_widget.rowCount()):
                row_data = []
                for column in range(self.table_widget.columnCount()):
                    item = self.table_widget.item(row, column)
                    if item is not None:
                        row_data.append(item.text())
                    else:
                        row_data.append('')
                writer.writerow(row_data)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
