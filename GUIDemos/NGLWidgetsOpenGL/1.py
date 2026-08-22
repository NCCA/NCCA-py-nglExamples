# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'MainWindow.ui'
##
## Created by: Qt User Interface Compiler version 6.10.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (
    QCoreApplication,
    QMetaObject,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
)
from rgbcolourwidget import RGBColourWidget


class Ui_MainWidget(object):
    def setupUi(self, MainWidget):
        if not MainWidget.objectName():
            MainWidget.setObjectName("MainWidget")
        MainWidget.resize(1024, 720)
        self.main_window_grid_layout = QGridLayout(MainWidget)
        self.main_window_grid_layout.setObjectName("main_window_grid_layout")
        self.transform_gb = QGroupBox(MainWidget)
        self.transform_gb.setObjectName("transform_gb")
        self.gridLayout = QGridLayout(self.transform_gb)
        self.gridLayout.setObjectName("gridLayout")
        self.label = QLabel(self.transform_gb)
        self.label.setObjectName("label")

        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)

        self.rotation_x = QDoubleSpinBox(self.transform_gb)
        self.rotation_x.setObjectName("rotation_x")
        self.rotation_x.setMinimum(-360.000000000000000)
        self.rotation_x.setMaximum(360.000000000000000)

        self.gridLayout.addWidget(self.rotation_x, 1, 0, 1, 1)

        self.rotation_y = QDoubleSpinBox(self.transform_gb)
        self.rotation_y.setObjectName("rotation_y")
        self.rotation_y.setMinimum(-360.000000000000000)
        self.rotation_y.setMaximum(360.000000000000000)

        self.gridLayout.addWidget(self.rotation_y, 1, 1, 1, 1)

        self.rotation_z = QDoubleSpinBox(self.transform_gb)
        self.rotation_z.setObjectName("rotation_z")
        self.rotation_z.setMinimum(-360.000000000000000)
        self.rotation_z.setMaximum(360.000000000000000)

        self.gridLayout.addWidget(self.rotation_z, 1, 2, 1, 1)

        self.label_2 = QLabel(self.transform_gb)
        self.label_2.setObjectName("label_2")

        self.gridLayout.addWidget(self.label_2, 2, 0, 1, 1)

        self.scale_x = QDoubleSpinBox(self.transform_gb)
        self.scale_x.setObjectName("scale_x")
        self.scale_x.setMinimum(-4.000000000000000)
        self.scale_x.setMaximum(4.000000000000000)
        self.scale_x.setSingleStep(0.010000000000000)
        self.scale_x.setValue(1.000000000000000)

        self.gridLayout.addWidget(self.scale_x, 3, 0, 1, 1)

        self.scale_y = QDoubleSpinBox(self.transform_gb)
        self.scale_y.setObjectName("scale_y")
        self.scale_y.setMinimum(-4.000000000000000)
        self.scale_y.setMaximum(4.000000000000000)
        self.scale_y.setSingleStep(0.010000000000000)
        self.scale_y.setValue(1.000000000000000)

        self.gridLayout.addWidget(self.scale_y, 3, 1, 1, 1)

        self.scale_z = QDoubleSpinBox(self.transform_gb)
        self.scale_z.setObjectName("scale_z")
        self.scale_z.setMinimum(-4.000000000000000)
        self.scale_z.setMaximum(4.000000000000000)
        self.scale_z.setSingleStep(0.010000000000000)
        self.scale_z.setValue(1.000000000000000)

        self.gridLayout.addWidget(self.scale_z, 3, 2, 1, 1)

        self.label_3 = QLabel(self.transform_gb)
        self.label_3.setObjectName("label_3")

        self.gridLayout.addWidget(self.label_3, 4, 0, 1, 1)

        self.position_x = QDoubleSpinBox(self.transform_gb)
        self.position_x.setObjectName("position_x")
        self.position_x.setMinimum(-20.000000000000000)
        self.position_x.setMaximum(20.000000000000000)
        self.position_x.setSingleStep(0.010000000000000)

        self.gridLayout.addWidget(self.position_x, 5, 0, 1, 1)

        self.position_y = QDoubleSpinBox(self.transform_gb)
        self.position_y.setObjectName("position_y")
        self.position_y.setMinimum(-20.000000000000000)
        self.position_y.setMaximum(20.000000000000000)
        self.position_y.setSingleStep(0.010000000000000)

        self.gridLayout.addWidget(self.position_y, 5, 1, 1, 1)

        self.position_z = QDoubleSpinBox(self.transform_gb)
        self.position_z.setObjectName("position_z")
        self.position_z.setMinimum(-20.000000000000000)
        self.position_z.setMaximum(20.000000000000000)
        self.position_z.setSingleStep(0.010000000000000)

        self.gridLayout.addWidget(self.position_z, 5, 2, 1, 1)

        self.main_window_grid_layout.addWidget(self.transform_gb, 0, 1, 1, 1)

        self.draw_gb = QGroupBox(MainWidget)
        self.draw_gb.setObjectName("draw_gb")
        self.draw_layout = QGridLayout(self.draw_gb)
        self.draw_layout.setObjectName("draw_layout")
        self.colour_button = QPushButton(self.draw_gb)
        self.colour_button.setObjectName("colour_button")

        self.draw_layout.addWidget(self.colour_button, 3, 0, 1, 1)

        self.verticalSpacer = QSpacerItem(
            20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding
        )

        self.draw_layout.addItem(self.verticalSpacer, 5, 0, 1, 1)

        self.object_selection = QComboBox(self.draw_gb)
        self.object_selection.addItem("")
        self.object_selection.addItem("")
        self.object_selection.addItem("")
        self.object_selection.setObjectName("object_selection")

        self.draw_layout.addWidget(self.object_selection, 0, 0, 1, 1)

        self.wireframe = QCheckBox(self.draw_gb)
        self.wireframe.setObjectName("wireframe")

        self.draw_layout.addWidget(self.wireframe, 1, 0, 1, 1)

        self.frame = RGBColourWidget(self.draw_gb)
        self.frame.setObjectName("frame")
        self.frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame.setFrameShadow(QFrame.Shadow.Raised)

        self.draw_layout.addWidget(self.frame, 4, 0, 1, 1)

        self.main_window_grid_layout.addWidget(self.draw_gb, 1, 1, 1, 1)

        self.horizontalSpacer = QSpacerItem(
            40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )

        self.main_window_grid_layout.addItem(self.horizontalSpacer, 0, 0, 1, 1)

        self.retranslateUi(MainWidget)

        QMetaObject.connectSlotsByName(MainWidget)

    # setupUi

    def retranslateUi(self, MainWidget):
        MainWidget.setWindowTitle(
            QCoreApplication.translate("MainWidget", "ngl Qt Demo", None)
        )
        self.transform_gb.setTitle(
            QCoreApplication.translate("MainWidget", "Transform", None)
        )
        self.label.setText(QCoreApplication.translate("MainWidget", "Rotation", None))
        self.label_2.setText(QCoreApplication.translate("MainWidget", "Scale", None))
        self.label_3.setText(QCoreApplication.translate("MainWidget", "Position", None))
        self.draw_gb.setTitle(QCoreApplication.translate("MainWidget", "Draw", None))
        self.colour_button.setText(
            QCoreApplication.translate("MainWidget", "Choose Colour", None)
        )
        self.object_selection.setItemText(
            0, QCoreApplication.translate("MainWidget", "Teapot", None)
        )
        self.object_selection.setItemText(
            1, QCoreApplication.translate("MainWidget", "Sphere", None)
        )
        self.object_selection.setItemText(
            2, QCoreApplication.translate("MainWidget", "Cube", None)
        )

        self.wireframe.setText(
            QCoreApplication.translate("MainWidget", "WireFrame", None)
        )

    # retranslateUi
