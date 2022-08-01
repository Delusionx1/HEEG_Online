"""PySide6 port of the Dynamic Spline example from Qt v5.x"""
import sys

from PySide6.QtCharts import QChart, QChartView
from PySide6.QtGui import QPainter,QLinearGradient,QColor, QGradient,QFont
from PySide6.QtWidgets import QApplication, QMainWindow,QLabel, QVBoxLayout,QWidget
from PySide6.QtCore import QPoint,Qt,QMargins
import matplotlib.pyplot as plt
from chart import Chart
import numpy as np
from DataLoadingAndPreProcessing import standardlizeSig
from numpy import inf

if __name__ == "__main__":
    EEG_EOG_Cat_real = np.load(r'D:\Pytorch_learning\EOG_learning\EEG_EOG_Cat.npy')
    EEG_EOG_Cat_classification = np.load(r'D:\Pytorch_learning\EOG_learning\EEG_EOG_Cat_real.npy')
    Errp_all_channels = np.load(r'D:\Pytorch_learning\EOG_learning\Errp_all_channels.npy')
    print(EEG_EOG_Cat_real[[[0][1:-1]],:].shape)
    a = QApplication(sys.argv)
    window = QMainWindow()
    EEG_EOG_Cat = np.zeros((EEG_EOG_Cat_real.shape[0], EEG_EOG_Cat_real.shape[1]))
    for i in range(EEG_EOG_Cat_real.shape[0]):
        if(i!=1):
            EEG_EOG_Cat[i,:] = standardlizeSig(EEG_EOG_Cat_real[i,:])
        else:
            EEG_EOG_Cat[i,:] = EEG_EOG_Cat_real[i,:]
    print(np.min(EEG_EOG_Cat), np.max(EEG_EOG_Cat))
    chart = Chart(EEG_EOG_Cat[:][:],min = np.min(EEG_EOG_Cat), max = np.max(EEG_EOG_Cat),real_data=EEG_EOG_Cat_real,EEG_EOG_Cat_classification=EEG_EOG_Cat_classification, Errp_data = Errp_all_channels)
    chart.legend().hide()
    chart.setMargins(QMargins(0,0,0,0))
    chart.layout().setContentsMargins(0,0,0,0)
    chart.setBackgroundRoundness(2)
    chart_view = QChartView(chart)
    chart_view.setContentsMargins(0,0,0,0)
    chart_view.setRenderHint(QPainter.Antialiasing)
    
    layout = QVBoxLayout()
    layout.setContentsMargins(0,0,0,0)
    layout.setSpacing(0)

    layout.addWidget(chart_view)
    widget = QWidget()
    widget.setLayout(layout)
    window.setCentralWidget(widget)

    window.resize(1500, 600)
    window.show()
    sys.exit(a.exec())