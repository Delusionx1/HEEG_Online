from pprint import pprint
import random
from types import new_class

from PySide6.QtCharts import QChart, QSplineSeries, QValueAxis,QAreaSeries,QLineSeries
from PySide6.QtCore import Qt, QTimer, Slot, QPointF, QRect
from PySide6.QtGui import QPen, QPainter,QLinearGradient,QGradient,QColor
from PySide6.QtWidgets import QApplication
import sys
import numpy as np
import pickle
import scipy.signal as signal
from scipy.signal import cheb2ord
from FBCSP import FBCSP
from Classifier import Classifier
from sklearn.svm import SVR
from FilterBank import FilterBank
from scipy import stats
import statistics
from time import sleep
import socket

class Chart(QChart):
    def __init__(self, data_package = None,real_data = None, parent = None,min = 1,max = 0,EEG_EOG_Cat_classification=None, Errp_data = None):
        super().__init__(QChart.ChartTypeCartesian, parent, Qt.WindowFlags())
        self.SAMPLE_COUNT = 3000
        self._timer = QTimer()
        self.data_package = data_package
        self.real_data =real_data
        self.min = min
        self.max = max
        self.EEG_EOG_Cat_classification = EEG_EOG_Cat_classification
        self.errp_data = Errp_data

        self.series_area_0 = QLineSeries()
        self.series_area_1 = QLineSeries()
        self.series_area_0.append(QPointF(2550, (data_package.shape[0]) * (max-min)))
        self.series_area_0.append(QPointF(2950, (data_package.shape[0]) * (max-min)))
        self.series_area_1.append(QPointF(2550, min+ 2*(max-min)))
        self.series_area_1.append(QPointF(2950, min+ 2*(max-min)))
        self.area_series = QAreaSeries(self.series_area_0, self.series_area_1)
        self.gradient = QLinearGradient(QPointF(0, 0), QPointF(0, 1))
        self.gradient.setColorAt(0.0, QColor(0,0,255,50))
        self.gradient.setColorAt(0.5, QColor(0,0,255,50))
        self.gradient.setColorAt(1, QColor(0,0,255,50))
        self.gradient.setCoordinateMode(QGradient.ObjectBoundingMode)
        self.area_series.setBrush(self.gradient)
        block_pen = QPen(Qt.transparent)
        self.area_series.setPen(block_pen)

        self.series_area_2 = QLineSeries()
        self.series_area_3 = QLineSeries()
        self.series_area_2.append(QPointF(2550, max))
        self.series_area_2.append(QPointF(2950, max))
        self.series_area_3.append(QPointF(2550, min))
        self.series_area_3.append(QPointF(2950, min))
        self.area_series_eog = QAreaSeries(self.series_area_2, self.series_area_3)
        gradient_eog = QLinearGradient(QPointF(0, 0), QPointF(0, 1))
        gradient_eog.setColorAt(0.0, QColor(255,0,100,150))
        gradient_eog.setColorAt(0.5, QColor(255,0,100,150))
        gradient_eog.setColorAt(1, QColor(255,0,100,150))
        gradient_eog.setCoordinateMode(QGradient.ObjectBoundingMode)
        self.area_series_eog.setBrush(gradient_eog)
        self.area_series_eog.setPen(block_pen)
        self.addSeries(self.area_series_eog)
        # self.area_series_eog.setVisible(False)

        self.series_area_4 = QLineSeries()
        self.series_area_5 = QLineSeries()
        self.series_area_4.append(QPointF(2550, data_package.shape[0] * (max-min)))
        self.series_area_4.append(QPointF(2950, data_package.shape[0] * (max-min)))
        self.series_area_5.append(QPointF(2550, min))
        self.series_area_5.append(QPointF(2950, min))
        self.area_series_errp = QAreaSeries(self.series_area_4, self.series_area_5)
        self.gradient_errp = QLinearGradient(QPointF(0, 0), QPointF(0, 1))
        self.gradient_errp.setColorAt(0.0, QColor(255,255,255,150))
        self.gradient_errp.setColorAt(0.5, QColor(255,255,255,150))
        self.gradient_errp.setColorAt(1, QColor(255,255,255,150))
        self.gradient_errp.setCoordinateMode(QGradient.ObjectBoundingMode)
        self.area_series_errp.setBrush(self.gradient_errp)
        block_pen = QPen(Qt.transparent)
        self.area_series.setPen(block_pen)
        self.addSeries(self.area_series_errp)
        self.area_series_errp.setVisible(False)

        self.series_area_6 = QLineSeries()
        self.series_area_7 = QLineSeries()
        self.series_area_6.append(QPointF(2550, max+(max-min)))
        self.series_area_6.append(QPointF(2950, max+(max-min)))
        self.series_area_7.append(QPointF(2550, min+(max-min)))
        self.series_area_7.append(QPointF(2950, min+(max-min)))
        self.area_series_labels = QAreaSeries(self.series_area_6, self.series_area_7)
        self.gradient_labels = QLinearGradient(QPointF(0, 0), QPointF(0, 1))
        self.gradient_labels.setColorAt(0.0, QColor(0,0,255,255))
        self.gradient_labels.setColorAt(0.5, QColor(0,0,255,255))
        self.gradient_labels.setColorAt(1, QColor(0,0,255,255))
        self.gradient_labels.setCoordinateMode(QGradient.ObjectBoundingMode)
        self.area_series_labels.setBrush(self.gradient_labels)
        self.area_series_labels.setPen(block_pen)
        self.addSeries(self.area_series_labels)
        # self.area_series_labels.setVisible(False)

        self.area_series.setBrush(self.gradient)
        block_pen = QPen(Qt.transparent)
        self.area_series.setPen(block_pen)

        self._titles = []
        self._axisX = QValueAxis()
        self._axisY = QValueAxis()
        self._step = 0
        self._x = 5
        self._y = 1
        self._timer.timeout.connect(self.handleTimeout)
        self._timer.setInterval(50)
        self.green = QPen(0xFFFF55)
        self.green.setWidth(1.5)
        self._axisY.setGridLineVisible(False)
        self._axisX.setGridLineVisible(False)
        self._axisX.hide()
        self._axisX.setLabelsVisible(False)

        self.my_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, 0)
        self.ip = '127.0.0.1'
        self.port = 8888
        self.ip_address = (self.ip, self.port)


        self.addSeries(self.area_series)
        self.all_series = []
        self.all_buffers = []
        EOG_Pen = QPen(0x00FF00)
        EOG_Pen.setWidth(2)
        for i in range(self.data_package.shape[0]):
            self.temp_series = QLineSeries(self)
            if(i == 0 or i== 1):
                self.temp_series.setPen(EOG_Pen)
            else:
                self.temp_series.setPen(self.green)
            self.all_series.append(self.temp_series)

        for i in range(len(self.all_series)):
            self.addSeries(self.all_series[i])

        self.setBackgroundBrush(Qt.black)
        self.addAxis(self._axisX, Qt.AlignBottom)
        self.addAxis(self._axisY, Qt.AlignLeft)
        self.area_series.attachAxis(self._axisX)
        self.area_series.attachAxis(self._axisY)

        self.area_series_eog.attachAxis(self._axisX)
        self.area_series_eog.attachAxis(self._axisY)
        
        self.area_series_labels.attachAxis(self._axisX)
        self.area_series_labels.attachAxis(self._axisY)
        # self.area_series_labels.setVisible(False)
        
        self.area_series_errp.attachAxis(self._axisX)
        self.area_series_errp.attachAxis(self._axisY)
        # self.area_series.setVisible(False)

        self._axisX.setTickCount(self.SAMPLE_COUNT)
        self._axisX.setRange(0, self.SAMPLE_COUNT)
        self._axisY.setRange(min, (max-min)*len(self.all_series))

        self._timer.start()
        self.signal_index = 0
        self.flag_mi_decoding = False
        for i in range(len(self.all_series)):
            self.temp_buffer = [QPointF(x, i*(max-min)) for x in range(self.SAMPLE_COUNT)]
            self.all_buffers.append(self.temp_buffer)
            self.all_series[i].append(self.all_buffers[i])
        for i in range(len(self.all_series)):
            self.all_series[i].attachAxis(self._axisX)
            self.all_series[i].attachAxis(self._axisY)
        self.eog_flag = 0
        self.decoding_loops = 0
        with open('../fbcsp.pkl', 'rb') as f:
            m_filters = 2
            self.fbcsp = FBCSP(m_filters)
            self.fbcsp = pickle.load(f)
        self.freq =250
        with open('../Errp_rieman.pkl', 'rb') as f:
            self.errp_classifier = pickle.load(f)
        self.fbank = FilterBank(self.freq)
        fbank_coeff = self.fbank.get_filter_coeff()
        with open('../fbcsp_classifier.pkl', 'rb') as e:
            classifier_type = SVR(gamma='auto')
            self.fbcsp_classifier = Classifier(classifier_type)
            self.fbcsp_classifier = pickle.load(e)
        self.eog_triggered_time = 0
        self.mi_end = True
        self.start_position = -1
        self.eog_start_pos = 0
        self.errp_flag = False
        self.errp_decoding_loops = 0
        self.mi_results_temp = []
        self.errp_results_temp = []

    def get_multi_class_regressed(self, y_predicted):
        y_predict_multi = np.asarray([np.argmin(y_predicted[i,:]) for i in range(y_predicted.shape[0])])
        return y_predict_multi
        
    def identify_eog(self, eeg_data):
        feature = np.array([np.std(eeg_data), np.mean(eeg_data), np.min(eeg_data), np.max(eeg_data)]).reshape(1,-1)
        # print(feature)
        with open('../classifier.pkl', 'rb') as f:
            clf = pickle.load(f)
        result = int(clf.predict(feature))
        return result
    
    def errp_decoding(self):
        if(self.errp_decoding_loops<=4):
            errp_temp = self.errp_data[:,self.signal_index-450:self.signal_index-240]
            result = self.errp_classifier.predict(np.expand_dims(errp_temp, axis=0))[0]
            print('Errp results:',result)
            if(int(result) == 1):
                self.area_series_errp.setVisible(True)
            else:
                self.area_series_errp.setVisible(False)
            self.errp_decoding_loops += 1
            self.errp_results_temp.append(result)
        else:
            classification_result= stats.mode(self.errp_results_temp)[0][0]
            if(classification_result == 1):
                self.my_socket.sendto('6'.encode('utf-8'),self.ip_address)
            print('Classification result:', classification_result)

            self.errp_results_temp = []
            self.errp_decoding_loops = 0
            self.area_series_errp.setVisible(False)
            self.errp_flag = False
        return 0

    def change_label_color(self, label_data):
        classification_result= stats.mode(label_data)[0][0]
        if(classification_result == 0):
                self.gradient_labels.setColorAt(0.0, QColor(255,0,0,255))
                self.gradient_labels.setColorAt(0.5, QColor(255,0,0,255))
                self.gradient_labels.setColorAt(1, QColor(255,0,0,255))
                self.area_series_labels.setBrush(self.gradient_labels)
        if(classification_result == 1):
                self.gradient_labels.setColorAt(0.0, QColor(153,51,255,255))
                self.gradient_labels.setColorAt(0.5, QColor(153,51,255,255))
                self.gradient_labels.setColorAt(1, QColor(153,51,255,255))
                self.area_series_labels.setBrush(self.gradient_labels)
        if(classification_result == 2):
                self.gradient_labels.setColorAt(0.0, QColor(255,51,0,255))
                self.gradient_labels.setColorAt(0.5, QColor(255,51,0,255))
                self.gradient_labels.setColorAt(1, QColor(255,51,0,255))
                self.area_series_labels.setBrush(self.gradient_labels)
        if(classification_result == 3):
                self.gradient_labels.setColorAt(0.0, QColor(102,255,255,255))
                self.gradient_labels.setColorAt(0.5, QColor(102,255,255,255))
                self.gradient_labels.setColorAt(1, QColor(102,255,255,255))
                self.area_series_labels.setBrush(self.gradient_labels)
        if(classification_result == 6):
                self.gradient_labels.setColorAt(0.0, QColor(255,255,255,255))
                self.gradient_labels.setColorAt(0.5, QColor(255,255,255,255))
                self.gradient_labels.setColorAt(1, QColor(255,255,255,255))
                self.area_series_labels.setBrush(self.gradient_labels)
        if(classification_result == 7):
                self.gradient_labels.setColorAt(0.0, QColor(0,0,255,255))
                self.gradient_labels.setColorAt(0.5, QColor(0,0,255,255))
                self.gradient_labels.setColorAt(1, QColor(0,0,255,255))
                self.area_series_labels.setBrush(self.gradient_labels)
        if(classification_result == 0.5):
                self.gradient_labels.setColorAt(0.0, QColor(0,0,255,255))
                self.gradient_labels.setColorAt(0.5, QColor(0,0,255,255))
                self.gradient_labels.setColorAt(1, QColor(0,0,255,50))
                self.area_series_labels.setBrush(self.gradient_labels)
        return 0

    def mi_decoding(self):
        if(self.decoding_loops<=8):
            eeg_data = self.EEG_EOG_Cat_classification[1:23,self.signal_index-450:self.signal_index-50]
            n_classes = 4
            eeg_data = np.expand_dims(eeg_data, axis= 0)
            # print(eeg_data.shape)
            filtered_data_test = self.fbank.filter_data(eeg_data)
            # print(filtered_data_test.shape)
            y_test_predicted = np.zeros((1, n_classes), dtype=np.float)
            for j in range(n_classes):
                x_features_test = self.fbcsp.transform(filtered_data_test,class_idx=j)
                y_test_predicted[0,j] = self.fbcsp_classifier.predict(x_features_test[0].reshape(1,-1))
            # print(y_test_predicted.shape)
            y_test_predicted_multi = self.get_multi_class_regressed(y_test_predicted)
            result = y_test_predicted_multi[0]
            print(result)
            if(result==0):
                self.gradient.setColorAt(0.0, QColor(255,0,0,50))
                self.gradient.setColorAt(0.5, QColor(255,0,0,50))
                self.gradient.setColorAt(1, QColor(255,0,0,50))
                self.area_series.setBrush(self.gradient)
            if(result==1):
                self.gradient.setColorAt(0.0, QColor(153,51,255,50))
                self.gradient.setColorAt(0.5, QColor(153,51,255,50))
                self.gradient.setColorAt(1, QColor(153,51,255,50))
                self.area_series.setBrush(self.gradient)
            if(result==2):
                self.gradient.setColorAt(0.0, QColor(255,51,0,50))
                self.gradient.setColorAt(0.5, QColor(255,51,0,50))
                self.gradient.setColorAt(1, QColor(255,51,0,50))
                self.area_series.setBrush(self.gradient)
            if(result==3):
                self.gradient.setColorAt(0.0, QColor(102,255,255,50))
                self.gradient.setColorAt(0.5, QColor(102,255,255,50))
                self.gradient.setColorAt(1, QColor(102,255,255,50))
                self.area_series.setBrush(self.gradient)
            self.mi_results_temp.append(result)
            self.decoding_loops += 1
        else:
            classification_result= stats.mode(self.mi_results_temp)[0][0] + 2
            print('Classification result:', classification_result)
            self.my_socket.sendto(str(classification_result).encode('utf-8'),self.ip_address)
            self.mi_results_temp = []
            self.errp_flag = True
            self.start_position = -1
            self.mi_end = True
                    
            if(self.mi_end == True):
                self.decoding_loops = 0
                self.gradient.setColorAt(0.0, QColor(0,0,255,50))
                self.gradient.setColorAt(0.5, QColor(0,0,255,50))
                self.gradient.setColorAt(1, QColor(0,0,255,50))
                self.area_series.setBrush(self.gradient)
        return 0

    @Slot()
    def handleTimeout(self):
        y = (self._axisX.max() - self._axisX.min()) / self._axisX.tickCount()
        self._x += y

        start = 0
        available_samples = 20
        if(self.signal_index >= self.data_package.shape[1]- (2 * available_samples)):
            QApplication.exit()
        if (available_samples < self.SAMPLE_COUNT):
            start = self.SAMPLE_COUNT - available_samples
            for s in range(start):
                for i in range(len(self.all_series)):
                    self.all_buffers[i][s].setY(self.all_buffers[i][s + available_samples].y())
                    
        for s in range(start, self.SAMPLE_COUNT):
            for i in range(len(self.all_series)):
                value = self.data_package[i][self.signal_index]
                self.all_buffers[i][s].setY(value + i*(self.max - self.min))
            self.signal_index = self.signal_index + 1
        for i in range(len(self.all_series)):
            self.all_series[i].replace(self.all_buffers[i])
        current_data = self.data_package[1,self.signal_index-400:self.signal_index-200]
        # print(current_data)
        self.change_label_color(current_data)
        if(self.signal_index > 400):
            eog_data = self.EEG_EOG_Cat_classification[0,self.signal_index-450:self.signal_index-50]
            eog_result = self.identify_eog(eog_data)
            if(eog_result == 0):
                self.eog_triggered_time = 0
                self.area_series_eog.setVisible(False)
            else:
                if(self.mi_end == True):
                    self.eog_start_pos = self.signal_index
                # print(self.eog_start_pos)
                # print(self.signal_index)
                self.start_position = self.signal_index
                self.area_series_eog.setVisible(True)
                self.eog_triggered_time +=1
        # print(self.signal_index-self.eog_start_pos)
        # print('--',self.start_position+600, ' ', self.signal_index )
        if(self.start_position != -1 and self.start_position+650 > self.signal_index and (self.signal_index-self.eog_start_pos)>0):
            self.mi_end = False
            self.mi_decoding()
            # print(self.signal_index)
        if(self.start_position==-1):
            self.mi_end = True
            if(self.mi_end == True):
                self.decoding_loops = 0
                self.gradient.setColorAt(0.0, QColor(0,0,255,50))
                self.gradient.setColorAt(0.5, QColor(0,0,255,50))
                self.gradient.setColorAt(1, QColor(0,0,255,50))
                self.area_series.setBrush(self.gradient)
        if(self.errp_flag == True):
            self.errp_decoding()

