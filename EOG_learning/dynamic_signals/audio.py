"""PySide6 port of the charts/audio example from Qt v5.x"""

import sys
from PySide6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis, QAreaSeries
from PySide6.QtCore import QPointF, Slot
from PySide6.QtMultimedia import (QAudioDevice, QAudioFormat,
        QAudioSource, QMediaDevices)
from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox


SAMPLE_COUNT = 2000

RESOLUTION = 2


class MainWindow(QMainWindow):
    def __init__(self, device):
        super().__init__()

        self._series = QLineSeries()
        self._chart = QChart()
        
        # self.series_0 = QLineSeries()
        # self.series_1 = QLineSeries()

        # self.series_0.append(QPointF(0, 8))
        # self.series_0.append(QPointF(10, 8))
        # self.series_1.append(QPointF(0, 0))
        # self.series_1.append(QPointF(10, 0))
        # self._series2 = QAreaSeries(self.series_0, self.series_1)

        self._chart.addSeries(self._series)
        # self._chart.addSeries(self._series2)
        self._axis_x = QValueAxis()
        self._axis_x.setRange(0, SAMPLE_COUNT)
        self._axis_x.setLabelFormat("%g")
        self._axis_x.setTitleText("Samples")
        self._axis_y = QValueAxis()
        self._axis_y.setRange(-1, 1)
        self._axis_y.setTitleText("Audio level")
        self._chart.setAxisX(self._axis_x, self._series)
        self._chart.setAxisY(self._axis_y, self._series)
        self._chart.legend().hide()
        name = device.description()
        self._chart.setTitle(f"Data from the microphone ({name})")

        format_audio = QAudioFormat()
        format_audio.setSampleRate(8000)
        format_audio.setChannelCount(1)
        format_audio.setSampleFormat(QAudioFormat.UInt8)

        self._audio_input = QAudioSource(device, format_audio, self)
        self._io_device = self._audio_input.start()
        self._io_device.readyRead.connect(self._readyRead)

        self._chart_view = QChartView(self._chart)
        self.setCentralWidget(self._chart_view)

        self._buffer = [QPointF(x, 0) for x in range(SAMPLE_COUNT)]
        self._series.append(self._buffer)

    def closeEvent(self, event):
        if self._audio_input is not None:
            self._audio_input.stop()
        event.accept()

    @Slot()
    def _readyRead(self):
        data = self._io_device.readAll()
        available_samples = data.size() // RESOLUTION
        start = 0
        print(available_samples)
        if (available_samples < SAMPLE_COUNT):
            start = SAMPLE_COUNT - available_samples
            for s in range(start):
                self._buffer[s].setY(self._buffer[s + available_samples].y())

        data_index = 0
        for s in range(start, SAMPLE_COUNT):
            value = (ord(data[data_index]) - 128) / 128
            self._buffer[s].setY(value)
            data_index = data_index + RESOLUTION
        self._series.replace(self._buffer)


if __name__ == '__main__':
    app = QApplication(sys.argv)

    input_devices = QMediaDevices.audioInputs()
    if not input_devices:
        QMessageBox.warning(None, "audio", "There is no audio input device available.")
        sys.exit(-1)
    main_win = MainWindow(input_devices[0])
    main_win.setWindowTitle("audio")
    available_geometry = main_win.screen().availableGeometry()
    size = available_geometry.height()
    main_win.resize(size, size*0.75)
    main_win.show()
    sys.exit(app.exec())