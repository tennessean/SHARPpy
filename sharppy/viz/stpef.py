import numpy as np
import os
from PySide import QtGui, QtCore
import sharppy.sharptab as tab
import sharppy.databases.inset_data as inset_data
from sharppy.sharptab.constants import *

## routine written by Kelton Halbert and Greg Blumberg
## keltonhalbert@ou.edu and wblumberg@ou.edu

__all__ = ['backgroundSTPEF', 'plotSTPEF']

class backgroundSTPEF(QtGui.QFrame):
    '''
    Draw the background frame and lines for the Theta-E plot frame
    '''
    def __init__(self):
        super(backgroundSTPEF, self).__init__()
        self.initUI()


    def initUI(self):
        ## window configuration settings,
        ## sich as padding, width, height, and
        ## min/max plot axes
        self.setStyleSheet("QFrame {"
            "  background-color: rgb(0, 0, 0);"
            "  border-width: 1px;"
            "  border-style: solid;"
            "  border-color: #3399CC;}")
        if self.physicalDpiX() > 75:
            fsize = 10
        else:
            fsize = 11
        self.plot_font = QtGui.QFont('Helvetica', fsize + 1)
        self.box_font = QtGui.QFont('Helvetica', fsize)
        self.plot_metrics = QtGui.QFontMetrics( self.plot_font )
        self.box_metrics = QtGui.QFontMetrics(self.box_font)
        self.plot_height = self.plot_metrics.xHeight() + 5
        self.box_height = self.box_metrics.xHeight() + 5
        self.lpad = 0.; self.rpad = 0.
        self.tpad = 25.; self.bpad = 15.
        self.wid = self.size().width() - self.rpad
        self.hgt = self.size().height() - self.bpad
        self.tlx = self.rpad; self.tly = self.tpad
        self.brx = self.wid; self.bry = self.hgt
        self.probmax = 70.; self.probmin = 0.
        self.plotBitMap = QtGui.QPixmap(self.width()-2, self.height()-2)
        self.plotBitMap.fill(QtCore.Qt.black)
        self.plotBackground()

    def resizeEvent(self, e):
        '''
        Handles the event the window is resized
        '''
        self.initUI()
    
    def plotBackground(self):
        '''
        Handles painting the frame.
        '''
        ## initialize a painter object and draw the frame
        qp = QtGui.QPainter()
        qp.begin(self.plotBitMap)
        qp.setRenderHint(qp.Antialiasing)
        qp.setRenderHint(qp.TextAntialiasing)
        self.draw_frame(qp)
        qp.end()

    def setBlackPen(self, qp):
        color = QtGui.QColor('#000000')
        color.setAlphaF(.5)
        pen = QtGui.QPen(color, 0, QtCore.Qt.SolidLine)
        brush = QtGui.QBrush(QtCore.Qt.SolidPattern)
        qp.setPen(pen)
        qp.setBrush(brush)
        return qp

    def draw_frame(self, qp):
        '''
        Draw the background frame.
        qp: QtGui.QPainter object
        '''
        ## set a new pen to draw with
        EF1_color = "#006600"
        EF2_color = "#FFCC33"
        EF3_color = "#FF0000"
        EF4_color = "#FF00FF"

        pen = QtGui.QPen(QtCore.Qt.white, 2, QtCore.Qt.SolidLine)
        qp.setPen(pen)
        qp.setFont(self.plot_font)
        rect1 = QtCore.QRectF(1.5, 2, self.brx, self.plot_height)
        qp.drawText(rect1, QtCore.Qt.TextDontClip | QtCore.Qt.AlignCenter,
            'Conditional Tornado Probs based on STPC')

        qp.setFont(QtGui.QFont('Helvetica', 9))
        color = QtGui.QColor(EF1_color)
        pen = QtGui.QPen(color, 2, QtCore.Qt.SolidLine)
        qp.setPen(pen)
        rect1 = QtCore.QRectF(self.stpc_to_pix(.2), 2 + self.plot_height, 10, self.plot_height)
        qp.drawText(rect1, QtCore.Qt.TextDontClip | QtCore.Qt.AlignCenter,
            'EF1+')

        color = QtGui.QColor(EF2_color)
        pen = QtGui.QPen(color, 2, QtCore.Qt.SolidLine)
        qp.setPen(pen)
        rect1 = QtCore.QRectF(self.stpc_to_pix(1.1), 2 + self.plot_height, 10, self.plot_height)
        qp.drawText(rect1, QtCore.Qt.TextDontClip | QtCore.Qt.AlignCenter,
            'EF2+')

        color = QtGui.QColor(EF3_color)
        pen = QtGui.QPen(color, 2, QtCore.Qt.SolidLine)
        qp.setPen(pen)
        rect1 = QtCore.QRectF(self.stpc_to_pix(3.1), 2 + self.plot_height, 10, self.plot_height)
        qp.drawText(rect1, QtCore.Qt.TextDontClip | QtCore.Qt.AlignCenter,
            'EF3+')

        color = QtGui.QColor(EF4_color)
        pen = QtGui.QPen(color, 2, QtCore.Qt.SolidLine)
        qp.setPen(pen)
        rect1 = QtCore.QRectF(self.stpc_to_pix(6.1), 2 + self.plot_height, 10, self.plot_height)
        qp.drawText(rect1, QtCore.Qt.TextDontClip | QtCore.Qt.AlignCenter,
            'EF4+')

        pen = QtGui.QPen(QtCore.Qt.blue, 1, QtCore.Qt.DashLine)
        qp.setPen(pen)
        ytick_fontsize = 10
        y_ticks_font = QtGui.QFont('Helvetica', ytick_fontsize)
        qp.setFont(y_ticks_font)
        efstp_inset_data = inset_data.condSTPData()
        texts = efstp_inset_data['ytexts']
        spacing = self.bry / 10.
        y_ticks = np.arange(self.tpad, self.bry+spacing, spacing)
        for i in xrange(len(y_ticks)):
            pen = QtGui.QPen(QtGui.QColor("#0080FF"), 1, QtCore.Qt.DashLine)
            qp.setPen(pen)
            try:
                qp.drawLine(self.tlx, self.prob_to_pix(int(texts[i])), self.brx, self.prob_to_pix(int(texts[i])))
            except:
                continue
            color = QtGui.QColor('#000000')
            pen = QtGui.QPen(color, 1, QtCore.Qt.SolidLine)
            qp.setPen(pen)
            ypos = spacing*(i+1) - (spacing/4.)
            ypos = self.prob_to_pix(int(texts[i])) - ytick_fontsize/2
            rect = QtCore.QRect(self.tlx, ypos, 20, ytick_fontsize)
            pen = QtGui.QPen(QtCore.Qt.white, 1, QtCore.Qt.SolidLine)
            qp.setPen(pen)
            qp.drawText(rect, QtCore.Qt.TextDontClip | QtCore.Qt.AlignCenter, texts[i])

        width = self.brx / 12
        spacing = self.brx / 12
        center = np.arange(spacing, self.brx, spacing) - width/2.
        texts = efstp_inset_data['xticks']
        
        # Draw the x tick marks
        qp.setFont(QtGui.QFont('Helvetica', 8))
        for i in xrange(np.asarray(texts).shape[0]):
            color = QtGui.QColor('#000000')
            color.setAlpha(0)
            pen = QtGui.QPen(color, 1, QtCore.Qt.SolidLine)
            rect = QtCore.QRectF(center[i], self.prob_to_pix(-2), width, 4)
            # Change to a white pen to draw the text below the box and whisker plot
            pen = QtGui.QPen(QtCore.Qt.white, 1, QtCore.Qt.SolidLine)
            qp.setPen(pen)
            qp.drawText(rect, QtCore.Qt.TextDontClip | QtCore.Qt.AlignCenter, texts[i])
        
        # Draw the EF1+ stuff
        ef1 = efstp_inset_data['EF1+']
        color = QtGui.QColor(EF1_color)
        pen = QtGui.QPen(color, 3, QtCore.Qt.SolidLine)
        qp.setPen(pen)        
        for i in xrange(1, np.asarray(texts).shape[0], 1):
            qp.drawLine(center[i-1] + width/2, self.prob_to_pix(ef1[i-1]), center[i] + width/2, self.prob_to_pix(ef1[i]))

        # Draw the EF2+ stuff
        ef1 = efstp_inset_data['EF2+']
        color = QtGui.QColor(EF2_color)
        pen = QtGui.QPen(color, 3, QtCore.Qt.SolidLine)
        qp.setPen(pen)        
        for i in xrange(1, np.asarray(texts).shape[0], 1):
            qp.drawLine(center[i-1] + width/2, self.prob_to_pix(ef1[i-1]), center[i] + width/2, self.prob_to_pix(ef1[i]))

        # Draw the EF3+ stuff
        ef1 = efstp_inset_data['EF3+']
        color = QtGui.QColor(EF3_color)
        pen = QtGui.QPen(color, 3, QtCore.Qt.SolidLine)
        qp.setPen(pen)        
        for i in xrange(1, np.asarray(texts).shape[0], 1):
            qp.drawLine(center[i-1] + width/2, self.prob_to_pix(ef1[i-1]), center[i] + width/2, self.prob_to_pix(ef1[i]))

        # Draw the EF4+ stuff
        ef1 = efstp_inset_data['EF4+']
        color = QtGui.QColor(EF4_color)
        pen = QtGui.QPen(color, 3, QtCore.Qt.SolidLine)
        qp.setPen(pen)        
        for i in xrange(1, np.asarray(texts).shape[0], 1):
            qp.drawLine(center[i-1] + width/2, self.prob_to_pix(ef1[i-1]), center[i] + width/2, self.prob_to_pix(ef1[i]))


    def prob_to_pix(self, prob):
        scl1 = self.probmax - self.probmin
        scl2 = self.probmin + prob
        return self.bry - (scl2 / scl1) * (self.bry - self.tpad)

    def stpc_to_pix(self, stpc):
        spacing = self.brx / 12
        center = np.arange(spacing, self.brx, spacing)
        if stpc == 0:
            i = 0
        elif stpc >= 0.01 and stpc < .5:
            i = 1
        elif stpc >= .5 and stpc < 1:
            i = 2
        elif stpc >= 1 and stpc < 2:
            i = 3
        elif stpc >= 2 and stpc < 3:
            i = 4
        elif stpc >= 3 and stpc < 4:
            i = 5
        elif stpc >= 4 and stpc < 6:
            i = 6
        elif stpc >= 6 and stpc < 8:
            i = 7
        elif stpc >= 8 and stpc < 10:
            i = 8
        elif stpc >= 10 and stpc < 12:
            i = 9
        else: 
            i = 10
        return center[i]


class plotSTPEF(backgroundSTPEF):
    '''
    Plot the data on the frame. Inherits the background class that
    plots the frame.
    '''
    def __init__(self):
        super(plotSTPEF, self).__init__()
        self.prof = None

    def setProf(self, prof):
        self.prof = prof
        self.stpc = prof.stp_cin

        self.clearData()
        self.plotBackground()
        self.plotData()
        self.update()

    def resizeEvent(self, e):
        '''
        Handles when the window is resized
        '''
        super(plotSTPEF, self).resizeEvent(e)
        self.plotData()
    
    def paintEvent(self, e):
        super(plotSTPEF, self).paintEvent(e)
        qp = QtGui.QPainter()
        qp.begin(self)
        qp.drawPixmap(1, 1, self.plotBitMap)
        qp.end()

    def clearData(self):
        '''
        Handles the clearing of the pixmap
        in the frame.
        '''
        self.plotBitMap = QtGui.QPixmap(self.width(), self.height())
        self.plotBitMap.fill(QtCore.Qt.black)
    
    def plotData(self):
        '''
        Handles painting on the frame
        '''
        if self.prof is None:
            return

        ## this function handles painting the plot
        ## create a new painter obkect
        qp = QtGui.QPainter()
        self.draw_stp(qp)

    def draw_stp(self, qp):
        qp.begin(self.plotBitMap)
        qp.setRenderHint(qp.Antialiasing)
        qp.setRenderHint(qp.TextAntialiasing)
        stpc_pix = self.stpc_to_pix(self.stpc)
        pen = QtGui.QPen(QtGui.QColor("#FFFFFF"), 1.5, QtCore.Qt.DotLine)
        qp.setPen(pen)
        qp.drawLine(stpc_pix, self.prob_to_pix(0), stpc_pix, self.prob_to_pix(70))
        qp.end()


