import os,io
import sys
import cv2
#import h5py
import numpy as np
import time
# import leveldb
# import plyvel
#import cupy as cp
import rocksdb



def str2byte(s):
    s = s.encode()
    return s


def byte2str(b):
    b = b.decode()
    return b


class Equirectangular:
    def __init__(self, data_path, city):
        count = 0
        self.wrong=0
        NYC = ['Wall_Street','Union_Square', 'Hudson_River']
        Pits= ['CMU', 'Allegheny', 'South_Shore']
        if city in NYC:
            self.db = rocksdb.DB(data_path+'manhattan/data/', rocksdb.Options(create_if_missing=False), read_only=True)
        elif city in Pits:
            self.db = rocksdb.DB(data_path+'pittsburgh/data/', rocksdb.Options(create_if_missing=False), read_only=True)
        elif 'navigation' in city:
            self.db = rocksdb.DB(data_path+'touchdown/data/', rocksdb.Options(create_if_missing=False), read_only=True)
            
        # self.db = rocksdb.DB('/home/tmp/kiran/manhattan/data0/', rocksdb.Options(create_if_missing=False), read_only=True)
        # self.db = plyvel.DB('/lab/tmpig10f/kiran/manhattan/data0/')
        self._img=None
        self._height=0
        self._width=0
        self.rec = {}

    def get_image(self, pos):
        pos = str(pos[0]) + ',' + str(pos[1])
        self._img = cv2.imdecode(np.frombuffer(self.db.get(str2byte(pos)), np.uint8), -1)
        [self._height, self._width, _] = self._img.shape



    # @profile(precision=5)
    def GetPerspective(self, FOV, THETA):
        #
        # THETA is left/right angle, PHI is up/down angle, both in degree
        #

        # height is fixed
        if THETA in self.rec:
            lon,lat=self.rec[THETA]
            # temp = time.time()
            # persp = cv2.remap(self._img, lon.astype(np.float32), lat.astype(np.float32), cv2.INTER_CUBIC,
            #                   borderMode=cv2.BORDER_WRAP)
            persp = cv2.remap(self._img, lon.astype(np.float32), lat.astype(np.float32), cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_WRAP)
            # self.t2 += time.time() - temp

            return persp
        height = self._height
        width = int(float(FOV) / 360 * self._width)
        w_len = np.tan(np.radians(FOV / 2.0))

        equ_h = self._height
        equ_w = self._width
        equ_cx = (equ_w - 1) / 2.0

        h_len = np.tan(np.radians(FOV / 2.0))

        x_map = np.ones([height, width], np.float32)
        y_map = np.tile(np.linspace(-w_len, w_len, width), [height, 1])

        D = np.sqrt(x_map ** 2 + y_map ** 2)
        xyz = np.stack((x_map, y_map), axis=2) / np.repeat(D[:, :, np.newaxis], 2, axis=2)

        # Rotation transormation
        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        z_axis = np.array([0.0, 0.0, 1.0], np.float32)
        [R1, _] = cv2.Rodrigues(z_axis * np.radians(THETA))
        R1 = np.array(R1)

        xyz = xyz.reshape([height * width, 2]).T
        xyz = np.dot(R1[:2, :2], xyz).T

        lon = np.arctan2(xyz[:, 1], xyz[:, 0])
        lon = lon.reshape([height, width]) / np.pi * 180
        lon = lon / 180 * equ_cx + equ_cx

        lat = np.tile(np.linspace(0, height - 1, height), [width, 1]).T

        self.rec[THETA] = [lon, lat]
        persp = cv2.remap(self._img, lon.astype(np.float32), lat.astype(np.float32), cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_WRAP)

        return persp



