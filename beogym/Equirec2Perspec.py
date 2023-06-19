import os,io
import sys
import cv2
import h5py
import numpy as np
import time
# import leveldb
import plyvel
import cupy as cp



def str2byte(s):
    s = s.encode()
    return s


def byte2str(b):
    b = b.decode()
    return b


class Equirectangular:
    def __init__(self):
        self.db = ''
        count = 0
        self.wrong=0
        while self.db == '':
            try:
                if count>100:
                    break
                self.db = plyvel.DB('/home/tmp/kiran/manhattan/data' + str(count) + '/')
            except :
                count += 1
        self._img=None
        self._height=0
        self._width=0
        self.rec = {}

    def get_image(self, pos):
        pos = str(pos[0]) + ',' + str(pos[1])
        try:
            self._img = cv2.imdecode(np.frombuffer(self.db.get(str2byte(pos)), np.uint8), -1)
        except:
            count=0
            print(count)
            self.db = ''
            while self.db == '' and self._img is None:
                try:
                    if count>100:
                        continue
                    self.db = plyvel.DB('/home/tmp/kiran/manhattan/data' + str(count) + '/')
                    self._img = cv2.imdecode(np.frombuffer(self.db.get(str2byte(pos)), np.uint8), -1)
                except :
                    count += 1
            #self._img = cv2.imdecode(np.frombuffer(self.db.get(str2byte(pos)), np.uint8), -1)
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
        height = 208
        width = int(float(FOV) / 360 * self._width)
        w_len = cp.tan(cp.radians(FOV / 2.0))

        equ_h = self._height
        equ_w = self._width
        equ_cx = (equ_w - 1) / 2.0

        h_len = cp.tan(cp.radians(FOV / 2.0))

        x_map = cp.ones([height, width], cp.float32)
        y_map = cp.tile(cp.linspace(-w_len, w_len, width), [height, 1])

        D = cp.sqrt(x_map ** 2 + y_map ** 2)
        xyz = cp.stack((x_map, y_map), axis=2) / cp.repeat(D[:, :, cp.newaxis], 2, axis=2)

        # Rotation transormation
        y_axis = cp.array([0.0, 1.0, 0.0], cp.float32)
        z_axis = cp.array([0.0, 0.0, 1.0], cp.float32)
        [R1, _] = cv2.Rodrigues(cp.asnumpy(z_axis * cp.radians(THETA)))
        R1 = cp.array(R1)

        xyz = xyz.reshape([height * width, 2]).T
        xyz = cp.dot(R1[:2, :2], xyz).T

        lon = cp.arctan2(xyz[:, 1], xyz[:, 0])
        lon = lon.reshape([height, width]) / cp.pi * 180
        lon = lon / 180 * equ_cx + equ_cx

        lat = cp.tile(cp.linspace(0, height - 1, height), [width, 1]).T
        lon = cp.asnumpy(lon)
        lat = cp.asnumpy(lat)

        self.rec[THETA] = [lon, lat]
        persp = cv2.remap(self._img, lon.astype(np.float32), lat.astype(np.float32), cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_WRAP)

        return persp



