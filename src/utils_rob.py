#! /usr/bin/env python
import math
import numpy as np


class Vector3D:
    def __init__(self, x, y, z):
        self._x = x
        self._y = y
        self._z = z

    def __mul__(self, other):
        return Vector3D(self._x * other._x, self._y * other._y, self._z * other._z)

    def mag(self):
        return math.sqrt((self._x) ^ 2 + (self._y) ^ 2 + (self._z) ^ 2)

    def dot(self, other):
        temp = self * other
        return temp._x + temp._y + temp._z

    def cos_theta(self):
        # vector's cos(angle) with the z-axis
        return self.dot(Vector3D(0, 0, 1)) / self.mag()  # (0,0,1) is the z-axis unit vector

    def phi(self):
        # vector's
        return math.asin(self.dot(Vector3D(0, 0, 1)) / self.mag())

    def toString(self):
        return "({x}, {y}, {z})".format(x=self._x, y=self._y, z=self._z)


class Vector7D:
    def __init__(self, x, y, z, xx, yy, zz, ww):
        self._x = x
        self._y = y
        self._z = z
        self._xx = xx
        self._yy = yy
        self._zz = zz
        self._ww = ww

    def __mul__(self, other):
        raise Exception("Not Implemented Exception")

    def dot(self, other):
        temp = self * other
        return temp._x + temp._y + temp._z

    def getQuaternion(self):
        q = np.zeros((4, 1))
        q[0] = self._xx
        q[1] = self._yy
        q[2] = self._zz
        q[3] = self._ww
        return q

    def quat2r(self):
        R = np.zeros((3, 3))
        R[0][0] = 2 * (self._ww * self._ww + self._xx * self._xx) - 1
        R[0][1] = 2 * (self._xx * self._yy - self._ww * self._zz)
        R[0][2] = 2 * (self._xx * self._zz + self._ww * self._yy)
        R[1][0] = 2 * (self._xx * self._yy + self._ww * self._zz)
        R[1][1] = 2 * (self._ww * self._ww + self._yy * self._yy) - 1
        R[1][2] = 2 * (self._yy * self._zz - self._ww * self._xx)
        R[2][0] = 2 * (self._xx * self._zz - self._ww * self._yy)
        R[2][1] = 2 * (self._yy * self._zz + self._ww * self._xx)
        R[2][2] = 2 * (self._ww * self._ww + self._zz * self._zz) - 1
        return R

    def toString(self):
        return "({x}, {y}, {z})".format(x=self._x, y=self._y, z=self._z)


# *.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*
# Funzione per convertire la matrice di rotazione in quaternione
# Input:    Ree --> matrice di rotazione
# Output:   quat --> quaternione come numpy array (x,y,z,w)
# *.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*
def r2quat(Ree):
    r1 = Ree[0][0]
    r2 = Ree[1][1]
    r3 = Ree[2][2]
    r4 = r1 + r2 + r3
    j = 1
    rj = r1
    if r2 > rj:
        j = 2
        rj = r2
    if r3 > rj:
        j = 3
        rj = r3
    if r4 > rj:
        j = 4
        rj = r4
    pj = 2 * math.sqrt(1 + 2 * rj - r4)
    if j == 1:
        p1 = pj / 4
        p2 = (Ree[1][0] + Ree[0][1]) / pj
        p3 = (Ree[0][2] + Ree[2][0]) / pj
        p4 = (Ree[2][1] - Ree[1][2]) / pj
    elif j == 2:
        p1 = (Ree[1][0] + Ree[0][1]) / pj
        p2 = pj / 4
        p3 = (Ree[2][1] + Ree[1][2]) / pj
        p4 = (Ree[0][2] - Ree[2][0]) / pj
    elif j == 3:
        p1 = (Ree[0][2] + Ree[2][0]) / pj
        p2 = (Ree[2][1] + Ree[1][2]) / pj
        p3 = pj / 4
        p4 = (Ree[1][0] - Ree[0][1]) / pj
    else:
        p1 = (Ree[2][1] - Ree[1][2]) / pj
        p2 = (Ree[0][2] - Ree[2][0]) / pj
        p3 = (Ree[1][0] - Ree[0][1]) / pj
        p4 = pj / 4
    quat = np.array([p1, p2, p3, p4])  # lo scalare e' l'ultimo
    return quat


# *.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*
# Funzione per convertire il quaternione in matrice di rotazione
# Input:     q --> quaternione come (x,y,z,w)
# Output:    R --> matrice di rotazione
# *.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*
def quat2r(q):
    R = np.zeros((3, 3))
    R[0][0] = 2 * (q[3] * q[3] + q[0] * q[0]) - 1
    R[0][1] = 2 * (q[0] * q[1] - q[3] * q[2])
    R[0][2] = 2 * (q[0] * q[2] + q[3] * q[1])
    R[1][0] = 2 * (q[0] * q[1] + q[3] * q[2])
    R[1][1] = 2 * (q[3] * q[3] + q[1] * q[1]) - 1
    R[1][2] = 2 * (q[1] * q[2] - q[3] * q[0])
    R[2][0] = 2 * (q[0] * q[2] - q[3] * q[1])
    R[2][1] = 2 * (q[1] * q[2] + q[3] * q[0])
    R[2][2] = 2 * (q[3] * q[3] + q[2] * q[2]) - 1
    return R


# *.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*
# Funzione per convertire la matrice di rotazione in asse/angolo
# Input:    R --> matrice di rotazione
# Output:   r --> asse di rotazione
#           theta --> angolo di rotazione
# *.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*
def r2asseangolo(R):
    # r = Vector3()
    r = Vector3D(0, 0, 0)
    val = ((R[0][0] + R[1][1] + R[2][2] - 1) * 0.5) + 0.0
    theta = math.acos(min(max(val, -1.0), 1.0))
    if math.fabs(theta - math.pi) <= 0.00001:
        r.x = -1 * math.sqrt((R[0][0] + 1) * 0.5)
        r.y = math.sqrt((R[1][1] + 1) * 0.5)
        r.z = math.sqrt(1 - (r.x ** 2) - (r.y ** 2))
    else:
        if theta >= 0.00001:
            r.x = (R[2][1] - R[1][2]) / (2 * math.sin(theta))
            r.y = (R[0][2] - R[2][0]) / (2 * math.sin(theta))
            r.z = (R[1][0] - R[0][1]) / (2 * math.sin(theta))
    return [r, theta]

# *.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*
# Funzione per convertire asse/angolo in matrice di rotazione
# Input:    R --> matrice di rotazione
# Output:   r --> asse di rotazione
#           theta --> angolo di rotazione
# *.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*
def asseangolo2r(r, theta):
    # r = Vector3()
    a = math.cos(theta)
    b = math.sin(theta)
    R = np.zeros((3, 3))
    R[0][0] = (r.x ** 2) * (1-a) + a
    R[0][1] = r.x * r.y * (1-a) - r.z * b
    R[0][2] = r.x * r.z * (1-a) + r.y * b
    R[1][0] = r.x * r.y * (1-a) + r.z * b
    R[1][1] = (r.y ** 2) * (1-a) + a
    R[1][2] = r.y * r.z * (1-a) - r.x * b
    R[2][0] = r.x * r.z * (1-a) - r.y * b
    R[2][1] = r.y * r.z * (1-a) + r.x * b
    R[2][2] = (r.z ** 2) * (1-a) + a
    return R

# *.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*
# Funzione per convertire la matrice di rotazione in rotationVector
# Input:    R --> matrice di rotazione
# Output:   r_theta --> rotationVector [theta*rx, theta*ry, theta*rz]
# *.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*
def r2rotationVector(R):
    # r = Vector3()
    r = Vector3D(0, 0, 0)
    val = ((R[0][0] + R[1][1] + R[2][2] - 1) * 0.5) + 0.0
    theta = math.acos(min(max(val, -1.0), 1.0))
    if math.fabs(theta - math.pi) <= 0.00001:
        r.x = -1 * math.sqrt((R[0][0] + 1) * 0.5)
        r.y = math.sqrt((R[1][1] + 1) * 0.5)
        r.z = math.sqrt(1 - (r.x ** 2) - (r.y ** 2))
    else:
        if theta >= 0.00001:
            r.x = (R[2][1] - R[1][2]) / (2 * math.sin(theta))
            r.y = (R[0][2] - R[2][0]) / (2 * math.sin(theta))
            r.z = (R[1][0] - R[0][1]) / (2 * math.sin(theta))
    r_theta = Vector3D(0, 0, 0)
    r_theta.x = theta * r.x
    r_theta.y = theta * r.y
    r_theta.z = theta * r.z
    return r_theta
