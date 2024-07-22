import numpy as np

class geosot2D:
    delt = 1.0e-10
    LON_LAT_DMS_TABLE = [0, 256.0, 128.0, 64.0, 32.0, 16.0, 8.0, 4.0, 2.0, 1.0,
                         32 / 60.0, 16 / 60.0, 8 / 60.0, 4 / 60.0, 2 / 60.0, 1 / 60.0,
                         32 / 3600.0, 16 / 3600.0, 8 / 3600.0, 4 / 3600.0, 2 / 3600.0, 1 / 3600.0,
                         1 / 7200.0, 1 / 14400.0, 1 / 28800.0, 1 / 57600.0, 1 / 115200.0, 1 / 230400.0,
                         1 / 460800.0, 1 / 921600.0, 1 / 1843200.0, 1 / 3686400.0, 1 / 7372800.0]
    seconddecimalB = 0
    seconddecimalL = 0
    des_B = 1
    des_L = 1

    @staticmethod

    def ll2code(Lon, Lat, layer):
        fLat = abs(Lat)
        fLon = abs(Lon)
        int_binary2code_B = 0
        int_binary2code_L = 0

        assert not (fLat - 90.0 > geosot2D.delt or fLon - 180.0 > geosot2D.delt), "Input is out of bounds"

        if Lon < 0: int_binary2code_L = 1 << 31
        if Lat < 0: int_binary2code_B = 1 << 31
        if layer <= 9:
            int_B = (int(fLat)) >> (9 - layer)
            int_L = (int(fLon)) >> (9 - layer)
            int_binary2code_B |= int_B << (32 - layer)
            int_binary2code_L |= int_L << (32 - layer)
        elif layer <= 15:

            int_B = (int(fLat))
            int_L = (int(fLon))
            int_binary2code_B |= int_B << 23
            int_binary2code_L |= int_L << 23

            minuteB = (fLat - int(fLat)) * 60.0
            minuteL = (fLon - int(fLon)) * 60.0
            int_B = (int(minuteB)) >> (15 - layer)
            int_L = (int(minuteL)) >> (15 - layer)
            int_binary2code_B |= int_B << (32 - layer)
            int_binary2code_L |= int_L << (32 - layer)
        elif layer <= 21:

            int_B = (int(fLat))
            int_L = (int(fLon))
            int_binary2code_B |= int_B << 23
            int_binary2code_L |= int_L << 23

            minuteB = (fLat - int(fLat)) * 60.0
            minuteL = (fLon - int(fLon)) * 60.0
            int_B = (int(minuteB))
            int_L = (int(minuteL))
            int_binary2code_B |= int_B << 17
            int_binary2code_L |= int_L << 17

            secondB = (minuteB - int(minuteB)) * 60.0
            secondL = (minuteL - int(minuteL)) * 60.0
            int_B = (int(secondB)) >> (21 - layer)
            int_L = (int(secondL)) >> (21 - layer)
            int_binary2code_B |= int_B << (32 - layer)
            int_binary2code_L |= int_L << (32 - layer)
        else:

            int_B = int(fLat)
            int_L = int(fLon)
            int_binary2code_B |= int_B << 23
            int_binary2code_L |= int_L << 23

            minuteB = (fLat - int(fLat)) * 60.0
            minuteL = (fLon - int(fLon)) * 60.0
            int_B = int(minuteB)
            int_L = int(minuteL)
            int_binary2code_B |= int_B << 17
            int_binary2code_L |= int_L << 17

            secondB = (minuteB - int(minuteB)) * 60.0
            secondL = (minuteL - int(minuteL)) * 60.0
            int_B = int(secondB * 2048.0)
            int_L = int(secondL * 2048.0)
            int_binary2code_B |= int_B
            int_binary2code_L |= int_L

        zero = 32 - layer
        BinaryLonCode = int_binary2code_L >> zero << zero
        BinaryLatCode = int_binary2code_B >> zero << zero

        return BinaryLonCode, BinaryLatCode, layer

    @staticmethod
    def code2ll(BinaryLonCode, BinaryLatCode, layer):

        assert (geosot2D.JudgeCodeWhetherExist_2D(BinaryLatCode, BinaryLonCode)), 
        if BinaryLatCode >> 31 == 1:
            geosot2D.des_B = -1.0
            BinaryLatCode = BinaryLatCode << 1 >> 1
        if BinaryLonCode >> 31 == 1:
            geosot2D.des_L = -1.0
            BinaryLonCode = BinaryLonCode << 1 >> 1

        if layer <= 9:
            LatCode = BinaryLatCode >> 23
            LonCode = BinaryLonCode >> 23
        elif layer <= 15:

            degreeB = BinaryLatCode >> 23
            degreeL = BinaryLonCode >> 23

            minuteB = int((np.uint32(BinaryLatCode << 9) >> (41 - layer) << (15 - layer)))
            minuteL = int((np.uint32(BinaryLonCode << 9) >> (41 - layer) << (15 - layer)))
            LatCode = float(degreeB) + float(minuteB) / 60.0
            LonCode = float(degreeL) + float(minuteL) / 60.0
        elif layer <= 21:

            degreeB = BinaryLatCode >> 23
            degreeL = BinaryLonCode >> 23

            minuteB = int(np.uint32(BinaryLatCode << 9) >> 26)
            minuteL = int(np.uint32(BinaryLonCode << 9) >> 26)
§’
            secondB = int(np.uint32(BinaryLatCode << 15) >> (47 - layer) << (21 - layer))
            secondL = int(np.uint32(BinaryLonCode << 15) >> (47 - layer) << (21 - layer))
            LatCode = float(degreeB) + float(minuteB) / 60.0 + float(secondB) / 3600.0
            LonCode = float(degreeL) + float(minuteL) / 60.0 + float(secondL) / 3600.0
        else:

            degreeB = int(BinaryLatCode) >> 23
            degreeL = int(BinaryLonCode) >> 23

            minuteB = int(np.uint32(BinaryLatCode << 9) >> 26)
            minuteL = int(np.uint32(BinaryLonCode << 9) >> 26)
§’
            secondB = int(np.uint32(BinaryLatCode << 15) >> 26)
            secondL = int(np.uint32(BinaryLonCode << 15) >> 26)

            for i in range(layer - 21):
                geosot2D.seconddecimalB += (float(np.uint32(BinaryLatCode << (21 + i)) >> 31)) / (
                    float(pow(2.0, i + 1)))
                geosot2D.seconddecimalL += (float(np.uint32(BinaryLonCode << (21 + i)) >> 31)) / (
                    float(pow(2.0, i + 1)))
            LatCode = float(degreeB) + float(minuteB) / 60.0 + float(secondB + geosot2D.seconddecimalB) / 3600.0
            LonCode = float(degreeL) + float(minuteL) / 60.0 + float(secondL + geosot2D.seconddecimalB) / 3600.0

        LatCode *= geosot2D.des_B
        LonCode *= geosot2D.des_L
        return LonCode, LatCode

    @staticmethod
    def getlevel(interval):
        for i in range(len(geosot2D.LON_LAT_DMS_TABLE)):
            if geosot2D.LON_LAT_DMS_TABLE[i] > interval and interval > geosot2D.LON_LAT_DMS_TABLE[i + 1]:
                return i + 1

    @staticmethod
    def JudgeCodeWhetherExist_2D(BinaryLatCode, BinaryLonCode):
        lat1 = (90 - (np.uint32(BinaryLatCode << 1) >> 24)) >> 8 & 1
        lon1 = (180 - (np.uint32(BinaryLonCode << 1) >> 24)) >> 8 & 1
        lat2 = (np.uint32(BinaryLatCode << 9) >> 28) ^ 15
        lon2 = (np.uint32(BinaryLonCode << 9) >> 28) ^ 15
        lat3 = (np.uint32(BinaryLatCode << 15) >> 28) ^ 15
        lon3 = (np.uint32(BinaryLonCode << 15) >> 28) ^ 15
        exist = (not lat1) and (not lon1) and lat2 and lon2 and lat3 and lon3
        return exist


if __name__ == '__main__':
    geosot2D = geosot2D()

    BinaryLatCode, BinaryLonCode, layer = geosot2D.ll2code(79.365, 159.548, 13)
    print(BinaryLatCode, BinaryLonCode, layer)
