import shutil
import cv2
from osgeo import ogr, osr, gdal
import shapefile as shp
import os
import numpy as np
import pandas as pd
import Data_Processing.SHP as shp_m
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import random



dic_ESA = {80:4,10:1,20:1,30:1,40:1,60:1,90:1,95:1,100:1,50:5}
dic_cls = {1:'公园绿地',2:'户外运动场所',3:'交通场地空间',4:'水体',5:"非城市开放空间"}

def correct():
    proj = osr.SpatialReference()
    proj.ImportFromEPSG(4326)
    wkt = proj.ExportToWkt()
    coding = "UTF-8"

    for country in os.listdir(r"J:\全球开放空间预测结果\运动场预测结果处理\pre_sports_correct\sport_aoi_ESA_value"):
        for city in os.listdir(f"J:\\全球开放空间预测结果\\运动场预测结果处理\\pre_sports_correct\\sport_aoi_ESA_value\\{country}"):
            print(country,city)
            save_shp = f"J:\\全球开放空间预测结果\\运动场预测结果处理\\pre_sports_correct\\sport_aoi_ESA_value\\{country}\\{city}\\{city}_pre_ESA_correct.shp"
            output_shp = shp.Writer(save_shp)
            output_shp.field("idx", "N", 10)
            output_shp.field("type", "C", 100)
            output_shp.field("type_id", "N", 10)

            idx = 1
            shp_path = f"J:\\全球开放空间预测结果\\运动场预测结果处理\\pre_sports_correct\\sport_aoi_ESA_value\\{country}\\{city}\\{city}_pre_ESA.shp"
            r = shp.Reader(shp_path)
            for shapeRec in r.iterShapeRecords():
                value = shapeRec.record["value"]
                area = shapeRec.record["Area"]
                if value in [40,60]:#,10,20,,30,60,90,95
                    type_id = 1
                elif value in [80]:
                    type_id = 4
                elif area<500:
                    type_id = dic_ESA[value]
                else:
                    type_id =2
                type = dic_cls[type_id]
                output_shp.record(idx,type,type_id)
                output_shp.shape(shapeRec.shape)
                idx+=1

            f = open(save_shp.replace(".shp", ".prj"), 'w')
            g = open(save_shp.replace(".shp", ".cpg"), 'w')
            g.write(coding)
            f.write(wkt)
            g.close()
            f.close()



if __name__ == '__main__':
    correct()