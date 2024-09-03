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

proj = osr.SpatialReference()
proj.ImportFromEPSG(4326)  # 4326-GCS_WGS_1984; 4490- GCS_China_Geodetic_Coordinate_System_2000
wkt = proj.ExportToWkt()
coding = "UTF-8"


def get_pre_result_china():
    result_type_id_dic = {'公园绿地': 1, '户外运动场所': 2, '交通场地空间': 3, '水体': 4, "非城市开放空间": 5}
    flag = 0

    for city in os.listdir(r"F:\开放空间数据处理\中国"):
        print(city)
        save_doc = f"J:\\全球开放空间预测结果\\预测结果correct_shp\\中国\\{city}"
        if not os.path.exists(save_doc):
            os.makedirs(save_doc)
        save_shp = f"{save_doc}\\{city}.shp"
        if os.path.exists(save_shp):
            continue

        w = shp.Writer(save_shp)
        w.field("idx", "N", 10)
        w.field("type", "C", 100)
        w.field("type_id", "N", 10)
        w.field("flag", "N", 10)
        idx = 0

        old_shp = shp.Reader(f"F:\\开放空间预测结果\\中国_result\\{city}\\OS_{city}.shp")
        for shapeRec in old_shp.iterShapeRecords():
            pre_type = shapeRec.record["type"]
            if pre_type == "户外运动场所":
                continue
            pre_id = result_type_id_dic[pre_type]
            w.record(idx,pre_type, pre_id, flag)
            w.shape(shapeRec.shape)
            idx += 1
        old_shp.close()

        sport_shp = shp.Reader(f"J:\\全球开放空间预测结果\\运动场预测结果处理\\pre_sports_correct\\sport_aoi_ESA_value\\中国\\{city}\\{city}_pre_ESA_correct.shp")
        for shapeRec in sport_shp.iterShapeRecords():
            sport_type = shapeRec.record["type"]
            type_id = shapeRec.record["type_id"]
            w.record(idx,sport_type, type_id, flag)
            w.shape(shapeRec.shape)
            idx += 1

        sport_shp.close()

        f = open(save_shp.replace(".shp", ".prj"), 'w')
        g = open(save_shp.replace(".shp", ".cpg"), 'w')
        g.write(coding)
        f.write(wkt)
        g.close()


def get_pre_result_foreign_1():
    """
    Japan Tokyo_Yokohama
    United_States LosAngeles
    United_States NewYork


    United_States Dallas_FortWorth
    """

    result_type_id_dic = {'公园绿地': 1, '户外运动场所': 2, '交通场地空间': 3, '水体': 4, "非城市开放空间": 5}
    flag = 0
    for country in os.listdir(r"J:\全球开放空间预测结果\预测结果_tif"):
        for city in os.listdir(f"J:\\全球开放空间预测结果\\预测结果_tif\\{country}"):
            print(country,city)
            old_pre_shp_path= f"J:\\全球开放空间预测结果\\预测结果_tif\\{country}\\{city}\\OS_pre_shp_merge\\{city}_pre.shp"
            if not os.path.exists(old_pre_shp_path):
                continue

            save_doc = f"I:\\全球开放空间结果处理\\预测结果correct_shp\\{country}\\{city}"
            if not os.path.exists(save_doc):
                os.makedirs(save_doc)
            save_shp = f"{save_doc}\\{city}.shp"
            if os.path.exists(save_shp):
                continue

            w = shp.Writer(save_shp)
            w.field("idx", "N", 10)
            w.field("type", "C", 100)
            w.field("type_id", "N", 10)
            w.field("flag", "N", 10)
            idx = 0

            old_shp = shp.Reader(old_pre_shp_path)
            for shapeRec in old_shp.iterShapeRecords():
                pre_type = shapeRec.record["type"]
                if pre_type == "户外运动场所":
                    continue
                pre_id = result_type_id_dic[pre_type]
                w.record(idx,pre_type, pre_id, flag)
                w.shape(shapeRec.shape)
                idx += 1
            old_shp.close()

            sport_shp = shp.Reader(f"J:\\全球开放空间预测结果\\运动场预测结果处理\\pre_sports_correct\\sport_aoi_ESA_value\\{country}\\{city}\\{city}_pre_ESA_correct.shp")
            for shapeRec in sport_shp.iterShapeRecords():
                sport_type = shapeRec.record["type"]
                type_id = shapeRec.record["type_id"]
                w.record(idx,sport_type, type_id, flag)
                w.shape(shapeRec.shape)
                idx += 1
            sport_shp.close()

            f = open(save_shp.replace(".shp", ".prj"), 'w')
            g = open(save_shp.replace(".shp", ".cpg"), 'w')
            g.write(coding)
            f.write(wkt)
            g.close()


def get_pre_result_foreign_merge_big(country,city):
    """
    Japan Tokyo_Yokohama
    United_States LosAngeles
    United_States NewYork


    United_States Dallas_FortWorth
    """

    result_type_id_dic = {'公园绿地': 1, '户外运动场所': 2, '交通场地空间': 3, '水体': 4, "非城市开放空间": 5}
    dic = {1: '公园绿地', 2: '户外运动场所', 3: '交通场地空间', 4: '水体', 5: "非城市开放空间"}


    print(country,city)

    save_doc = f"I:\\全球开放空间结果处理\\大文件处理\\{country}\\{city}"
    if not os.path.exists(save_doc):
        os.makedirs(save_doc)


    save_shp_1 = f"{save_doc}\\{city}_1.shp"
    save_shp_2 = f"{save_doc}\\{city}_2.shp"
    save_shp_3 = f"{save_doc}\\{city}_3.shp"


    w = shp.Writer(save_shp_1)
    w.field("type", "C", 100)
    w.field("type_id", "N", 10)

    w_2 = shp.Writer(save_shp_2)
    w_2.field("type", "C", 100)
    w_2.field("type_id", "N", 10)

    w_3 = shp.Writer(save_shp_3)
    w_3.field("type", "C", 100)
    w_3.field("type_id", "N", 10)



    shp_doc = f"J:\\全球开放空间预测结果\\预测结果_tif\\Japan\\Tokyo_Yokohama\\OS_pre_shp"
    shp_list= []
    for file in os.listdir(shp_doc):
        if os.path.splitext(file)[1] == ".shp":
            shp_list.append(os.path.join(shp_doc,file))
    num = len(shp_list)
    i = 0
    for shp_f in tqdm(shp_list) :
        r = shp.Reader(shp_f)
        for shapeRec in r.iterShapeRecords():
            value = int(shapeRec.record["value"])
            if value == 2:
                continue
            type = dic[value]
            if i<num/3:
                w.record(type,value)
                w.shape(shapeRec.shape)
            elif i<2*num/3:
                w_2.record(type, value)
                w_2.shape(shapeRec.shape)
            else:
                w_3.record(type, value)
                w_3.shape(shapeRec.shape)
        i+=1


    f = open(save_shp_1.replace(".shp", ".prj"), 'w')
    g = open(save_shp_1.replace(".shp", ".cpg"), 'w')
    g.write(coding)
    f.write(wkt)
    g.close()
    f.close()

    f = open(save_shp_2.replace(".shp", ".prj"), 'w')
    g = open(save_shp_2.replace(".shp", ".cpg"), 'w')
    g.write(coding)
    f.write(wkt)
    g.close()
    f.close()

    f = open(save_shp_3.replace(".shp", ".prj"), 'w')
    g = open(save_shp_3.replace(".shp", ".cpg"), 'w')
    g.write(coding)
    f.write(wkt)
    g.close()
    f.close()

def get_pre_result_foreign_2(country,city):
    result_type_id_dic = {'公园绿地': 1, '户外运动场所': 2, '交通场地空间': 3, '水体': 4, "非城市开放空间": 5}
    dic = {1: '公园绿地', 2: '户外运动场所', 3: '交通场地空间', 4: '水体', 5: "非城市开放空间"}

    save_doc = f"I:\\全球开放空间结果处理\\大文件处理\\{country}\\{city}"
    if not os.path.exists(save_doc):
        os.makedirs(save_doc)


    save_shp = f"{save_doc}\\{city}.json"

    driver1 = ogr.GetDriverByName("GeoJSON")
    driver = ogr.GetDriverByName("ESRI Shapefile")
    geomtype = ogr.wkbMultiPolygon
    SpatialReference = osr.SpatialReference()
    SpatialReference.ImportFromEPSG(4326)

    data_source = driver1.CreateDataSource(save_shp)
    layer = data_source.CreateLayer("os", srs=SpatialReference, geom_type=geomtype)

    idFiled = ogr.FieldDefn("idx", ogr.OFTInteger)
    layer.CreateField(idFiled)

    typeFiled = ogr.FieldDefn("type", ogr.OFTString)
    typeFiled.SetWidth(100)
    layer.CreateField(typeFiled)

    type_idFiled = ogr.FieldDefn("type_id", ogr.OFTInteger)
    layer.CreateField(type_idFiled)

    flagFiled = ogr.FieldDefn("flag", ogr.OFTInteger)
    layer.CreateField(flagFiled)

    layerDefn = layer.GetLayerDefn()

    shp_doc = f"J:\\全球开放空间预测结果\\预测结果_tif\\{country}\\{city}\\OS_pre_shp"
    shp_list= []
    for file in os.listdir(shp_doc):
        if os.path.splitext(file)[1] == ".shp":
            shp_list.append(os.path.join(shp_doc,file))

    idx = 0
    for shp_f in tqdm(shp_list) :
        ds = driver.Open(os.path.join(shp_doc, shp_f))
        tmp_layer = ds.GetLayer(0)
        for feature in tmp_layer:
            value = int(feature.GetField("value"))
            if value==2:
                continue
            type = dic[value]
            geom = feature.GetGeometryRef()
            new_feature = ogr.Feature(layerDefn)
            new_feature.SetGeometry(geom)
            new_feature.SetField('idx', idx)
            new_feature.SetField('type', type)
            new_feature.SetField('type_id', value)
            new_feature.SetField('flag', 0)

            layer.CreateFeature(new_feature)
            idx+=1
            new_feature =None
            geom = None
        ds.Destroy()

    sport_shp = f"J:\\全球开放空间预测结果\\运动场预测结果处理\\pre_sports_correct\\sport_aoi_ESA_value\\{country}\\{city}\\{city}_pre_ESA_correct.shp"
    ds = driver.Open(sport_shp)
    tmp_layer = ds.GetLayer(0)
    for feature in tmp_layer:
        type = feature.GetField("type")
        type_id = int(feature.GetField("type_id"))
        geom = feature.GetGeometryRef()
        new_feature = ogr.Feature(layerDefn)
        new_feature.SetGeometry(geom)
        new_feature.SetField('idx', idx)
        new_feature.SetField('type', type)
        new_feature.SetField('type_id', type_id)
        new_feature.SetField('flag', 0)

        layer.CreateFeature(new_feature)
        idx += 1
        new_feature = None
        geom = None
    ds.Destroy()
    data_source.Destroy()



def get_pre_result_foreign_one_city(country,city):
    result_type_id_dic = {'公园绿地': 1, '户外运动场所': 2, '交通场地空间': 3, '水体': 4, "非城市开放空间": 5}
    flag = 0

    old_pre_shp_path= f"J:\\全球开放空间预测结果\\预测结果_tif\\{country}\\{city}\\OS_pre_shp_merge\\{city}_pre.shp"
    if not os.path.exists(old_pre_shp_path):
        return
    save_doc = f"K:\\全球开放空间结果处理\\预测结果correct_shp\\{country}\\{city}"
    if not os.path.exists(save_doc):
        os.makedirs(save_doc)

    save_shp = f"{save_doc}\\{city}.shp"
    if os.path.exists(save_shp):
        return

    w = shp.Writer(save_shp)
    w.field("idx", "N", 10)
    w.field("type", "C", 100)
    w.field("type_id", "N", 10)
    w.field("flag", "N", 10)
    idx = 0

    old_shp = shp.Reader(old_pre_shp_path)
    for shapeRec in old_shp.iterShapeRecords():
        pre_type = shapeRec.record["type"]
        if pre_type == "户外运动场所":
            continue
        pre_id = result_type_id_dic[pre_type]
        w.record(idx,pre_type, pre_id, flag)
        w.shape(shapeRec.shape)
        idx += 1
    old_shp.close()

    sport_shp = shp.Reader(f"J:\\全球开放空间预测结果\\运动场预测结果处理\\pre_sports_correct\\sport_aoi_ESA_value\\{country}\\{city}\\{city}_pre_ESA_correct.shp")
    for shapeRec in sport_shp.iterShapeRecords():
        sport_type = shapeRec.record["type"]
        type_id = shapeRec.record["type_id"]
        w.record(idx,sport_type, type_id, flag)
        w.shape(shapeRec.shape)
        idx += 1
    sport_shp.close()

    f = open(save_shp.replace(".shp", ".prj"), 'w')
    g = open(save_shp.replace(".shp", ".cpg"), 'w')
    g.write(coding)
    f.write(wkt)
    g.close()

if __name__ == '__main__':
    # """
    # Japan Tokyo_Yokohama
    # United_States LosAngeles
    # United_States NewYork
    #
    #
    # United_States Dallas_FortWorth
    # """
    # get_pre_result_foreign_2("United_States","LosAngeles")
    # get_pre_result_foreign_2("United_States", "NewYork")
    get_pre_result_foreign_one_city("Egypt","Cairo")



