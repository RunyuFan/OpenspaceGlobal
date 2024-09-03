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
result_type_id_dic = {'公园绿地': 1, '户外运动场所': 2, '交通场地空间': 3, '水体': 4, "非城市开放空间": 5}
result_type_flag = {'公园绿地': 2, '户外运动场所': 4, '交通场地空间': 5, '水体': 3, "非城市开放空间": 1}


def get_merge_result_1():
    for country in os.listdir(r"I:\全球开放空间结果处理\预测结果correct_shp"):
        for city in os.listdir(f"I:\\全球开放空间结果处理\\预测结果correct_shp\\{country}"):
            print(country,city)
            save_doc = f"I:\\全球开放空间结果处理\\预测结果correct_shp_merge\\{country}\\{city}"
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


            pre_shp = shp.Reader(f"I:\\全球开放空间结果处理\\预测结果correct_shp\\{country}\\{city}\\{city}.shp")
            for shapeRec in pre_shp.iterShapeRecords():
                w.record(*shapeRec.record)
                w.shape(shapeRec.shape)
            pre_shp.close()

            osm_shp = shp.Reader(f"K:\\开放空间\\{country}\\aoi_byGrid\\{country}_{city}.shp")
            for shapeRec in osm_shp.iterShapeRecords():
                idx = shapeRec.record["idx"]
                osm_type = shapeRec.record["type"]
                osm_type_id = result_type_id_dic[osm_type]
                flag = shapeRec.record["flag"]
                w.record(idx,osm_type,osm_type_id,flag)
                w.shape(shapeRec.shape)
            osm_shp.close()

            f = open(save_shp.replace(".shp", ".prj"), 'w')
            g = open(save_shp.replace(".shp", ".cpg"), 'w')
            g.write(coding)
            f.write(wkt)
            g.close()

def get_merge_result_2(country,city):
    dic = {1: '公园绿地', 2: '户外运动场所', 3: '交通场地空间', 4: '水体', 5: "非城市开放空间"}

    save_doc = f"I:\\全球开放空间结果处理\\预测结果correct_shp_merge\\{country}\\{city}"
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
        tmp_layer = None
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
    tmp_layer = None
    ds.Destroy()

    osm_shp = f"K:\\开放空间\\{country}\\aoi_byGrid\\{country}_{city}.shp"
    ds = driver.Open(osm_shp)
    tmp_layer = ds.GetLayer(0)
    for feature in tmp_layer:
        osm_idx = feature.GetField("idx")
        osm_type = feature.GetField("type")
        osm_type_id = result_type_id_dic[osm_type]
        flag = int(feature.GetField("flag"))


        geom = feature.GetGeometryRef()
        new_feature = ogr.Feature(layerDefn)
        new_feature.SetGeometry(geom)
        new_feature.SetField('idx', osm_idx)
        new_feature.SetField('type', osm_type)
        new_feature.SetField('type_id', osm_type_id)
        new_feature.SetField('flag', flag)

        layer.CreateFeature(new_feature)
        new_feature = None
        geom = None
    tmp_layer = None
    ds.Destroy()
    data_source.Destroy()


def get_merge_result_LosAngeles():
    dic = {1: '公园绿地', 2: '户外运动场所', 3: '交通场地空间', 4: '水体', 5: "非城市开放空间"}

    save_doc = r"I:\全球开放空间结果处理\预测结果correct_shp_merge\United_States\LosAngeles"
    if not os.path.exists(save_doc):
        os.makedirs(save_doc)


    save_shp = f"{save_doc}\\LosAngeles.json"

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

    shp_doc = r"J:\全球开放空间预测结果\预测结果_tif\United_States\LosAngeles\OS_pre_shp"
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
        tmp_layer = None
        ds.Destroy()

    sport_shp = r"J:\全球开放空间预测结果\运动场预测结果处理\pre_sports_correct\sport_aoi_ESA_value\United_States\LosAngeles\LosAngeles_pre_ESA_correct.shp"
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
    tmp_layer = None
    ds.Destroy()

    osm_shp = r"K:\开放空间\United_States\aoi_byGrid\United_States_LosAngeles_grid.shp"
    ds = driver.Open(osm_shp)
    tmp_layer = ds.GetLayer(0)
    for feature in tmp_layer:
        osm_id = feature.GetField("id")
        osm_type = feature.GetField("type")
        if osm_type=="其他":
            continue
        osm_type_id = result_type_id_dic[osm_type]
        flag = result_type_flag[osm_type]

        geom = feature.GetGeometryRef()
        new_feature = ogr.Feature(layerDefn)
        new_feature.SetGeometry(geom)
        new_feature.SetField('idx', osm_id)
        new_feature.SetField('type', osm_type)
        new_feature.SetField('type_id', osm_type_id)
        new_feature.SetField('flag', flag)

        layer.CreateFeature(new_feature)
        new_feature = None
        geom = None
    tmp_layer = None
    ds.Destroy()

    road_shp = r"K:\开放空间\United_States\aoi_byGrid\United_States_LosAngeles_roads_buffer_grid.shp"
    ds = driver.Open(road_shp)
    tmp_layer = ds.GetLayer(0)
    for feature in tmp_layer:
        geom = feature.GetGeometryRef()
        new_feature = ogr.Feature(layerDefn)
        new_feature.SetGeometry(geom)
        new_feature.SetField('idx', idx)
        new_feature.SetField('type', "交通场地空间")
        new_feature.SetField('type_id', 3)
        new_feature.SetField('flag', 5)

        layer.CreateFeature(new_feature)
        new_feature = None
        geom = None
        idx+=1
    tmp_layer = None
    ds.Destroy()

    data_source.Destroy()


def get_merge_result_one_city(country,city):

    save_doc = f"K:\\全球开放空间结果处理\\预测结果correct_shp_merge\\{country}\\{city}"
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


    pre_shp = shp.Reader(f"K:\\全球开放空间结果处理\\预测结果correct_shp\\{country}\\{city}\\{city}.shp")
    for shapeRec in pre_shp.iterShapeRecords():
        w.record(*shapeRec.record)
        w.shape(shapeRec.shape)
    pre_shp.close()

    # osm_shp = shp.Reader(f"K:\\开放空间\\{country}\\aoi_byGrid\\{country}_{city}.shp")
    osm_shp = shp.Reader(r"J:\全球开放空间预测结果\世界OSM_road_shp\Egypt\Egypt_Cairo.shp")
    for shapeRec in osm_shp.iterShapeRecords():
        idx = shapeRec.record["idx"]
        osm_type = shapeRec.record["type"]
        osm_type_id = result_type_id_dic[osm_type]
        flag = shapeRec.record["flag"]
        w.record(idx,osm_type,osm_type_id,flag)
        w.shape(shapeRec.shape)
    osm_shp.close()

    f = open(save_shp.replace(".shp", ".prj"), 'w')
    g = open(save_shp.replace(".shp", ".cpg"), 'w')
    g.write(coding)
    f.write(wkt)
    g.close()




def bigggg():
    # get_merge_result_2("Japan", "Tokyo_Yokohama")
    get_merge_result_LosAngeles()
    # get_merge_result_2("United_States", "NewYork")
    # get_merge_result_2("United_States", "Dallas_FortWorth")
if __name__ == '__main__':
    get_merge_result_one_city("Egypt","Cairo")


