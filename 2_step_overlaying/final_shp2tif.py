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


# 缺少获取shp文件坐标系的步骤
def vector2raster(inputfilePath, outputfile, resp,project):
    sf = shp.Reader(inputfilePath)
    # 读取shp四至
    min_x, min_y, max_x, max_y = sf.bbox

    tifrow = int((max_x - min_x) / resp)
    tifcol = int((max_y - min_y) / resp)

    vector = ogr.Open(inputfilePath)

    # layer = vector.GetLayer()
    layer = vector.ExecuteSQL(f"select * from {vector.GetLayer().GetName()} order by flag")

    targetDataset = gdal.GetDriverByName('GTiff').Create(outputfile, tifrow, tifcol, 1, gdal.GDT_Byte,options=["TILED=YES", "COMPRESS=LZW"])
    transform = (min_x, resp, 0, max_y, 0, -resp)

    targetDataset.SetGeoTransform(transform)
    targetDataset.SetProjection(project)
    band = targetDataset.GetRasterBand(1)
    NoData_value = 0
    band.SetNoDataValue(NoData_value)
    band.FlushCache()
    gdal.RasterizeLayer(targetDataset, [1], layer,options=["ATTRIBUTE=type_id"])
    targetDataset = None


def vector2raster_json(json_file, outputfile, resp,project):
    driver = ogr.GetDriverByName("GeoJSON")
    data_source = driver.Open(json_file)
    inlayer = data_source.GetLayer()

    extent = inlayer.GetExtent()
    (min_x, max_x, min_y, max_y) = extent

    # 读取shp四至
    tifrow = int((max_x - min_x) / resp)
    tifcol = int((max_y - min_y) / resp)


    layer = data_source.ExecuteSQL(f"select * from {inlayer.GetName()} order by flag")
    targetDataset = gdal.GetDriverByName('GTiff').Create(outputfile, tifrow, tifcol, 1, gdal.GDT_Byte,options=["TILED=YES", "COMPRESS=LZW"])
    transform = (min_x, resp, 0, max_y, 0, -resp)

    targetDataset.SetGeoTransform(transform)
    targetDataset.SetProjection(project)
    band = targetDataset.GetRasterBand(1)
    NoData_value = 0
    band.SetNoDataValue(NoData_value)
    band.FlushCache()
    gdal.RasterizeLayer(targetDataset, [1], layer,options=["ATTRIBUTE=type_id"])
    targetDataset = None
    data_source = None


def china_os_result():
    img = gdal.Open(r"J:\开放空间世界image\Brazil\Brasilia\209.tif")
    projection = img.GetProjection()
    transform = img.GetGeoTransform()
    # print(transform)
    # print(projection)
    for city in os.listdir(r"J:\全球开放空间预测结果\World_OS_Result_correct\China"):
        print(city)
        input_shp = f"K:\\全球开放空间结果处理\\预测结果correct_shp_merge\\China\\{city}\\{city}.shp"
        save_doc = f"J:\全球开放空间预测结果\\World_OS_Result_correct\\China\\{city}"
        if not os.path.exists(save_doc):
            os.makedirs(save_doc)
        save_raster = f"{save_doc}\\{city}.tif"
        if os.path.exists(save_raster):
            continue
        vector2raster(input_shp,save_raster,transform[1],projection)


def foreign_os_result():
    img = gdal.Open(r"J:\开放空间世界image\Brazil\Brasilia\209.tif")
    projection = img.GetProjection()
    transform = img.GetGeoTransform()
    # print(transform)
    # print(projection)
    for country in os.listdir(r"I:\全球开放空间结果处理\预测结果correct_shp_merge"):
        for city in os.listdir(f"I:\\全球开放空间结果处理\\预测结果correct_shp_merge\\{country}"):

            print(country,city)
            input_shp = f"I:\\全球开放空间结果处理\\预测结果correct_shp_merge\\{country}\\{city}\\{city}.shp"
            save_doc = f"J:\全球开放空间预测结果\\World_OS_Result_correct\\{country}\\{city}"
            if not os.path.exists(save_doc):
                os.makedirs(save_doc)
            save_raster = f"{save_doc}\\{city}.tif"
            if os.path.exists(save_raster):
                continue
            vector2raster(input_shp,save_raster,transform[1],projection)


def foreign_os_result_one_city(country,city):
    img = gdal.Open(r"J:\开放空间世界image\Brazil\Brasilia\209.tif")
    projection = img.GetProjection()
    transform = img.GetGeoTransform()
    # print(transform)
    # print(projection)

    input_shp = f"K:\\全球开放空间结果处理\\预测结果correct_shp_merge\\{country}\\{city}\\{city}.shp"
    save_doc = f"J:\全球开放空间预测结果\\World_OS_Result_correct\\{country}\\{city}"
    if not os.path.exists(save_doc):
        os.makedirs(save_doc)
    save_raster = f"{save_doc}\\{city}.tif"
    if os.path.exists(save_raster):
        return
    vector2raster(input_shp,save_raster,transform[1],projection)


def big(country,city):
    print(country,city)
    img = gdal.Open(r"J:\开放空间世界image\Brazil\Brasilia\209.tif")
    projection = img.GetProjection()
    transform = img.GetGeoTransform()


    input_shp = f"I:\\全球开放空间结果处理\\预测结果correct_shp_merge\\{country}\\{city}\\{city}.json"
    save_doc = f"J:\全球开放空间预测结果\\World_OS_Result_correct\\{country}\\{city}"
    if not os.path.exists(save_doc):
        os.makedirs(save_doc)
    save_raster = f"{save_doc}\\{city}.tif"
    if os.path.exists(save_raster):
        return
    vector2raster_json(input_shp, save_raster, transform[1], projection)



def osm2tif():
    img = gdal.Open(r"J:\开放空间世界image\Brazil\Brasilia\209.tif")
    projection = img.GetProjection()
    transform = img.GetGeoTransform()


    for country in os.listdir(r"J:\全球开放空间预测结果\World_OS_Result_correct"):
        for city in os.listdir(f"J:\\全球开放空间预测结果\\World_OS_Result_correct\\{country}"):
            print(country,city)
            input_shp = f"K:\\开放空间\\{country}\\aoi_byGrid\\{country}_{city}.shp"
            if country == "China":
                continue
                # input_shp = f"I:\\开放空间数据处理\\中国_osm\\{city}\\{city}_osm_road.shp"
            save_doc = f"J:\\全球开放空间预测结果\\世界OSM_road_tif\\{country}"
            if not os.path.exists(save_doc):
                os.makedirs(save_doc)
            save_raster = f"{save_doc}\\{country}_{city}.tif"
            if os.path.exists(save_raster):
                continue
            vector2raster(input_shp,save_raster,transform[1],projection)

def osm2tif_china():
    dic = {"上海市": "Shanghai", "北京市": "Beijing", "深圳市": "Shenzhen", "成都市": "Chengdu", "广州市": "Guangzhou",
           "重庆市": "Chongqing", "天津市": "Tianjin", "武汉市": "Wuhan", "东莞市": "Dongguan", "西安市": "Xi_an",
           "杭州市": "Hangzhou", "佛山市": "Foshan", "南京市": "Nanjing", "沈阳市": "Shenyang", "青岛市": "Qingdao",
           "济南市": "Jinan", "长沙市": "Changsha", "哈尔滨市": "Harbin", "郑州市": "Zhengzhou", "昆明市": "Kunming",
           "大连市": "Dalian", "苏州市": "Suzhou", "厦门市": "Xiamen", "泉州市": "Quanzhou", "合肥市": "Hefei",
           "太原市": "Taiyuan",
           "乌鲁木齐市": "Urumqi", "无锡市": "Wuxi", "福州市": "Fuzhou", "宁波市": "Ningbo", "贵阳市": "Guiyang",
           "南昌市": "Nanchang",
           "常州市": "Changzhou", "中山市": "Zhongshan", "石家庄市": "Shijiazhuang", "长春市": "Changchun",
           "香港特别行政区": "Hong_Kong",
           "台北市": "Taipei", "温州市": "Wenzhou"}

    img = gdal.Open(r"J:\开放空间世界image\Brazil\Brasilia\209.tif")
    projection = img.GetProjection()
    transform = img.GetGeoTransform()
    country = "China"

    for city in os.listdir(f"I:\开放空间数据处理\中国_osm"):
        print(country,city)
        city_e = dic[city]
        input_shp = f"I:\\开放空间数据处理\\中国_osm\\{city}\\{city_e}.shp"
        save_doc = f"J:\\全球开放空间预测结果\\世界OSM_road_tif\\{country}"
        if not os.path.exists(save_doc):
            os.makedirs(save_doc)
        save_raster = f"{save_doc}\\{country}_{city_e}.tif"
        if os.path.exists(save_raster):
            continue
        vector2raster(input_shp,save_raster,transform[1],projection)

if __name__ == '__main__':
    # from get_final_result_shp import bigggg
    # """
    # Japan Tokyo_Yokohama
    # United_States LosAngeles
    # United_States NewYork
    #
    #
    # United_States Dallas_FortWorth
    # """
    # # big("Japan", "Tokyo_Yokohama")
    # # big("United_States", "LosAngeles")
    # big("United_States", "NewYork")
    # big("United_States", "Dallas_FortWorth")
    foreign_os_result_one_city("Egypt","Cairo")
    # china_os_result()