from osgeo import ogr, osr, gdal
import shapefile as shp
import os
import numpy as np
import cv2
import shutil
import pandas as pd
import Data_Processing.SHP as shp_m
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import geopandas


def raster2shp(input_raster,shp_save_doc):
    basename = os.path.basename(input_raster)
    outshp= os.path.join(shp_save_doc,basename.replace(".tif",".shp"))
    if os.path.exists(outshp):
        return 0

    inraster = gdal.Open(input_raster)
    im_data = inraster.GetRasterBand(1)



    if not os.path.exists(shp_save_doc):
        os.makedirs(shp_save_doc)

    SpatialReference = osr.SpatialReference()
    if inraster.GetProjection():
        SpatialReference.ImportFromWkt(inraster.GetProjection())
    else:
        SpatialReference.ImportFromEPSG(4326)


    driver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(outshp):  # 若文件已存在，则删除
        driver.DeleteDataSource(outshp)

    ds = driver.CreateDataSource(outshp)  # 创建Shape文件
    geomtype = ogr.wkbMultiPolygon
    options = ["ENCODING=UTF-8"]
    layer = ds.CreateLayer(os.path.splitext(basename)[0], srs=SpatialReference, geom_type=geomtype,options=options)
    layer.CreateField(ogr.FieldDefn('value', ogr.OFTReal))

    # gdal.FPolygonize(im_data, None, layer, 0)
    gdal.Polygonize(im_data, None, layer, 0)
    ds.SyncToDisk()
    ds.Destroy()



def save_tif_2(data, outpath, im_width, im_height, bandsNum, gdaltype, gt, proj,nodata,zip=None):
    format = "GTiff"
    driver = gdal.GetDriverByName(format)
    if zip == None:
        dst_ds = driver.Create(outpath, im_width, im_height, bandsNum, gdaltype)
    else:
        dst_ds = driver.Create(outpath, im_width, im_height, bandsNum, gdaltype, options=["TILED=YES", "COMPRESS=LZW"])
    dst_ds.SetGeoTransform(gt)
    dst_ds.SetProjection(proj)
    if bandsNum != 1:
        for i in range(bandsNum):
            if nodata != None:
                dst_ds.GetRasterBand(i+1).SetNoDataValue(nodata)
            dst_ds.GetRasterBand(i+1).WriteArray(data[i,:, :])

    else:
        if nodata != None:
            dst_ds.GetRasterBand(1).SetNoDataValue(nodata)
        dst_ds.GetRasterBand(1).WriteArray(data)


def png2tif(png_path,flag_tif_path,save_path,nodata=None):
    img = cv2.imdecode(np.fromfile(png_path, dtype=np.uint8), 0)

    # img= cv2.cvtColor(image, cv2.IMREAD_GRAYSCALE)
    img = np.array(img)

    dataset = gdal.Open(flag_tif_path)
    proj = dataset.GetProjection()
    gt = dataset.GetGeoTransform()

    if nodata==None:
        nodata = dataset.GetRasterBand(1).GetNoDataValue()
    # numpy的数据类型，转换为gdal的数据类型
    gdaltype = 1
    # im_width, im_height = dataset.RasterXSize, dataset.RasterYSize
    im_height, im_width= img.shape
    save_tif_2(img, save_path, im_width, im_height, 1, gdaltype, gt, proj, nodata)


def label_png2tif(png_doc,save_doc,flag_doc):
    if not os.path.exists(save_doc):
        os.makedirs(save_doc)
    id_list = []
    for file in os.listdir(png_doc):
        if os.path.splitext(file)[1] == ".png":
            name = os.path.splitext(file)[0]
            idx = name.split("_")[0]
            if idx not in id_list:
                id_list.append(idx)
    for id in tqdm(id_list):
        png_path = f"{png_doc}\\{id}_pred.png"
        flag_tif_path = f"{flag_doc}\\{id}.tif"
        save_path = f"{save_doc}\\{id}.tif"
        if os.path.exists(save_path):
            continue
        png2tif(png_path,flag_tif_path,save_path,15)


def label_tif2shp(tif_doc,save_shp_doc):
    if not os.path.exists(save_shp_doc):
        os.makedirs(save_shp_doc)
    for file in os.listdir(tif_doc):
        if os.path.splitext(file)[1] == ".tif":
            tif_path = os.path.join(tif_doc,file)
            raster2shp(tif_path,save_shp_doc)


def merge_label_shp(save_shp,shp_doc):
    dic = {0:'公园绿地',1:'户外运动场所',2:'交通场地空间',3:'水体',4:"非城市开放空间"}
    output_shp = shp.Writer(save_shp)
    shp_list = []
    flag = 0
    for shp_f in os.listdir(shp_doc):
        if os.path.splitext(shp_f)[1]==".shp":
            shp_list.append(os.path.join(shp_doc,shp_f))
    id = 1
    for shp_f in tqdm(shp_list):
        r = shp.Reader(shp_f)
        if flag==0:
            for field in r.fields[1:]:
                output_shp.field(*field)
            output_shp.field("OS_id","N",10)
            output_shp.field("type", "C", 100)
            flag = 1
        for shapeRec in r.iterShapeRecords():
            value = shapeRec.record["value"]
            type = dic[value]
            output_shp.record(*shapeRec.record,id,type)
            output_shp.shape(shapeRec.shape)
            id += 1


    proj = osr.SpatialReference()
    proj.ImportFromEPSG(4326)  # 4326-GCS_WGS_1984; 4490- GCS_China_Geodetic_Coordinate_System_2000
    wkt = proj.ExportToWkt()
    coding = "UTF-8"

    f = open(save_shp.replace(".shp", ".prj"), 'w')
    g = open(save_shp.replace(".shp", ".cpg"), 'w')
    g.write(coding)
    f.write(wkt)
    g.close()
    f.close()

def result():
    for country in os.listdir(r"L:\全球开放空间预测结果\预测结果_png"):
        for city in os.listdir(f"L:\\全球开放空间预测结果\\预测结果_png\\{country}"):#
            print(country,"___________",city)
            label_png2tif(f"L:\\全球开放空间预测结果\\预测结果_png\\{country}\\{city}",
                          f"L:\\全球开放空间预测结果\\预测结果_shp\\{country}\\{city}\\OS_tif",
                          f"L:\\开放空间世界image\\{country}\\{city}")
            label_tif2shp(f"L:\\全球开放空间预测结果\\预测结果_shp\\{country}\\{city}\\OS_tif",
                          f"L:\\全球开放空间预测结果\\预测结果_shp\\{country}\\{city}\\OS_shp")
            if os.path.exists(f"L:\\全球开放空间预测结果\\预测结果_shp\\{country}\\{city}\\OS_{city}.shp"):
                continue
            merge_label_shp(f"L:\\全球开放空间预测结果\\预测结果_shp\\{country}\\{city}\\OS_{city}.shp",
                            f"L:\\全球开放空间预测结果\\预测结果_shp\\{country}\\{city}\\OS_shp")


if __name__ == '__main__':
    result()