# -*- encoding:utf-8 -*-
import arcpy
import os


join_feature = u"J:\\全球开放空间预测结果\\运动场预测结果处理\\pre_sports_correct\\sport_points_ESA_split\\sports_points_ESA_value.shp"
for country in os.listdir(u"L:\\ESSD验证数据\\样本点选择\\户外运动场所\\pre_sports_shp"):
    for city in os.listdir(u"L:\\ESSD验证数据\\样本点选择\\户外运动场所\\pre_sports_shp\\"+country):
        print(country,city)
        sports_path = u"L:\\ESSD验证数据\\样本点选择\\户外运动场所\\pre_sports_shp\\"+country + u"\\" + city + u"\\" +city +u"_pre_area.shp"
        save_doc = save_path = u"J:\\全球开放空间预测结果\\运动场预测结果处理\\pre_sports_correct\\sport_aoi_ESA_value\\"+country + u"\\" + city
        if not os.path.exists(save_doc):
            os.makedirs(save_doc)
        save_path = save_doc+ u"\\" +city +u"_pre_ESA.shp"
        arcpy.analysis.SpatialJoin(sports_path , join_feature, save_path,join_operation=u"JOIN_ONE_TO_ONE",join_type=u"KEEP_COMMON",match_option=u"INTERSECT")