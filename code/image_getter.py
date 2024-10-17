import ee
import numpy as np
import requests
import io
from utils import mask_s2_clouds, gee_image_to_np_iamge, np_image_to_simple_np_2d_image, np_image_get_bands, np_image_show, mask_viirs_nighttime, np_normalize
from api_geoportal import get_geoportal_lt_map

class ImageGetter:
    def __init__(self):
        ee.Authenticate()
        ee.Initialize()
    
    def get_gee_sentinel_2(self, lon, lat, from_date, to_date, diameter = 500, scale = 10):
        # https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED
        point = ee.Geometry.Point([lon, lat])
        buffer = point.buffer(diameter/2).bounds();
        
        copernicus = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        copernicus = copernicus.filterBounds(buffer)
        copernicus = copernicus.filterDate(from_date, to_date)
        copernicus = copernicus.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 5))
        copernicus = copernicus.map(mask_s2_clouds)
        copernicus = copernicus.mean()

        copernicus_img = copernicus.reproject(crs='EPSG:3346', scale=scale)
        copernicus_img = gee_image_to_np_iamge(copernicus_img, buffer, scale)
        copernicus_img, copernicus_bands = np_image_to_simple_np_2d_image(copernicus_img)
        return copernicus_img, copernicus_bands
    
    def get_gee_viirs(self, lon, lat, from_date, to_date, diameter = 500, scale = 10):
        # https://developers.google.com/earth-engine/datasets/catalog/NOAA_VIIRS_DNB_MONTHLY_V1_VCMSLCFG
        point = ee.Geometry.Point([lon, lat])
        buffer = point.buffer(diameter/2).bounds();
        
        viirs = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG')
        viirs = viirs.filterBounds(buffer)
        viirs = viirs.filterDate(from_date, to_date)
        viirs = viirs.map(mask_viirs_nighttime)
        viirs = viirs.mean()

        viirs_img = viirs.reproject(crs='EPSG:3346', scale=scale)
        viirs_img = gee_image_to_np_iamge(viirs_img, buffer, scale)
        viirs_img, viirs_bands = np_image_to_simple_np_2d_image(viirs_img)
        return viirs_img, viirs_bands
    
    def get_100m_vilnius_pop(self, lon, lat, from_date, to_date, diameter = 500, scale = 10):
        # https://maps.vilnius.lt/teritoriju-planavimas
        # https://gis.vplanas.lt/arcgis/rest/services/Interaktyvus_zemelapis/Teritoriju_planavimas/MapServer/identify
        # projects/ginsed2019/assets/vilnius_pop_den_1ha
        point = ee.Geometry.Point([lon, lat])
        buffer = point.buffer(diameter/2).bounds();
        
        vilnius_pop_den = ee.FeatureCollection("projects/ginsed2019/assets/vilnius_pop_den_1ha");
        vilnius_pop_den = vilnius_pop_den.filterBounds(buffer);
        vilnius_pop_den = vilnius_pop_den.reduceToImage(properties = ['pop_den_1ha'], reducer = ee.Reducer.first());
        
        vilnius_pop_den = vilnius_pop_den.reproject(crs='EPSG:3346', scale=scale)
        vilnius_pop_den = gee_image_to_np_iamge(vilnius_pop_den, buffer, scale)
        vilnius_pop_den, vilnius_pop_den_bands = np_image_to_simple_np_2d_image(vilnius_pop_den)
        return vilnius_pop_den, vilnius_pop_den_bands
    
    def get_1000m_lithuania_pop(self, lon, lat, from_date, to_date, diameter = 500, scale = 10):
        # https://www.geoportal.lt/geoportal/duomenu-paieska#queryText=Gyventoj%C5%B3%20ir%20b%C5%ABst%C5%B3%20sura%C5%A1ymas%202021%20m.%20%E2%80%93%20gyventojai
        # projects/ginsed2019/assets/Gyventou_surasymas_2021_GRID_1km_simple
        point = ee.Geometry.Point([lon, lat])
        buffer = point.buffer(diameter/2).bounds();
        
        popPoints = ee.FeatureCollection("projects/ginsed2019/assets/Gyventou_surasymas_2021_GRID_1km_simple");
        popPoints = popPoints.filterBounds(buffer);
        popPoints = popPoints.reduceToImage(properties = ['POP'], reducer = ee.Reducer.first());
        # Focal mode is not good aproch
        # popPoints = popPoints.focal_mode(radius=1000,kernelType='square',units='meters')
        popPoints = popPoints.reproject(crs='EPSG:3346', scale=scale)
        popPoints = gee_image_to_np_iamge(popPoints, buffer, scale)
        popPoints, popPoints_bands = np_image_to_simple_np_2d_image(popPoints)
        return popPoints, popPoints_bands
    
    def get_geoportal(self, lon, lat, period, diameter, resolution):
        # https://www.geoportal.lt/map/
        res = get_geoportal_lt_map(lat, lon, diameter, resolution, period)
        return res, ["R", "G", "B"]
        
    
if False:
    plt_width = 20
    if True:
        lon = 25.279652
        lat = 54.687157
        from_date = '2021-07-01'
        to_date = '2021-09-29'
        diameter = 20000
        scale = 132.29
        name = "Vilnius"
    if False:
        lon = 21.4645
        lat = 55.4642
        from_date = '2021-07-01'
        to_date = '2021-09-29'
        diameter = 500
        scale = 2.64
    resol_text = f"\nResolution: {scale} $m^2$ per pixel"
    perio_text = f"\nPeriod: from {from_date} to {to_date}"
    
    ig = ImageGetter()
    sentinel_2_img, sentinel_2_bands = ig.get_gee_sentinel_2(lon, lat, from_date, to_date, diameter, scale)
    viirs_img, viirs_bands = ig.get_gee_viirs(lon, lat, from_date, to_date, diameter, scale)
    vilnius_pop_den, vilnius_pop_den_bands = ig.get_100m_vilnius_pop(lon, lat, from_date, to_date, diameter, scale)
    lithunia_pop, lithuania_pop_bands = ig.get_1000m_lithuania_pop(lon, lat, from_date, to_date, diameter, scale)
    
    np_image_show(np.log(np_image_get_bands(sentinel_2_img, sentinel_2_bands, ['B4', 'B3', 'B2']) + 1), title = f"Log-transformed RGB image of {name}{resol_text}{perio_text}", width_cm=plt_width, name = "log_rgb_vilnius_132_2021")
    np_image_show(np.log(np_image_get_bands(viirs_img, viirs_bands, ['avg_rad']) + 1), title = f"Log-transformed average radiance image of {name}{resol_text}{perio_text}", width_cm=plt_width, name = "log_radiance_vilnius_132_2021")
    np_image_show(np.log(np_image_get_bands(vilnius_pop_den, vilnius_pop_den_bands, ['first']) + 1), title = f"Log-transformed population dencity per 100 $m^2$ image of {name}{resol_text}\nPeriod: Unknown", width_cm=plt_width, name = "log_pop_den_vilnius_132_2021")
    np_image_show(np.log(np_image_get_bands(lithunia_pop, lithuania_pop_bands, ['first']) + 1), title = f"Log-transformed population count per 1000 $m^2$ image of {name}{resol_text}\nPeriod: 2021", width_cm=plt_width, name = "log_pop_count_vilnius_132_2021")
        
    # True color
    np_image_show(np_image_get_bands(sentinel_2_img, sentinel_2_bands, ['B4', 'B3', 'B2']), title = f"True color image of {name}{resol_text}{perio_text}", width_cm=plt_width, name = "rgb_vilnius_132_2021")
    # False color (urban)
    np_image_show(np_image_get_bands(sentinel_2_img, sentinel_2_bands, ['B12', 'B11', 'B4']), title = f"False color (urban) image of {name}{resol_text}{perio_text}", width_cm=plt_width, name = "false_col_urban_vilnius_132_2021")
    # NDVI
    tmp = np_image_get_bands(sentinel_2_img, sentinel_2_bands, ['B8', 'B4'])
    np_image_show((tmp[:,:,0] - tmp[:,:,1]) / (tmp[:,:,0] + tmp[:,:,1]), title = f"NDVI image of {name}{resol_text}{perio_text}", width_cm=plt_width, name = "ndvi_vilnius_132_2021")
    # Moisture index
    tmp = np_image_get_bands(sentinel_2_img, sentinel_2_bands, ['B8', 'B11'])
    np_image_show((tmp[:,:,0] - tmp[:,:,1]) / (tmp[:,:,0] + tmp[:,:,1]), title = f"Moisture index image of {name}{resol_text}{perio_text}", width_cm=plt_width, name = "moisture_vilnius_132_2021")
    # SWIR
    np_image_show(np_image_get_bands(sentinel_2_img, sentinel_2_bands, ['B12', 'B8A', 'B4']), title = f"SWIR image of {name}{resol_text}{perio_text}", width_cm=plt_width, name = "swir_vilnius_132_2021")
    # NDWI
    tmp = np_image_get_bands(sentinel_2_img, sentinel_2_bands, ['B3', 'B8'])
    np_image_show((tmp[:,:,0] - tmp[:,:,1]) / (tmp[:,:,0] + tmp[:,:,1]), title = f"NDWI image of {name}{resol_text}{perio_text}", width_cm=plt_width, name = "ndwi_vilnius_132_2021")
    # NDSI
    tmp = np_image_get_bands(sentinel_2_img, sentinel_2_bands, ['B3', 'B11'])
    np_image_show((tmp[:,:,0] - tmp[:,:,1]) / (tmp[:,:,0] + tmp[:,:,1]), title = f"NDSI image of {name}{resol_text}{perio_text}", width_cm=plt_width, name = "ndsi_vilnius_132_2021")
    
    geoportal_img, geoportal_img_bands = ig.get_geoportal(lon, lat, '2021-2023', diameter, f'{scale}m')
    np_image_show(np_image_get_bands(geoportal_img, geoportal_img_bands, ['R', 'G', 'B']), title = f"RGB image of {name} from geoportal{resol_text}\nPeriod: from 2021 to 2023", width_cm=plt_width, name = "geoportal_vilnius_132_2021")
    