import numpy as np
import requests
from PIL import Image
import io
import math
from pyproj import Transformer

class EPSG4326_TO_EPSG3346:
  def __init__(self):
    self.transformer = Transformer.from_crs("EPSG:4326", "EPSG:3346", always_xy=True)

  def transform(self, lat, lon):
    """
    Transforms coordinates from EPSG:4326 to EPSG:3346.

    EPSG:3346 is a projected coordinate reference system used in Lithuania.
    EPSG:4326 is a geographic coordinate reference system used in the world.
    EPSG:4326 uses degrees as units, while EPSG:3346 uses meters as units.

    Args:
      lat: Latitude.
      lon: Longitude.

    Returns:
      Transformed coordinates.
    """
    return self.transformer.transform(lon, lat)

transformer = EPSG4326_TO_EPSG3346()

def standardize_rgb(rgb_matrix):
    # Ensure the input is float type for calculations
    rgb_matrix = rgb_matrix.astype(float)

    # Standardize each channel separately
    for channel in range(3):  # 0 for R, 1 for G, 2 for B
        channel_values = rgb_matrix[:,:,channel]
        mean = np.mean(channel_values)
        std = np.std(channel_values)
        rgb_matrix[:,:,channel] = (channel_values - mean) / std
        
    return (rgb_matrix + 1) / 2
        
def meaters_to_pixels(meaters, resolution):
    pixels = math.ceil(meaters / resolution)
    return pixels

def pixels_to_meaters(pixels, resolution):
    meaters = pixels * resolution
    return meaters

def devide_into_parts(x, max_x, offset_x):
    complete_parts = int(x // max_x)
    remainder = x % max_x
    res = np.full(complete_parts, max_x)
    if remainder > 0:
        res = np.append(res, remainder)
    csum = np.insert(np.cumsum(res)[:-1], 0, 0) + offset_x
    res = list(zip(csum, res))
    return res

def devide_into_parts_xy(x, y, width, height, max_width, max_height):
    x_parts = devide_into_parts(width, max_width, x)
    y_parts = devide_into_parts(height, max_height, y)
    res = [[{'x': x_part[0], 'y': y_part[0], 'width': x_part[1], 'height': y_part[1]} for x_part in x_parts] for y_part in y_parts]
    return res, (len(x_parts), len(y_parts))
        
def get_params(x, y, width, height, resolution, max_px_width, max_px_height, scale):
    parts, dims = devide_into_parts_xy(x, y, width, height, pixels_to_meaters(max_px_width, resolution), pixels_to_meaters(max_px_height, resolution))
    return [[{
        'bbox': f'{part["x"]},{part["y"]},{part["x"]+part["width"]},{part["y"]+part["height"]}',
        'format': 'png',
        'transparent': 'false',
        'f': 'image',
        'mapScale': f'{scale}',
        'size': f'{meaters_to_pixels(part["width"], resolution)},{meaters_to_pixels(part["height"], resolution)}',
    } for part in x_parts] for x_parts in parts], dims, (meaters_to_pixels(width, resolution), meaters_to_pixels(height, resolution))


class GeoportalAPI:
  def __init__(self):
      self.period = {
          '1995-1999': 'https://www.geoportal.lt/arcgis/rest/services/NZT/ORT10LT_1995_2001/MapServer',
          '2005-2006': 'https://www.geoportal.lt/arcgis/rest/services/NZT/ORT10LT_2005_2006/MapServer',
          '2009-2010': 'https://www.geoportal.lt/arcgis/rest/services/NZT/ORT10LT_2009_2010/MapServer',
          '2012-2013': 'https://www.geoportal.lt/arcgis/rest/services/NZT/ORT10LT_2012_2013/MapServer',
          '2015-2017': 'https://www.geoportal.lt/arcgis/rest/services/NZT/ORT10LT_2015/MapServer',
          '2018-2020': 'https://www.geoportal.lt/arcgis/rest/services/NZT/ORT10LT_2018_2020/MapServer',
          '2021-2023': 'https://www.geoportal.lt/arcgis/rest/services/NZT/ORT10LT_2021_2023/MapServer'
      }
      self.period_info = {}
      self.period_lods_info = {}
      self.generator_log = {0: 0}

  def get_period_names(self):
      return list(self.period.keys())

  def get_period_info(self, period):
      if period not in self.period_info:
          response = requests.get(f"{self.period[period]}?f=json")
          self.period_info[period] = response.json()
      return self.period_info[period]

  def get_period_lods_info(self, period):
      if period not in self.period_lods_info:
          self.period_lods_info[period] = {lod['level']: lod for lod in self.get_period_info(period)['tileInfo']['lods']}
      return self.period_lods_info[period]

  def __get_map_from_bottom_left_corner(self, period, params, rgb_standardized):
      """
      Get map from geoportal.lt

      :param x: x coordinate of bottom left corner in EPSG3346
      :param y: y coordinate of bottom left corner in EPSG3346
      :param width: width of map in meters
      :param height: height of map in meters
      :param period: period of map
      :param lod: level of detail of map

      :return: map as numpy array
      """
      response = requests.get(f"{self.period[period]}/export", params=params)
      map_as_png = response.content
      img = Image.open(io.BytesIO(map_as_png))
      img = img.convert('RGB')
      map_as_matrix = np.array(img)
      if rgb_standardized:
        map_as_matrix = standardize_rgb(map_as_matrix)
      return map_as_matrix

  def __print_progress(self, current, total, predicted_size, real_size):
      progress = f"{current:5d}/{total:5d}"
      pred_size = f"predicted size: {predicted_size:10s}"
      real_sz = f"real size: {real_size:10s}"
      progress_line = f"{progress} | {pred_size} | {real_sz}"
      #print(progress_line)

  def __get_map_from_bottom_left_corner_generator_x(self, period, params, gen_key, dims, rgb_standardized):
      for param in params:
          maps = []
          for p in period:
              map = self.__get_map_from_bottom_left_corner(p, param, rgb_standardized)
              self.generator_log[gen_key] += 1
              self.__print_progress(self.generator_log[gen_key], dims[0] * dims[1] * len(period), param['size'], str(map.shape))
              maps.append(map)
          map = np.stack(maps, axis = 2)
          yield map

  def __get_map_from_bottom_left_corner_generator_y(self, period, params, dims, rgb_standardized):
      key = max(self.generator_log.keys()) + 1
      self.generator_log[key] = 0
      self.__print_progress(self.generator_log[key], dims[0] * dims[1] * len(period), "-", "-")
      for x_param in params:
          yield self.__get_map_from_bottom_left_corner_generator_x(period, x_param, key, dims, rgb_standardized)

  def get_map_from_bottom_left_corner_generator(self, x, y, width, height, period, lod, rgb_standardized = True):
      period_info = self.get_period_info(period[0]) # FIXME: look at all pertiods
      period_lod_info = self.get_period_lods_info(period[0])[lod]
      params, dims, dims_pixel = get_params(x, y, width, height, period_lod_info['resolution'], period_info['maxImageWidth'], period_info['maxImageHeight'], period_lod_info['scale'])
      return self.__get_map_from_bottom_left_corner_generator_y(period, params[::-1], dims, rgb_standardized), dims, dims_pixel

  def get_map_from_bottom_left_corner(self, x, y, width, height, period, lod, rgb_standardized = True):
      maps, dims, dims_pixel = self.get_map_from_bottom_left_corner_generator(x, y, width, height, period, lod, rgb_standardized)
      map = np.concatenate([np.concatenate(list(x_s), axis = 1) for x_s in maps], axis = 0)
      return map, dims_pixel

  def get_map_from_center_generator(self, x, y, width, height, period, lod, rgb_standardized = True):
      x = x - width / 2
      y = y - height / 2
      return self.get_map_from_bottom_left_corner_generator(x, y, width, height, period, lod, rgb_standardized)

  def get_map_from_center(self, x, y, width, height, period, lod, rgb_standardized = True):
      x = x - width / 2
      y = y - height / 2
      return self.get_map_from_bottom_left_corner(x, y, width, height, period, lod, rgb_standardized)

api = GeoportalAPI()
transformer = EPSG4326_TO_EPSG3346()
resolutions_dict = {
    '529.16m': 1,
    '264.58m': 2,
    '132.29m': 3,
    '66.14m': 4,
    '26.45m': 5,
    '13.22m': 6,
    '6.61m': 7,
    '2.64m': 8,
    '1.32m': 9,
    '0.52m': 10,
    '0.23m': 11,
    '0.13m': 12
}

def get_geoportal_lt_map(lat, lon, diameter, resolution, period):
    """
    Retrieves a map from the Lithuanian Geoportal based on given parameters.

    This function fetches a map image centered on the provided latitude and longitude,
    with a specified diameter, resolution, and time period.

    Parameters:
    lat (float): Latitude of the center point.
    lon (float): Longitude of the center point.
    diameter (float): Diameter of the area to be mapped, in meters.
    resolution (str): Resolution of the map image. Must be one of the valid resolutions.
    period (str): Time period of the map data. Must be one of the valid periods.

    Returns:
    numpy.ndarray: A 3D array representing the map image.

    List of valid periods:
        '2021-2023';
        '2018-2020';
        '2015-2017';
        '2012-2013';
        '2009-2010';
        '2005-2006';
        '1995-1999'.

    List of valid resolutions:
        '529.16m';
        '264.58m';
        '132.29m';
         '66.14m';
         '26.45m';
         '13.22m';
          '6.61m';
          '2.64m';
          '1.32m';
          '0.52m';
          '0.23m';
          '0.13m'.

    Example:
    >>> map_image = get_geoportal_lt_map(54.6872, 25.2797, 1000, '26.45m', '2021-2023')
    """
    x, y = transformer.transform(lat, lon)
    map = api.get_map_from_center(x, y, diameter, diameter, [period], resolutions_dict[resolution], False)
    map = map[0][:,:,0,:]
    return map


if False:
    lat = 55.4642
    lon = 21.4645
    diameter = 100
    period = '2021-2023'
    resolution = '0.13m'
    map = get_geoportal_lt_map(lat, lon, diameter, resolution, period)
