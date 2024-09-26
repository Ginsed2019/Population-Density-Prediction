import requests
from bs4 import BeautifulSoup
import json
import re
import pandas as pd

# Data from https://www.facebook.com/legacy/notes/642765649176985/

def list_of_dicts_to_csv(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

def unescape(s):
    s = s.replace('\\"', '"')
    s = s.replace('\\\\', '\\')
    s = s.replace('\\n', ' ')
    s = s.replace('\\r', '')
    return s

def niekonaujo_location2dict(loc, dict = {}):
    global zzz
    zzz = loc
    res = {
        **dict,
        'name': loc[5][0][1][0],
        'desc': loc[5][1][1][0] if ((len(loc[5]) >= 2) and (loc[5][1] is not None)) else None,
        'lat': loc[1][0][0][0] if (loc[1] is not None) else loc[2][0][0][0][0][0],
        'lon': loc[1][0][0][1] if (loc[1] is not None) else loc[2][0][0][0][0][1]
    }
    return res

def niekonaujo_country2dict_array(country, dict = {}):
    dict = {**dict, 'subsource': country[2]}
    locations = [niekonaujo_location2dict(loc, dict) for loc in country[12][0][13][0]]
    return locations

def niekonaujo2dict_array(json, source):
    countries = json[1][6]
    dict = {'source': source}
    res = [loc for country in countries for loc in niekonaujo_country2dict_array(country, dict)]
    return res

def niekonaujo_url2json(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    script_tag = soup.find('script', string=re.compile('var _pageData'))
    script_content = script_tag.string
    page_data = re.search(r'var \_pageData = "(.+?)(?<!\\)";', script_content).group(1)
    page_data = unescape(page_data)
    page_data = json.loads(page_data)
    return page_data

def get_niekonaujo_as_pd():
    url = "https://www.google.com/maps/d/u/0/viewer?mid=1Az6PkPxpnUrXqQ83ncNQoSZa-L8&femb=1&ll=49.210260177154%2C33.579373950000004&z=4"
    json_data = niekonaujo_url2json(url)
    locs = niekonaujo2dict_array(json_data, "niekonaujo")
    df = pd.DataFrame(locs)
    return df

def get_fb1_as_pd():
    url = "https://www.google.com/maps/d/u/0/viewer?mid=1DMNV0VkWoFpwcVnkXqq_7V9lRpA&ll=55.54977308818169%2C24.37769785000001&z=7"
    json_data = niekonaujo_url2json(url)
    locs = niekonaujo2dict_array(json_data, "fb1")
    df = pd.DataFrame(locs)
    return df

def get_dvarai_as_pd():
    url = "https://www.google.com/maps/d/u/0/viewer?mid=1afR3M4lZZ5sN7Ec1dLWHegPbNr8&ll=54.88931537260346%2C24.868004851562517&z=6"
    json_data = niekonaujo_url2json(url)
    locs = niekonaujo2dict_array(json_data, "dvarai")
    df = pd.DataFrame(locs)
    return df

def get_dvarai2_as_pd():
    url = "https://www.google.com/maps/d/u/0/viewer?mid=1RlEBzmmFiJ4otBtYCyZlpD9MjDY&ll=55.310687910343795%2C24.112769000000007&z=8"
    json_data = niekonaujo_url2json(url)
    locs = niekonaujo2dict_array(json_data, "dvarai2")
    df = pd.DataFrame(locs)
    return df

def get_vaiduokliai_as_pd():
    url = "https://www.google.com/maps/d/u/0/viewer?mid=1XcX902WI2qr5hMlK3WrnRMGbot0&ll=54.98112933981097%2C23.232513949999998&z=7"
    json_data = niekonaujo_url2json(url)
    locs = niekonaujo2dict_array(json_data, "vaiduokliai")
    df = pd.DataFrame(locs)
    return df

def get_truristo_as_pd():
    url = "https://www.google.com/maps/d/u/0/viewer?hl=lt&mid=1vASPNoEr2e_tSdsp0tmc1noJOMI&ll=55.39936060694531%2C22.756247070221&z=7"
    json_data = niekonaujo_url2json(url)
    locs = niekonaujo2dict_array(json_data, "truristo")
    df = pd.DataFrame(locs)
    return df

def get_moltovolinija_as_pd():
    url = "https://www.google.com/maps/d/u/0/viewer?msa=0&hl=lt&ie=UTF8&ll=54.95971420174872%2C22.390059000000004&spn=1.973305%2C2.673042&t=h&source=embed&mid=1q7QBoVtlB844rF4k_2LS93kc_Hs&z=8"
    json_data = niekonaujo_url2json(url)
    locs = niekonaujo2dict_array(json_data, "moltovolinija")
    df = pd.DataFrame(locs)
    return df

if (False):
    df = pd.concat([
        get_niekonaujo_as_pd(),
        get_fb1_as_pd(),
        get_dvarai_as_pd(),
        get_dvarai2_as_pd(),
        get_vaiduokliai_as_pd(),
        get_truristo_as_pd(),
        get_moltovolinija_as_pd()
    ], ignore_index=True)
    df.to_csv("abandoned_building_locations.csv", index=False)
