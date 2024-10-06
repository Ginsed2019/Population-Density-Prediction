from api_geoportal import get_geoportal_lt_map
import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import base64
import io
from PIL import Image
import re
import requests
import numpy as np
from config import GOOGLE_MAPS_API_KEY
from io import BytesIO

def numpy_to_b64(arr):
    im = Image.fromarray(arr)
    buffer = io.BytesIO()
    im.save(buffer, format="PNG")
    encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{encoded_image}"

def make_url_markdown(text):
    url_pattern = r'(https?://\S+)'
    def replace_url(match):
        url = match.group(1)
        return f'[{url}]({url})'
    return re.sub(url_pattern, replace_url, text)

def make_urls_clickable(text):
    return dcc.Markdown(text, link_target='_blank')

def get_static_map(lat, lon, zoom=6, size="400x400"):
    base_url = "https://maps.googleapis.com/maps/api/staticmap"
    c_lat = 55.19
    c_lon = 23.54
    params = {
        "center": f"{c_lat},{c_lon}",
        "zoom": zoom,
        "size": size,
        "markers": f"color:red|{lat},{lon}",
        "key": GOOGLE_MAPS_API_KEY  # Replace with your actual API key
    }
    response = requests.get(base_url, params=params)
    img = Image.open(io.BytesIO(response.content))
    return np.array(img)

def get_street_view_image(lat, lon, api_key, heading):
    base_url = "https://maps.googleapis.com/maps/api/streetview"
    params = {
        "size": "600x400",
        "location": f"{lat},{lon}",
        "heading": heading,
        "key": api_key
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        img = Image.open(BytesIO(response.content)).convert('RGB')
        return np.array(img)
    return None

def get_nearby_landmarks(lat, lon, api_key):
    base_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "location": f"{lat},{lon}",
        "radius": 1000,  # Search within 1km
        "type": "tourist_attraction",
        "key": api_key
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        return response.json().get('results', [])
    return []

def get_place_photos(place_id, api_key, max_photos=2):
    base_url = "https://maps.googleapis.com/maps/api/place/details/json"
    params = {
        "place_id": place_id,
        "fields": "photo",
        "key": api_key
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        photos = response.json().get('result', {}).get('photos', [])
        return [photo['photo_reference'] for photo in photos[:max_photos]]
    return []

def get_place_photo(photo_reference, api_key):
    base_url = "https://maps.googleapis.com/maps/api/place/photo"
    params = {
        "maxwidth": 600,
        "photo_reference": photo_reference,
        "key": api_key
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        img = Image.open(BytesIO(response.content)).convert('RGB')
        return np.array(img)
    return None

def get_location_images(lat, lon, api_key, total_images=4):
    images = []
    
    # Get landmark images
    landmarks = get_nearby_landmarks(lat, lon, api_key)
    if landmarks:
        closest_landmark = landmarks[0]
        photo_references = get_place_photos(closest_landmark['place_id'], api_key, max_photos=2)
        for photo_ref in photo_references:
            img = get_place_photo(photo_ref, api_key)
            if img is not None:
                images.append(img)
    
    # Fill the rest with Street View images
    street_view_count = total_images - len(images)
    for i in range(street_view_count):
        heading = i * (360 / street_view_count)
        img = get_street_view_image(lat, lon, api_key, heading)
        if img is not None:
            images.append(img)
    
    return images

server_storage = {}

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], assets_folder='assets')

app.config.suppress_callback_exceptions = True

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H4("Menu"),
            dcc.Upload(
                id='upload-data',
                children=html.Div(['Drag and Drop or ', html.A('Select Files')])
            ),
            html.Div(id='file-menu'),
            html.Div(id='file-row-menu'),
            dcc.Dropdown(id='source-filter', multi=True, placeholder="Select source(s)", value=["niekonaujo"]),
            dbc.Button("Previous", id="prev-button"),
            dbc.Button("Next", id="next-button")
        ], width=3),
        dbc.Col([
            html.Div(id='image-display')
        ], width=9)
    ])
])

@callback(
    Output('file-menu', 'children'),
    Output('source-filter', 'options'),
    Input('upload-data', 'filename')
)
def update_file_menu(filename):
    if filename is None: filename = '../data/abandoned_building_locations.csv'
    server_storage['locations_row'] = 0
    df = pd.read_csv(filename, encoding='utf8')
    # df = df[df['source'] != "truristo"]
    # df = df[df['source'] != "dvarai2"]
    # df = df[df['source'] != "dvarai"]
    # df = df[df['source'] != "moltovolinija"]
    # df = df[df['source'] == "niekonaujo"]
    df = df.sample(frac=1).reset_index(drop=True)
    server_storage['locations'] = df
    sources = df['source'].unique().tolist()
    return html.Div([
        html.P(f"Selected: '{filename}'")
    ]), sources

@callback(
    Output('file-row-menu', 'children'),
    Output('image-display', 'children'),
    Input('prev-button', 'n_clicks'),
    Input('next-button', 'n_clicks'),
    Input('source-filter', 'value'),
    State('file-row-menu', 'children')
)
def update_file_row_menu(prev_clicks, next_clicks, selected_sources, current_children):
    if not ("locations" in server_storage): return html.Div(), html.Div()
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
    if selected_sources:
        filtered_df = server_storage['locations'][server_storage['locations']['source'].isin(selected_sources)]
    else:
        filtered_df = server_storage['locations']
        
    if 'filtered_row' not in server_storage:
        server_storage['filtered_row'] = 0

    if button_id == 'prev-button' and prev_clicks is not None:
        server_storage['filtered_row'] = server_storage['filtered_row'] - 1
        if (server_storage['filtered_row'] < 0): server_storage['filtered_row'] = len(filtered_df) - 1
    elif button_id == 'next-button' and next_clicks is not None:
        server_storage['filtered_row'] = server_storage['filtered_row'] + 1
        if (server_storage['filtered_row'] > len(filtered_df) - 1): server_storage['filtered_row'] = 0
        
    server_storage['filtered_row'] = min(server_storage['filtered_row'], len(filtered_df) - 1)
    
    current_row = filtered_df.iloc[server_storage['filtered_row']]
    
    file_row_menu = html.Div([
        html.P(f"Row {server_storage['filtered_row'] + 1} of {len(filtered_df)}"),
        html.Table([
            html.Tr([html.Th("Source:"), html.Td(f"{current_row['source']}")]),
            html.Tr([html.Th("Lat, Lon:"), html.Td(f"{current_row['lat']}, {current_row['lon']}")]),
            html.Tr([html.Th("Name:"), html.Td(f"{current_row['name']}")]),
            html.Tr([html.Th("Description:"), html.Td(make_urls_clickable(current_row['desc']))]),
            html.Tr([html.Th("Google maps:"), html.Td(make_urls_clickable(f"http://maps.google.com/maps?q={current_row['lat']},{current_row['lon']}"))])
        ])
    ])
    
    image_array = get_geoportal_lt_map(current_row['lat'], current_row['lon'], 100, '0.13m', '2021-2023')
    image_b64 = numpy_to_b64(image_array)
    
    map_image_array = get_static_map(current_row['lat'], current_row['lon'])
    map_image_b64 = numpy_to_b64(map_image_array)
    
    image_arrays = get_location_images(current_row['lat'], current_row['lon'], GOOGLE_MAPS_API_KEY)
    
    image_display = html.Div([
        dbc.Row([
            dbc.Col([
                html.Img(src=image_b64, style={'width': '100%', 'height': 'auto', 'object-fit': 'contain'})
            ], width=8),
            dbc.Col([
                html.Img(src=map_image_b64, style={'width': '100%', 'height': 'auto', 'object-fit': 'contain', 'margin-bottom': '10px'}),
                html.Img(src=numpy_to_b64(image_arrays[0]), style={'width': '100%', 'height': 'auto', 'object-fit': 'contain'})
            ], width=4)
        ]),
        dbc.Row([
            dbc.Col([
                html.Img(src=numpy_to_b64(image_arrays[1]), style={'width': '100%', 'height': 'auto', 'object-fit': 'contain'})
            ], width=4),
            dbc.Col([
                html.Img(src=numpy_to_b64(image_arrays[2]), style={'width': '100%', 'height': 'auto', 'object-fit': 'contain'})
            ], width=4),
            dbc.Col([
                html.Img(src=numpy_to_b64(image_arrays[3]), style={'width': '100%', 'height': 'auto', 'object-fit': 'contain'})
            ], width=4)
        ])
    ])
    
    return file_row_menu, image_display


# app.layout = dbc.Container([
#     dbc.Row([
#         dbc.Col([
#             html.H4("Menu"),
#             dcc.Upload(
#                 id='upload-data',
#                 children=html.Div([
#                     'Drag and Drop or ',
#                     html.A('Select Files')
#                 ]),
#                 style={
#                     'width': '100%',
#                     'height': '60px',
#                     'lineHeight': '60px',
#                     'borderWidth': '1px',
#                     'borderStyle': 'dashed',
#                     'borderRadius': '5px',
#                     'textAlign': 'center',
#                     'margin': '10px'
#                 },
#             ),
#             html.Div(id='file-info'),
#             dbc.Button("Previous", id="prev-button", n_clicks=0),
#             dbc.Button("Next", id="next-button", n_clicks=0),
#         ], width=3),
#         dbc.Col([
#             html.H4("Data View"),
#             html.Div(id='data-view')
#         ], width=9)
#     ])
# ], fluid=True)

# def parse_contents(contents, filename):
#     content_type, content_string = contents.split(',')
#     decoded = base64.b64decode(content_string)
#     try:
#         if 'csv' in filename:
#             df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
#         elif 'xls' in filename:
#             df = pd.read_excel(io.BytesIO(decoded))
#         else:
#             return html.Div([
#                 'Unsupported file type.'
#             ])
#     except Exception as e:
#         print(e)
#         return html.Div([
#             'There was an error processing this file.'
#         ])

#     return df, html.Div([
#         html.H5(filename),
#         html.H6(f"{len(df)} rows"),
#         html.H6(f"{len(df.columns)} columns")
#     ])

# @app.callback(
#     Output('file-info', 'children'),
#     Output('data-view', 'children'),
#     Input('upload-data', 'contents'),
#     State('upload-data', 'filename'),
#     prevent_initial_call=True
# )
# def update_output(contents, filename):
#     if contents is None:
#         return None, None
#     df, file_info = parse_contents(contents, filename)
#     return file_info, None

# current_row = 0

# @app.callback(
#     Output('data-view', 'children'),
#     Input('prev-button', 'n_clicks'),
#     Input('next-button', 'n_clicks'),
#     State('upload-data', 'contents'),
#     State('upload-data', 'filename'),
#     prevent_initial_call=True
# )
# def update_row(prev_clicks, next_clicks, contents, filename):
#     global current_row
#     if contents is None:
#         return None
    
#     df, _ = parse_contents(contents, filename)
    
#     ctx = dash.callback_context
#     if not ctx.triggered:
#         button_id = 'No clicks yet'
#     else:
#         button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
#     if button_id == 'next-button' and current_row < len(df) - 1:
#         current_row += 1
#     elif button_id == 'prev-button' and current_row > 0:
#         current_row -= 1
    
#     row_data = df.iloc[current_row].to_dict()
#     return html.Div([
#         html.H5(f"Row {current_row + 1}"),
#         html.Pre(str(row_data))
#     ])

if __name__ == '__main__':
    app.run_server(debug=True, port=8061)