import requests
import numpy as np
import io
import matplotlib.pyplot as plt
import pandas
import inflection

def mask_s2_clouds(image):
  """Masks clouds in a Sentinel-2 image using the QA band.

  Args:
      image (ee.Image): A Sentinel-2 image.

  Returns:
      ee.Image: A cloud-masked Sentinel-2 image.
  """
  qa = image.select('QA60')

  # Bits 10 and 11 are clouds and cirrus, respectively.
  cloud_bit_mask = 1 << 10
  cirrus_bit_mask = 1 << 11

  # Both flags should be set to zero, indicating clear conditions.
  mask = (
      qa.bitwiseAnd(cloud_bit_mask)
      .eq(0)
      .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
  )

  return image.updateMask(mask)

def mask_viirs_nighttime(image):
    """
    Masks low-quality pixels in a VIIRS Nighttime image using the cf_cvg band.
    
    Args:
        image (ee.Image): A VIIRS Nighttime image.
    
    Returns:
        ee.Image: A masked VIIRS Nighttime image.
    """
    # Select the cloud-free coverage band
    cf_cvg = image.select('cf_cvg')
    
    # Set a threshold for the minimum number of observations
    # You may need to adjust this threshold based on your specific requirements
    min_observations = 3
    
    # Create a mask where cf_cvg is greater than or equal to the threshold
    mask = cf_cvg.gte(min_observations)
    
    # Apply the mask to the image
    return image.updateMask(mask)

def gee_image_to_np_iamge(gee_image, buffer, scale):
    url = gee_image.getDownloadURL({
        'scale': scale,
        'region': buffer.getInfo(),
        'format': 'NPY'
    })
    response = requests.get(url)
    gee_arr = np.load(io.BytesIO(response.content), allow_pickle=True)
    return gee_arr

def np_image_to_simple_np_2d_image(np_image):
    bands = np_image.dtype.names
    shape = (np_image.shape[0], np_image.shape[1], len(bands))
    arr3d = np.zeros(shape, dtype=np.float32)
    for b, band in enumerate(bands):
        arr3d[:, :, b] = np_image[band]
    return arr3d, bands

def np_image_get_bands(np_image, bands, select):
    return np.take(np_image, np.where(np.isin(bands, select))[0], axis=-1)

def np_normalize(np_image):
    np_image = np_image - np_image.min()
    np_image = np_image / np_image.max()
    return np_image

def np_image_show(np_image, normalize = True, title = None, width_cm = 8, name = None):
    if normalize: np_image = np_normalize(np_image)
    aspect_ratio = np_image.shape[0] / np_image.shape[1]
    if width_cm:
        # Convert cm to inches (1 inch = 2.54 cm)
        width_inches = width_cm / 2.54
        height_inches = width_inches * aspect_ratio
        plt.figure(figsize=(width_inches, height_inches))
    else:
        plt.figure()
    plt.imshow(np_image)
    plt.title(title if title else f'Image of size: {np_image.shape}')
    plt.axis('off')
    
    if name:
        None
        #plt.imsave(f"../plots/raw_{name}.png", np_image)
        #plt.savefig(f"../plots/plt_{name}.png")
        
    plt.show()
    return None

if False:
    gyv = pandas.read_csv('../data/Gyventou_surasymas_2021_(GRID_1km).csv')
    gyv[['X', 'Y']] = gyv['GRID_ID'].str.extract(r'1x1kmX(\d*)Y(\d*)').astype(int)
    gyv = gyv[['X', 'Y', 'POP']]
    gyv[['X']] = gyv[['X']] * 10
    gyv[['Y']] = gyv[['Y']] * 100    
    gyv.to_csv('../data/Gyventou_surasymas_2021_GRID_1km_simple.csv', index=False)
    

