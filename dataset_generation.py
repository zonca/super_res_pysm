#NOTE: To avoid automatic execution set the 'execute' to False

execute = False

import numpy as np
import pysm3
from pysm3 import units as u
import healpy as hp
import matplotlib.pyplot as plt

import pixell

from PIL import Image

#PySM simulation
def pysm_realiz(nside,seed):
    dust = pysm3.ModifiedBlackBodyRealization(
        nside=nside,
        amplitude_modulation_temp_alm="dust_gnilc/raw/gnilc_dust_temperature_modulation_alms_lmax768.fits.gz",
        amplitude_modulation_pol_alm="dust_gnilc/raw/gnilc_dust_polarization_modulation_alms_lmax768.fits.gz",
        largescale_alm="dust_gnilc/raw/gnilc_dust_largescale_template_logpoltens_alm_nside2048_lmax1024_complex64.fits.gz",
        small_scale_cl="dust_gnilc/raw/gnilc_dust_small_scales_logpoltens_cl_lmax16384.fits.gz",
        largescale_alm_mbb_index="dust_gnilc/raw/gnilc_dust_largescale_template_beta_alm_nside2048_lmax1024.fits.gz",
        small_scale_cl_mbb_index="dust_gnilc/raw/gnilc_dust_small_scales_beta_cl_lmax16384_2023.06.06.fits.gz",
        largescale_alm_mbb_temperature="dust_gnilc/raw/gnilc_dust_largescale_template_Td_alm_nside2048_lmax1024.fits.gz",
        small_scale_cl_mbb_temperature="dust_gnilc/raw/gnilc_dust_small_scales_Td_cl_lmax16384_2023.06.06.fits.gz",
        freq_ref="353 GHz",
        seeds=[seed, seed + 1000, seed + 2000],
        max_nside=8192,
)
    sky = pysm3.Sky(nside=nside, component_objects=[dust])
    freq = 353 * u.GHz
    m = sky.get_emission(freq)
    return m

#transfer the sky map to 2D
def sky_to_2D(healpix_reso,m):
    car_reso = (np.pi / np.round(np.pi / healpix_reso.value)) * u.radian
    #print(healpix_reso.to(u.arcmin), car_reso.to(u.arcmin))

    m_car = pysm3.apply_smoothing_and_coord_transform(
        m, output_car_resol=car_reso, return_healpix=False, return_car=True
    )

    #print('map in cartesian shape:',m_car.shape)
    return m_car

## Crop image to a a square centered in the middle
def crop_image_func(m_car,img_size):
    start_x=int(len(m_car[0,0,:])/2-img_size/2)#500
    start_y=int(len(m_car[0,:,0])/2-img_size/2)#
    #print('The image starts at x',start_x,'and y ',start_y)

    crop_image=m_car[0,start_y:start_y+img_size,start_x:start_x+img_size]
    return crop_image

def get_image(nside,seed,healpix_reso,img_size):
    #PySM realization
    m = pysm_realiz(nside,seed)
    #Transfert sky to 2D
    m_car=sky_to_2D(healpix_reso,m)
    #crop image
    crop_image=crop_image_func(m_car,img_size)
    return crop_image

def get_pillow_img(in_image):
    #scale the array to 0-255 (need it to create the image with pillow uint8)
    img_scaled = (in_image /in_image.max()) * 255
    img_scaled= img_scaled.astype(np.uint8)
    #create the pillow image from the array
    img_pil=Image.fromarray(img_scaled)
    #flip the image as we did with the 'origin=lower' in pyplot
    flip_img=img_pil.transpose(Image.FLIP_TOP_BOTTOM)
    return flip_img

if not execute:
    import sys
    sys.exit("Stopping the execution")

################## Parameters

dataset_name='256_set2'
root_dataset_folder='/home/javierhn/datasets/galactic_dust_realization/'
dataset_folder=root_dataset_folder+dataset_name+'/'
dataset_folder_test=root_dataset_folder+dataset_name+'_test/'

num_train_images=1000
num_test_images=100
img_size=256
nside = 256  # higher for high resolution
healpix_reso = hp.nside2resol(nside) * u.radian
npix = hp.nside2npix(nside)
print(healpix_reso.to(u.arcmin))
print(npix / 1e6, "Mpix")

#save a map in HDF to preserve the wcs
seed=101 #just some number
crop_image=crop_image=get_image(nside,seed,healpix_reso,img_size)
file_map_name=root_dataset_folder+dataset_name+'_map.hdf'
pixell.enmap.write_map(file_map_name,crop_image,fmt='hdf')

#create the TRAINING DATA
for i in range(num_train_images):
    seed=i
    crop_image=crop_image=get_image(nside,seed,healpix_reso,img_size)
    pil_img=get_pillow_img(crop_image) #convert the image to pillow
    file_name=dataset_folder+'{:04d}'.format(i)+'.png'
    #plt.imsave(fname=file_name, arr=crop_image, vmin=0, vmax=1000, origin="lower", format='png') #imsave save the array "as is"
    #write the image monochromatic in a png file
    pil_img.save(file_name)


#create the TEST DATA
for i in range(num_test_images):
    seed=i+num_train_images #to avoid the same seed
    crop_image=crop_image=get_image(nside,seed,healpix_reso,img_size)
    pil_img=get_pillow_img(crop_image) #convert the image to pillow
    file_name=dataset_folder_test+'{:04d}'.format(i)+'.png'
    #plt.imsave(fname=file_name, arr=crop_image, vmin=0, vmax=1000, origin="lower", format='png') #imsave save the array "as is"
    pil_img.save(file_name)
