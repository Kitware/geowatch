import os, sys, glob
import xml.etree.ElementTree as ET
import gdal
import numpy as np
import pandas as pd

def print_usage():
   print("Usage: python s2_baseline_scene.py input_folder_safe")
   return

def find_baseline_scene(input_folder):
   # Find and collect all xml meta files
   pfname_xml_list = glob.glob(input_folder + '/*.SAFE/GRANULE/*/MTD_TL.xml', recursive=True)

   header = ['granule_id', 'proc_ver', 'sun_zenith_angle', 'cloud', 'coverage', 'mgrs_tile_id']
   df = pd.DataFrame(columns = header)

   for pfname_xml in pfname_xml_list:
       tmp_dict = {}

       # Granule ID
       path_to_xml = os.path.dirname(pfname_xml)
       granule_id = os.path.basename(path_to_xml)
       # Checking granule id
       if len(granule_id)==0:
          continue
       tmp_dict['granule_id'] = granule_id

       # MGRS tile id from granule id without T
       tmp_dict['mgrs_tile_id'] = granule_id.split('_')[1][1:]

       # Reading XML file
       tree = ET.parse(pfname_xml)
       root = tree.getroot()

       # Getting cloud cover 0-1
       cloud = 1. # default value
       for x in root.iter("CLOUDY_PIXEL_PERCENTAGE"):
            cloud = float(x.text) / 100.
            break
       tmp_dict['cloud'] = cloud

       # Mean sun zenith angle 0-90
       sun_zenith_angle = 90. # default value
       for x in root.iter('Mean_Sun_Angle'):
           for c in x:
               if (c.tag == 'ZENITH_ANGLE'):
                   sun_zenith_angle = float(c.text)
       tmp_dict['sun_zenith_angle'] = sun_zenith_angle

       # Baseline processing
       proc_ver = '02.01' # default value
       for x in root.iter("TILE_ID"):
           tile_id = x.text
           break
       proc_ver = tile_id.split('_')[-1]
       tmp_dict['proc_ver'] = float(proc_ver[1:]) # assuming 'N02.01

       # Coverage of the scene 0-1
       # Selecting a B04 
       pfname_b04 = glob.glob(os.path.join(path_to_xml, 'IMG_DATA') + '/*_B04.jp2')
       coverage = 0.
       if len(pfname_b04)==1:
           ds = gdal.Open(pfname_b04[0])
           if not (ds is None): # checking if file is valid
               arr = ds.GetRasterBand(1).ReadAsArray()
               coverage = np.sum(arr>0)/(1.*arr.shape[0]*arr.shape[1])

               ds = None
               arr = None
       tmp_dict['coverage'] = coverage
        
       # Adding to the dataframe
       df = df.append(tmp_dict, ignore_index=True)
       print(granule_id, proc_ver, sun_zenith_angle, cloud, coverage, tmp_dict['mgrs_tile_id'])

       tree = None
       #break
    
   norm_value_proc_ver = df['proc_ver'].max()
   #print(norm_value_proc_ver)
   df['score'] = 0.25*df['coverage'] + 0.25*(1-df['cloud']) + \
                 0.25*(1-df['sun_zenith_angle']/90.) + 0.25*( df['proc_ver']/norm_value_proc_ver)

   print(df)
   mgrs_tile_list = pd.unique(df['mgrs_tile_id'].values)

   for mgrs_tile in mgrs_tile_list:
       fname_ref_scene = '%s.baseline.scene.csv'%(mgrs_tile)
       df_tmp = df[df['mgrs_tile_id']==mgrs_tile]
       df_tmp.loc[[df_tmp['score'].idxmax()]].to_csv(os.path.join(input_folder, fname_ref_scene), index=False)

if __name__ == '__main__':
   if len(sys.argv) < 2:
       print_usage()
       sys.exit(1)

   input_folder = sys.argv[1]

   find_baseline_scene(input_folder)
