import tensorflow as tf
import glob
from tqdm import tqdm

def _float32_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def write_record(image_path, mask_path):
    ID = image_path.split('/')[-1].split('.')[0]
    feature = {'image':  _bytes_feature(tf.compat.as_bytes(open(image_path, 'rb').read())),
                'mask':   _bytes_feature(tf.compat.as_bytes(open(mask_path, 'rb').read())),
                'ID':  _bytes_feature(ID.encode('utf-8')),
        }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())

if __name__ == '__main__':
    all_masks = sorted(glob.glob('../../../ATR_curate/masks/*.png'))
    all_images = sorted(glob.glob('../../../ATR_curate/images/*.jpg'))
    all_edge_maps_1 = sorted(glob.glob('../../../ATR_curate/images/*.jpg'))
    all_edge_maps_2 = sorted(glob.glob('../../../ATR_curate/images/*.jpg'))
    num_data = len(all_images)
    
    FILEPATH =  f"/content/atr_{num_data}.record"
    writer = tf.io.TFRecordWriter(FILEPATH)
    for image_path, mask_path in tqdm(zip(all_images, all_masks)):
        write_record(image_path, mask_path)
    print("writing completed")
    writer.close()