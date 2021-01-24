import  tensorflow as  tf 
import os 

def start_tpu():
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print("Running on TPU ", tpu.cluster_spec().as_dict()["worker"])
    except ValueError:
        raise BaseException(
            "ERROR: Not connected to a TPU runtime; please see the previous cell in this notebook for instructions!"
        )
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
    return tpu_strategy

def get_weights(opts):
    if not os.path.isdir(opts.weight_path):
        print('Downloading Weights')
        cmd = f"/content/{opts.exp_name}/weights {opts.root_path}"
        os.system(cmd) 
    else:
        print('Weights are present in the directory')