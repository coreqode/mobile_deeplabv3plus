import argparse

def get_parser():
    """docstring for argparse"""
    args = argparse.ArgumentParser()
    args.add_argument("-augment", action="store_true")  # colab_tpu/gpu/debug
    args.add_argument("-restart", action="store_true")
    args.add_argument("-tfrecord", action="store_true")
    args.add_argument("--mode", type=str, default="debug")
    args.add_argument("--lr_schedule", type=list, default = None )
    args.add_argument("--loss_weights", type=list, default=None)
    args.add_argument("--train_batch_size", type=int, default=4 * 8)
    args.add_argument("--val_batch_size", type=int, default=4 * 8)
    args.add_argument("--epochs", type=int, default=50)
    args.add_argument("--learning_rate", type=float, default=0.00010)
    args.add_argument("--num_data", type=int, default=None)  # optional
    args.add_argument("--input_height", type=int, default=512)
    args.add_argument("--input_width", type=int, default=256)
    args.add_argument("--save_freq", type=int, default=2)
    args.add_argument("--model", type=str, default="mobilenet")  # xception
    args.add_argument("--exp_name", type=str, default="atr_wce_512_256_mobilenet_edge_loss_1")
    args.add_argument("--dump_parse_path", type=str,
                      default="/content/opts.txt")
    args.add_argument("--save_path", type=str,
                      default="/content/")
    args.add_argument("--tensorboard_logs", type=str,
                      default="gs://experiments_logs/grapy/")
    args.add_argument("--weight_path", type=str, default='/content/weights/')
    args.add_argument("--tfrecord_path", type=str,
                      default="/content/atr_17706.record")

    opts = args.parse_args()
    return opts
