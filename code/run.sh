# pip install progress

# python main.py -tfrecord -augment --mode colab_tpu --exp_name atr_512_256_mobilenet_edge_loss_1
python main.py -tfrecord -augment -restart --mode colab_tpu --exp_name atr_512_256_mobilenet_edge_loss_1 --learning_rate 0.00001