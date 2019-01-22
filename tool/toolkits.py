import os
import numpy as np

def initialize_GPU(args):
    # Initialize GPUs
    import tensorflow as tf
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session


# set up multiprocessing
def set_mp(processes=8):
    import multiprocessing as mp

    def init_worker():
        import signal
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    global pool
    try:
        pool.terminate()
    except:
        pass

    if processes:
        pool = mp.Pool(processes=processes, initializer=init_worker)
    else:
        pool = None
    return pool


# vggface2 dataset
def get_vggface2_imglist(args):
    def get_datalist(s):
        file = open('{}'.format(s), 'r')
        datalist = file.readlines()
        imglist = []
        labellist = []
        for i in datalist:
            linesplit = i.split(' ')
            imglist.append(linesplit[0])
            labellist.append(int(linesplit[1][:-1]))
        return imglist, labellist

    print('==> calculating image lists...')
    # Prepare training data.
    imgs_list_trn, lbs_list_trn = get_datalist(args.trn_meta)
    imgs_list_trn = [os.path.join(args.data_path, i) for i in imgs_list_trn]
    imgs_list_trn = np.array(imgs_list_trn)
    lbs_list_trn = np.array(lbs_list_trn)

    # Prepare validation data.
    imgs_list_val, lbs_list_val = get_datalist(args.val_meta)
    imgs_list_val = [os.path.join(args.data_path, i) for i in imgs_list_val]
    imgs_list_val = np.array(imgs_list_val)
    lbs_list_val = np.array(lbs_list_val)

    return imgs_list_trn, lbs_list_trn, imgs_list_val, lbs_list_val


def display(imgs):
    import pylab as plt
    f, ax = plt.subplots(2, 2)
    for i in range(2):
        for j in range(2):
            ax[i, j].imshow(imgs[i*2+j].astype('uint8'))
            print('image mean: {}, std: {}'.format(np.mean(imgs[i*2+j]), np.std(imgs[i*2+j])))
    plt.show()

