from __future__ import absolute_import
from __future__ import print_function
import os
import sys
import pdb
import argparse
import utils as ut
import numpy as np

sys.path.append('../tool')
import toolkits

parser = argparse.ArgumentParser()
parser.add_argument('--net', default='resnet50', type=str)
parser.add_argument('--loss', default='softmax', choices=['softmax'], type=str)
parser.add_argument('--aggregation', default='avg', choices=['avg'], type=str)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--mode', default='eval', choices=['train', 'eval'], type=str)
parser.add_argument('--benchmark', default='ijbb', choices=['ijbb', 'ijbc'], type=str)

parser.add_argument('--feature_dim', default=512, choices=[512], type=int)
parser.add_argument('--data_path', default='path_to_the_ijbb', type=str)

global args
args = parser.parse_args()


def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]


def get_data_path():
    print('==> get data path, template id and media id.')

    def get_datalist(s):
        file = open(s, 'r')
        ijbb_meta = file.readlines()
        faceid, tid, mid = [], [], []
        for j in ijbb_meta:
            jsplit = j.split()
            faceid += [jsplit[0]]
            tid += [int(jsplit[1])]
            mid += [int(jsplit[-1])]

        faceid, template_id, media_id = map(np.array, [faceid, tid, mid])
        return faceid, template_id, media_id

    # ==> put the image paths into a long list, and break down into sublists as indicated by batch_size
    tid_mid_path = '../meta/{}_face_tid_mid.txt'.format(args.benchmark)
    faces, templates, medias = get_datalist(tid_mid_path)
    facepaths = np.array([os.path.join(args.data_path, f) for f in faces])
    return facepaths, templates, medias


def get_verification_label():
    # =============================================================
    # load meta information for template-to-template verification.
    # tid --> template id,  label --> 1/0
    # format:
    #           tid_1 tid_2 label
    # =============================================================
    print('==> get verification template pair and label.')
    tid_pair_path = '../meta/{}_template_pair_label.txt'.format(args.benchmark)
    file = open(tid_pair_path, 'r')
    meta = file.readlines()
    Y, p1, p2 = [], [], []
    for m in meta:
        msplit = m.split()
        Y += [int(msplit[-1])]
        p1 += [int(msplit[0])]
        p2 += [int(msplit[1])]
    Y, p1, p2 = map(np.array, [Y, p1, p2])
    return Y, p1, p2


def initialize_model():
    # Set basic environments.
    # Initialize GPUs
    toolkits.initialize_GPU(args)

    # ==> loading the pre-trained model.
    import model
    model_eval = None
    if args.aggregation == 'avg':
        if args.loss == 'softmax':
            model_eval = model.Vggface2_ResNet50(mode=args.mode)
        else:
            raise IOError('==> unknown loss type.')
    else:
        raise IOError('==> unknown aggregation mode.')

    print('test: {}_{}_{} on {} benchmark.'.format(args.net, args.aggregation, args.loss, args.benchmark))

    if args.resume:
        if os.path.isfile(args.resume):
            model_eval.load_weights(args.resume, by_name=True)
            print('==> successfully loaded the model {}'.format(args.resume))
        else:
            raise IOError('==> can not find the model to load {}'.format(args.resume))
    return model_eval


def image_encoding(model, facepaths):
    print('==> compute image-level feature encoding.')
    num_faces = len(facepaths)
    face_feats = np.empty((num_faces, args.feature_dim))
    imgpaths = facepaths.tolist()
    imgchunks = list(chunks(imgpaths, args.batch_size))

    for c, imgs in enumerate(imgchunks):
        im_array = np.array([ut.load_data(path=i, shape=(224, 224, 3), mode='eval') for i in imgs])
        f = model.predict(im_array, batch_size=args.batch_size)
        start = c * args.batch_size
        end = min((c + 1) * args.batch_size, num_faces)
        face_feats[start:end] = f
        if c % 500 == 0:
            print('-> finish encoding {}/{} images.'.format(c*args.batch_size, num_faces))
    return face_feats


def template_encoding(templates, medias, img_norm_feats):
    # ==========================================================
    # 1. face image --> l2 normalization.
    # 2. compute media encoding.
    # 3. compute template encoding.
    # 4. save template features.
    # ==========================================================
    print('==> compute template-level feature encoding.')

    uq_temp = np.unique(templates)
    num_temp = len(uq_temp)
    tmp_feats = np.empty((num_temp, args.feature_dim))

    for c, uqt in enumerate(uq_temp):
        (ind_t,) = np.where(templates == uqt)
        face_norm_feats = img_norm_feats[ind_t]
        faces_media = medias[ind_t]
        uqm, counts = np.unique(faces_media, return_counts=True)
        media_norm_feats = []

        for u,ct in zip(uqm, counts):
            (ind_m,) = np.where(faces_media == u)
            if ct < 2:
                media_norm_feats += [face_norm_feats[ind_m]]
            else:
                media_norm_feats += [np.sum(face_norm_feats[ind_m], 0, keepdims=True)]

        media_norm_feats = np.array(media_norm_feats)
        media_norm_feats = media_norm_feats / np.sqrt(np.sum(media_norm_feats ** 2, -1, keepdims=True))
        template_norm_feats = np.sum(media_norm_feats, 0)
        tmp_feats[c] = template_norm_feats
        if c % 500 == 0:
            print('-> finish encoding {}/{} templates.'.format(c, num_temp))
    return tmp_feats


def verification(unique_templates, tmp_feats, p1, p2):
    print('==> compute template verification results.')
    # ==========================================================
    #         Loading the Template-specific Features.
    # ==========================================================
    uq_temp = unique_templates
    score = np.zeros((len(p1),))
    # ==========================================================
    #         Compute set-to-set Similarity Score.
    # ==========================================================
    total_pairs = np.array(range(len(p1)))
    batchsize = 256
    sublists = [total_pairs[i:i + batchsize] for i in range(0, len(p1), batchsize)]
    num_sublists = len(sublists)

    for c, s in enumerate(sublists):
        t1 = p1[s]
        t2 = p2[s]
        ind1 = np.squeeze(np.array([np.where(uq_temp == j) for j in t1]))
        ind2 = np.squeeze(np.array([np.where(uq_temp == j) for j in t2]))

        inp1 = tmp_feats[ind1]
        inp2 = tmp_feats[ind2]

        v1 = inp1 / np.sqrt(np.sum(inp1 ** 2, -1, keepdims=True))
        v2 = inp2 / np.sqrt(np.sum(inp2 ** 2, -1, keepdims=True))

        similarity_score = np.sum(v1 * v2, -1)
        score[s] = similarity_score
        if c % 500 == 0: print('-> finish {}/{} pair verification.'.format(c, num_sublists))
    return score


def compute_ROC(labels, scores, roc_path):
    print('==> compute ROC.')
    import sklearn.metrics as skm
    from scipy import interpolate
    fpr, tpr, thresholds = skm.roc_curve(labels, scores)
    fpr_levels = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    f_interp = interpolate.interp1d(fpr, tpr)
    tpr_at_fpr = [f_interp(x) for x in fpr_levels]
    roc_txt = roc_path[:-3] + 'txt'
    file = open('../result/{}'.format(roc_txt), 'w')
    for (far, tar) in zip(fpr_levels, tpr_at_fpr):
        print('TAR @ FAR = {} : {}'.format(far, tar))
        file.write('TAR @ FAR = {}: {}\n'.format(far, tar))
    file.close()


if __name__ == '__main__':
    facepaths, templates, medias = get_data_path()
    groundtruth, template_1, template_2 = get_verification_label()
    unique_templates = np.unique(templates)

    model_eval = initialize_model()
    face_feats = image_encoding(model_eval, facepaths)
    template_feats = template_encoding(templates, medias, face_feats)
    score = verification(unique_templates, template_feats, template_1, template_2)

    score_path = args.resume.split(os.sep)[-2] + '_dim{}_scores.npy'.format(args.feature_dim)
    np.save('../result/{}'.format(score_path), score)
    compute_ROC(groundtruth, score, score_path)
