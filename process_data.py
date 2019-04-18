import os, h5py
import numpy as np
import pickle


def normalize(data):
    if len(data) == 0:
        return data
    data = np.array(data)
    min_price = data.min()
    max_price = data.max()
    if min_price < max_price:
        k = 1.0 / (max_price - min_price)
        data = k * (data - min_price)
    return data

def generate_pkl(type):
    with h5py.File("./data/digits/%s/images/digitStruct.mat" % type, 'r') as f:
        name = f['/digitStruct/name'][:]
        bbox = f['digitStruct']['bbox'][:]
        print(name.shape)
        img_path = []
        img_info = []
        for i in range(name.shape[0]):
            img0 = f[name[i][0]][()].astype(np.uint8)
            img0 = str(img0.tostring(), 'utf8')
            img_path.append(img0)
            if i % 1000 == 0:
               print(i)
            info_list = []
            ctn_digit = f[bbox[i][0]]['label'].shape[0]
            if ctn_digit == 1:
                info = {}
                height = f[bbox[i][0]]['height']
                width = f[bbox[i][0]]['width']
                label = f[bbox[i][0]]['label']
                left = f[bbox[i][0]]['left']
                top = f[bbox[i][0]]['top']

                info['height'] = height[0][0]
                info['width'] = width[0][0]
                info['label'] = label[0][0]
                if info['label'] > 9:
                    print('label > 9')
                info['left'] = left[0][0]
                info['top'] = top[0][0]
                info_list.append(info)
            else:
                for d in range(ctn_digit):
                    info = {}  # ['height', 'label', 'left', 'top', 'width']
                    height = f[bbox[i][0]]['height']
                    width = f[bbox[i][0]]['width']
                    label = f[bbox[i][0]]['label']
                    left = f[bbox[i][0]]['left']
                    top = f[bbox[i][0]]['top']

                    info['height'] = f[height[d][0]][0][0]
                    info['width'] = f[width[d][0]][0][0]
                    info['label'] = f[label[d][0]][0][0]
                    if info['label'] > 9:
                        print('label > 9')
                    info['left'] = f[left[d][0]][0][0]
                    info['top'] = f[top[d][0]][0][0]
                    info_list.append(info)
            img_info.append(info_list)


        with open('./data/%s.pkl' % type, 'wb') as f:
            pickle.dump([img_path, img_info], file=f, protocol=2)

def generate_torch_data(in_path, out_path, type):
    import pickle
    with open(in_path, 'rb') as f:
        data = pickle.load(file=f)
    img_path = data[0]
    img_info = data[1]
    img_file = open(os.path.join(out_path, 'img.txt'), 'w')

    for i in range(len(img_path)):
        img_file.write('./data/digits/%s/images/%s\n'%(type, img_path[i]))
        fname =  img_path[i].replace('png', 'txt')
        with open(os.path.join(out_path, 'labels', fname), 'w') as bfile:
            for fo in img_info[i]:
                norm = np.array([fo['left'], fo['top'], fo['width'], fo['height']])
                norm = norm / np.linalg.norm(norm)
                if int(fo['label']) < 0 or int(fo['label']) > 9:
                    print(fo['label'])
                    
                bfile.write('%d %f %f %f %f\n' % (int(fo['label']-1), norm[0], norm[1], norm[2], norm[3]))
    img_file.close()


#generate_pkl('train')
#generate_pkl('test')
generate_torch_data('./data/train.pkl', './data/digits/train', 'train')
generate_torch_data('./data/test.pkl', './data/digits/test', 'test')
