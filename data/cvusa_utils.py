import os
from data.custom_transforms import *
import torch
import _pickle as cPickle


class CVUSA(torch.utils.data.Dataset):
    """
    Creates CVUSA dataset class for PyTorch dataloader.
    
    Args:
        root: Root folder of CVUSA dataset
        sate_size: Size of the satellite images to be fed into the model
        pano_size: Size of the panorama (street-view) images to be fed into the model
        use_polar: If True, satellite images will become the polar transformed ones, otherwise the output is the street, satellite and polar images
        transform_op: Data augmentation
    """

    def __init__(self, root, csv_file, sate_size=(256, 256), pano_size=(616, 112), use_polar=False, name=None, transform_op=None):
        self.root = root
        self.name = name
        self.use_polar = use_polar
        self.sate_size = pano_size if use_polar else sate_size
        self.pano_size = pano_size if use_polar else pano_size
        self.csv_path = os.path.join(root, csv_file)
        self.transform_op = transform_op
        self.filename = "CVUSA_preprocessed"

        # Load image list
        csv_path = os.path.join(root, 'splits', csv_file)
        with open(csv_path, 'r') as f:
            lines = f.readlines()
            pano_ims, sate_ims, polar_ims, item_ids, pano_ann = [], [], [], [], []
            for line in lines:
                
                items = line.strip().split(',')
                if(len(items) == 1): # the small sanity check training set has a different separator in the csv
                    items = line.strip().split(';')
                    
                item_id = (items[0].split('/')[-1]).split('.')[0]
                if use_polar:
                      # adjusted because of different folder structure
                    sate_ims.append(items[0].replace('bingmap', 'polarmap').replace('/19', '')) #.replace('jpg', 'png')
                else:
                    sate_ims.append(items[0])
                    polar_ims.append(items[0].replace('bingmap', 'polarmap').replace('/19', '')) #.replace('jpg', 'png')
                
                item_ids.append(item_id)
                pano_ims.append(items[1])
                pano_ann.append(items[2])
        
        self.pano_ims, self.sate_ims, self.pano_ann, self.item_ids = pano_ims, sate_ims, pano_ann, item_ids
        self.polar_ims = polar_ims if (use_polar is False) else None
        self.num = len(self.pano_ims)
        print('Load data from {}, total {}'.format(csv_path, self.num))               

    @classmethod
    def load_im(self, im_path, resize=None):
        im = cv2.imread(im_path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        
        if resize:
            im = cv2.resize(im, resize, interpolation=cv2.INTER_CUBIC)
        im = np.array(im, dtype=np.float32)
        return im
        
    def load(self):
        f = open(os.path.join(self.root, self.filename+'.p'), 'rb')
        tmp_dict = cPickle.load(f)
        f.close()          
        self.__dict__.update(tmp_dict) 


    def save(self):
        f = open(os.path.join(self.root, self.filename+'.p'), 'wb')
        cPickle.dump(self.__dict__, f, protocol=4)
        f.close()


    def __getitem__(self, index):
        # Triplet construction
        pos_id = index
        sate_path = os.path.join(self.root, self.sate_ims[pos_id])
        pano_path = os.path.join(self.root, self.pano_ims[pos_id])
        polar_path = os.path.join(self.root, self.polar_ims[pos_id]) if (self.polar_ims is not None) else None
        
        # Load and process images        
        sate_im = self.load_im(sate_path, resize=self.sate_size)
        pano_im = self.load_im(pano_path, resize=self.pano_size)
        polar_im= self.load_im(polar_path, resize=self.pano_size) if (polar_path is not None) else None

        if polar_im is not None:
            sample = {'satellite': sate_im, 'street': pano_im, 'polar': polar_im}
        else:
            sample = {'satellite': sate_im, 'street': pano_im}

        if self.transform_op:
            sample = self.transform_op(sample)
        sample['im_path'] = (sate_path, pano_path)
        sample['item_id'] = self.item_ids[pos_id]
        return sample

    def __len__(self):
        return self.num

    def __repr__(self):
        fmt_str = 'CVUSA \n'
        fmt_str += 'Pair cvs path: {}\n'.format(self.csv_path)
        fmt_str += 'Number of data pairs: {}\n'.format(self.__len__())
        fmt_str += 'Dataset root : {}\n'.format(self.root)
        fmt_str += 'Image Transforms: {}\n'.format(self.transform_op.__repr__().replace('\n', '\n    '))
        return fmt_str

def convert_image_np(inp):
    """Convert a Tensor to numpy image."""

    if inp.ndim == 3:
        inp = inp.unsqueeze(0).numpy().transpose((0, 2, 3, 1))
    elif inp.ndim == 4:
        inp = inp.numpy().transpose((0, 2, 3, 1))
    else:
        raise ValueError('The input should be an image (ndim=3) or an array of images (ndim=4)!')
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for i in range(inp.shape[0]):        
        inp[i] = std * inp[i] + mean
        inp[i] = np.clip(inp[i], 0, 1)
    return inp