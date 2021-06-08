'''
    Parse the config file
    ----------
    file : json
        config file
    '''
def parser(file):
    class dotdict(dict):
        """dot.notation access to dictionary attributes"""
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

    import yaml
    return (dotdict(yaml.load(open(file))))

def resize_image_mlvl(args, image, level):
    return (image[:, :, (args.mlvl_borders[level]):(args.patch_size[0] - args.mlvl_borders[level]),
            (args.mlvl_borders[level]):(args.patch_size[1] - args.mlvl_borders[level]),
            (args.mlvl_borders[level]):(args.patch_size[2] - args.mlvl_borders[level])]) \
            [:, :, ::args.mlvl_strides[level], ::args.mlvl_strides[level], ::args.mlvl_strides[level]]