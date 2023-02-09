def NAME_TO_WIDTH(name):
    map = {
        'mn01': 0.1,
        'mn02': 0.2,
        'mn04': 0.4,
        'mn05': 0.5,
        'mn06': 0.6,
        'mn08': 0.8,
        'mn10': 1.0,
        'mn12': 1.2,
        'mn14': 1.4,
        'mn16': 1.6,
        'mn20': 2.0,
        'mn30': 3.0,
        'mn40': 4.0
    }
    try:
        w = map[name[:4]]
    except:
        w = 1.0

    return w
