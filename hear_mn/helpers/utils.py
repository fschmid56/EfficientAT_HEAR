def NAME_TO_WIDTH(name):
    map = {
        'mn04': 0.4,
        'mn05': 0.5,
        'mn10': 1.0,
        'mn20': 2.0,
        'mn30': 3.0,
        'mn40': 4.0
    }
    try:
        w = map[name[:4]]
    except:
        w = 1.0

    return w
