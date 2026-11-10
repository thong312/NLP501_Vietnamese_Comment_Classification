from vihsd.data import normalize


def segment(text):
    # PyVi ships its segmentation resources: inference does not download Java/models.
    from pyvi import ViTokenizer
    return ViTokenizer.tokenize(normalize(text))
