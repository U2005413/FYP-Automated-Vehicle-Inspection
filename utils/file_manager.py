from PIL import Image


def get_filename_wo_ext(filename):
    index = filename.rindex(".")
    return filename[:index]


def save_image_arr(image_arr, save_path):
    im = Image.fromarray(image_arr)
    im.save(save_path)
