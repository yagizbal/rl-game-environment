import numpy as np
import os
import pygame

def load_map(name):
    """
    This function takes a name and loads the map with that name.
    name -> name of the map, type: str
    """
    return np.load(f"saved_arrays/{name}.npy"), np.load(f"saved_entities/{name}.npy")


def queue_maps():
    """
    This function takes all the maps in the saved_images folder and queues their loading. 
    There are extras available in saved_arrays that are not found in saved_images. 
    The purpose of this function is to queue the loading of all the maps in saved_images.
    """
    already_saved = (os.listdir("saved_images"))
    for i in already_saved:
        img = Image.open(f"saved_images/{i}")
        img.show()

def load_images(name_list,k,directory=None):
    """
    This function takes a list of names, a k value and a directory, and loads the images from the directory.
    name_list -> list of names of the tile images, type: list   
    k -> k value, which is the number of the first image, type: int
    directory -> directory from which the images are loaded, type: str
    """

    image_names_dict = {}
    image_map_dict = {}
    image_number = 0
    
    for index,image_name in enumerate(name_list):
        image_names_dict[index+k] = image_name
        img = pygame.image.load(f"{directory}/{image_name}.png").convert_alpha()
        image_map_dict[index+k] = img
        image_number+=1

    return name_list, image_names_dict,image_map_dict,image_number