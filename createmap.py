import numpy as np
import random
from perlin_noise import PerlinNoise

from drawmap import *


def map_maker(scale,object_):
    '''This function creates a map of a given scale and fills it with a given object. 
    For example, if scale=1 and object_=0, the map will be 100x100 and filled with 0s.
    scale -> scale of the map, type: int
    object_ -> object to be used for the map, type: int
    '''
    return np.full((100*scale,100*scale),object_)


def executor(array, steps):
    """
    This function takes an array and a list of steps, and applies the steps to the array.
    The steps are dictionaries with the following format:
    {"method":method,"args":{"arg1":arg1,"arg2":arg2}}
    map_array -> array to be modified, type: np.array
    steps -> list of steps to be applied, type: list
    """
    temp_array = array
    for step in steps:
        temp_array=step["method"](temp_array,**step["args"])
    return temp_array


def sigmoid_(array,num,float=True):
    """
    This function takes an array and a number, and applies the sigmoid function to the array.
    The sigmoid function is 1/(1+e^-x), where x is the array.
    array -> array to be modified, type: np.array
    num -> number to be multiplied by the array, type: int
    """
    return num / (1+np.exp(-array))

def bias_(array,bias):
    """
    This function takes an array and a bias, and adds the bias to the array.
    array -> array to be modified, type: np.array
    bias -> bias to be added, type: int
    """
    return array+bias

def perlin_(array,octaves,seed,object,bias=0,float=False):
    """
    This function takes an array, an octaves value, a seed, an object, a bias and a float value.
    array -> array to be modified, type: np.array
    octaves -> octaves value for the perlin noise, type: int
    seed -> seed for the perlin noise, type: int
    object -> object to be used for the perlin noise, type: int
    bias -> bias to be added to the perlin noise, type: int
    float -> whether the perlin noise should be applied as a float or as an int, type: bool
    """
    grid = array.shape
    perlin = PerlinNoise(octaves=octaves, seed=seed)
    temp = ([[perlin([_/grid[0], __/grid[1]]) for __ in range(grid[0])] for _ in range(grid[1])])
    temp = np.array(temp)+bias
    
    if float==False:
        temp = (np.rint(sigmoid_(temp))).astype(int)
        temp_w = np.where(temp==1)
        for x in range(len(temp_w[0])):
            array[temp_w[0][x]][temp_w[1][x]] = object
    else:
        array = array+temp
    return array

def convert_float_to_int(array,segments):
    """This function rounds the values of an array to the nearest integer, then converts them to integers between 0 and segments-1
    array -> array to be converted
    segments -> number of segments to be converted to; meaning that if segments=4, the values will be converted to 0,1,2,3
    """
    array= np.rint(array*(segments-1))
    array[array>segments-1]=segments-1
    array[array<0]=0
    return array

def convolute(array,kernel,stride):
    """
    This function takes an array, a kernel size and a stride, and convolutes the array.
    array -> array to be convoluted, type: np.array
    kernel -> size of the kernel, type: int
    stride -> size of the stride, type: int
    """
    rows = array.shape[0]
    columns = array.shape[1]

    ls = []
    for row in range(rows-kernel):
        ls1=[]
        for i in range(columns-kernel):
            tile_value=np.sum(array[0+(row*stride):kernel+(row*stride),0+(i*stride):kernel+(stride*i)])/(kernel*kernel)
            ls1.append(tile_value)
        ls.append(ls1)
    ls=np.array(ls)
    return ls

def border(array,bordersize,object):
    """
    This function takes an array, a bordersize and an object, and adds a border to the array.
    map_array -> array to be bordered, type: np.array
    bordersize -> size of the border, type: int
    object -> object to be used for the border, type: int
    """
    grid = array.shape
    x = grid[0]+bordersize
    y = grid[1]+bordersize
    temp = np.full((x,y),object)
    temp[int(bordersize/2):-int(bordersize/2),int(bordersize/2):-int(bordersize/2)]=array
    return temp

def upscale_array(array,upscale):
    """
    This function takes an array and an upscale value, and upscales the array.
    array -> array to be upscaled, type: np.array
    upscale -> upscale value, type: int
    """
    grid = array.shape
    x = grid[0]*upscale
    y = grid[1]*upscale
    temp = np.zeros((x,y))
    for i in range(grid[0]):
        for j in range(grid[1]):
            temp[i*upscale:i*upscale+upscale,j*upscale:j*upscale+upscale]=array[i,j]
    return temp

def create_map(amount,scale,upscale):
    for i in range(amount):
        #scale=5
        #upscale=25

        array_ = map_maker(scale=scale,object_= 0)
        array_ = perlin_(array_, octaves=random.choice([1,4,10]), seed=random.randint(0,100), object=0, bias=0, float=True)   


        array_ = bias_(array_,(random.random()*1.2))
        #create a line of going through the upper 1/4 of the map, as tall as 1/15th of the map
        array_[int(array_.shape[0]/4):int(array_.shape[0]/4)+int(array_.shape[0]/20),:]=0 
        #create a line of going through the lower 1/4 of the map, as tall as 1/15th of the map
        array_[int(array_.shape[0]/4)*3:int(array_.shape[0]/4)*3+int(array_.shape[0]/20),:]=0


        array_ = upscale_array(array_,upscale=upscale)
        array_ = convolute(array_,kernel=int(scale*(10*upscale/2)),stride=1) #convolute to smoothe the transitions

        array_[array_<0]=0

        """After creating the map of floats and deleting the outliers, we break it down into integers which will be used for the color dict.
        """


        array_integer = convert_float_to_int(array_,segments=10)
        array_integer = border(array_integer,70,10)
        

        save_map(array_integer, draw=True, float=False)
        img = draw_map(array_integer, float=False)

#create_map(amount=10,scale=5,upscale=5)


def populate_map(map_full,amount,tile_bias_list,object,entity_full=None):
    """
    This function takes a map array, an amount, a tile bias list, an object and an object dict, and populates the map with the object.
    The tile bias list is a list of tuples, where the first element is the tile and the second element is the bias, for example:
    tile_bias_list = [(1,0.5),(2,0.2),(3,0.3)], meaning that the tile 1 will be used 50% of the time, the tile 2 will be used 20% of the time, and the tile 3 will be used 30% of the time.
    the object is the object to be used for the population, for example 1 for a bush, 2 for a fruit bush, 3 for a tree.
    map_array -> array to be populated, type: np.array
    amount -> amount of objects to be populated, type: int
    tile_bias_list -> list of tiles and biases, type: list, format: [(tile,bias),(tile,bias)]
    object -> object to be populated, type: int
    """

    if entity_full is None:
        entity_full = np.zeros((map_full.shape[0],map_full.shape[1]))

    tile_occurrences = {}
    for tile, bias in tile_bias_list:
        tile_occurrences[int(tile)] = len(np.where(map_full == int(tile))[0])

    #multiply the occurrences by the bias
    for tile, bias in tile_bias_list:
        tile_occurrences[int(tile)] = int(tile_occurrences[int(tile)]*bias)

    #now pick tiles from the dict according to their calculated value
    #if there is 100 tiles with bias 0.5, and 200 tiles with bias 0.25, they will be used at the same rate
    
    choices = []
    for i in range(amount):
        choice = np.random.choice(list(tile_occurrences.keys()), p=np.array(list(tile_occurrences.values())) / sum(tile_occurrences.values()))
        choices.append(choice)
    
    #get number of occurrences of each choice
    choices_occurrences = {}
    for choice in choices:
        choices_occurrences[choice] = choices.count(choice)
    
    print(choices_occurrences)
    
    #now place the objects
    for choice in choices_occurrences:
        tile_positions = np.where(map_full == choice)
        for i in range(choices_occurrences[choice]):
            index = random.randint(0, len(tile_positions[0]) - 1)
            row, col = tile_positions[0][index], tile_positions[1][index]
            entity_full[row][col] = object


    return entity_full



