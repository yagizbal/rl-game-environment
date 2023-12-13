import numpy as np
from PIL import Image, ImageDraw
import os

def draw_map(map_array,color_dict=None,float=False,value=1):
    if color_dict==None:    
        color_dict = {0:(8,0,255), #deep water
                      1:(0,170,255), #water
                      2: (0,230,255), #shallow water
                      3: (3,81,28), #fertile1
                      4: (3, 120, 38), #fertile2
                      5: (3, 150, 48), #fertile3
                      6: (3, 190, 50), #fertile4
                      7: (100, 175, 15), #desert1
                      8: (130, 175, 15), #desert2
                      9: (190, 210, 5), #desert3
                      10: (0,0,0) #border
                      }
                      
    if float==True:
        color_dict = {}
        for i in range(255):
            color_dict[i]=(i,i,i)
        map_array = np.rint((map_array*255)/value)

    if float==False:
        map_array = np.rint(map_array).astype(int)
    
    img = Image.new("RGB", (map_array.shape))
    img1 = ImageDraw.Draw(img)  
    img1.rectangle( [(0,0),(map_array.shape)] ,fill="white",outline="white")

    for element in color_dict:
        b,a = np.where(map_array==element)
        color = color_dict.get(element)
        for index,x in enumerate(a):
            y = b[index]
            img.putpixel((x,y),(color))
    
    return img


def save_map(map_array, entity_array, color_dict=None,draw=True,float=False,flip=False,settings=False,value=1,extend=""):      
    """
    """      
    already_saved = (os.listdir("saved_arrays"))
    a_ls = []
    for i in already_saved:
        l,b = i.split("-")
        a_ls.append(int(l))
    if len(a_ls)==0:
        a_ls.append(0)

    a_ls.sort()
    last_number = int(a_ls[-1])
    next_number = last_number+1

    generated_name = f"{next_number}-{extend}"
    if draw==True:
        img = draw_map(map_array,color_dict,float,value=value)        
        img.save(f"saved_images/{generated_name}.png")

        #save array in saved_arrays as .npy
        np.save(f"saved_arrays/{generated_name}.npy",map_array)
        #save settings in saved_settings as .txt
        if settings==True:
            with open(f"saved_settings/{generated_name}.txt","w") as f:
                f.write(f"float={float}\nvalue={value}\nflip={flip}\n")
        
    if entity_array is not None:
        np.save(f"saved_entities/{generated_name}.npy",entity_array)
  