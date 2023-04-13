import pydot
import cv2
import os
from graphviz import Digraph 

def render_graph(dot_file, out_name):
    g = pydot.graph_from_dot_file(dot_file)[0]
    g.write_png(out_name)


def show_save(scene, index, path = "./graphviz"):
    img = scene.rgb.copy()
    for obj in scene.objects:
        img = cv2.putText(img, obj.label, (obj.box.x, obj.box.y), fontFace=0, fontScale=1, color=(0,0,0xff))
        img = cv2.rectangle(img, (obj.box.x, obj.box.y), (obj.box.x+obj.box.w, obj.box.y+obj.box.h), (0,0,0xff), 2)
    cv2.imshow("show", img)
    while 1:
        key = cv2.waitKey(1) & 0xff
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(os.path.join(path, "i" + str(index) + ".png"), img)
            render_graph(os.path.join(path, "input.dot"), 
                os.path.join(path, "G" + str(index) + ".png"))
            break
    cv2.destroyWindow("show")

def save_scene_graph_gui(num_objects, save_idx):
    from sim2realVL.data.sim_dataset import get_sim_rgbd_scenes
    ds = get_sim_rgbd_scenes()
    for i in range(len(ds)):
        if len(ds[i].objects) == num_objects:
            show_save(ds[i], save_idx)
            