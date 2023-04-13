from ns_man.structs import *
from ns_man.utils.image_proc import * 

import pandas as pd
import subprocess
import os
import cv2
from random import sample
from functools import lru_cache


class SimScenesDataset:
    def __init__(self, images_path: str, csv_path: str, ignore_labels: List[str] = ['trash']):
        self.ignore_labels = ignore_labels
        self.root = images_path
        self.csv_path = csv_path
        self.table = pd.read_table(csv_path)
        self.image_ids = self.table['image_id'].tolist()
        self.labels = [row.split(',') for row in self.table['label'].tolist()]
        self.contours =  [[eval(x.strip("()")) for x in row.split("),")] for row in self.table["RGB_contour"].tolist()]
        self.contours = [[np.int0([[[c[i], c[i+1]]] for i in range(0, len(c)-1, 2)]) for c in row] for row in self.contours]
        # self.pos_2d = [[float(x.strip("()")) for x in p.split(',')] for p in self.table['2D_position'].tolist()]
        # self.pos_2d = [[float(x.strip("()")) for x in p.split(',')] for p in self.table['2D_position'].tolist()]
        self._pos_3d = [[float(x.strip("()")) for x in p.split(',')] for p in self.table['3D_pose'].tolist()]
        self._pos_3d = [[Pose3d(*[p[j] for j in range(i, i+6)]) for i in range(0, len(p)-5, 6)] for p in self._pos_3d]
        # moments = [[cv2.moments(c) for c in row] for row in self.contours]
        # self.centers = [[(int(M['m10']/M['m00']), int(M['m01']/M['m00'])) for M in row] for row in moments]
        # self.boxes = [[Box(*cv2.boundingRect(c)) for c in row] for row in self.contours]
        # self.rects = [[Rectangle(*sum(cv2.boxPoints(cv2.minAreaRect(c)).tolist(), [])) for c in row] for row in self.contours]          
        #self.categories = [[CATEGORY_MAP[l.split('_')[0]] for l in labs] for labs in self.labels]
        # self.objects = [[ObjectProto(l, co, p) for l, co, p in zip(ls, cos, ps)] 
        #                 for ls, cos, ps in zip(self.labels, self.contours, self.pos_3d)]
        
        self.objects, self.boxes, self.rects, self.centers, self.pos_3d = [], [], [], [], []
        for ls, cos, ps in zip(self.labels, self.contours, self._pos_3d):
            _objects_this_scene = []
            _boxes_this_scene = []
            _rects_this_scene = []
            _centers_this_scene = []
            _poses_this_scene = []
            for l, co, p in zip(ls, cos, ps):
                if l not in ignore_labels:
                    _objects_this_scene.append(ObjectProto(l, co, list(p)))
                    _boxes_this_scene.append(Box(*cv2.boundingRect(co)))
                    _rects_this_scene.append(Rectangle(*sum(cv2.boxPoints(cv2.minAreaRect(co)).tolist(), [])))
                    _M = cv2.moments(co)
                    _centers_this_scene.append((int(_M['m10']/_M['m00']), int(_M['m01']/_M['m00'])))
                    _poses_this_scene.append(p)
            self.objects.append(_objects_this_scene)
            self.boxes.append(_boxes_this_scene)
            self.rects.append(_rects_this_scene)
            self.centers.append(_centers_this_scene)
            self.pos_3d.append(_poses_this_scene)


    def get_image(self, n: int) -> array:
        return cv2.imread(os.path.join(self.root, "RGB", str(self.image_ids[n]) + '.png'))

    def get_image_from_id(self, image_id: int) -> array:
        return cv2.imread(os.path.join(self.root, "RGB", str(image_id) + '.png'))

    def get_depth(self, n: int) -> array:
        return np.load(os.path.join(self.root, "Depth", str(self.image_ids[n]) + '.npy'))

    def get_depth_from_id(self, image_id: int) -> array:
        return np.load(os.path.join(self.root, "Depth", str(image_id) + '.npy'))

    def get_rgbd(self, n: int) -> Tuple[array, array]:
        return self.get_image(n), self.get_depth(n)

    def get_rgbd_from_id(self, image_id: int) -> Tuple[array, array]:
        return self.get_image_from_id(image_id), self.get_depth_from_id(image_id)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, n: int) -> Scene:
        return Scene(environment="sim", 
                     image_id=self.image_ids[n], 
                     objects=self.objects[n])

    def get_from_id(self, n: int) -> Scene:
        return self.__getitem__(self.image_ids.index(n))

    def show(self, n: int):
        scene = self.__getitem__(n)
        img = self.get_image(n).copy()
        for i, obj in enumerate(scene.objects):
            x, y, w, h = self.boxes[n][i]
            rect = self.rects[n][i]
            rect = np.int0([(rect.x1, rect.y1), (rect.x2, rect.y2), (rect.x3, rect.y3), (rect.x4, rect.y4)])
            #img = cv2.putText(img, obj.label, (x, y), fontFace=0, fontScale=1, color=(0,0,0xff))
            img = cv2.putText(img, str(1+i), (x, y), fontFace=0, fontScale=1, thickness=2, color=(0,0,0xff))
            img = cv2.rectangle(img, (x, y), (x+w, y+h), (0xff, 0, 0), 2)
            img = cv2.drawContours(img, [rect], 0, (0,0,0xff), 2)
            img = cv2.drawContours(img, [obj.contour], 0, (0,0xff, 0), 1)
        show(img, str(self.image_ids[n]) + ".png")

    def show_selected(self, n: int):
        scene = self.__getitem__(n)
        img = self.get_image(n).copy()
        for i, obj in enumerate(scene.objects):
            x, y, w, h = self.boxes[n][i]
            rect = self.rects[n][i]
            rect = np.int0([(rect.x1, rect.y1), (rect.x2, rect.y2), (rect.x3, rect.y3), (rect.x4, rect.y4)])
            #img = cv2.putText(img, obj.label, (x, y), fontFace=0, fontScale=1, color=(0,0,0xff))
            img = cv2.putText(img, str(1+i), (x, y), fontFace=0, fontScale=1, thickness=2, color=(0,0,0xff))
            img = cv2.rectangle(img, (x, y), (x+w, y+h), (0xff, 0, 0), 2)
            img = cv2.drawContours(img, [rect], 0, (0,0,0xff), 2)
            img = cv2.drawContours(img, [obj.contour], 0, (0,0xff, 0), 1)
        show(img, str(self.image_ids[n]) + ".png")

    def save(self, image_id: int, path: str):
        scene = self.get_from_id(image_id)
        img = self.get_image_from_id(scene.image_id).copy()
        n = self.image_ids.index(scene.image_id)
        for i, obj in enumerate(scene.objects):
            x, y, w, h = self.boxes[n][i]
            rect = self.rects[n][i]
            rect = np.int0([(rect.x1, rect.y1), (rect.x2, rect.y2), (rect.x3, rect.y3), (rect.x4, rect.y4)])
            #img = cv2.putText(img, obj.label, (x, y), fontFace=0, fontScale=1, color=(0,0,0xff))
            img = cv2.putText(img, str(1+i), (x, y), fontFace=0, fontScale=1, thickness=2, color=(0,0,0xff))
            #img = cv2.rectangle(img, (x, y), (x+w, y+h), (0xff, 0, 0), 2)
            #img = cv2.drawContours(img, [rect], 0, (0,0,0xff), 2)
            img = cv2.drawContours(img, [obj.contour], 0, (0,0,0xff), 2)
        #show(img, str(self.image_ids[n]) + ".png")
        print(f'Saving in {path}')
        cv2.imwrite(path, img)

    def save_selected_blur(self, image_id, keep, path):
        scene = self.get_from_id(image_id)
        img = self.get_image_from_id(scene.image_id).copy()
        H, W = img.shape[0:2]
        #blur = cv2.medianBlur(img.copy(), 21)
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = np.clip(np.float32(v / 4), 0, 127).astype(np.uint8)
        blur = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)

        n = self.image_ids.index(scene.image_id)
        for idx in keep:
            i = idx -1
            obj = scene.objects[i]
            #for i, obj in enumerate(scene.objects):
            x, y, w, h = self.boxes[n][i]
            rect = self.rects[n][i]
            rect = np.intp([(rect.x1, rect.y1), (rect.x2, rect.y2), (rect.x3, rect.y3), (rect.x4, rect.y4)])
            #blur = cv2.putText(blur, str(1+i), (x, y), fontFace=0, fontScale=1, thickness=3, color=(0,0,0xff))
            #blur = cv2.drawContours(blur, [rect], 0, (0,0,0xff), 2)
            mask = cv2.drawContours(np.zeros((H,W), dtype=np.uint8), [obj.contour], 0, 0xff, -1)
            blur[mask == 0xff] = img[mask == 0xff]
            blur = cv2.drawContours(blur, [obj.contour], 0, (0,0,0xff), -1)
            #blur[mask == 0xff, 2] = 0xff
            #blur[mask == 0xff, 0] = 0
            # blur[mask == 0xff, 1] = 0
        show(blur, str(self.image_ids[n]) + "_selected.png")
        print(f'Saving in {path}')
        cv2.imwrite(path, blur)

    def show_id(self, id: int):
        self.show(self.image_ids.index(id))

    def inspect(self):
        for n in range(self.__len__()):
            self.show(n)

    def massage(self, from_, to_):
        drop = []
        for i in range(from_, to_):
            scene = self.__getitem__(i)
            img = self.get_image(i).copy()
            for obj in scene.objects:
                x, y, w, h = obj.box.x, obj.box.y, obj.box.w, obj.box.h
                rect = obj.rectangle
                rect = np.int0([(rect.x1, rect.y1), (rect.x2, rect.y2), (rect.x3, rect.y3), (rect.x4, rect.y4)])
                img = cv2.putText(img, obj.label, (x, y), fontFace=0, fontScale=1, color=(0,0,0xff))
                img = cv2.rectangle(img, (x, y), (x+w, y+h), (0xff,0,0  ), 2)    
                img = cv2.drawContours(img, [rect], 0, (0,0,0xff), 2)
                img = cv2.drawContours(img, [obj.contour], 0, (0,0xff,0), 2)
            cv2.imshow(str(self.image_ids[i]), img)
            while True:
                key = cv2.waitKey(1) & 0xff
                if key == ord('q'):
                    break
                elif key == ord('m'):
                    drop.append(self.image_ids[i])
                    break
            cv2.destroyWindow(str(self.image_ids[i]))
        return drop


def get_sim_rgbd_scenes(split: Maybe[str]=None):
    if split == 'train':
        return SimScenesDataset("/home/p300488/dual_arm_ws/DATASET", "/home/p300488/dual_arm_ws/DATASET/data.tsv")
    elif split == 'dev':
        return SimScenesDataset("/home/p300488/dual_arm_ws/DATASET_DEV", "/home/p300488/dual_arm_ws/DATASET_DEV/data.tsv")
    elif split is None:
        return (
                SimScenesDataset("/home/p300488/dual_arm_ws/DATASET", "/home/p300488/dual_arm_ws/DATASET/data.tsv"),
                SimScenesDataset("/home/p300488/dual_arm_ws/DATASET_DEV", "/home/p300488/dual_arm_ws/DATASET_DEV/data.tsv")
               )


def get_sim_rgbd_objects(split: Maybe[str]=None):
    ds = get_sim_rgbd_scenes(split=split)
    if split is not None:
        ds = [ds]
    ret = []
    for dataset in ds:
        all_crops, all_labels = [], []
        chunk = [0, (len(dataset) if split == 'dev' else 5000)]
        for i in range(chunk[0], chunk[1]):
            scene = dataset[i]
            #print(scene)
            rgb = dataset.get_image(i)
            crops = [crop_contour(rgb, o.contour) for o in scene.objects]
            labels =[o.label for o in scene.objects]
            all_crops.extend(crops)
            all_labels.extend(labels)
        ret.append(list(zip(all_crops, all_labels)))
    return ret if split is not None else ret[0]
