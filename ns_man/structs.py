from typing import *
from typing import Optional as Maybe
from typing import Mapping as Map 

from numpy import array 
from torch import Tensor
from torch import float as floatt 
from torch import long as longt 

from dataclasses import dataclass 
from abc import ABC 

Tensors = Tuple[Tensor, ...]


@dataclass 
class Box:
    # bounding box coordinates in (x,y) cv2-like frame
    x: int 
    y: int
    w: int 
    h: int 

    def __iter__(self):
        return iter([self.x, self.y, self.w, self.h])


@dataclass
class Rectangle:
    # min area rectangle coordinates in (x,y) cv2-like frame
    x1: int 
    y1: int
    x2: int 
    y2: int 
    x3: int 
    y3: int
    x4: int 
    y4: int 

    def __iter__(self):
        return iter([self.x1, self.y1, self.x2, self.y2, self.x3, self.y3, self.x4, self.y4])


@dataclass
class Box3d:
    # 3D bounding box coordinates in world frame (centroid, dimensions)
    x: float
    y: float
    z: float
    l: float
    w: float
    h: float

    def __iter__(self):
        return iter([self.x, self.y, self.z, self.l, self.w, self.h])


@dataclass
class Pose3d:
    # Pose in world frame (position, yaw rotation) 
    x: float 
    y: float 
    z: float 
    roll: float
    pitch: float
    yaw: float 

    def __iter__(self):
        return iter([self.x, self.y, self.z, self.roll, self.pitch, self.yaw])


@dataclass
class ObjectProto:
    label: str
    contour: array 
    pose_3d: Pose3d
    #position_2d: Tuple[float, float]

    def __iter__(self):
        return iter([self.label, self.contour, self.pose_3d])
@dataclass
class Object:
    label: str 
    category: str 
    RGB_center: Tuple[float, float]
    RGB_rectangle: Rectangle 
    RGB_box: Box
    position_3d: Box3d 
    orientation: Tuple[float, ...] 
    size: float 
    material: str 
    color: str 
    special: Maybe[List[str]]


@dataclass
class Scene:
    environment: str
    image_id: str
    objects: Sequence[ObjectProto]

    @property
    def labels(self):
        return [o.label for o in self.objects]

    @property
    def categories(self):
        return [o.category for o in self.objects]

    @property
    def boxes(self):
        return [o.box for o in self.objects]

    # def get_crops(self):
    #     return [self.rgb[o.box.y : o.box.y+o.box.h, o.box.x : o.box.x+o.box.w] for o in self.objects]


@dataclass
class SceneRGB(Scene):
    image: array

    def get_crops(self):
        return [self.image[o.box.y : o.box.y+o.box.h, o.box.x : o.box.x+o.box.w] for o in self.objects]


@dataclass
class SceneGraph:
    nodes: Sequence[Object]
    edges: Dict[str, List[int]]

    @property
    def objects(self):
        return self.nodes

    @property
    def relationships(self):
        return self.relationships


@dataclass
class AnnotatedScene:
    environment: str 
    image_id: str 
    objects: Sequence[Object]
    relationships: Dict[str, List[int]]

    @property
    def labels(self):
        return [o.label for o in self.objects]

    @property
    def categories(self):
        return [o.category for o in self.objects]

    @property
    def positions(self):
        return [o.position_3d for o in self.objects]

    @property
    def rectangles(self):
        return [o.RGB_rectangle for o in self.objects]

    @property
    def boxes(self):
        return [o.RGB_box for o in self.objects]
    
    @property
    def centers_RGB(self):
        return [o.RGB_center for o in self.objects]

    @property
    def centers_3d(self):
        return [o.position_3d[0:3] for o in self.objects]

    @property
    def rotations(self):
        return [o.rotation for o in self.objects]
        
    @property
    def colors(self):
        return [o.color for o in self.objects]

    @property
    def materials(self):
        return [o.material for o in self.objects]

    @property
    def special_tags(self):
        return [o.special for o in self.objects]  