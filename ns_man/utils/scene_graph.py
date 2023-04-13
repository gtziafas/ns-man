from ns_man.structs import *
from ns_man.utils.viz import *

import numpy as np
import pandas as pd
import os
import time 
from functools import reduce
import pydot
from graphviz import Digraph 
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import json
from operator import mul


WORKSPACE_DIMENSIONS = [1.8, 1.2, 1.]
WORKSPACE_COORDS = [[-0.9, 0.9], [0.55, 1.75], [0., 1.0]]
RELATIONS = [
    "LEFT" ,
    "RIGHT" ,
    "BEHIND",
    "FRONT" ,
    "CLOSE" ,
    "FAR",
    "BIGGER",
    "SMALLER",
    "NEXT"  
]
HRELATIONS = ["CLOSER", "FURTHER"]
SIZE_NORM = .0805
REL_SYNONYMS = {'left': ['left', 'left of', 'left side of'],
 'right': ['right', 'right of', 'right side of'],
 'close': ['closer', 'nearer', 'more reachable'],
 'far': ['further', 'more far away', 'less reachable', 'more far', 'far away'],
 'behind': ['behind'],
 'front': ['in front of', 'in front', 'front of'],
 'bigger': ['bigger', 'larger'],
 'smaller': ['smaller'],
 'next': ['next to', 'close to', 'next', 'close']}

HREL_SYNONYMS = {
	'closer' : ['closer to than', 'closer than', 'closer'],
	'further': ['further from than', 'further than', 'further']
}


def pairwise_rel_str(nodes, edges):
    N = len(nodes)
    mat = np.zeros((N,N), dtype='<U36')
    for i in range(N):
        mat[i, i] = ",".join(nodes[i].attributes)
        for j in range(N):
            if edges[i,j,0]:
                mat[i,j] = 'left' if not mat[i,j] else mat[i,j] + ",left"
            if edges[i,j,1]:
                mat[i,j] = 'right' if not mat[i,j] else mat[i,j] + ",right"
            if edges[i,j,2]:
                mat[i,j] = 'behind' if not mat[i,j] else mat[i,j] + ",behind"
            if edges[i,j,3]:
                mat[i,j] = 'front' if not mat[i,j] else mat[i,j] + ",front"
            if edges[i,j,4]:
                mat[i,j] = 'closer' if not mat[i,j] else mat[i,j] + ",closer"
            if edges[i,j,5]:
                mat[i,j] = 'further' if not mat[i,j] else mat[i,j] + ",further"
            if edges[i,j,6]:
                mat[i,j] = 'bigger' if not mat[i,j] else mat[i,j] + ",bigger"
            if edges[i,j,7]:
                mat[i,j] = 'smaller' if not mat[i,j] else mat[i,j] + ",smaller"
            if edges[i,j,8]:
                mat[i,j] = 'next to' if not mat[i,j] else mat[i,j] + ",next"
    #rows = [['{' + str(int(row[0])) + ', ' +  str(int(row[1])) + ', ' +  str(round(row[2],3)) + '}' for row in self.edges[i,:,:].tolist()] for i in range(len(self.nodes))] 
    return mat


def compute_pairwise_dist(nodes: Dict[str, Any]):
    N = len(nodes)
    centers = np.asarray([normalize_coords(o['position_3d']) for o in nodes], dtype=np.float32)
    differences = centers[:, np.newaxis] - centers
    #normalize_factor = np.linalg.norm(WORKSPACE_DIMENSIONS)
    return np.linalg.norm(differences, axis=-1)


def vectorize_edges(graph: Dict[str, Any], skip_hrel=True, add_distance=False):
	n_items, n_edges = len(graph['objects']), len(graph['relationships'].keys())
	if skip_hrel:
		n_edges -= 2
	x_rel = np.zeros((n_edges, n_items, n_items), dtype=np.float32)
	for rel_idx, (_, connections) in enumerate(graph['relationships'].items()):
		if skip_hrel and rel_idx >= n_edges:
			break
		for i, targets in enumerate(connections):
			for target in targets:
				x_rel[rel_idx, i, target] = 1
	x_rel = x_rel.transpose((2, 1, 0)) # N x N x R
	# add extra distance in last index key
	if add_distance:
		x_rel = np.concatenate([x_rel, 
		compute_pairwise_dist(graph['relationships'])[:,:,np.newaxis]], axis=-1)
	return x_rel


def normalize_coords(center):
	x, y, z = center
	xx = (x - WORKSPACE_COORDS[0][0]) / WORKSPACE_DIMENSIONS[0]
	yy = (y - WORKSPACE_COORDS[1][0]) / WORKSPACE_DIMENSIONS[1]
	zz = (z - WORKSPACE_COORDS[2][0]) / WORKSPACE_DIMENSIONS[2]
	return [xx, yy, zz]



class SceneGraph:

    def __init__(self, nodes: List[Object], edges: Dict[str, List[Any]]):
        self.nodes = nodes 
        self.edges = {key: edges[key] for key in [rel for rel in edges.keys() if rel not in ['closer', 'further']]}

    def print(self):
        maxchars = 18
        header = ''.join([' '*maxchars] + [o.label + ' ' * (maxchars - len(o.label)) for o in self.nodes]) 
        rows = [['{' + str(int(row[0])) + ', ' +  str(int(row[1])) + ', ' +  str(round(row[2],3)) + '}' for row in self.edges[i,:,:].tolist()] for i in range(len(self.nodes))] 
        rows = [''.join([self.nodes[i].label + ' ' * (maxchars - len(self.nodes[i].label))] + [x + ' ' * (maxchars - len(x)) for x in rows[i]]) for i in range(len(rows))]
        print('\n'.join([header, *rows]))

    def render(self, path: str):
        with open('input.dot', 'w+') as f:
            items = ['item{} [ label="{}"];'.format(i, self.nodes[i].label) for i in range(len(self.nodes))]
            self_loops = ['item{} -> item{} [ label="{}"];'.format(i+1, i+1, "\n".join([self.nodes[i].color, 
                self.nodes[i].material])) for i in range(len(self.nodes))]
            edges = [['item{} -> item{} [ label={}];'.format(i+1, 1 + j, str(round(self.edges[i,j,-1],4))) for j in range(i+1,len(self.nodes))] for i in range(0,len(self.nodes))]
            
            f.write("digraph G {")
            f.write('\n')
            f.write('  ')
            f.write('\n  '.join([*items, *self_loops, *sum(edges, [])]))
            f.write('\n')
            f.write('}')
        render_graph('input.dot', path)

    def pairwise_euclidean(self):
        coordinates = np.asarray([normalize_coords(o.position_3d) for o in self.nodes])
        differences = coordinates[:, np.newaxis] - coordinates
        return np.linalg.norm(differences, axis=-1)

    def vectorize_edges(self):
        Gvec = SceneGraph(nodes=self.nodes, edges=self.edges)
        n_edges, n_items = len(Gvec.edges), len(Gvec.nodes)
        x_rel = np.zeros((n_edges, n_items, n_items), dtype=int)
        for rel_idx, (_, connections) in enumerate(Gvec.edges.items()):
            for i, target in enumerate(connections):
                x_rel[rel_idx, i, target] = 1
        Gvec.edges = x_rel.transpose((2, 1, 0)) # N x N x |R|
        # add an extra index for distances
        Gvec.edges = np.concatenate([Gvec.edges, self.pairwise_euclidean()[:,:,np.newaxis]], axis=-1)
        return Gvec

@dataclass
class SceneGraphVectorized:
    nodes: array # N x Dv 
    edges: array # N x N x |R|


class SceneGraphParserGrundTruth(ABC):
    def __init__(self, config_path: str, maps_csv: str):
        self.config_path = config_path
        with open(config_path) as f:
            self.cfg = json.load(f)

        self.maps = pd.read_table(maps_csv)
        self.color_map = {k:v for k,v in zip(self.maps['label'], self.maps['color'])}
        self.class_map = {k:v for k,v in zip(self.maps['label'], self.maps['class'])}
        self.material_map = {k:v for k,v in zip(self.maps['label'], self.maps['material'])}
        # use area (length * height) as size indicator
        self.size_map = {k:v for k,v in zip(self.maps["label"], 
            [reduce(mul, list(eval(d))[0::2], 1) for d in self.maps["dimensions"]])}
        self.special_map = {
            k: (v != '-') for k, v in zip(self.maps["label"], self.maps["special"])
        }
        
        self.behind_threshold = self.cfg['behind_threshold']
        self.next_threshold = self.cfg['next_threshold']
        self.size_threshold = self.cfg['size_threshold']
        self.use_3d = self.cfg['use_3d']
        self.vectorized = self.cfg['vectorized']
        self.function_map = {
            'left'   :   self.compare_left,
            'right'  :   self.compare_right,
            'behind' :   self.compare_behind,
            'front'  :   self.compare_front,
            'close'  :   self.compare_close,
            'far'    :   self.compare_far,
            'bigger' :   self.compare_bigger,
            'smaller':   self.compare_smaller,
            'next'   :   self.compare_next
            # 'closer' :   self.compare_closer,
            # 'further':   self.compare_further
        }

    def __call__(self, scene: Scene):
        if self.vectorized:
            return self.call_vectorized(scene)
        else:
            return self.call(scene)

    def compute_box(self, o: ObjectProto) -> Box:
        return Box(*cv2.boundingRect(o.contour))

    def compute_size(self, o: ObjectProto) -> float:
        # buggy
        return cv2.contourArea(o.contour) / reduce(mul, self.cfg['image_resolution'], 1)

    def compute_center(self, o: ObjectProto) -> Tuple[int, int]:
        M = cv2.moments(o.contour)
        return int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])

    def compute_rectangle(self, o: ObjectProto) -> Rectangle:
        pts = cv2.boxPoints(cv2.minAreaRect(o.contour))
        return Rectangle(*sum(pts.tolist(), []))

    def call(self, scene: Scene) -> SceneGraph:
        num_objects = len(scene.objects)
        # FOR WASH-v1
        # nodes = [Object(label=o.label,
        #                  category=self.class_map[o.label],
        #                  RGB_center=list(self.compute_center(o)),
        #                  RGB_rectangle=[int(x) for x in list(self.compute_rectangle(o))],
        #                  RGB_box = list(self.compute_box(o)),
        #                  # TODO: replace with actual 3d data from newest data gen
        #                  rotation=None,
        #                  #size=self.compute_size(o),
        #                  size=self.size_map[o.label],
        #                  position_3d=[*o.position_2d, 0., *([None] * 3)],
        #                  material=self.material_map[o.label],
        #                  color=self.color_map[o.label],
        #                  special=self.special_map[o.label] 
        #                  ) 
        #     for o in scene.objects]

        # FOR GD
        nodes = [Object(label=o.label,
                         category=self.class_map[o.label],
                         RGB_center=list(self.compute_center(o)),
                         RGB_rectangle=[int(x) for x in list(self.compute_rectangle(o))],
                         RGB_box = list(self.compute_box(o)),
                         orientation=list(o.pose_3d)[3:],
                         #size=self.compute_size(o),
                         size=self.size_map[o.label],
                         position_3d=list(o.pose_3d)[0:3],
                         material=self.material_map[o.label],
                         color=self.color_map[o.label],
                         special=self.special_map[o.label] 
                         ) 
            for o in scene.objects]

        #edges = np.empty((num_objects, num_objects, len(self.spatial_concepts)))
        edges = {k: [] for k in self.function_map.keys()}

        for spt_conc, _fn in self.function_map.items():
            for i, o in enumerate(nodes):
                _current = []
                for j in range(num_objects):
                    if i == j:
                        continue
                    if _fn(o, nodes[j]):
                        _current.append(j)
                edges[spt_conc].append(_current)

        # manually add hyper-edge relations
        edges = {**edges, **self.compute_hyper_edges(nodes)}

        return AnnotatedScene(environment=scene.environment,
                              image_id=scene.image_id,
                              objects=nodes,
                              relationships=edges,
                              )

    def call_vectorized(self, scene: Scene) -> SceneGraphVectorized:
        raise NotImplementedError

    def compare_right(self, o1: Object, o2: Object) -> bool:
        return True if self.right(o1) < self.left(o2) else False 

    def compare_left(self, o1: Object, o2: Object) -> bool:
        return True if self.left(o1) > self.right(o2) else False       

    def compare_far(self, o1: Object, o2: Object) -> bool:
        #return True if self.top(o1) > self.bottom(o2) else False  
        return True if o1.RGB_box[1] > o2.RGB_box[1] else False

    def compare_close(self, o1: Object, o2: Object) -> bool:
        #return True if self.bottom(o1) < self.top(o2) else False     
        return True if o1.RGB_box[1] < o2.RGB_box[1] else False

    def compare_front(self, o1: Object, o2: Object) -> bool:
        (x1, y1), (x2, y2) = o1.RGB_center, o2.RGB_center
        overlap = self.right(o1) >= self.left(o2) if x1<=x2 else self.right(o2) >= self.left(o1)
        return True if (overlap and y1<y2 and self.top(o2)-self.bottom(o1)<=self.behind_threshold) else False
    
    def compare_behind(self, o1: Object, o2: Object) -> bool:
        (x1, y1), (x2, y2) = o1.RGB_center, o2.RGB_center
        overlap = self.right(o1) >= self.left(o2) if x1<=x2 else self.right(o2) >= self.left(o1)
        return True if (overlap and y1>y2 and self.top(o1)-self.bottom(o2)<=self.behind_threshold) else False

    def compare_smaller(self, o1: Object, o2: Object) -> bool:
        # if self.use_3d:
        #     dA = (o1.position_3d.l * o1.position_3d.w * o1.position_3d.h) - \
        #          (o2.position_3d.l * o2.position_3d.w * o2.position_3d.h)
        #     dA /= reduce(mul, self.cfg['workspace_dimensions'], 1)
        # else:
        #     dA = o1.size - o2.size
        
        #return True if o1.size - o2.size > self.size_threshold else False   
        return o1.size > o2.size

    def compare_bigger(self, o1: Object, o2: Object) -> bool:
        # if self.use_3d:
        #     dA = (o1.position_3d.l * o1.position_3d.w * o1.position_3d.h) - \
        #          (o2.position_3d.l * o2.position_3d.w * o2.position_3d.h)
        #     dA /= reduce(mul, self.cfg['workspace_dimensions'], 1)
        # else:
        #     dA = o1.size - o2.size
        
        #return True if o1.size - o2.size < - self.size_threshold else False       
        return o1.size < o2.size 

    def compare_next(self, o1: Object, o2: Object) -> bool:
        dist = self.distance(o1, o2)
        return True if dist <= self.next_threshold else False     

    def compute_hyper_edges(self, nodes: List[Object]) -> Dict[str, List[int]]:
        N = len(nodes)
        centers = array([normalize_coords(o.position_3d[0:3]) for o in nodes])
        dst = (lambda x: np.sqrt(np.square(x).sum(1)))(np.expand_dims(centers, 2) - centers.T).T
        dst_tile = (np.tile(np.expand_dims(dst, 1), (1,N,1)) - np.tile(np.expand_dims(dst, 0), (N, 1, 1)))
        closer = (dst_tile < 0).astype(np.int32)
        further = (dst_tile > 0).astype(np.int32)
         
        # un-vectorize and squeeze to desired list notation
        closer = [
            [
                [idx for idx, val in enumerate(b) if val and idx not in [i, j]]
                for j, b in enumerate(a)
            ]
            for i, a in enumerate(closer.tolist())
        ]

        further = [
            [
                [idx for idx, val in enumerate(b) if val and idx not in [i, j]]
                for j, b in enumerate(a)
            ]
            for i, a in enumerate(further.tolist())
        ]
        return {'closer': closer, 'further': further}

    def left(self, o: Object) -> int:
        return min([x for x in o.RGB_rectangle[0::2]])

    def right(self, o: Object) -> int:
        return max([x for x in o.RGB_rectangle[0::2]])

    def top(self, o: Object) -> int:
        return min([x for x in o.RGB_rectangle[1::2]])

    def bottom(self, o: Object) -> int:
        return max([x for x in o.RGB_rectangle[1::2]])

    def distance(self, o1: Object, o2: Object) -> float:
        if self.use_3d:
            x1, y1, z1 = normalize_coords(o1.position_3d)
            x2, y2, z2 = normalize_coords(o2.position_3d)
            dst = np.linalg.norm([z2 - z1, y2 - y1, x2 - x1]) 
            # normalize_factor = np.linalg.norm(self.cfg['workspace_dimensions'])  
        else:
            dst =  np.linalg.norm([o2.RGB_center[1] - o1.RGB_center[1],
                                  o2.RGB_center[0] - o1.RGB_center[0]
                                 ])
            normalize_factor = np.linalg.norm(self.cfg['image_resolution'])
            dst /= normalize_factor
        return dst 

    def parse_dataset(self, dataset: List[Scene]) -> List[AnnotatedScene]:
        return list(map(self.__call__, dataset))

    def parse_dataset_dict(self, 
                           dataset: List[Scene], 
                           info: Maybe[Dict[str, Any]] = None,
                           save: Maybe[str] = None) -> Dict[str, Any]:
        graphs = self.parse_dataset(dataset)
        data = {'info': info,
                'scenes':[
                    {
                        'image_index': dataset[i].image_id, 
                        'objects': [o.__dict__ for o in g.objects],
                        'relationships': g.relationships,
                        'image_filename': str(dataset[i].image_id) + '.png',
                        'split': info['split'] if info is not None else None,
                        'directions': None
                    }
                for i, g in enumerate(graphs)]
            }
        if save is not None:
            with open(save, 'w') as f:
                json.dump(data, f)
        return data



class SceneGraphParser(ABC):
    def __init__(self, cfg: Dict[str, Any], maps: Dict[str, Any]):
        self.cfg = cfg
        self.maps = maps
        self.color_map = {int(k):v for k,v in self.maps['color'].items()}
        self.material_map = {int(k):v for k,v in self.maps['material'].items()}
        self.category_map = {int(k):v for k,v in self.maps['category'].items()}

        self.behind_threshold = self.cfg['behind_threshold']
        self.next_threshold = self.cfg['next_threshold']
        self.size_threshold = self.cfg['size_threshold']
        self.use_3d = self.cfg['use_3d']
        self.vectorized = self.cfg['vectorized']
        self.function_map = {
            'left'   :   self.compare_left,
            'right'  :   self.compare_right,
            'behind' :   self.compare_behind,
            'front'  :   self.compare_front,
            'close'  :   self.compare_close,
            'far'    :   self.compare_far,
            'bigger' :   self.compare_bigger,
            'smaller':   self.compare_smaller,
            'next'   :   self.compare_next
            # 'closer' :   self.compare_closer,
            # 'further':   self.compare_further
        }

    def __call__(self, *args, **kwargs):
        if self.vectorized:
            return self.call_vectorized(*args, **kwargs)
        else:
            return self.call(*args, **kwargs)

    def compute_box(self, contour: array) -> Box:
        return Box(*cv2.boundingRect(contour))

    def compute_size(self, contour: array) -> float:
        # buggy
        return cv2.contourArea(contour) / reduce(mul, self.cfg['image_resolution'], 1)

    def compute_center(self, contour: array) -> Tuple[int, int]:
        M = cv2.moments(contour)
        return int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])

    def compute_rectangle(self, contour: array) -> Rectangle:
        pts = cv2.boxPoints(cv2.minAreaRect(contour))
        return Rectangle(*sum(pts.tolist(), []))

    def call(self, 
            object_attributes: List[Dict[str, int]],
            object_contours: List[array]
            ) -> SceneGraph:
        assert len(object_attributes) == len(object_contours)
        num_objects = len(object_attributes)

        nodes = [Object(label=None,
                         category=self.category_map[att['category']],
                         RGB_center=list(self.compute_center(contour)),
                         RGB_rectangle=[int(x) for x in list(self.compute_rectangle(contour))],
                         RGB_box = list(self.compute_box(contour)),
                         # TODO: replace with actual 3d data from newest data gen
                         orientation=att['orientation'],
                         #size=self.compute_size(o),
                         size=att['size'],
                         position_3d=att['position_3d'],
                         material=self.material_map[att['material']],
                         color=self.color_map[att['color']],
                         special=False 
                         ) 
            for att, contour in zip(object_attributes, object_contours)]

        #edges = np.empty((num_objects, num_objects, len(self.spatial_concepts)))
        edges = {k: [] for k in self.function_map.keys()}

        for spt_conc, _fn in self.function_map.items():
            for i, o in enumerate(nodes):
                _current = []
                for j in range(num_objects):
                    if i == j:
                        continue
                    if _fn(o, nodes[j]):
                        _current.append(j)
                edges[spt_conc].append(_current)

        # manually add hyper-edge relations
        edges = {**edges, **self.compute_hyper_edges(nodes)}

        return SceneGraph(nodes=[o.__dict__ for o in nodes], edges=edges)

    def call_vectorized(self, scene: Scene) -> SceneGraphVectorized:
        raise NotImplementedError

    def compare_right(self, o1: Object, o2: Object) -> bool:
        return True if self.right(o1) < self.left(o2) else False 

    def compare_left(self, o1: Object, o2: Object) -> bool:
        return True if self.left(o1) > self.right(o2) else False       

    def compare_far(self, o1: Object, o2: Object) -> bool:
        return True if o1.RGB_center[1] > o2.RGB_center[1] else False  

    def compare_close(self, o1: Object, o2: Object) -> bool:
        return True if o1.RGB_center[1] < o2.RGB_center[1] else False     

    def compare_front(self, o1: Object, o2: Object) -> bool:
        (x1, y1), (x2, y2) = o1.RGB_center, o2.RGB_center
        overlap = self.right(o1) >= self.left(o2) if x1<=x2 else self.right(o2) >= self.left(o1)
        return True if (overlap and y1<y2 and self.top(o2)-self.bottom(o1)<=self.behind_threshold) else False
    
    def compare_behind(self, o1: Object, o2: Object) -> bool:
        (x1, y1), (x2, y2) = o1.RGB_center, o2.RGB_center
        overlap = self.right(o1) >= self.left(o2) if x1<=x2 else self.right(o2) >= self.left(o1)
        return True if (overlap and y1>y2 and self.top(o1)-self.bottom(o2)<=self.behind_threshold) else False

    def compare_smaller(self, o1: Object, o2: Object) -> bool:
        # if self.use_3d:
        #     dA = (o1.position_3d.l * o1.position_3d.w * o1.position_3d.h) - \
        #          (o2.position_3d.l * o2.position_3d.w * o2.position_3d.h)
        #     dA /= reduce(mul, self.cfg['workspace_dimensions'], 1)
        # else:
        #     dA = o1.size - o2.size
        return True if o1.size - o2.size > self.size_threshold else False   

    def compare_bigger(self, o1: Object, o2: Object) -> bool:
        # if self.use_3d:
        #     dA = (o1.position_3d.l * o1.position_3d.w * o1.position_3d.h) - \
        #          (o2.position_3d.l * o2.position_3d.w * o2.position_3d.h)
        #     dA /= reduce(mul, self.cfg['workspace_dimensions'], 1)
        # else:
        #     dA = o1.size - o2.size
        return True if o1.size - o2.size < - self.size_threshold else False       

    def compare_next(self, o1: Object, o2: Object) -> bool:
        dist = self.distance(o1, o2)
        return True if dist <= self.next_threshold else False     

    def compute_hyper_edges(self, nodes: List[Object]) -> Dict[str, List[int]]:
        N = len(nodes)
        centers = array([o.position_3d[0:2] for o in nodes])
        normalize_factor = np.linalg.norm(array(self.cfg['workspace_dimensions'][0:2]))
        dst = (lambda x: np.sqrt(np.square(x).sum(1)))(np.expand_dims(centers, 2) - centers.T).T / normalize_factor
        closer = (np.tile(np.expand_dims(dst, 1), (1,N,1)) - np.tile(np.expand_dims(dst, 0), (N, 1, 1)) < 0).astype(np.int32)
        further = (1 - closer).astype(np.int32)

        # un-vectorize and squeeze to desired list notation
        closer = [
            [
                [idx for idx, val in enumerate(b) if val and idx not in [i, j]]
                for j, b in enumerate(a)
            ]
            for i, a in enumerate(closer.tolist())
        ]

        further = [
            [
                [idx for idx, val in enumerate(b) if val and idx not in [i, j]]
                for j, b in enumerate(a)
            ]
            for i, a in enumerate(further.tolist())
        ]
        return {'closer': closer, 'further': further}

    def left(self, o: Object) -> int:
        return min([x for x in o.RGB_rectangle[0::2]])

    def right(self, o: Object) -> int:
        return max([x for x in o.RGB_rectangle[0::2]])

    def top(self, o: Object) -> int:
        return min([x for x in o.RGB_rectangle[1::2]])

    def bottom(self, o: Object) -> int:
        return max([x for x in o.RGB_rectangle[1::2]])

    def distance(self, o1: Object, o2: Object) -> float:
        if self.use_3d:
            (x1, y1, z1, _, _, _) = o1.position_3d
            (x2, y2, z2, _, _, _) = o2.position_3d
            dst = np.linalg.norm([z2 - z1, y2 - y1, x2 - x1]) 
            normalize_factor = np.linalg.norm(self.cfg['workspace_dimensions'])  
        else:
            dst =  np.linalg.norm([o2.RGB_center[1] - o1.RGB_center[1],
                                  o2.RGB_center[0] - o1.RGB_center[0]
                                 ])
            normalize_factor = np.linalg.norm(self.cfg['image_resolution'])
        return dst / normalize_factor