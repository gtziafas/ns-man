from ns_man.structs import *

import random
import json
from collections import Counter


def invert_listed_dict(d: Dict[str, List[Any]]):
    d_inv = {}
    for k, v in d.items():
        for w in v:
            d_inv[w] = k
    return d_inv


class HOTSExecutor:
    """Symbolic program executor for RGBD-Objects"""

    def __init__(self, 
                 metadata: Dict[str, Any], 
                 synonyms: Dict[str, Any], 
                 train_scene_json: str=None, 
                 val_scene_json: str=None,
                 with_synonyms: bool=True
    ):
        if train_scene_json is not None and val_scene_json is not None:        
            self.scenes = {
                'train': json.load(open(train_scene_json))['scenes'],
                'val': json.load(open(val_scene_json))['scenes']
            }
            for split in self.scenes.keys():
                self.scenes[split] = {s["image_index"]: s for s in self.scenes[split]}

        # #self.vocab = utils.load_vocab(vocab_json)
        #self.vocab = json.load(open(vocab_json))
        self.metadata = json.load(open(metadata))
        self.synonyms = json.load(open(synonyms))
        self._setup_vocab()
        self.with_synonyms = with_synonyms
        self.answer_candidates = {
            'count': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
            'equal_color': [True, False],
            'equal_integer': [True, False],
            'equal_material': [True, False],
            'equal_category': [True, False],
            #'equal_size': [True, False],
            'exist': [True, False],
            'greater_than': [True, False],
            'less_than': [True, False],
            'query_color': self.colors,
            'query_material': self.materials,
            #'query_size': ['small', 'large'],
            'query_category': self.categories,
            'same_color': [True, False],
            'same_material': [True, False],
            #'same_size': [True, False],
            'same_category': [True, False]
        }

        self.modules = {}
        self._register_modules()

    def _setup_vocab(self):
        self.colors = self.metadata['types']['Color']
        self.color_synonyms = {k:v for k,v in self.synonyms.items() if k in self.colors}
        self.materials = self.metadata['types']['Material']
        self.materials_synonyms = {k:v for k,v in self.synonyms.items() if k in self.materials}
        self.categories = self.metadata['types']['Category']
        self.categories_synonyms = {k:v for k,v in self.synonyms.items() if k in self.categories}
        self.spatial_concepts = self.metadata['types']['Relation']
        self.spatial_synonyms = {k:v for k,v in self.synonyms.items() if k in self.spatial_concepts}
        self.spatial_synonyms_inv = invert_listed_dict(self.spatial_synonyms)
        self.location_concepts = self.metadata['types']['Location']
        self.location_synonyms = {k:v for k,v in self.synonyms.items() if k in self.location_concepts}
        self.location_synonyms_inv = invert_listed_dict(self.location_synonyms)
        self.hyper_relations = self.metadata['types']['HyperRelation']
        self.hyper_relations_synonyms = {k:v for k,v in self.synonyms.items() if k in self.hyper_relations}
        self.hyper_relations_synonyms_inv = invert_listed_dict(self.hyper_relations_synonyms)
        self.special_tags = self.metadata['types']['Special']
        self.special_synonyms = {k:v for k,v in self.synonyms.items() if k in self.special_tags}
        self.all_color_words = list(set(sum(list(self.color_synonyms.values()), []) + self.colors))
        self.all_material_words = list(set(sum(list(self.materials_synonyms.values()), []) + self.materials))
        self.all_category_words = list(set(sum(list(self.categories_synonyms.values()), []) + self.categories))
        #self.all_category_words = self.all_category_words + [w+'s' for w in self.all_category_words]
        self.all_spatial_words = list(set(sum(list(self.spatial_synonyms.values()), []) + self.spatial_concepts))
        self.all_location_words = list(set(sum(list(self.location_synonyms.values()), []) + self.location_concepts))
        self.all_hyper_relation_words = list(set(sum(list(self.hyper_relations_synonyms.values()), []) + self.hyper_relations))
        self.all_special_words = sum(list(self.special_synonyms.values()), [])
        #self.all_special_words = self.all_special_words + [w+'s' for w in self.all_special_words]
        self.attribute_concepts = ["category", "color", "material"]

    def run(self, 
            program: List[str], 
            scene_graph: Dict[str, Any],
            guess: bool=False, 
            debug: bool=False
            ):
        # init answer and current node
        ans, temp = None, None

        # Lookups
        self.objects = scene_graph['objects']
        self.relationships = scene_graph['relationships']

        self.exe_trace = []
        for token in program:
            if token == 'scene':
                if temp is not None:
                    ans = 'error'
                    break
                temp = ans
                ans = list(range(len(self.objects)))
            elif token in self.modules:
                module = self.modules[token]
                if token.startswith('same') or token.startswith('relate'):
                    ans = module(ans, list(range(len(self.objects))))
                else:
                    ans = module(ans, temp)
                if ans == 'error':
                    break
            self.exe_trace.append([token, str(ans)])
            if debug:
                print(token)
                print('ans:')
                self._print_debug_message(ans)
                print('temp: ')
                self._print_debug_message(temp)
                print()
        ans = str(ans)

        # if ans == 'error' and guess:
        #     final_module = self.vocab['prog2id'][program[0]]
        #     if final_module in self.answer_candidates:
        #         ans = random.choice(self.answer_candidates[final_module])
        return ans

    def run_dataset(self, program: List[str], index: int, split: str, guess=False, debug=False):
        assert self.modules and self.scenes, 'Must have scene annotations and define modules first'
        assert split == 'train' or split == 'val'
        scene = self.scenes[split][index]
        return self.run(program, scene, guess, debug)

    def _print_debug_message(self, program):
        if type(program) == list:
            for o in program:
                print(self._object_info(o))
        elif type(program) == dict:
            print(self._object_info(program))
        else:
            print(program)

    def _object_info(self, obj):
        return '%s %s %s %s at %s' % (self.objects[obj]['category'], self.objects[obj]['color'], self.objects[obj]['material'], self.objects[obj]['size'], str(self.objects[obj]['RGB_center']))
    
    def _register_modules(self):
        self.modules['count'] = self.count
        self.modules['equal_color'] = self.equal_color
        self.modules['equal_integer'] = self.equal_integer
        self.modules['equal_material'] = self.equal_material
        self.modules['equal_category'] = self.equal_category
        # self.modules['equal_size'] = self.equal_size
        self.modules['exist'] = self.exist
        for color in (self.colors if not self.with_synonyms else self.all_color_words):
            self.modules[f'filter_color[{color}]'] = self.filter_color(color)
        for material in (self.materials if not self.with_synonyms else self.all_material_words):
            self.modules[f'filter_material[{material}]'] = self.filter_material(material)
        for category in (self.categories if not self.with_synonyms else self.all_category_words):
            self.modules[f'filter_category[{category}]'] = self.filter_category(category)
            self.modules[f'filter_category[{category}s]'] = self.filter_category(category)
        self.modules['greater_than'] = self.greater_than
        self.modules['less_than'] = self.less_than
        self.modules['intersect'] = self.intersect
        for attribute in self.attribute_concepts:
            self.modules[f'query_{attribute}'] = self.query(attribute)
            self.modules[f'same_{attribute}'] = self.same(attribute)
        for relation in (self.spatial_concepts if not self.with_synonyms else self.all_spatial_words):
            self.modules[f'relate[{relation}]'] = self.relate(relation)
        for relation in (self.location_concepts if not self.with_synonyms else self.all_location_words):
            self.modules[f'locate[{relation}]'] = self.locate(relation)
        for relation in (self.hyper_relations if not self.with_synonyms else self.all_hyper_relation_words):
            self.modules[f'hyper_relate[{relation}]'] = self.hyper_relate(relation)
        for special_word in (self.special_tags if not self.with_synonyms else self.all_special_words):
            fn = "ground" if not self.with_synonyms else "filter_category"
            _unique = True if not self.with_synonyms else False
            self.modules[f'{fn}[{special_word}]'] = self.ground(special_word, _unique)
            self.modules[f'{fn}[{special_word}s]'] = self.ground(special_word, _unique)
        self.modules['union'] = self.union
        self.modules['unique'] = self.unique

    def count(self, scene, _):
        if type(scene) == list:
            return len(scene)
        return 'error'
    
    def equal_color(self, color1, color2):
        if type(color1) == str and color1 in self.colors and type(color2) == str and color2 in self.colors:
            if color1 == color2:
                return True
            else:
                return False
        return 'error'
    
    def equal_integer(self, integer1, integer2):
        if type(integer1) == int and type(integer2) == int:
            if integer1 == integer2:
                return True
            else:
                return False
        return 'error'
    
    def equal_material(self, material1, material2):
        if type(material1) == str and material1 in self.materials and type(material2) == str and material2 in self.materials:
            if material1 == material2:
                return True
            else:
                return False
        return 'error'
    
    def equal_category(self, category1, category2):
        if type(category1) == str and category1 in self.categories and type(category2) == str and category2 in self.categories:
            if category1 == category2:
                return True
            else:
                return False
        return 'error'
    
    def exist(self, scene, _):
        if type(scene) == list:
            if len(scene) != 0:
                return True
            else:
                return False
        return 'error'

    def filter_color(self, color):
        if color in self.colors:
            condition = lambda o: self.objects[o]['color'] == color
        elif color in self.all_color_words:
            condition = lambda o: color in self.color_synonyms[self.objects[o]['color']]
        else:
            print(f'Unknown color {color}') 
        def _filter_color(scene, _):
            if type(scene) == list:
                output = []
                for o in scene:
                    if condition(o):
                        output.append(o)
                return output
            return 'error'  
        return _filter_color    
    
    def filter_material(self, material):
        if material in self.materials:
            condition = lambda o: self.objects[o]['material'] == material
        elif material in self.all_material_words:
            condition = lambda o: material in self.materials_synonyms[self.objects[o]['material']]
        else:
            print(f'Unknown material {material}') 
        def _filter_material(scene, _):
            if type(scene) == list:
                output = []
                for o in scene:
                    if condition(o):
                        output.append(o)
                return output
            return 'error'  
        return _filter_material

    def filter_category(self, category):
        # if category[-1] == "s" and category[:-1] in self.all_category_words:
        #     category = category[:-1] # plural
        #     condition = lambda o: self.objects[o]['category'] == category
        if category in self.categories:
            condition = lambda o: self.objects[o]['category'] == category
        elif category in self.all_category_words:
            condition = lambda o: category in self.categories_synonyms[self.objects[o]['category']]
        else:
            print(f'Unknown category {category}') 
        def _filter_category(scene, _):
            if type(scene) == list:
                output = []
                for o in scene:
                    if condition(o):
                        output.append(o)
                return output
            return 'error'  
        return _filter_category

    def ground(self, word, unique=True):
        if not self.with_synonyms:
            assert word in self.special_tags, f'Unknown special tag {word}'
            condition = lambda o: self.objects[o]['special'] and self.objects[o]['label'] == word
        else:
            assert word in self.all_special_words, f'Unknown special tag {word}'
            condition = lambda o: self.objects[o]['special'] and word in self.special_synonyms[self.objects[o]['label']]
        def _ground(scene, _):
            if type(scene) == list:
                output = []
                for o in scene:
                    if condition(o):
                        output.append(o)
                if len(output) == 1:
                    return output[0] if unique else output
            return 'error'
        return _ground

    def greater_than(self, integer1, integer2):
        if type(integer1) == int and type(integer2) == int:
            if integer1 < integer2:
                return True
            else:
                return False
        return 'error'
    
    def less_than(self, integer1, integer2):
        if type(integer1) == int and type(integer2) == int:
            if integer1 > integer2:
                return True
            else:
                return False
        return 'error'
    
    def intersect(self, scene1, scene2):
        if type(scene1) == list and type(scene2) == list:
            output = []
            for o in scene1:
                if o in scene2:
                    output.append(o)
            return output
        return 'error'
    
    def query(self, concept):
        assert concept in self.attribute_concepts, f'Unknown attribute {concept}'
        def _query_attribute(obj, _):
            if type(obj) == int:
                return self.objects[obj][concept]
            return 'error' 
        return _query_attribute

    def same(self, concept):
        assert concept in self.attribute_concepts, f'Unknown attribute {concept}'
        def _same_attribute(obj, scene):
            if type(obj) == int and type(scene) == list:
                output = []
                for o in scene:
                    if self.objects[o][concept] == self.objects[obj][concept] and o != obj:
                        output.append(o)
                return output
            return 'error'
        return _same_attribute

    def relate(self, concept):
        if not self.with_synonyms:
            assert concept in self.spatial_concepts, f'Unknown spatial concept {concept}'
        else:
            assert concept in self.all_spatial_words, f'Unknown spatial concept {concept}'
            concept = self.spatial_synonyms_inv[concept] if concept in self.spatial_synonyms_inv.keys() else concept
        def _relate(obj, _):
            if type(obj) == int:
                return self.relationships[concept][obj]
            return 'error'
        return _relate

    def hyper_relate(self, concept):
        if not self.with_synonyms:
            assert concept in self.hyper_relations, f'Unknown hyper-relation concept {concept}'
        else:
            assert concept in self.all_hyper_relation_words, f'Unknown hyper-relation concept {concept}'
            concept = self.hyper_relations_synonyms_inv[concept] if concept in self.hyper_relations_synonyms_inv.keys() else concept
        def _hyper_relate(obj1, obj2):
            if type(obj1) == int and type(obj2) == int:
                return self.relationships[concept][obj2][obj1]
            return 'error'
        return _hyper_relate

    def locate(self, concept):
        if not self.with_synonyms:
            assert concept in self.location_concepts, f'Unknown location concept {concept}'
        else:
            assert concept in self.all_location_words, f'Unknown location concept {concept}'
            concept = self.location_synonyms_inv[concept] if concept in self.location_synonyms_inv.keys() else concept
        assert concept[:-1] in self.spatial_concepts
        concept = concept[:-1]
        def _locate(scene, _):
            if type(scene) == list:
                output = []
                for o in scene:
                    output.extend(list(set(self.relationships[concept][o]) & set(scene)))
                if len(output) > 0:
                    return Counter(output).most_common()[0][0]        
            return 'error'
        return _locate
    
    def union(self, scene1, scene2):
        if type(scene1) == list and type(scene2) == list:
            output = list(scene2)
            for o in scene1:
                if o not in scene2:
                    output.append(o)
            return output
        return 'error'
    
    def unique(self, scene, _):
        if type(scene) == list and len(scene) > 0:
            return scene[0]
        return 'error'
