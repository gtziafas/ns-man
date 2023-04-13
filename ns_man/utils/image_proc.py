from ns_man.structs import * 

import cv2 
import numpy as np 

MORPHOLOGY_MAP = {
    'erode': cv2.MORPH_ERODE,
    'dilate': cv2.MORPH_DILATE,
    'open': cv2.MORPH_OPEN,
    'close': cv2.MORPH_CLOSE
}


def show(img: array, legend: Maybe[str] = None):
    legend = 'unlabeled' if legend is None else legend
    cv2.imshow(legend, img)
    while 1:
        key = cv2.waitKey(1) & 0xff
        if key == ord('q'):
            break
    cv2.destroyWindow(legend)


def show_many(imgs: List[array], legends: Maybe[List[str]] = None):
    assert len(imgs) == len(legends)
    legends = [l if l is not None else 'image' + str(i) for i, l in enumerate(legends)]
    print([f'{l}: {i.shape}'for l, i in zip(legends, imgs)])
    for i, l in zip(imgs, legends):
        cv2.imshow(l, i)
    while 1:
        key = cv2.waitKey(1) & 0xff
        if key == ord('q'):
            break
    destroy()


def compute_histograms(imgs: List[array]) -> List[array]:
    hists = []
    for c in imgs:
        channels = [c[...,0], c[...,1], c[...,2]]
        for i, ch in enumerate(channels):
                ch = ch.astype(float) / 0xff
                ch_hist = np.histogram(ch, bins=256, range=(0,1))[0]
                ch_hist = ch_hist / (img_size[0] * img_size[1])
                channels[i] = ch_hist
        hists.append(np.concatenate(channels))
    return hists 


def x1y1x2y2_to_xywh(box: Tuple[int, ...]) -> Box:
    # convert np-style to cv2-style coordinates for box
    y1, y2, x1, x2 = box
    return Box(x = x1, y = y1, w = x2 - x1, h = y2 - y1)


def crop_box(img: array, box: Box) -> array:
    return img[box.y : box.y + box.h, box.x : box.x + box.w]


def crop_rectangle(img: array, rect: Rectangle) -> array:
    rect = np.int0([(rect.x1, rect.y1), (rect.x2, rect.y2), (rect.x3, rect.y3), (rect.x4, rect.y4)])
    mask = np.zeros((img.shape[0], img.shape[1], 3))
    mask = cv2.drawContours(mask, [rect], 0, (0xff,0xff,0xff), -1)
    img[mask==0] = 0
    box = cv2.boundingRect(rect)
    return crop_box(img, Box(*box))


def box_center(box: Box) -> Tuple[int, int]:
    return box.x + box.w // 2, box.y + box.h // 2


def crop_contour(img: array, contour: array, fill_with: int=0) -> array:
    mask = np.zeros((img.shape[0], img.shape[1], 3))
    mask = cv2.drawContours(mask, [contour], 0, (0xff,0xff,0xff), -1)
    mask = cv2.drawContours(mask, [contour], 0, (0,0,0), 3)
    box = cv2.boundingRect(contour)
    mask = np.where(mask == 0xff, img, fill_with)
    return crop_box(mask, Box(*box))


def threshold(img: array) -> array:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 0xff, 0)
    return thresh 


def depth_eval(pt: Tuple[int, int], depth: array, ws: int = 40) -> float:
    window =  crop_box(depth, Box(x=pt[0], y=pt[1], w=ws, h=ws))
    return window[window>0].mean()


# pad image with zeros in the center of a desired resolution frame
def pad_with_frame(imgs: List[array], desired_shape: Tuple[int, int], fill_with: int=0) -> List[array]:
    H, W = desired_shape
    
    def _pad_with_frame(img: array) -> array:
        # construct a frame of desired resolution
        frame = np.ones((H, W, 3), dtype=np.uint8) * fill_with

        # paste image in the center of the frame
        starty, startx = max(0, (H - img.shape[0]) // 2), max(0, (W - img.shape[1]) // 2)
        frame[starty : starty + min(H, img.shape[0]), startx :  startx + min(W, img.shape[1]), :] = img
        return frame

    return list(map(_pad_with_frame, imgs))


def filter_large(desired_shape: Tuple[int, int]) -> Callable[[List[array]], List[array]]:
    H, W = desired_shape

    def _filter_large(imgs: List[array]) -> List[array]: 
        # identify images larger than desired resolution
        large_idces = [idx for idx, i in enumerate(imgs) if i.shape[0] > H or i.shape[1] > W]

        # crop tight boxes for that imges
        cropped_imgs = crop_boxes_dynamic([imgs[idx] for idx in large_idces])

        # return all thresholded and inverted and large properly replaced
        return [thresh_invert(img) if i not in large_idces else cropped_imgs[large_idces.index(i)] for i, img in enumerate(imgs)]
    
    return _filter_large


def crop_boxes_dynamic(imgs: List[array]) -> List[array]:
    # find contours for each image
    contours = [cv2.findContours(i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] for i in imgs]
    
    # sort by area and keep largest
    contours = [sorted(conts, key=lambda c: cv2.contourArea(c), reverse=True)[0] for conts in contours]
    
    # compute largests contour bounding box
    boxes = list(map(cv2.boundingRect, contours))

    # crop tight box
    return [crop_box(i, Box(*b)) for i, b in zip(imgs, boxes)]


def crop_boxes_fixed(desired_shape: Tuple[int, int], fill_with: int=0) -> Callable[[List[array]], List[array]]:
    H, W = desired_shape

    def _crop_boxes_fixed(imgs: List[array]) -> List[array]:
        imgs = list(imgs)
        
        # identify images larger than desired resolution
        large_idces = [idx for idx, i in enumerate(imgs) if i.shape[0] > H or i.shape[1] > W]

        # identify maximum dimension of large images
        large_maxdims = [max(imgs[i].shape[0:2]) for i in large_idces]

        for idx, img in enumerate(imgs):
            # if large, resize to desired shape while maintaining original ratio
            if idx in large_idces:
                maxdim = large_maxdims[large_idces.index(idx)]
                img = pad_with_frame([img], (maxdim, maxdim), fill_with)[0]
                imgs[idx] = cv2.resize(img, (H, W))
            # if small, pad to desired shape
            else:
                imgs[idx] = pad_with_frame([img], (H, W), fill_with)[0]

        return imgs
    
    return _crop_boxes_fixed


def destroy():
    cv2.destroyAllWindows()