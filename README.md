# (Unfinished) Beverage Detection Model

Author: Jose Manuel G. Dolot

Made as a requirement for EEE 197Z

### What is in this repo
Within this repo are primarily the python files related to processing Datasets-- I primarily worked with the notebook file so feel free to look through it.

### Why did I not finish?
The major problem encountered was finding a way to convert the given dataset from *VGG Image Annotation format* to *COCO format*. I explored and researched a lot of ways to do so which all proved to be challenging since the implementation of this specific dataset is quite different than other references. Aside from using VIA, this dataset was processed through wandb functions, unlike the provided PyTorch Object Detection Tutorial and other references online which showed a fairly direct way (through torch.utils.data, dictionary functions, etc.) of processing datasets.

### The goal is to be able to make a dictionary following this format:

```
{'imagefilename.jpg':
    {
    "boxes":[xmin, ymin, xmax, ymax],
     "labels":class_id,
     "image_id":image_id
     "area":[(xmax-xmin) * (ymax-ymin)]
     "iscrowd":0
     }
]
```

What we have right now is something like this (based on test_dict and train_dict):

```
{ 'path/to/imagefilename.jpg': [xmin, xmax, ymin, ymax, class_id]}
```

### On the bright side, I was able to accomplish making that dictionary
After exploring and sifting through different solutions such as converting the whole "drinks" dataset to COCO through Roboflow, sorting the dataset folder itself, exploring the annotation apps in the VIA website, and a lot more, I realized I hadn't studied the actual demo files provided in this course-- to be specific, the `label_utils.py`. With the knowledge I gained from researching around the web, I edited two functions within the helper module:

```
def get_label_dictionary_coco(labels, keys):
    """Associate key (filename) to value (box coords, class)"""
    dictionary = {}
    for key in keys:
        dictionary[key] = [] # empty boxes

    for label in labels:
        if len(label) != 6:
            print("Incomplete label:", label[0])
            continue

        value = label[1:]

        if value[0]==value[1]:
            continue
        if value[2]==value[3]:
            continue

        if label[-1]==0:
            print("No object labelled as bg:", label[0])
            continue

        # box coords are float32
        value = value.astype(np.float32)
        target = {}
        bndboxes = [value[0], value[2], value[1], value[3]]
        target["boxes"] = [bndboxes]
        target["labels"] = [value[4]]
        target["image_id"] = int(label[0][:-4])
        target["area"] = [(value[1]-value[0])*(value[3]-value[2])]
        target["iscrowd"] = [0]
        # filename is key
        key = label[0]
        # boxes = bounding box coords and class label
        #boxes = dictionary[key]
        #boxes.append(value)
        dictionary[key] = target

    # remove dataset entries w/o labels
    for key in keys:
        if len(dictionary[key]) == 0:
            del dictionary[key]

    return dictionary

def build_label_dictionary_coco(path):
    """Build a dict with key=filename,
    value={
      "boxes":[xmin, ymin, xmax, ymax],
      "labels":class_id,
      #"masks":masks,
      "image_id":image_id
      "area":(xmax-xmin) * (ymax-ymin)
      "iscrowd":0
    }"""
    labels = load_csv(path)
    dir_path = os.path.dirname(path)
    # skip the 1st line header
    labels = labels[1:]
    # keys are filenames
    keys = np.unique(labels[:,0])
    #keys = [os.path.join(dir_path, key) for key in keys]
    dict = get_label_dictionary_coco(labels, keys) 
    #for key in dictionary.keys():
        #dict[key] = np.array(dictionary[key])
        #e.g. {0010069.jpg:{Key1:Value1, Key2:Value2, ...}}

    #classes = np.unique(labels[:,-1]).astype(int).tolist()
    # insert background label 0
    #classes.insert(0, 0)
    #print("Num of unique classes:: ", classes)
    return dict
```

This would output the required dictionary when called through:
```
test_dict_coco = label_utils.build_label_dictionary_coco(
    config['test_split'])
train_dict_coco = label_utils.build_label_dictionary_coco(
    config['train_split'])
```

###And that's it for my progress!
For posterity, I was initially planning to use Yolov5 as my object detection model.
