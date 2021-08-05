import os
import json
import shutil

if __name__ == "__main__":
    print('helloo')

def get_all_files(root_folder, ext):
    # Create a list of file and subdirectories names in the given directory 

    file_list = os.listdir(root_folder)
    all_files = []

    # Iterate over all the entries
    for entry in file_list:
        filepath = os.path.join(root_folder, entry)
        filepath = os.path.abspath(filepath)

        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(filepath):
            all_files = all_files + get_file_list(filepath, ext)
        elif filepath.endswith(ext):
            all_files.append(filepath)
                
    return all_files

def get_file_list(folder, ext):
    # create a list of file and subdirectories names in the given directory 
    file_list = os.listdir(folder)
    all_files = []

    # Iterate over all the entries
    for entry in file_list:
            fullPath = os.path.join(folder, entry)
            # If entry is a directory then get the list of files in this directory 
            if os.path.isdir(fullPath):
                all_files = all_files + get_file_list(fullPath, ext)
            elif entry.endswith(ext):
                all_files.append(fullPath)
                
    return all_files 

def convert_via_to_txt_annotations(via_folder, dst_folder):
    [result_dict, file_paths] = load_json_via_all(via_folder)
    
    id = 0

    for file_path in file_paths:
        patient_result = result_dict[file_path]
        patient_txtfile = str(id).zfill(7) + ".txt"
        patient_txtfile = os.path.join(dst_folder, patient_txtfile)

        with open(patient_txtfile, 'w+') as txtfile:
            for region in patient_result:
                out_line = "{}  [{}, {}, {}, {}]\n".format(region[4], region[0], region[1], region[2], region[3])
                txtfile.write(out_line)

        id += 1

        
def load_json_via_all(folder, change_to_ext = "", annotation_ext = 'json'):
    result_dict = {}
    filepath_list = get_all_files(folder, annotation_ext)
    complete_path_list = []
    
    for filepath in filepath_list:         
        img_annotations = load_json_via(filepath)

        sorted_path_list = sorted(img_annotations.items())
        sorted_path_list = [x[0] for x in sorted_path_list]
        complete_path_list.extend(sorted_path_list)

        for img_path in img_annotations:
            key = img_path

            if change_to_ext != "":
                base = os.path.splitext(img_path)[0]
                key = base + "." + change_to_ext

            result_dict[key] = img_annotations[img_path]  

    return [result_dict, complete_path_list]


def load_json_via(json_file):
    # Multiple image annotations can be stored in a single file.

    with open(json_file, 'r') as f:
        via_json_file = json.load(f)

    regions_dict = {}     
    base_folder = json_file.replace('.json', '') + "\\"

    for via_img_metadata in via_json_file['_via_img_metadata'].items():
        filename = base_folder + via_img_metadata[1]['filename']
        filename = os.path.abspath(filename)
        regions_dict[filename] = []

        for region in via_img_metadata[1]['regions']:
            category = region['region_attributes']['Class']
            x = region['shape_attributes']['x']
            y = region['shape_attributes']['y']
            w = region['shape_attributes']['width']
            h = region['shape_attributes']['height']

            regions_dict[filename].append([x, y, w, h, category])

    return regions_dict


def load_txt_annotations_multiple(folder):
    filepath_list = get_all_files(folder, 'txt')
    
    result_dict = {}

    for filepath in filepath_list:
        filename = os.path.abspath(filepath)
        #filename = filename.replace('txt', img_extension)
        
        #if file_list and filename not in file_list:
        #    continue # Don't add filename to dictionary.

        img_annotations = load_txt_annotation(filepath)       
        result_dict[filename] = img_annotations

    return result_dict


def load_txt_annotation(txt_file):
    lines = [l.strip() for l in open(txt_file, 'r').readlines()]

    regions = []
    for l in lines:
        line_split = l.split('[')
        category = line_split[0].strip()
        [x, y, w, h] = ([int(float(n)) for n in line_split[1].split(']')[0].split(', ')])

        regions.append([x, y, w, h, category])
    
    return regions    
  
