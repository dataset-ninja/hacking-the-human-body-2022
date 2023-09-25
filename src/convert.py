import supervisely as sly
import os
from collections import defaultdict
import csv
import numpy as np
from dataset_tools.convert import unpack_if_archive
import src.settings as s
from urllib.parse import unquote, urlparse
from supervisely.io.fs import get_file_name, get_file_name_with_ext

from tqdm import tqdm

def download_dataset(teamfiles_dir: str) -> str:
    """Use it for large datasets to convert them on the instance"""
    api = sly.Api.from_env()
    team_id = sly.env.team_id()
    storage_dir = sly.app.get_data_dir()

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, str):
        parsed_url = urlparse(s.DOWNLOAD_ORIGINAL_URL)
        file_name_with_ext = os.path.basename(parsed_url.path)
        file_name_with_ext = unquote(file_name_with_ext)

        sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
        local_path = os.path.join(storage_dir, file_name_with_ext)
        teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

        fsize = api.file.get_directory_size(team_id, teamfiles_dir)
        with tqdm(
            desc=f"Downloading '{file_name_with_ext}' to buffer...",
            total=fsize,
            unit="B",
            unit_scale=True,
        ) as pbar:        
            api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)
        dataset_path = unpack_if_archive(local_path)

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, dict):
        for file_name_with_ext, url in s.DOWNLOAD_ORIGINAL_URL.items():
            local_path = os.path.join(storage_dir, file_name_with_ext)
            teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

            if not os.path.exists(get_file_name(local_path)):
                fsize = api.file.get_directory_size(team_id, teamfiles_dir)
                with tqdm(
                    desc=f"Downloading '{file_name_with_ext}' to buffer...",
                    total=fsize,
                    unit="B",
                    unit_scale=True,
                ) as pbar:
                    api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)

                sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
                unpack_if_archive(local_path)
            else:
                sly.logger.info(
                    f"Archive '{file_name_with_ext}' was already unpacked to '{os.path.join(storage_dir, get_file_name(file_name_with_ext))}'. Skipping..."
                )

        dataset_path = storage_dir
    return dataset_path
    
def count_files(path, extension):
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                count += 1
    return count
    
def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    ### Function should read local dataset and upload it to Supervisely project, then return project info.###
    images_path = os.path.join("Hacking the Human Body","train_images")
    masks_path = os.path.join("Hacking the Human Body","train_mclass_masks")
    csv_test = os.path.join("Hacking the Human Body","test.csv")
    cvs_train = os.path.join("Hacking the Human Body","train.csv")
    batch_size = 5
    ds_name_train = "train"
    ds_name_test = "test"

    test_tags = defaultdict()
    train_tags = defaultdict()

    with open(csv_test, "r") as file:
            csvreader = csv.reader(file)
            for idx, row in enumerate(csvreader):
                if idx != 0:
                    test_tags[row[0]] = {"organ": row[1], "tissue_thickness" :row[5]}
                else:
                    tag_names_test = ["organ","tissue_thickness"]

    with open(cvs_train, "r") as file:
            csvreader = csv.reader(file)
            for idx, row in enumerate(csvreader):
                if idx != 0:
                    train_tags[row[0]] = {"tissue_thickness" :row[6],"age":row[8],"sex":row[9]}
                else:
                    tag_names_train = ["tissue_thickness","age","sex"]

    tag_dict = {"test":test_tags,"train":train_tags}




    def create_ann(image_path):
        labels = []
        tags = []

        image_name = get_file_name_with_ext(image_path)
        img_mp = sly.imaging.image.read(image_path)[:, :, 0]
        img_height = img_mp.shape[0]
        img_wight = img_mp.shape[1]
        image_name_nf = image_name.split('.')[0]
        tags_single = tag_dict[ds][image_name_nf]
        for key,value in tags_single.items():
            tag = [sly.Tag(tagmeta, value=value) for tagmeta in tag_metas if tagmeta.name == key]
            tags.extend(tag)

        mask_path = os.path.join(masks_path, image_name)
        if ds == "train":
            mask_np = sly.imaging.image.read(mask_path)[:, :, 0]
            

            pixels = np.unique(mask_np)[1:]
            for pixel in pixels:
                obj_class = pixel_to_class.get(pixel)
                mask = mask_np == pixel
                curr_bitmap = sly.Bitmap(mask)
                curr_label = sly.Label(curr_bitmap, obj_class)
                labels.append(curr_label)

        return sly.Annotation(img_size=(img_height, img_wight), labels=labels,img_tags=tags)


    prostate = sly.ObjClass("prostate", sly.Bitmap)
    spleen = sly.ObjClass("spleen", sly.Bitmap)
    lung = sly.ObjClass("lung", sly.Bitmap)
    kidney = sly.ObjClass("kidney", sly.Bitmap)
    largeintestine = sly.ObjClass("largeintestine", sly.Bitmap)

    pixel_to_class = {1: prostate, 2: spleen, 3: lung, 4: kidney, 5: largeintestine}

    tag_metas = [sly.TagMeta(name, sly.TagValueType.ANY_STRING) for name in ["organ","tissue_thickness","age","sex"]]

    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)
    meta = sly.ProjectMeta(obj_classes=[prostate, spleen, lung, kidney, largeintestine],tag_metas=tag_metas)
    api.project.update_meta(project.id, meta.to_json())


    images_names_train = os.listdir(images_path)
    image_test = os.listdir(os.path.join("Hacking the Human Body","test"))

    dataset_train = api.dataset.create(project.id, "train", change_name_if_conflict=True)
    dataset_test = api.dataset.create(project.id, "test", change_name_if_conflict=True)

    progress = sly.Progress("Create dataset {}".format("test"), len(images_names_train)+1)

    project_dict = {"test":image_test,"train":images_names_train}


    for ds in project_dict:
        if ds == "test":
            dataset = dataset_test
            images_path = os.path.join("Hacking the Human Body","test")
        else:
            dataset = dataset_train
            images_path = os.path.join("Hacking the Human Body","train_images")
        images_names = project_dict[ds]
        for img_names_batch in sly.batched(images_names, batch_size=batch_size):
            images_pathes_batch = [os.path.join(images_path, image_name) for image_name in img_names_batch]

            img_infos = api.image.upload_paths(dataset.id, img_names_batch, images_pathes_batch)
            img_ids = [im_info.id for im_info in img_infos]

            anns_batch = [create_ann(image_path) for image_path in images_pathes_batch]
            api.annotation.upload_anns(img_ids, anns_batch)

            progress.iters_done_report(len(img_names_batch))

    return project
