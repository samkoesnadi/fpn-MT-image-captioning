"""
Convert dataset IU X-ray to COCO dataset
"""

from common.common_imports import *
from common.common_definitions import *
import xml.etree.ElementTree as ET
import random
from shutil import copyfile
from tqdm import tqdm

def convert_store_to_coco_val_train(directory_to_walk, amount_of_validation=500,):
	list_of_files = os.listdir(directory_to_walk)
	random.shuffle(list_of_files)

	# seperate to valid and train datasets
	val_files = list_of_files[:amount_of_validation]
	train_files = list_of_files[amount_of_validation:]

	# run conversion for val_files and train_files
	logging.info("Begin conversion to COCO format...")
	convert_store_format_to_coco(val_files, directory_to_walk, DATADIR, DATATYPE_VAL)
	convert_store_format_to_coco(train_files, directory_to_walk, DATADIR, DATATYPE_TRAIN)
	logging.info("End conversion to COCO format...")

def convert_store_format_to_coco(list_of_files, parentDir_string, dataDir, dataType, imgId_start=1000, annotationId_start=1000):
	coco_json = {
		"info": {},
		"licenses": [],
		"images": [],
		"annotations": []
	}

	licenses_list = []
	images_list = []
	annotations_list = []

	imgId = imgId_start
	licenseId = 1
	annotationId = annotationId_start

	# mkdir the folder to store images
	imgsDir = os.path.join(dataDir, "images", dataType)
	try:
		os.mkdir(imgsDir, 755)
	except FileExistsError:
		pass

	for file in tqdm(list_of_files, desc=dataType):
		if file.endswith(".xml"):
			tree = ET.parse(os.path.join(parentDir_string, file))
			root = tree.getroot()

			# first loop, store the dataset info
			if not coco_json["info"]:
				coco_json["info"]["description"] = (root.find("./title").text)
				coco_json["info"]["date_created"] = (root.find("./articleDate").text)
				coco_json["info"]["contributor"] = (root.find("./publisher").text)

			license_url = (root.find("./licenseURL").text)
			license_type = (root.find("./licenseType").text)
			findings = (root.find(".//AbstractText[@Label=\"FINDINGS\"]").text)
			impression = (root.find(".//AbstractText[@Label=\"IMPRESSION\"]").text)

			# if None then convert to empty string
			findings = '' if findings is None else findings
			impression = '' if impression is None else impression

			# iterate through the image and write
			for imgPath in root.findall("./parentImage"):
				imgPath = imgPath.attrib["id"] + ".png"
				license = {
					"url": license_url,
					"id": licenseId,
					"name": license_type
				}
				image = {
					"license": licenseId,
					"file_name": imgPath,
					"id": imgId
				}
				annotation = {
					"image_id": imgId,
					"id": annotationId,
					"caption": (impression + ' ' + findings)
				}

				licenses_list.append(license)
				images_list.append(image)
				annotations_list.append(annotation)

				# copy image to the imgsDir path
				imgFullPath = os.path.join(dataDir, "images", "nlmcxr", imgPath)
				copyfile(imgFullPath, os.path.join(imgsDir, imgPath))

				imgId += 1
				licenseId += 1
				annotationId += 1

	# put the list in the root coco_json
	coco_json["licenses"] = licenses_list
	coco_json["images"] = images_list
	coco_json["annotations"] = annotations_list

	# store json
	with open(os.path.join(dataDir, "annotations", "captions_" + dataType + ".json"), 'w') as f_json:
		json.dump(coco_json, f_json)

if __name__ == "__main__":
	# convert iuxray raw format to coco
	convert_store_to_coco_val_train("datasets/iuxray/annotations_raw_xml/nlmcxr/ecgen-radiology/")
