from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse,HttpResponse
import numpy as np
import urllib
import json
import cv2
import os
import json
import operator
import face_recognition
from django.conf import settings
from .color_transfer import color_transfer
import datetime
import base64
from PIL import Image
import requests
import io
from .sudoku import sudoku_labels,sudoku_samples,solver,generate
import pyrebase
from . import firebase_config
from wsgiref.util import FileWrapper
from gtts import gTTS

firebase=pyrebase.initialize_app(firebase_config.CONFIG)
db = firebase.database()


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def _grab_image(path=None, stream=None, url=None):
    if path is not None:
        image = cv2.imread(path)
    else:
        if url is not None:
            resp = urllib.urlopen(url)
            data = resp.read()
        elif stream is not None:
            data = stream.read()
        image = np.asarray(bytearray(data), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

@csrf_exempt
def googleTextToSpeech(request):
	returndata = {"success": False}
	if request.method == "POST":
		if request.POST.get("text", None):
			text=request.POST["text"]
		else:
			returndata["error"] = "No text provided."
			return JsonResponse(returndata)
		try:
			tts = gTTS(text, lang='en')
			fp = io.BytesIO()
			tts.write_to_fp(fp)
			fp.seek(0) 
			response = HttpResponse(FileWrapper(fp), content_type='audio/mp3')
			return response
		except:
			return JsonResponse(returndata)

@csrf_exempt
def detect_firebase_db(request):
	returndata = {"success": False}
	detection_method="cnn" # face detection model to use: either `hog` or `cnn`
	firebase=False
	if request.method == "POST":
		if request.POST.get("firebase", None):
			firebase=bool(request.POST.get("firebase", False))
		if firebase==False and request.FILES.get("data", None) is None:
			returndata["error"] = "No data file provided."
			return JsonResponse(returndata)
		if request.FILES.get("image", None) is not None:
			image = _grab_image(stream=request.FILES["image"])
		else:
			url = request.POST.get("url", None)
			if url is None:
				returndata["error"] = "No URL provided."
				return JsonResponse(returndata)
			image = _grab_image(url=url)
		print("[INFO] loading encodings...")
		if firebase==False:
			myfile = request.FILES.get("data", None)
			file = myfile.read().decode('utf-8')
			data = json.loads(file)
		else:
			data = json.loads(db.child("image_encoding").get().val())
		rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		print("[INFO] recognizing faces...")
		boxes = face_recognition.face_locations(rgb,model=detection_method)
		print(boxes)
		encodings = face_recognition.face_encodings(rgb, boxes)
		names = []
		for encoding in encodings:
			matches = face_recognition.compare_faces(data["encodings"],encoding,tolerance=0.54)
			print("matches", matches)
			name = "Unknown"
			if True in matches:
				matchedIdxs = [i for (i, b) in enumerate(matches) if b]
				counts = {}
				for i in matchedIdxs:
					name = data["names"][i]
					counts[name] = counts.get(name, 0) + 1
				name = max(counts, key=counts.get)
			names.append(name)
		print(names)
		returndata.update({"success": True,"name":names})
	return JsonResponse(returndata)


@csrf_exempt
def train_firebase_db(request):
	returndata = {"success": False}
	knownEncodings = []
	knownNames = []
	detection_method="cnn" # face detection model to use: either `hog` or `cnn`
	resume=False
	name="Unknown"
	fp = io.BytesIO()
	firebase=False
	if request.method == "POST":
		if request.POST.get("resume", None):
			resume=bool(request.POST.get("resume", False))
		if request.POST.get("firebase", None):
			firebase=bool(request.POST.get("firebase", False))
		if firebase==False and resume==True and request.FILES.get("data", None) is None:
			returndata["error"] = "No data file provided."
			return JsonResponse(returndata)
		if request.FILES.get("image", None) is not None:
			image = _grab_image(stream=request.FILES["image"])
		else:
			url = request.POST.get("url", None)
			if url is None:
				returndata["error"] = "No URL provided."
				return JsonResponse(returndata)
			image = _grab_image(url=url)
		if request.POST.get("name", None):
			name=request.POST.get("name", None)
		rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		boxes = face_recognition.face_locations(rgb,
			model=detection_method)
		encodings = face_recognition.face_encodings(rgb, boxes)
		for encoding in encodings:
			knownEncodings.append(encoding)
			knownNames.append(name)
		print("[INFO] serializing encodings...")
		if not resume:
			data = {"encodings": knownEncodings, "names": knownNames}
			if firebase==False:
				fp.write(json.dumps(data, cls=NumpyEncoder).encode())
				fp.seek(0) 
				response = HttpResponse(FileWrapper(fp), content_type='pickle')
				return response
			else:
				db.child("image_encoding").set(json.dumps(data, cls=NumpyEncoder)) 
				returndata.update({"success": True})
		else:
			print("[INFO] Resuming...")
			if firebase==False:
				myfile = request.FILES.get("data", None)
				file = myfile.read().decode('utf-8')
				file_encodings = json.loads(file)
			else:
				file_encodings = json.loads(db.child("image_encoding").get().val())
			encodings = file_encodings.get("encodings") 
			names = file_encodings.get("names")
			# print("new")
			for items in knownEncodings:
				encodings.append(items)
			for items in knownNames:
				names.append(items)
			data = {"encodings": encodings, "names": names}
			if firebase==False:
				fp.write(json.dumps(data, cls=NumpyEncoder).encode())
				fp.seek(0) 
				response = HttpResponse(FileWrapper(fp), content_type='pickle')
				return response
			else:
				db.child("image_encoding").set(json.dumps(data, cls=NumpyEncoder)) 
				returndata.update({"success": True})
	return JsonResponse(returndata)



@csrf_exempt
def train_file(request):
	returndata = {"success": False}
	knownEncodings = []
	knownNames = []
	detection_method="cnn" # face detection model to use: either `hog` or `cnn`
	resume=False
	name="Unknown"
	fp = io.BytesIO()
	if request.method == "POST":
		if request.POST.get("resume", None):
			resume=bool(request.POST.get("resume", False))
		if request.FILES.get("image", None) is not None:
			image = _grab_image(stream=request.FILES["image"])
		else:
			url = request.POST.get("url", None)
			if url is None:
				returndata["error"] = "No URL provided."
				return JsonResponse(returndata)
			image = _grab_image(url=url)
		if request.POST.get("name", None):
			name=request.POST.get("name", None)
		rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		boxes = face_recognition.face_locations(rgb,
			model=detection_method)
		encodings = face_recognition.face_encodings(rgb, boxes)
		for encoding in encodings:
			knownEncodings.append(encoding)
			knownNames.append(name)
		print("[INFO] serializing encodings...")
		if not resume:
			data = {"encodings": knownEncodings, "names": knownNames}
			jsonData = json.dumps(data, cls=NumpyEncoder)
			binaryData = jsonData.encode()
			fp.write(binaryData)
			fp.seek(0) 
			response = HttpResponse(FileWrapper(fp), content_type='pickle')
			return response
		else:
			print("[INFO] Resuming...")
			file_encodings =json.loads(db.child("image_encoding").get().val())
			encodings =file_encodings.get("encodings") 
			names = file_encodings.get("names")
			print("new")
			for items in knownEncodings:
				encodings.append(items)
			for items in knownNames:
				names.append(items)
			data = {"encodings": encodings, "names": names}
			db.child("image_encoding").set(json.dumps(data, cls=NumpyEncoder)) 
			returndata.update({"success": True})
	return JsonResponse(returndata)



@csrf_exempt
def colortransfer(request):
	json_request=False
	returndata = {"success": False}
	if request.method == "POST":
		if request.POST.get("json", False):
			json_request=json.loads(request.POST["json"].lower())
		if request.FILES.get("source_image", None) is not None:
			source_image = _grab_image(stream=request.FILES["source_image"])
		else:
			source_url = request.POST.get("source_url", None)
			if source_url is None:
				returndata["error"] = "No URL provided."
				return JsonResponse(returndata)
			source_image = _grab_image(url=source_url)
		if request.FILES.get("target_image", None) is not None:
			target_image = _grab_image(stream=request.FILES["target_image"])
		else:
			target_url = request.POST.get("target_url", None)
			if target_url is None:
				returndata["error"] = "No URL provided."
				return JsonResponse(returndata)
			target_image = _grab_image(url=url)
		transfer = color_transfer(source_image, target_image, clip=True, preserve_paper=True)
		transfer=cv2.cvtColor(transfer, cv2.COLOR_BGR2RGB)
		img =  Image.fromarray(transfer, 'RGB')
		response = HttpResponse(content_type="image/jpeg")
		img.save(response, "JPEG") 
		if not json_request:
			return response
		img_str = base64.b64encode(response.getvalue())
		img_base64 = (bytes("data:image/jpeg;base64,", encoding='utf-8') + img_str).decode("utf-8")
		returndata.update({"success": True,"image":img_base64}) 
	return JsonResponse(returndata) 



@csrf_exempt
def sudoku_extractor(request):
	returndata = {"success": False}
	if request.method == "POST":
		if request.FILES.get("image", None) is not None:
			img = _grab_image(stream=request.FILES["image"])
		else:
			url = request.POST.get("url", None)
			if url is None:
				returndata["error"] = "No URL provided."
				return JsonResponse(returndata)
			img = _grab_image(url=url)
		model = cv2.ml.KNearest_load(os.path.join(settings.BASE_DIR,'webapi','sudoku','model.xml')) 

		## START Crop sudoku from puzzle
		p_img = img.copy() # Load an color image in grayscale
		#Process the img to find contour
		gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		gray=cv2.GaussianBlur(gray, (5, 5), 0)
		thresh=cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 2)
		contours_, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#Get all the contours in the img
		contour = None
		maxArea = 0
		#Find the largest contour(Sudoku Grid)
		for c in contours_:
			area = cv2.contourArea(c)
			if area > 25000:
				peri = cv2.arcLength(c, True)
				polygon = cv2.approxPolyDP(c, 0.01*peri, True)
				if area>maxArea and len(polygon)==4:
					contour = polygon
					maxArea = area
		#Draw the contour and extract Sudoku Grid
		if contour is not None:
			cv2.drawContours(img, [contour], 0, (0, 255, 0), 2)
			points = np.vstack(contour).squeeze()
			print(points)
			x,y=list(),list()
			for i in range(4):
				x.append(points[i][0])
				y.append(points[i][1])
			top_left_x, top_left_y = min(x), min(y)
			bot_right_x, bot_right_y = max(x), max(y)
			img = img[top_left_y:bot_right_y+1, top_left_x:bot_right_x+1]
		## END Crop sudoku from puzzle
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			ret, thresh = cv2.threshold(gray, 200, 255, 1)
			kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
			dilated = cv2.dilate(thresh, kernel)
			contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
			boxes = []
			for i in range(len(hierarchy[0])):
				if hierarchy[0][i][3] == 0:
					boxes.append(hierarchy[0][i])
			height, width = img.shape[:2]
			box_h = height / 9
			box_w = width / 9
			number_boxes = []
			soduko = np.zeros((9, 9), np.int32)
			for j in range(len(boxes)):
				if boxes[j][2] != -1:
					x, y, w, h = cv2.boundingRect(contours[boxes[j][2]])
					number_boxes.append([x, y, w, h])
					number_roi = gray[y : y + h, x : x + w]
					resized_roi = cv2.resize(number_roi, (20, 40))
					thresh1 = cv2.adaptiveThreshold(resized_roi, 255, 1, 1, 11, 2)
					normalized_roi = thresh1 / 255.0
					sample1 = normalized_roi.reshape((1, 800))
					sample1 = np.array(sample1, np.float32)
					retval, results, neigh_resp, dists = model.findNearest(sample1, 1)
					number = int(results.ravel()[0])
					cv2.putText(
						img,
						str(number),
						(x + w + 1, y + h - 20),
						3,
						2.0,
						(255, 0, 0),
						2,
						cv2.LINE_AA,
					)
					soduko[int(y / box_h)][int(x / box_w)] = number
			# print(soduko)
			returndata.update({"success": True,"sudoku": json.dumps(soduko,cls=NumpyEncoder)})
		else:
			returndata.update({"error": "No sudoku puzzle found in image!"})
	return JsonResponse(returndata)

@csrf_exempt
def sudoku_solver(request):
	returndata = {"success": False}
	if request.method == "POST":
		if request.POST.get("sudoku", False):
			sudoku=json.loads(request.POST["sudoku"])
		else:
			returndata["error"] = "No sudoku puzzle provided."
			return JsonResponse(returndata)
		if(np.array(sudoku).shape==(9,9,)):
			solve=solver.Solver(sudoku)
			isSolved=solve.solve()
			solution=solve.solution()
			if(isSolved!=False):
				returndata.update({"success": True,"solution":json.dumps(solution,cls=NumpyEncoder)}) 
			else:
				returndata.update({"error":"Error! Please enter a valid sudoku game!"})
		else:
			returndata.update({"error":"Error! Please enter a valid sudoku game!"})
	return JsonResponse(returndata)

# [[8, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 3, 6, 0, 0, 0, 0, 0], [0, 7, 0, 0, 9, 0, 2, 0, 0], [0, 5, 0, 0, 0, 7, 0, 0, 0], [0, 0, 0, 0, 4, 5, 7, 0, 0], [0, 0, 0, 1, 0, 0, 0, 3, 0], [0, 0, 1, 0, 0, 0, 0, 6, 8], [0, 0, 8, 5, 0, 0, 0, 1, 0], [0, 9, 0, 0, 0, 0, 4, 0, 0]]

@csrf_exempt
def sudoku_generator(request):
	returndata = {"success": False}
	if request.method == "POST":
		if request.POST.get("blank",None):
			blank=int(request.POST["blank"])
		else:
			returndata["error"] = "No of blank fields not specified!"
			return JsonResponse(returndata)
		puzzle=generate.sudoku(blank)
		returndata.update({"success": True,"sudoku":json.dumps(puzzle,cls=NumpyEncoder)})
	return JsonResponse(returndata)