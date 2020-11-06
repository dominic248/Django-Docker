# Django in Docker
## Docker file for Django with OpenCV-4 and dlib
### Create Docker Image
```
docker build -t <container-name> -f Dockerfile .
```
### Run Docker Image
```
docker run -it -p 8888:8888 <container-name>
```

---

## Setting up Django API Project for Face Recognition (first time only)
```
virtualenv -p python3.6 env
source ./env/bin/activate
sudo apt-get install cmake
pip install -r requirements.txt
```

---

## Running the Django API Project for Face Recognition
```
source ./env/bin/activate
python src/manage.py runserver
```

---

## Creating a public server for Django API project
Download [ngrok](https://ngrok.com/) and extract files on your PC and export the path to ngrok file to the System Environment PATH variable and run
```
ngrok http 8000
```
You'll get a link to access the Django API Project outside the home network.

---

## API endpoints: https://delta24-mini-django-api-v1.herokuapp.com/
<details>
    <summary><strong>NOTE</strong></summary>
    <blockquote>
    <ol>
        <li>Use 100x100 image dimension on these endpoints to avoid memory leak, due to large numpy array. (For face training and face recognition API's)</li>
        <li>Server sleeps after 30 minutes of inactivity and will take time to restart on new requests after 30 minutes of inactivity.</li>
    </ol>
    </blockquote>
</details>

### **Face Training API:**
> **API endpoint:** ```/face_detection/train/firebase/``` <br>
> **Method:** POST <br>
> **Parameters:**<br>
> 1. **```image```**<br>
> **Type:** Image<br>
> **Alternative:** ```url``` (Image URL)<br>
> **Description:** Single image file with 1 person in image.
> 2. **```name```**<br>
> **Type:** String<br>
> **Description:** Name of the person.
> 3. **```resume```** (optional)<br>
> **Type:** Boolean<br>
> **Description:** Resume training or overwrite all previous trainings.

### **Face Recognition API:**
> **API endpoint:** ```/face_detection/detect/firebase/```<br>
> **Method:** POST <br>
> **Parameters:**<br>
> 1. **```image```**<br>
> **Type:** Image<br>
> **Alternative:** ```url``` (Image URL)<br>
> **Description:** Single image file with 1 person in image.

### **Color Transfer API:**
> **API endpoint:** ```/colortransfer/```<br>
> **Method:** POST <br>
> **Parameters:**<br>
> 1. **```source_url```**<br>
> **Type:** Image<br>
> **Alternative:** ```source_url``` (Image URL)<br>
> **Description:** Single image file or URL of image.
> 2. **```target_image```**<br>
> **Type:** Image<br>
> **Alternative:** ```target_url``` (Image URL)<br>
> **Description:** Single image file or URL of image.
> 3. **```json```**<br>
> **Type:** Boolean<br>
> **Description:** To get image as Base64 URL set it as ```true``` and for Image response set it as ```false```.

### **Sudoku Solver API:**
> **API endpoint:** ```/sudoku/solver/```<br>
> **Method:** POST <br>
> **Parameters:**<br>
> 1. **```sudoku```**<br>
> **Type:** 2D-Array/List<br>
> **Description:** 2D-Array/List, with blank cells of sudoku filled with 0.

### **Sudoku Extractor API:**
> **API endpoint:** ```/sudoku/extractor/```<br>
> **Method:** POST <br>
> **Parameters:**<br>
> 1. **```image```**<br>
> **Type:** Image<br>
> **Alternative:** ```url``` (Image URL)<br>
> **Description:** Single image file with a cropped sudoku puzzle.
