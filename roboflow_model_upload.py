from roboflow import Roboflow

rf = Roboflow(api_key=secret)  # My Roboflow API key
project = rf.workspace(
    # Change to your project name
    "shashank-p3ytm").project("football-field-detection-f07vi-e8dgd")

project.version("1").deploy(
    model_type="yolov8", model_path=f"runs\\pose\\train")
