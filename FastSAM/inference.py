model_path = "./fastsam/test75/weights/best.pt"
image_path = "../datasets/test/cat_dog.png"

!python ../FastSAM/Inference.py --model_path {model_path} --img_path {image_path}
