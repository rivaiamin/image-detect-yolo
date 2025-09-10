setup:
	# download yolov8s-cls.pt
	wget https://github.com/ultralytics/assets/releases/download/v8.0/yolov8s-cls.pt

	# install dependencies
	pip install -r requirements.txt

	# create directories
	mkdir -p uploads
	mkdir -p logs
	mkdir -p dataset/train
	mkdir -p dataset/val

	sudo chgrp www-data uploads
	sudo chmod g+w uploads
	sudo chgrp www-data logs
	sudo chmod g+w logs

run:
	python web_app.py

train:
	python main.py