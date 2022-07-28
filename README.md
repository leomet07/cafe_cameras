# Cafe Cameras

A fast motion tracking system for detecting how/if pedestrains walk.

It doesn't even have to be pedestrians, it could be any type of movement.

## Run Locally

Clone the project

```bash
git clone https://github.com/leomet07/cafe_cameras
```

Go to the project directory

```bash
cd cafe_cameras
```

Install dependencies

```bash
pip install -r requirements.txt
```

Start the program

```bash
python main.py
```

## Environment Variables

To run this project, you will NOT need to add environment variables to your .env file

Instead, add camera configuration to a `cameras.json` file

An example `cameras.json` :

```json
[
	{
		"url": "rtsp://user:password@1.1.1.1:PORT/xyz/abc?channel=X&subtype=Y",
		"dimensions": {
			"top_left_point": {
				"x": 500,
				"y": 120
			},
			"width": 1000,
			"height": 450
		}
	}
]
```

## Random information

The start and end times are recording in UTC time.

## Privacy Implications

While this project does "track" pedestrians, it does not record induvidual data or any identifiable information. It would record a moving blob and a pedestrian in the same way. This project is also FOSS, and the source code is public. You may check for yourselves!