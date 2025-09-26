# Object Measurement Tool with ArUco Markers

This project provides a computer vision solution for measuring objects in images using ArUco markers as a reference for scale. It can detect and measure various shapes including rectangles, squares, circles, triangles, and polygons, providing accurate measurements in centimeters.

## Features

- Automatic object detection and measurement
- Support for multiple shapes (rectangles, squares, circles, triangles, polygons)
- ArUco marker-based scale calibration
- Detailed measurements including:
  - Width and height
  - Perimeter
  - Area
  - Object classification
- Interactive visualization with matplotlib
- Detailed measurement reports
- Automatic result saving

## Requirements

### Python Version
This project requires Python 3.7.x due to compatibility requirements with the dependencies. We recommend using Python 3.7.9 for optimal compatibility.

### Dependencies
The project requires the following Python packages:
```
contourpy==1.3.2
cycler==0.12.1
fonttools==4.60.0
kiwisolver==1.4.9
matplotlib==3.7.1
numpy==1.24.4
opencv-contrib-python==4.8.0.76
opencv-python==4.8.0.76
packaging==25.0
pillow==11.3.0
pyparsing==3.2.5
python-dateutil==2.9.0.post0
six==1.17.0
```

## Installation

1. Clone or download this repository to your local machine.

2. Create a virtual environment (recommended):
```bash
python -m venv venv
```

3. Activate the virtual environment:
   - Windows:
   ```bash
   .\venv\Scripts\activate
   ```
   - Linux/Mac:
   ```bash
   source venv/bin/activate
   ```

4. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Prepare your image:
   - Place an ArUco marker (5x5 cm) in the image alongside the objects you want to measure
   - Ensure good lighting and clear visibility of both the marker and objects
   - Save the image in the project directory

2. Configure the input:
   - Open `measure_shapes.py`
   - Update the `IMAGE_PATH` variable with your image filename
   - Adjust `ARUCO_REAL_SIZE_CM` if using a different size marker (default is 5.0 cm)

3. Run the program:
```bash
python measure_shapes.py
```

4. View results:
   - The program will display an interactive matplotlib window with measurements
   - A high-resolution output image will be saved in the `result/` directory
   - Detailed measurements will be printed in the console

## Output

The program generates:
- An interactive visualization window
- A saved PNG file with measurements in the `result/` directory
- Console output with detailed measurements and detection information

## Troubleshooting

1. If no ArUco marker is detected:
   - Ensure the marker is clearly visible in the image
   - Check lighting conditions
   - Verify the marker is a 5x5 ArUco marker

2. If objects are not being detected:
   - Ensure good contrast between objects and background
   - Check if objects are clearly visible and not overlapping
   - Adjust lighting conditions if necessary

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.