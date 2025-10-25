import cv2
from VideoAnalyzer import VideoAnalyzer
from Sketcher import Sketcher

model = cv2.imread('res/input/target.jpg')
va = VideoAnalyzer('res/input/video.mp4', model, (325,309), 6, 50)
sketcher = Sketcher(50, '"')
va.analyze('res/output/debug.mp4', sketcher)
print('frame_total', va.telemetry_sections.get('frame_total'))
print('run_clock', va.telemetry_sections.get('run_clock'))
print('post_processing', va.telemetry_sections.get('post_processing'))
