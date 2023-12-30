import os
from extract_frames import FrameExtractor
from temperature_brightness_contrast import ImageProcessor
from background_remove import BackgroundRemover

###################################################################################################################################################
# pip install opencv-contrib-python
# pip install opencv-python

# ./raw_video folder needed with the .mp4 to process

# input: nameofvideo.mp4 video of the face
# output: nameofvideo folder inside ./extracted_frames with all frames extracted + ./output folder inside with processed frames 
###################################################################################################################################################


class Preprocessor:
    def __init__(self, video_dir, video_name):
        self.video_dir = video_dir
        self.video_name = video_name
        self.frame_extractor = FrameExtractor()
        
        # Definir las rutas de entrada y salida para ImageProcessor
        video_name_without_extension = os.path.splitext(self.video_name)[0]
        self.input_dir_for_processor = os.path.join(self.frame_extractor.extracted_frames_dir, video_name_without_extension)
        self.output_dir_for_processor = os.path.join(self.input_dir_for_processor, "processed")
        
        self.image_processor = ImageProcessor(input_folder=self.input_dir_for_processor, output_folder=self.output_dir_for_processor)

        self.background_remover = BackgroundRemover(input_folder=self.output_dir_for_processor, output_folder=self.output_dir_for_processor)

    def process_video(self):
        # Extract frames from the video
        video_path = os.path.join(self.video_dir, self.video_name)
        self.frame_extractor.extract_frames(video_path, self.input_dir_for_processor)  # Added the output directory

        # Process the extracted frames using ImageProcessor
        self.image_processor.process_images()

        # Remove the background
        for image_name in os.listdir(self.output_dir_for_processor):
            if image_name.endswith('.png'):
                self.background_remover.process_image(image_name)

        return self.image_processor.output_folder

"""
preprocessor = Preprocessor(video_dir='./raw_video', video_name='test.mp4') # Name of the video to preprocess
processed_dir = preprocessor.process_video()
print(f"Processed images are saved in: {processed_dir}")
"""
