from PIL import Image
from transformers import BlipProcessor
from transformers import BlipForConditionalGeneration


class ImageCaption:

    def __init__(self):
        self.processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )

        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )

    def generate_caption(self, image_path):

        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt")

        output = self.model.generate(**inputs)

        caption = self.processor.decode(
            output[0],
            skip_special_tokens=True
        )

        return caption