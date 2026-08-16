from faster_whisper import WhisperModel


class WhisperService:

    def __init__( self ):

        self.model = WhisperModel(  "base",device="cpu"    )


    def transcribe(self, audio_path):

        segments, info = self.model.transcribe(audio_path )

        text = " ".join(
            segment.text
            for segment in segments
        )

        return {
            "text": text.strip(),
            "language": info.language
        }