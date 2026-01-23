import logging
import sys,os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../grpc_pb/')
import json

try:
    from ugot.grpc_pb import audio_pb2,audio_pb2_grpc
except:
    pass
try:
    import audio_pb2,audio_pb2_grpc
except:
    pass

from base_client import GrpcClient

class AudioClient(GrpcClient):
    def __init__(self, address):
        super().__init__(address)

        self.client = audio_pb2_grpc.AudioServiceGrpcStub(channel=self.channel)

    def setAudioAsr(self, begin_vad = 3000, end_vad = 1500, duration = 60000):

        input_data = audio_pb2.AudioAsrRequest()
        input_data.begin_vad = begin_vad
        input_data.end_vad = end_vad
        input_data.duration = duration

        response = self.client.setAudioAsr(input_data)
        return response
    
    def getAsrAndDoa(self, begin_vad = 3000, end_vad = 1500, duration = 60000):

        input_data = audio_pb2.AudioAsrRequest()
        input_data.begin_vad = begin_vad
        input_data.end_vad = end_vad
        input_data.duration = duration

        response = self.client.getAsrAndDoa(input_data)
        return response.data

    def setAudioTts(self, text, voice_type):

        input_data = audio_pb2.AudioTtsRequest()
        input_data.text = text
        input_data.voice_type = voice_type  # 声音类型 0-女声，1-男声

        response = self.client.setAudioTts(input_data)
        return response

    def setAudioNlp(self, text):

        input_data = audio_pb2.AudioNlpRequest()
        input_data.text = text

        response = self.client.setAudioNlp(input_data)
        return response

    def playAudioFile(self, audio_file, audio_type):

        input_data = audio_pb2.AudioPlayRequest()
        input_data.audio_type = audio_type  # 0 表示上传音频, 1 表示录音,2 表示内部音效
        if isinstance(audio_file, str):
            input_data.audio_file = audio_file
        elif isinstance(audio_file, tuple):
            input_data.audio_file = '/'.join(audio_file)

        response = self.client.playAudioFile(input_data)
        return response

    def stopPlayAudio(self):

        input_data = audio_pb2.AudioEmptyRequest()

        response = self.client.stopPlayAudio(input_data)
        return response

    def getDirectionOfAudio(self):

        input_data = audio_pb2.AudioDirectionRequest()
        input_data.duration = 3000

        response = self.client.getDirectionOfAudio(input_data)
        return response

    def enableAudioDirection(self):

        input_data = audio_pb2.AudioEmptyRequest()

        response = self.client.enableAudioDirection(input_data)
        return response

    def disableAudioDirection(self):

        input_data = audio_pb2.AudioEmptyRequest()

        response = self.client.disableAudioDirection(input_data)
        return response

    def __del__(self):
        # # 销毁时候停止所有声音
        try:
            self.stopPlayAudio()
        except Exception as e:
            # logging.debug('audio stopPlayAudio error:')
            pass
        pass
