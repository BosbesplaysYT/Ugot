import logging
import sys,os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../grpc_pb/')
import json

try:
    from ugot.grpc_pb import vision_pb2,vision_pb2_grpc
except:
    pass

try:
    import vision_pb2,vision_pb2_grpc
except:
   pass

from base_client import GrpcClient

# AI视觉
MODELNAME_KNN = 'custom_knn' # knn
MODELNAME_POSE = 'human_pose' # 人体姿态
MODELNAME_WORD = 'word_recognition' # 文字识别
MODELNAME_COLOR = 'color_recognition' # 颜色识别
MODELNAME_APRILTAG = 'apriltag_qrcode' # ArpilTag
MODELNAME_QRCODE = 'apriltag_qrcode' # 二维码
MODELNAME_EXPRESSION = 'face_attribute' # 表情识别
MODELNAME_CHARACTERISTIC = 'face_attribute' # 人脸特征
MODELNAME_LICENSE_PLATE = 'lpd_recognition' # 车牌识别
MODELNAME_GESTURE = 'gesture' # 手势识别
MODELNAME_TRAFFIC = 'traffic_sign' # 交通识别，交通标识以及斑马线
MODELNAME_OBJECT_RECOGNITION = 'object_recognition' # 物体识别
MODELNAME_FACE_RECOGNITION = 'face_recognition' # 人脸识别
MODELNAME_TRACK_RECOGNITION = 'line_recognition' # 单双轨
MODELNAME_COLOR_TRACK = 'color_tracking' # 自定义颜色
MODELNAME_TOY_RECOGNITION = 'toy_recognition' # 公仔识别

MODEL_REFLECT_LIST = {
    MODELNAME_KNN, # = 0 # knn
    MODELNAME_POSE, # = 1 # 人体姿态
    MODELNAME_WORD, # = 2 # 文字识别
    MODELNAME_COLOR, # = 3 # 颜色识别
    MODELNAME_APRILTAG, # = 4 # ArpilTag
    MODELNAME_QRCODE, # = 5 # 二维码
    MODELNAME_EXPRESSION, # = 6 # 表情识别
    MODELNAME_CHARACTERISTIC, # = 7 # 人脸特征
    MODELNAME_LICENSE_PLATE, # = 8 # 车牌识别
    MODELNAME_GESTURE, # = 9 # 手势识别
    MODELNAME_TRAFFIC, # = 10 # 交通识别，交通标识以及斑马线
    MODELNAME_OBJECT_RECOGNITION, # = 11 # 物体识别
    MODELNAME_FACE_RECOGNITION, # = 12 # 人脸识别
    MODELNAME_TRACK_RECOGNITION, # = 13 # 单双轨
    # MODELNAME_COLOR_TRACK, # = 14 # 自定义颜色
    MODELNAME_TOY_RECOGNITION, # = 15 # 公仔识别
}

class VisionClient(GrpcClient):
    def __init__(self, address):
        super().__init__(address)

        self.camera_client_id = -1

        self.client = vision_pb2_grpc.AIVisionServiceGrpcStub(channel=self.channel)

    def load_models(self, models):

        input_data = vision_pb2.LoadModelRequest()
        modellist = []
        if isinstance(models, list):
            for idx in models:
                if idx in MODEL_REFLECT_LIST:
                    modellist.append(idx)
                else:
                    print('unknown model of {}'.format(idx))
        elif isinstance(models, str):
            modellist.append(models)
        else:
            return False

        for model in modellist:
            info = input_data.models.add()
            info.model = model
        response = self.client.loadModel(input_data)

        return response.code == 0

    def release_models(self, models = None):

        if models is None:
            return self.unloadAllModels()


        input_data = vision_pb2.ReleaseModelRequest()
        modellist = []
        if isinstance(models, list):
            for idx in models:
                if idx in MODEL_REFLECT_LIST:
                    modellist.append(idx)
                else:
                    print('unknown model of {}'.format(idx))
        elif isinstance(models, str):
            modellist.append(models)
        else:
            return False

        for model in modellist:
            input_data.models.append(model)
        response = self.client.releaseModel(input_data)

        return response.code == 0

    """ knn
        """

    def knn_identify(self, model_name):
        """推理,在times时间内，用模型model_name识别
           结果是否为label

        Args:
            model_name(string): 模型
        Returns:
        """
        model = MODELNAME_KNN

        self.knn_load(model_name)

        input_data = vision_pb2.InferenceRequest()
        input_data.need_pic = False
        info = input_data.models.add()
        info.model = model
        para = {"model_name": model_name}
        info.para = json.dumps(para)
        response = self.client.doModelInference(input_data)
        for item in response.data.inference:
            if item.model == model:
                return item.data
        return None

    def knn_load(self, model_name):
        model = MODELNAME_KNN
        # 加载模型
        input_data = vision_pb2.LoadModelRequest()
        models = [model]
        for model in models:
            info = input_data.models.add()
            info.model = model
            para = {"model_name": model_name}
            info.para = json.dumps(para)
        response = self.client.loadModel(input_data)

    def knn_train(self, model_name, label_list):
        model = MODELNAME_KNN

        input_data = vision_pb2.TrainAndSaveRequest()
        input_data.model = model
        input_data.para.name = model_name

        for label in label_list:
            input_data.para.categoryNames.append(label)
        response = self.client.trainAndSave(input_data)
        for feature in response:  # 流式返回的结果
            yield feature

    def knn_rename(self, src_name, dst_name):
        model = MODELNAME_KNN
        input_data = vision_pb2.SetModelParaRequest()
        input_data.model = model
        input_data.invoke = "KnnModelRename"
        para = {"src_name":src_name, "dst_name":dst_name}
        input_data.para = json.dumps(para)
        response = self.client.setModelPara(input_data)
        if response.code != 0:
            print('knn_rename: {}'.format(response.msg))
        return response

    def knn_delete(self, model_name):
        model = MODELNAME_KNN
        input_data = vision_pb2.SetModelParaRequest()
        input_data.model = model
        input_data.invoke = "KnnModelAndLabelDelete"
        para = {"model_name":model_name}
        input_data.para = json.dumps(para)
        response = self.client.setModelPara(input_data)
        code = response.code
        if code != 0:
            if code == 1001:
                print('knn_delete: {}'.format('file name not exist!'))
        return response

    def knn_query(self):
        model = MODELNAME_KNN
        input_data = vision_pb2.SetModelParaRequest()
        input_data.model = model
        input_data.invoke = "knnModelAndLabelQuery"
        para = {}
        input_data.para = json.dumps(para)
        response = self.client.setModelPara(input_data)
        if response.code != 0:
            print('knn_query: {}'.format(response.msg))
        return response

    """ 人体姿态
    """


    def pose_identify(self):
        """人体姿态推理
        Args:

        Returns:
            keypoints
        """
        model = MODELNAME_POSE

        input_data = vision_pb2.InferenceRequest()
        input_data.need_pic = False
        info = input_data.models.add()
        info.model = model
        para = {}
        info.para = json.dumps(para)
        response = self.client.doModelInference(input_data)
        for item in response.data.inference:
            if item.model == model:
                return item.data
        return None

    """ 文字识别
    """


    def word_identify(self):
        """文字识别

        Args:
        Returns:
            文字识别结果
        """
        model = MODELNAME_WORD

        input_data = vision_pb2.InferenceRequest()
        input_data.need_pic = False
        info = input_data.models.add()
        info.model = model
        response = self.client.doModelInference(input_data)
        for item in response.data.inference:
            if item.model == model:
                return item.data
        return None

    """ 颜色
    """


    def color_identify(self):
        """颜色识别
        """
        model = MODELNAME_COLOR

        input_data = vision_pb2.InferenceRequest()
        input_data.need_pic = False
        info = input_data.models.add()
        info.model = model
        response = self.client.doModelInference(input_data)
        for item in response.data.inference:
            if item.model == model:
                return item.data
        return None

    """ AprilTag
    """


    def apriltag_inference(self):
        """AprilTag
        """
        model = MODELNAME_APRILTAG

        input_data = vision_pb2.InferenceRequest()
        input_data.need_pic = False
        info = input_data.models.add()
        info.model = model
        response = self.client.doModelInference(input_data)
        for item in response.data.inference:
            if item.model == model:
                return item.data
        return None

    """ 二维码
    """


    def qrcode_inference(self):
        """二维码
        """
        model = MODELNAME_QRCODE

        input_data = vision_pb2.InferenceRequest()
        input_data.need_pic = False
        info = input_data.models.add()
        info.model = model
        response = self.client.doModelInference(input_data)
        for item in response.data.inference:
            if item.model == model:
                return item.data
        return None

    """ 表情识别
    """


    def expression_inference(self):
        """表情识别
        """
        model = MODELNAME_EXPRESSION

        input_data = vision_pb2.InferenceRequest()
        input_data.need_pic = False
        info = input_data.models.add()
        info.model = model
        response = self.client.doModelInference(input_data)
        for item in response.data.inference:
            if item.model == model:
                return item.data
        return None

    """ 人脸特征
    """


    def face_characteristic_inference(self):
        """人脸特征
        """
        model = MODELNAME_CHARACTERISTIC

        input_data = vision_pb2.InferenceRequest()
        input_data.need_pic = False
        info = input_data.models.add()
        info.model = model
        response = self.client.doModelInference(input_data)
        for item in response.data.inference:
            if item.model == model:
                return item.data
        return None

    """ 车牌识别
    """


    def license_plate_inference(self):
        """车牌识别
        """
        model = MODELNAME_LICENSE_PLATE

        input_data = vision_pb2.InferenceRequest()
        input_data.need_pic = False
        info = input_data.models.add()
        info.model = model
        response = self.client.doModelInference(input_data)
        for item in response.data.inference:
            if item.model == model:
                return item.data
        return None

    """ 手势识别
    """


    def gesture_inference(self):
        """手势识别
        """
        model = MODELNAME_GESTURE

        input_data = vision_pb2.InferenceRequest()
        input_data.need_pic = False
        info = input_data.models.add()
        info.model = model
        response = self.client.doModelInference(input_data)
        for item in response.data.inference:
            if item.model == model:
                return item.data
        return None

    """ 交通识别
    """


    def traffic_inference(self):
        """交通识别
        """
        model = MODELNAME_TRAFFIC

        input_data = vision_pb2.InferenceRequest()
        input_data.need_pic = False
        info = input_data.models.add()
        info.model = model
        response = self.client.doModelInference(input_data)
        for item in response.data.inference:
            if item.model == model:
                return item.data
        return None

    """ 人脸识别"""
    def face_recognition_inference(self):
        """人脸识别
        """
        model = MODELNAME_FACE_RECOGNITION

        input_data = vision_pb2.InferenceRequest()
        input_data.need_pic = False
        info = input_data.models.add()
        info.model = model
        # para = {"model_name":model_name}
        # info.para = json.dumps(para)
        response = self.client.doModelInference(input_data)
        for item in response.data.inference:
            if item.model == model:
                return item.data
        return None
    
    def face_recognition_insert_data(self, img, name):
        model = MODELNAME_FACE_RECOGNITION

        input_data = vision_pb2.SetModelParaRequest()
        input_data.model = model
        input_data.invoke = "extractionFeatureAndSave"
        para = {"img":img, "name":name}
        input_data.para = json.dumps(para)
        response = self.client.setModelPara(input_data)
        return response
    
    def face_recognition_get_all_names(self):
        model = MODELNAME_FACE_RECOGNITION

        input_data = vision_pb2.SetModelParaRequest()
        input_data.model = model
        input_data.invoke = "getAllFace"
        para = {}
        input_data.para = json.dumps(para)
        response = self.client.setModelPara(input_data)
        if response.code == 0:
            data = json.loads(response.data)
            if data is None:
                return []
            return data
        return []
    
    def face_recognition_delete_name(self, name):
        model = MODELNAME_FACE_RECOGNITION

        input_data = vision_pb2.SetModelParaRequest()
        input_data.model = model
        input_data.invoke = "deleteFeature"
        para = {"name": name}
        input_data.para = json.dumps(para)
        response = self.client.setModelPara(input_data)
        return response


    """ 单双轨识别"""
    def track_recognition_inference(self):
        """单双轨识别
        """
        model = MODELNAME_TRACK_RECOGNITION

        input_data = vision_pb2.InferenceRequest()
        input_data.need_pic = False
        info = input_data.models.add()
        info.model = model
        # para = {"model_name":model_name}
        # info.para = json.dumps(para)
        response = self.client.doModelInference(input_data)
        for item in response.data.inference:
            if item.model == model:
                return item.data
        return None

    def set_track_color_line(self, color, line_type):
        model = MODELNAME_TRACK_RECOGNITION

        input_data = vision_pb2.SetModelParaRequest()
        input_data.model = model
        input_data.invoke = "setTrackingColorAndLineType"
        para = {"color": color, "line_type": line_type}
        input_data.para = json.dumps(para)
        response = self.client.setModelPara(input_data)

    """ 自定义颜色识别"""

    def color_track_inference(self, color_name):
        """自定义颜色识别
        """
        model = MODELNAME_COLOR_TRACK

        input_data = vision_pb2.InferenceRequest()
        input_data.need_pic = False
        info = input_data.models.add()
        info.model = model
        para = {"color_name": color_name}
        info.para = json.dumps(para)
        response = self.client.doModelInference(input_data)
        for item in response.data.inference:
            if item.model == model:
                return item.data
        return None
    
    """ 公仔识别 """
    def toy_recognition_inference(self):
        model = MODELNAME_TOY_RECOGNITION

        input_data = vision_pb2.InferenceRequest()
        input_data.need_pic = False
        info = input_data.models.add()
        info.model = model
        response = self.client.doModelInference(input_data)
        for item in response.data.inference:
            if item.model == model:
                return item.data
        return None



    def unloadAllModels(self):

        # 释放模型
        input_data = vision_pb2.ReleaseModelRequest()
        # input_data.models.append("")
        response = self.client.releaseModel(input_data)
        return response.code == 0

    def startAutoInference(self, models):
        # 开始自动推理
        input_data = vision_pb2.AutoInferenceRequest()
        input_data.models.append("aaa")
        response = self.client.startAutoInference(input_data)

        for feature in response:  # 流式返回的结果
            yield feature.msg

    def get_camera_number(self):
        input_data = vision_pb2.EmptyRequest()
        
        response = self.client.getCameraNumber(input_data)
        return response

    def openCamera(self):
        input_data = vision_pb2.OpenCameraRequest()
        input_data.extra = ''
        
        response = self.client.openCamera(input_data)
        return response.clientId
    
    def closeCamera(self, clientid):
        input_data = vision_pb2.CloseCameraRequest()
        input_data.clientId = clientid
        input_data.extra = ''

        response = self.client.closeCamera(input_data)
        return response
    
    def readCameraData(self):
        input_data = vision_pb2.ReadCameraDataRequest()
        input_data.extra = ''
        
        response = self.client.readCameraData(input_data)
        return response

    def __del__(self):
        # # 销毁时候卸载所有模型
        try:
            self.unloadAllModels()
            if self.camera_client_id > 0:
                self.closeCamera(self.camera_client_id)
        except Exception as e:
            # logging.debug('vision unloadAllModels error:')
            pass
