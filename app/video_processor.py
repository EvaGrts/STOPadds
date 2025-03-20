import cv2
import os
import numpy as np
import onnxruntime as ort
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import torch  # Pour détecter le GPU

class VideoProcessor:
    def __init__(self, video_path, onnx_model_path, engine_model_path, output_path):
        self.video_path = video_path
        self.onnx_model_path = onnx_model_path
        self.engine_model_path = engine_model_path
        self.output_path = output_path
        self.metrics_text = []
        # self.use_tensorrt = torch.cuda.is_available()  # Vérifie si un GPU est dispo
        self.use_tensorrt = False
        self.model = self.load_model()

    def load_model(self):
        """Charge le bon modèle selon la présence d'un GPU"""
        if self.use_tensorrt:
            return self.load_tensorrt_model()
        else:
            return self.load_onnx_model()

    def load_tensorrt_model(self):
        """Charge un modèle TensorRT"""
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(self.engine_model_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def load_onnx_model(self):
        """Charge un modèle ONNX"""
        return ort.InferenceSession(self.onnx_model_path, providers=['CUDAExecutionProvider'] if self.use_tensorrt else ['CPUExecutionProvider'])

    def preprocess_frame(self, frame):
        """Prépare une image pour l'inférence"""
        frame_resized = cv2.resize(frame, (640, 640))  # Modifier selon ton modèle
        frame_normalized = frame_resized.astype(np.float32) / 255.0
        frame_transposed = np.transpose(frame_normalized, (2, 0, 1))  # (HWC) → (CHW)
        return np.expand_dims(frame_transposed, axis=0).astype(np.float32)  # (1, C, H, W)

    def infer_frame(self, frame):
        """Effectue l'inférence sur une frame"""
        frame_input = self.preprocess_frame(frame)
       
        if self.use_tensorrt:
            # Inférence avec TensorRT
            context = self.model.create_execution_context()
            input_shape = frame_input.shape
            input_size = np.prod(input_shape) * np.dtype(np.float32).itemsize

            d_input = cuda.mem_alloc(input_size)
            d_output = cuda.mem_alloc(input_size)
            bindings = [int(d_input), int(d_output)]

            cuda.memcpy_htod(d_input, frame_input)
            context.execute_v2(bindings)

            output = np.empty(input_shape, dtype=np.float32)
            cuda.memcpy_dtoh(output, d_output)
            
            #return output.squeeze()

        else:
            # Inférence avec ONNX
            input_name = self.model.get_inputs()[0].name
            output_name = self.model.get_outputs()[0].name
            output = self.model.run([output_name], {input_name: frame_input})[0]
            confidences = output[0][4, :]
            bboxes=output[0][0:4,:]

            sorted_indices = confidences.argsort()[::-1]
            confidences = confidences[sorted_indices]
            bboxes = bboxes[:, sorted_indices]


            if(confidences[0]>0.25):
                print(confidences[0])
                x_center, y_center, width, height = bboxes[:, 0]
                x_min=int(x_center - width/2)
                x_max=int(x_center + width/2)
                y_min=int(y_center - height/2)
                y_max=int(y_center + height/2)
                # rectangle vert
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)


            return frame

    def process_video(self):
        """Traite une vidéo image par image avec TensorRT ou ONNX"""
        if not self.video_path:
            self.metrics_text.append("Aucune vidéo sélectionnée !")
            return

        self.metrics_text.append("Traitement de la vidéo en cours...")
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            self.metrics_text.append("Erreur : Impossible d'ouvrir la vidéo.")
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = self.infer_frame(frame)  # 🔹 Inférence sur l'image
            out.write(processed_frame)

        cap.release()
        out.release()

        if os.path.exists(self.output_path):
            print(f"Vidéo traitée enregistrée : {self.output_path}")
        else:
            print("Erreur : La vidéo traitée n'a pas été générée.")


        self.metrics_text.append("Traitement terminé !")



