import sys
import cv2
import threading
import time
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QTextEdit, QCheckBox, QHBoxLayout, QSpacerItem, QSizePolicy
from PyQt6.QtGui import QPixmap, QImage, QIcon
from PyQt6.QtCore import Qt
from video_processor import VideoProcessor

class VideoProcessingApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Suppression des panneaux publicitaires")
        self.video_path = None
        self.processed_video_path = None
        self.video_width = None
        self.video_height = None
        
        # Layout configuration
        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()
        
        # Ajout de marges et d'espacements
        left_layout.setContentsMargins(20, 20, 20, 20)
        left_layout.setSpacing(15)
        
        # Upload button
        self.upload_btn = QPushButton("📂 Charger une vidéo")
        self.upload_btn.setStyleSheet("padding: 10px;")
        self.upload_btn.clicked.connect(self.upload_video)
        left_layout.addWidget(self.upload_btn)
        
        # Metrics display
        self.metrics_label = QLabel("Logs:")
        left_layout.addWidget(self.metrics_label)
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        left_layout.addWidget(self.metrics_text)
        
        # Control buttons
        self.process_btn = QPushButton("⚙️ Lancer le traitement IA")
        self.process_btn.setStyleSheet("padding: 10px;")
        self.process_btn.clicked.connect(self.process_video)
        left_layout.addWidget(self.process_btn)
        
        self.show_original = QCheckBox("🎞️ Afficher vidéo traitée")
        self.show_original.stateChanged.connect(self.toggle_video)
        left_layout.addWidget(self.show_original)
        
        self.play_btn = QPushButton("▶️ Lire la vidéo")
        self.play_btn.setStyleSheet("padding: 10px;")
        self.play_btn.clicked.connect(self.play_video)
        left_layout.addWidget(self.play_btn)
        
        self.download_btn = QPushButton("💾 Télécharger la vidéo traitée")
        self.download_btn.setStyleSheet("padding: 10px;")
        self.download_btn.clicked.connect(self.download_video)
        left_layout.addWidget(self.download_btn)
        
        self.quit_btn = QPushButton("❌ Quitter")
        self.quit_btn.setStyleSheet("padding: 10px;")
        self.quit_btn.clicked.connect(self.close)
        left_layout.addWidget(self.quit_btn)
        
        # Ajouter un espace vide pour pousser les éléments vers le haut
        left_layout.addStretch()
        
        # Video display frames
        self.original_video_label = QLabel("🎥 Vidéo originale")
        self.original_video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(self.original_video_label)
        
        self.video_label = QLabel("📽️ Vidéo traitée")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(self.video_label)
        
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)
        self.setLayout(main_layout)
    
    def upload_video(self):
        file_dialog = QFileDialog()
        self.video_path, _ = file_dialog.getOpenFileName(self, "Sélectionner une vidéo", "", "Video Files (*.mp4 *.avi *.mov)")
        if self.video_path:
            self.metrics_text.append(f"Vidéo chargée : {self.video_path}")
            cap = cv2.VideoCapture(self.video_path)
            ret, frame = cap.read()
            cap.release()
            if ret:
                h, w, _ = frame.shape
                scale_factor = min(800 / w, 600 / h)
                self.video_width = int(w * scale_factor)
                self.video_height = int(h * scale_factor)
            threading.Thread(target=self.display_original_video).start()
    
    def process_video(self):
        if not self.video_path:
            self.metrics_text.append("Aucune vidéo sélectionnée !")
            return
        video_processor=VideoProcessor(self.video_path,"model/best.onnx","model/best.engine","output/output.mp4",0.25)
        video_processor.process_video()

        self.processed_video_path = "output/output.mp4"
        self.metrics_text.append(f"Vidéo traitée disponible : {self.processed_video_path}")

        
    def play_video(self):
        video_to_play = self.processed_video_path if self.show_original.isChecked() else self.video_path 
        if not video_to_play:
            self.metrics_text.append("Aucune vidéo disponible !")
            return
        
        threading.Thread(target=self.display_video, args=(video_to_play, self.video_label)).start()
        threading.Thread(target=self.display_video, args=(self.video_path, self.original_video_label)).start()
    
    def display_video(self, path, label):
        cap = cv2.VideoCapture(path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.video_width, self.video_height))
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            label.setPixmap(pixmap)
            
            time.sleep(1 / 30)  # Approx 30 fps
        
        cap.release()
    
    def display_original_video(self):
        self.display_video(self.video_path, self.original_video_label)
    
    def toggle_video(self):
        self.metrics_text.append("Affichage de la vidéo modifié")
    
    def download_video(self):
        if self.processed_video_path:
            file_dialog = QFileDialog()
            save_path, _ = file_dialog.getSaveFileName(self, "Enregistrer la vidéo", "", "MP4 files (*.mp4)")
            if save_path:
                with open(self.processed_video_path, "rb") as src, open(save_path, "wb") as dst:
                    dst.write(src.read())
                self.metrics_text.append(f"Vidéo téléchargée : {save_path}")
        else:
            self.metrics_text.append("Aucune vidéo traitée à télécharger !")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoProcessingApp()
    window.show()
    sys.exit(app.exec())
