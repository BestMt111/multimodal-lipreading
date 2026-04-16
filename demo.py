import os
import hydra
import cv2
import imageio

import torch
import torchaudio
import torchvision
from datamodule.transforms import AudioTransform, VideoTransform
from datamodule.av_dataset import cut_or_pad
from preparation.utils import save2vid

class InferencePipeline(torch.nn.Module):
    def __init__(self, cfg, detector="retinaface"):
        super(InferencePipeline, self).__init__()
        self.modality = cfg.data.modality
        if self.modality in ["audio", "audiovisual"]:
            self.audio_transform = AudioTransform(subset="test")
        if self.modality in ["video", "audiovisual"]:
            if detector == "mediapipe":
                from preparation.detectors.mediapipe.detector import LandmarksDetector
                from preparation.detectors.mediapipe.video_process import VideoProcess
                self.landmarks_detector = LandmarksDetector()
                self.video_process = VideoProcess(convert_gray=False)
            elif detector == "retinaface":
                from preparation.detectors.retinaface.detector import LandmarksDetector
                from preparation.detectors.retinaface.video_process import VideoProcess
                self.landmarks_detector = LandmarksDetector(device="cuda:3")
                self.video_process = VideoProcess(convert_gray=False)
            self.video_transform = VideoTransform(subset="test")

        if cfg.data.modality in ["audio", "video"]:
            from lightning import ModelModule
        elif cfg.data.modality == "audiovisual":
            from lightning_av import ModelModule
        self.modelmodule = ModelModule(cfg)
        self.modelmodule.model.load_state_dict(torch.load(cfg.pretrained_model_path, map_location=lambda storage, loc: storage))
        self.modelmodule.eval()


    def forward(self, data_dir):
        data_dir = os.path.abspath(data_dir)
        assert os.path.isdir(data_dir), f"data_dir_name: {data_dir} does not exist."

        data_list = os.listdir(data_dir)
        transcript = []
        ground_truth = []
        for data in data_list:
            data_path = data_dir + '/' + data

            if data_path[-4:] != '.mp4':
                assert "read file error, not mp4 data."

            ground_truth.append(data_path[:-4])

            if self.modality in ["audio", "audiovisual"]:
                audio, sample_rate = self.load_audio(data_path)
                audio = self.audio_process(audio, sample_rate)
                audio = audio.transpose(1, 0)
                audio = self.audio_transform(audio)

            if self.modality in ["video", "audiovisual"]:
                # video是一个（T, H, W, C)的数据
                video = self.load_video(data_path)
                landmarks = self.landmarks_detector(video)
                # 唇部区域裁剪，H和W裁剪为96，此时通道仍然为3
                video = self.video_process(video, landmarks)

                # save2vid(data[:-4]+"_p.mp4", video, 25)

                video = torch.tensor(video)
                # 转换为（T, C, H, W）
                video = video.permute((0, 3, 1, 2))
                # 经过了中心裁剪，灰度转换等，通道数变为1
                video = self.video_transform(video)

            if self.modality == "video":
                with torch.no_grad():
                    output = self.modelmodule(video)
                    transcript.append(output)
            elif self.modality == "audio":
                with torch.no_grad():
                    output = self.modelmodule(audio)
                    transcript.append(output)

            elif self.modality == "audiovisual":
                print(len(audio), len(video))
                assert 530 < len(audio) // len(video) < 670, "The video frame rate should be between 24 and 30 fps."

                rate_ratio = len(audio) // len(video)
                if rate_ratio == 640:
                    pass
                else:
                    print(f"The ideal video frame rate is set to 25 fps, but the current frame rate ratio, calculated as {len(video)*16000/len(audio):.1f}, which may affect the performance.")
                    audio = cut_or_pad(audio, len(video) * 640)
                with torch.no_grad():
                    transcript.append(self.modelmodule(video, audio))

            print('ground_truth: ' + data[:-4] + '\n' + 'transcript: ' + output)
        return

    def load_audio(self, data_filename):
        waveform, sample_rate = torchaudio.load(data_filename, normalize=True)
        return waveform, sample_rate

    def load_video(self, data_filename):
        return torchvision.io.read_video(data_filename, pts_unit="sec")[0].numpy()

    def audio_process(self, waveform, sample_rate, target_sample_rate=16000):
        if sample_rate != target_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, target_sample_rate
            )
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform


@hydra.main(version_base="1.3", config_path="configs", config_name="config_demo")
def main(cfg):
    pipeline = InferencePipeline(cfg)
    pipeline(cfg.file_path_dir)
    # for i in range(len(transcript)):
    #     print('ground_truth: ' + ground_truth[i] +'\n' + 'transcript: ' + transcript[i])
    # print(f"torchranscript: {transcript}")


if __name__ == "__main__":
    main()
