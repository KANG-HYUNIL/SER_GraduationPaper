import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
from omegaconf import DictConfig

class AudioPipeline:
    def __init__(self, cfg: DictConfig):
        """
        Initializes the audio processing pipeline based on Hydra config.
        Args:
            cfg (DictConfig): Data configuration (src/configs/data/default.yaml)
        """
        self.sample_rate = cfg.sample_rate
        self.duration = cfg.duration
        self.n_mels = cfg.n_mels
        self.n_fft = cfg.n_fft
        self.hop_length = cfg.hop_length
        self.target_length = int(self.sample_rate * self.duration)
        self.normalize = cfg.normalize
        # Resizing Resolution(고정 해상도) 설정
        self.resize_height = cfg.get("resize_height", 128)
        self.resize_width = cfg.get("resize_width", 512)

        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=cfg.f_min,
            f_max=cfg.f_max
        )
        
        self.amplitude_to_db = T.AmplitudeToDB()

    # def _pad_truncate(self, waveform: torch.Tensor) -> torch.Tensor:
    #     """
    #     Pads or truncates the waveform to a fixed length.
        
    #     Args:
    #         waveform (torch.Tensor): Input audio tensor of shape (Channels, Time) or (1, Time).
            
    #     Returns:
    #         torch.Tensor: Fixed length tensor of shape (Channels, Target_Length).
    #     """
    #     if waveform.shape[1] > self.target_length:
    #         return waveform[:, :self.target_length]
    #     elif waveform.shape[1] < self.target_length:
    #         pad_amount = self.target_length - waveform.shape[1]
    #         return F.pad(waveform, (0, pad_amount), "constant")
    #     return waveform

    def transform(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        """
        Applies the full pipeline: Resample -> Pad/Crop -> MelSpectrogram -> Log -> Normalize
        
        Args:
            waveform (torch.Tensor): Raw audio tensor of shape (Channels, Time).
            sr (int): Sample rate of the input waveform.
            
        Returns:
            torch.Tensor: Log-Mel Spectrogram of shape (Channels, n_mels, TimeFrames).
        """
        # 1. Resample if necessary
        if sr != self.sample_rate:
            resampler = T.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        # 2. Mix to mono if necessary (though RAVDESS is usually mono)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 3. Trim silence (Optional - skipping for simple baseline, but good to have)
        # waveform = torchaudio.functional.wad(waveform) 

        # 4. Pad or Truncate
        # Modified: Return variable length for Dynamic Padding (Collate Fn handling)
        # waveform = self._pad_truncate(waveform)

        # 5. Extract Log-Mel Spectrogram
        mel_spec = self.mel_spectrogram(waveform)
        log_mel_spec = self.amplitude_to_db(mel_spec)

        # 6. Normalize (Instance Normalization)
        if self.normalize:
            mean = log_mel_spec.mean()
            std = log_mel_spec.std()
            log_mel_spec = (log_mel_spec - mean) / (std + 1e-6)

        # 7. Bicubic Resizing (고정된 해상도로 변환)
        # 패딩 대신 보간법을 사용하여 신호의 맥락을 유지하면서 규격화합니다.
        # F.interpolate는 4D (Batch, Channel, Height, Width) 입력을 기대하므로 배치 차원 추가
        # log_mel_spec: (Channels, n_mels, TimeFrames) -> (1, Channels, n_mels, TimeFrames)
        log_mel_spec = log_mel_spec.unsqueeze(0)
        
        # Bicubic 보간법 적용 (시간축 왜곡 최소화)
        log_mel_spec = F.interpolate(
            log_mel_spec, 
            size=(self.resize_height, self.resize_width), 
            mode='bicubic', 
            align_corners=False
        )
        
        # 다시 원래 차원으로 복구: (Channels, n_mels, TimeFrames)
        # 결과물은 항상 (1, 128, 512) 형태가 됩니다.
        return log_mel_spec.squeeze(0)
