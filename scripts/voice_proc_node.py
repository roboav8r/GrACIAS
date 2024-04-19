#!/usr/bin/env python3

import os
import json
import yaml
import torch
import numpy as np
from pathlib import Path
import importlib
import librosa
import torchaudio
import torchaudio.transforms as T
from torchaudio.models.decoder import download_pretrained_files, ctc_decoder, cuda_ctc_decoder
from torchaudio.utils import download_asset
import sentencepiece as spm

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

from audio_common_msgs.msg import AudioDataStamped
from std_msgs.msg import String


class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        """Given a sequence emission over labels, get the best path string
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          str: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i] for i in indices])

class VoiceProcNode(Node):

    def __init__(self):
        super().__init__('voice_proc_node')

        self.subscription = self.create_subscription(AudioDataStamped, 'audio_data', self.audio_data_callback, 10)
        self.audio_scene_publisher = self.create_publisher(String, 'voice_data', 10)
        
        # Declare parameters with default values
        self.declare_parameter('n_channels', rclpy.Parameter.Type.INTEGER)
        self.declare_parameter('sample_rate', rclpy.Parameter.Type.INTEGER)
        self.declare_parameter('frame_size', rclpy.Parameter.Type.INTEGER)
        self.declare_parameter('voice_index', rclpy.Parameter.Type.INTEGER_ARRAY)

        # Retrieve parameters
        self.n_channels = self.get_parameter('n_channels').get_parameter_value().integer_value
        self.sample_rate = self.get_parameter('sample_rate').get_parameter_value().integer_value
        self.frame_size = self.get_parameter('frame_size').get_parameter_value().integer_value
        self.voice_idx = self.get_parameter('voice_index').get_parameter_value().integer_array_value

        # Audio data storage
        self.frame = torch.zeros([self.frame_size, self.n_channels],dtype=torch.float16)
        self.voice_channels = torch.zeros([self.frame_size, len(self.voice_idx)],dtype=torch.float16)
        # self.voice_channels = self.voice_channels.to('cuda')
        self.min_voice_len = 16800
        self.n_trailing_frames = 2 # prevent early cutoff. Only stop recording if silent for n consecutive frames.
        self.n_silent = 0 
        self.recording = False

        # Initialize torch params
        self.url_prefix = "https://huggingface.co/Zengwei/icefall-asr-librispeech-pruned-transducer-stateless7-ctc-2022-12-01"
        self.model_link = f"{self.url_prefix}/resolve/main/exp/cpu_jit.pt"
        self.model_path = self.download_asset_external(self.model_link, "cuda_ctc_decoder/cpu_jit.pt")

        self.bpe_link = f"{self.url_prefix}/resolve/main/data/lang_bpe_500/bpe.model"
        self.bpe_path = self.download_asset_external(self.bpe_link, "cuda_ctc_decoder/bpe.model")

        self.bpe_model = spm.SentencePieceProcessor()
        self.bpe_model.load(self.bpe_path)
        self.tokens = [self.bpe_model.id_to_piece(id) for id in range(self.bpe_model.get_piece_size())]

        torch.random.manual_seed(0)
        self.torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.initialize_asr_model()
        self.resampler = T.Resample(self.sample_rate, self.bundle.sample_rate, dtype=torch.float16)
        self.resampler = self.resampler.to('cuda')

    def download_asset_external(self, url, key):
        path = Path(torch.hub.get_dir()) / "torchaudio" / Path(key)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(url, path)
        return str(path)

    def initialize_asr_model(self):
        self.bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H

        # self.get_logger().info("Sample Rate: %s" % str(self.bundle.sample_rate))
        # self.get_logger().info("Labels: %s" % str(self.bundle.get_labels()))

        self.asr_model = self.bundle.get_model().to(self.torch_device)
        # self.asr_model.half()


        self.lm_files = download_pretrained_files("librispeech-4-gram")
        LM_WEIGHT = 3.23
        WORD_SCORE = -0.26

        self.beam_search_decoder = ctc_decoder(
            lexicon=self.lm_files.lexicon,
            tokens=self.lm_files.tokens,
            lm=self.lm_files.lm,
            nbest=3,
            beam_size=1500,
            lm_weight=LM_WEIGHT,
            word_score=WORD_SCORE,
        )

        self.greedy_decoder = GreedyCTCDecoder(labels=self.bundle.get_labels())
        self.cuda_decoder = cuda_ctc_decoder(self.tokens, nbest=10, beam_size=10, blank_skip_threshold=0.95)

        self.acoustic_model = torch.jit.load(self.model_path)
        self.acoustic_model.to(self.torch_device)
        self.acoustic_model.eval()

        self.get_logger().info("Decoder type: %s" % str(type(self.beam_search_decoder)))
        self.get_logger().info("Decoder dir: %s" % str(dir(self.beam_search_decoder)))
        

    def asr(self):

        # Resample to bundle sample rate
        # self.get_logger().info("Voice tensor resample input size: %s" % str(self.voice_tensor.T[0,:]))

        self.voice_tensor_resampled = self.resampler(self.voice_tensor.T).to(self.torch_device)
        torch.save(self.voice_tensor_resampled,'voice_data_resampled.pt')

        with torch.inference_mode():
            # emission, _ = self.asr_model(self.voice_tensor_resampled.float())

            # CPU beam search decoder
            # beam_search_result = self.beam_search_decoder(emission)
            # beam_search_transcript = " ".join(beam_search_result[0][0].words).strip()

            # greedy decoder
            # transcript = self.greedy_decoder(emission[0])

            # CUDA decoder
            feat = torchaudio.compliance.kaldi.fbank(self.voice_tensor_resampled.float(), num_mel_bins=80, snip_edges=False)
            feat = feat.unsqueeze(0)
            feat_lens = torch.tensor(feat.size(1), device=self.torch_device).unsqueeze(0)

            encoder_out, encoder_out_lens = self.acoustic_model.encoder(feat, feat_lens)
            nnet_output = self.acoustic_model.ctc_output(encoder_out)
            log_prob = torch.nn.functional.log_softmax(nnet_output, -1)

            results = self.cuda_decoder(log_prob, encoder_out_lens.to(torch.int32))
            transcript = self.bpe_model.decode(results[0][0].tokens).lower()

            self.get_logger().info("Transcript: %s" % transcript)


    def audio_data_callback(self, msg):
        # self.get_logger().info("Audio CB 1")

        # self.frame = torch.from_numpy(np.frombuffer(msg.audio.data,dtype=np.float16)).view(-1,self.n_channels)


        chunk = torch.from_numpy(np.frombuffer(msg.audio.data,dtype=np.float16)).view(-1,self.n_channels)

        # self.get_logger().info('Got chunk with size %s' % (str(chunk.size())))

        self.voice_chunk = chunk[:,self.voice_idx]

        # self.voice_channels = self.voice_channels.to('cuda')
        # self.get_logger().info("Audio CB 2")

        # run VAD on voice channels
        self.voice_data = torchaudio.functional.vad(self.voice_chunk.T, self.sample_rate, trigger_level=3.0, pre_trigger_time=0.2)

        # self.get_logger().info('Got voice_data with size %s' % (str(self.voice_data.size())))


        # If contains voice data
        if self.voice_data.size(1) != self.min_voice_len:

            self.n_silent = 0

            # self.get_logger().info('Voice data detected')

            # if already recording, append to existing voice tensor
            if self.recording:
                # self.get_logger().info('Continuing recording')
                self.voice_tensor = torch.cat((self.voice_tensor,self.voice_chunk),0)

            # If not recording, start recording with existing voice chunk
            else:
                # self.get_logger().info('Starting recording')
                self.recording = True
                self.voice_tensor = self.voice_chunk
                

        # If it doesn't contain voice data
        else:
            # self.get_logger().info('NO voice data detected')
            self.n_silent +=1

            # If recording, stop recording and process
            if self.recording:

                if self.n_silent >= self.n_trailing_frames:

                    # self.get_logger().info('Ending recording')
                    self.recording = False
                    torch.save(self.voice_tensor.T,'voice_data.pt')
                    self.voice_tensor = self.voice_tensor.to(self.torch_device)
                    self.asr()

                else: 
                    # self.get_logger().info('Continuing recording')
                    self.voice_tensor = torch.cat((self.voice_tensor,self.voice_chunk),0)

            # If not recording, do nothing
        


        # # Roll the frame, and replace oldest contents with new chunk
        # self.frame = torch.roll(self.frame, -chunk.size(0), 0)
        # self.frame[-chunk.size(0):,:] = -chunk

        # self.get_logger().info('Computed frame with size %s' % (str(self.frame.size())))

        # scene_msg = String()
        # scene_msg.data = "Class: %s, %s%%" % (self.audio_scene_labels[class_idx], conf.item())
        # self.audio_scene_publisher.publish(scene_msg)


def main(args=None):
    rclpy.init(args=args)
    voice_proc_node = VoiceProcNode()
    rclpy.spin(voice_proc_node)
    voice_proc_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()