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

class SceneVoiceNode(Node):

    def __init__(self):
        super().__init__('scene_voice_node')

        self.subscription = self.create_subscription(AudioDataStamped, 'audio_data', self.audio_data_callback, 10)
        self.scene_transcript_publisher = self.create_publisher(String, 'scene_voice_data', 10)
        
        # Declare parameters with default values
        self.declare_parameter('n_channels', rclpy.Parameter.Type.INTEGER)
        self.declare_parameter('audio_sample_rate', rclpy.Parameter.Type.INTEGER)
        self.declare_parameter('hop_size', rclpy.Parameter.Type.INTEGER)
        self.declare_parameter('frame_size', rclpy.Parameter.Type.INTEGER)
        self.declare_parameter('voice_index', rclpy.Parameter.Type.INTEGER_ARRAY)
        self.declare_parameter('lexicon_file', rclpy.Parameter.Type.STRING)

        # Retrieve parameters
        self.n_channels = self.get_parameter('n_channels').get_parameter_value().integer_value
        self.audio_sample_rate = self.get_parameter('audio_sample_rate').get_parameter_value().integer_value
        self.hop_size = self.get_parameter('hop_size').get_parameter_value().integer_value
        self.frame_size = self.get_parameter('frame_size').get_parameter_value().integer_value
        self.voice_index = self.get_parameter('voice_index').get_parameter_value().integer_array_value
        self.lexicon_file = self.get_parameter('lexicon_file').get_parameter_value().string_value

        # Audio data storage
        self.torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hop = torch.zeros([self.hop_size, self.n_channels],dtype=torch.float16)
        self.frame = torch.zeros([self.frame_size, self.n_channels],dtype=torch.float16)
        self.voice_frame = torch.zeros([self.frame_size, len(self.voice_index)],dtype=torch.float16)
        self.voice_frame_sum = torch.zeros([self.frame_size, 1],dtype=torch.float16)
        
        self.hop = self.hop.to(self.torch_device)
        self.frame = self.frame.to(self.torch_device)
        self.voice_frame = self.voice_frame.to(self.torch_device)
        self.voice_frame_sum = self.voice_frame_sum.to(self.torch_device)
        
        self.min_voice_len = 6615 # TODO compute this based on f_sampling, VAD times
        self.n_trailing_frames = 2 # prevent early cutoff. Only stop recording if silent for n consecutive frames.
        self.n_silent = 0 
        self.recording = False

        # Model params
        self.url_prefix = "https://huggingface.co/Zengwei/icefall-asr-librispeech-pruned-transducer-stateless7-ctc-2022-12-01"
        self.model_link = f"{self.url_prefix}/resolve/main/exp/cpu_jit.pt"
        self.model_path = self.download_asset_external(self.model_link, "cuda_ctc_decoder/cpu_jit.pt")
        self.bpe_link = f"{self.url_prefix}/resolve/main/data/lang_bpe_500/bpe.model"
        self.bpe_path = self.download_asset_external(self.bpe_link, "cuda_ctc_decoder/bpe.model")
        self.bpe_model = spm.SentencePieceProcessor()
        self.bpe_model.load(self.bpe_path)
        self.tokens = [self.bpe_model.id_to_piece(id) for id in range(self.bpe_model.get_piece_size())]
        torch.random.manual_seed(0)
        self.initialize_asr_model()
        self.resampler = T.Resample(self.audio_sample_rate, self.bundle.sample_rate, dtype=torch.float16)
        self.resampler = self.resampler.to(self.torch_device)

    def download_asset_external(self, url, key):
        path = Path(torch.hub.get_dir()) / "torchaudio" / Path(key)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(url, path)
        return str(path)

    def initialize_asr_model(self):
        self.bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        self.asr_model = self.bundle.get_model().to(self.torch_device)

        self.lm_files = download_pretrained_files("librispeech-4-gram")
        self.get_logger().info("LM Files: %s" % str(self.lm_files))
        LM_WEIGHT = 3.23
        WORD_SCORE = -0.26

        self.beam_search_decoder = ctc_decoder(
            lexicon=os.path.join(get_package_share_directory('situated_interaction'), self.lexicon_file),
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

        self.voice_tensor_resampled = self.resampler(self.voice_tensor).to(self.torch_device)
        torch.save(self.voice_tensor_resampled,'voice_data_resampled.pt')

        with torch.inference_mode():
            emission, _ = self.asr_model(self.voice_tensor_resampled.view(1,-1).float())

            # CPU beam search decoder
            beam_search_result = self.beam_search_decoder(emission.cpu())
            beam_search_transcript = " ".join(beam_search_result[0][0].words).strip()

            # greedy decoder
            # transcript = self.greedy_decoder(emission[0])

            # # CUDA decoder
            # feat = torchaudio.compliance.kaldi.fbank(self.voice_tensor_resampled.view(1,-1).float(), num_mel_bins=80, snip_edges=False)
            # feat = feat.unsqueeze(0)
            # feat_lens = torch.tensor(feat.size(1), device=self.torch_device).unsqueeze(0)

            # encoder_out, encoder_out_lens = self.acoustic_model.encoder(feat, feat_lens)
            # nnet_output = self.acoustic_model.ctc_output(encoder_out)
            # log_prob = torch.nn.functional.log_softmax(nnet_output, -1)

            # results = self.cuda_decoder(log_prob, encoder_out_lens.to(torch.int32))
            # transcript = self.bpe_model.decode(results[0][0].tokens).lower()

            self.get_logger().info("Transcript: %s" % beam_search_transcript)


    def audio_data_callback(self, msg):
        self.hop = torch.from_numpy(np.frombuffer(msg.audio.data,dtype=np.float16)).view(-1,self.n_channels)

        # Roll the frame, and replace oldest contents with new chunk
        self.frame = torch.roll(self.frame, -self.hop.size(0), 0)
        self.frame[-self.hop.size(0):,:] = -self.hop

        self.voice_frame = self.frame[:,self.voice_index]

        torch.save(self.voice_frame.T,'voice_frame.pt')

        self.voice_frame_sum = torch.mean(self.voice_frame, 1)
        self.voice_hop_sum = torch.mean(self.hop[:,self.voice_index],1)

        torch.save(self.voice_frame_sum,'voice_frame_sum.pt')
        torch.save(self.voice_hop_sum,'voice_hop_sum.pt')

        # run VAD on voice channels
        self.voice_data = torchaudio.functional.vad(self.voice_frame_sum, self.audio_sample_rate, trigger_level=3.0) # TODO - make these reconfigurable params

        # self.get_logger().info('Got voice_data with size %s' % (str(self.voice_data.size())))


        # If contains voice data
        if len(self.voice_data) > self.min_voice_len:

            self.n_silent = 0

            self.get_logger().info('Voice data detected')

            # if already recording, append to existing voice tensor
            if self.recording:
                # self.get_logger().info('Continuing recording')
                self.voice_tensor = torch.cat((self.voice_tensor,self.voice_hop_sum),0)

            # If not recording, start recording with existing voice chunk
            else:
                # self.get_logger().info('Starting recording')
                self.recording = True
                self.voice_tensor = self.voice_hop_sum
                

        # If it doesn't contain voice data
        else:
            # self.get_logger().info('NO voice data detected')
            self.n_silent +=1

            # If recording, stop recording and process
            if self.recording:

                if self.n_silent >= self.n_trailing_frames:

                    # self.get_logger().info('Ending recording')
                    self.recording = False
                    torch.save(self.voice_tensor,'voice_data.pt')
                    self.voice_tensor = self.voice_tensor.to(self.torch_device)
                    self.asr()

                else: 
                    # self.get_logger().info('Continuing recording')
                    self.voice_tensor = torch.cat((self.voice_tensor,self.voice_hop_sum),0)

            # If not recording, do nothing

def main(args=None):
    rclpy.init(args=args)
    scene_voice_node = SceneVoiceNode()
    rclpy.spin(scene_voice_node)
    scene_voice_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()