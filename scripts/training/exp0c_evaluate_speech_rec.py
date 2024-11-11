#!/usr/bin/env python

import os
import time
import numpy as np
import pandas as pd

import pyroomacoustics as pra
from IPython.display import Audio

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from ament_index_python.packages import get_package_share_directory

import torch, torchaudio
import torchaudio.transforms as T
from torchaudio.models.decoder import download_pretrained_files, ctc_decoder


# params
# filepath = '../../bags/e1_speech_dev'
filepath = '../../bags/e1_est_tuning'
actual_scene_labels = ['campus','courtyard','lab','lobby']

# sampling/acquisition
sample_rate = 44100
hop_size = 44100
frame_size = 44100 # publish 1 second every second
n_total_channels = 16
channel_indices_used = [0, 1, 2, 3, 4, 5, 6, 7]
n_channels_used = len(channel_indices_used)

speed_sound = 343.
n_sources = 2
n_fft = 512
f_min = 300
f_max = 8000
doa_dimension = 3
array_x_pos = [0.43, 0.43, -0.34, -0.34, 0., 0., 0., 0.]
array_y_pos = [-0.28, 0.28, 0.30, -0.30, -0.29, 0.29, -0.11, 0.11]
array_z_pos = [0.395, 0.395, 0.395, 0.395, 0.610, 0.610, 0.660, 0.660]
array_pos = np.array([array_x_pos,
                      array_y_pos]) # , array_z_pos
ssl_algo = 'MUSIC'
excess_front = int(np.ceil((n_fft-1)/2))
excess_back = int(np.floor((n_fft-1)/2))

# beamformer params

# VAD params
n_silent_frames = 2
trigger_time = 0.25
search_time = 0.25
allowed_gap = 0.25
pre_trigger_time = 0.25
min_voice_samples = 31200

# LM params
lexicon_package = 'situated_interaction'
lexicon_file = 'config/rocog_lexicon_full_phrases.txt'

# Tunable parameters
vad_trigger_levels = [3., 5., 7.]
am_bundles = ['WAV2VEC2_ASR_LARGE_960H']
lm_weights = [-3, -1., 0., 1., 3.]
word_scores = [-3, -1., 0., 1., 3.]
sil_scores = [-3, -1., 0., 1., 3.]


# Create beamformer object, compute beam weights
source_angle = 0
bf = pra.Beamformer(array_pos, sample_rate, n_fft)
bf.far_field_weights(source_angle)

# Compute voice activity detector object
torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                  
lm_files = download_pretrained_files("librispeech-4-gram")


# Helper functions
def typename(topic_name, topic_types):
    for topic_type in topic_types:
        if topic_type.name == topic_name:
            return topic_type.type
    raise ValueError(f"topic {topic_name} not in bag")


# Initialize results
word_error_results = {}

for trigger_level in vad_trigger_levels:
    
    # Create VAD object, get name    
    vad = torchaudio.transforms.Vad(sample_rate, 
                                    trigger_level=trigger_level, 
                                    trigger_time=trigger_time,
                                    search_time=search_time,
                                    allowed_gap=allowed_gap,
                                    pre_trigger_time=pre_trigger_time) 
    vad.to(torch_device)
    vad_str = 'tl_%s' % (str(trigger_level).replace('.','')) # VAD param

    for bundle_name in am_bundles:
        # Create acoustic model, get name
        exec('bundle = torchaudio.pipelines.%s' % bundle_name)
        asr_model = bundle.get_model().to(torch_device)
        
        if bundle_name == 'WAV2VEC2_ASR_BASE_960H':
            acoustic_model_str = 'am_base'
        elif bundle_name == 'WAV2VEC2_ASR_LARGE_960H':
            acoustic_model_str = 'am_large'
        else:
            error('Not a valid bundle name')

        resampler = T.Resample(sample_rate, bundle.sample_rate, dtype=torch.float16)
        resampler = resampler.to(torch_device)

        
        for lm_weight in lm_weights:

            for word_score in word_scores:

                for sil_score in sil_scores:

                    # Create language model, get name
                    beam_search_decoder = ctc_decoder(
                        lexicon=os.path.join(get_package_share_directory(lexicon_package), lexicon_file),
                        tokens=lm_files.tokens,
                        lm=lm_files.lm,
                        nbest=1,
                        beam_size=50,
                        lm_weight=lm_weight,
                        word_score=word_score,
                        sil_score=sil_score
                    )
                    lang_model_str = 'lm_%s_ws_%s_ss_%s' % (str(lm_weight).replace('.',''),str(word_score).replace('.',''),str(sil_score).replace('.',''))
                    
                    model_str = "%s_%s_%s)" % (vad_str, acoustic_model_str, lang_model_str)
                    print("Evaluating %s" % model_str)
                    
                    word_error_results[model_str] = {}
                    word_error_results[model_str]['overall'] = {}
                    word_error_results[model_str]['overall']['total_words'] = 0
                    word_error_results[model_str]['overall']['incorrect_words'] = 0
                    
                    # Iterate through each file
                    # Traverse filepath
                    for root, _, files in os.walk(filepath):
                    
                        # Examine each file
                        for file in files:
                            
                            # If it is an mcap, examine
                            if os.path.splitext(file)[-1] == '.mcap':
                    
                                # Get labels based on filepath
                                labels = os.path.normpath(root).split(os.sep)
                                command_actual = labels[-1]
                                cmd_mode_actual = labels[-2]
                                role_actual = labels[-3]
                                scene_actual = labels[-4]
                                iteration = os.path.splitext(file)[-2].split('_')[-1]
                    
                                scene_actual_idx = actual_scene_labels.index(scene_actual)
                    
                                if cmd_mode_actual not in ['gest_verb','verbal']:
                                    continue
                    
                                # Add new result dictionary keys
                                if scene_actual not in word_error_results[model_str].keys():
                                    word_error_results[model_str][scene_actual] = {}
                                    word_error_results[model_str][scene_actual]['total_words'] = 0
                                    word_error_results[model_str][scene_actual]['incorrect_words'] = 0
                                
                                # Create reader object           
                                reader = rosbag2_py.SequentialReader()            
                                reader.open(
                                    rosbag2_py.StorageOptions(uri=os.path.join(root,file), storage_id="mcap"),
                                    rosbag2_py.ConverterOptions(
                                        input_serialization_format="cdr", output_serialization_format="cdr"
                                    ),
                                )
                                topic_types = reader.get_all_topics_and_types()
                    
                    
                                # RESET
                                # Create empty audio frame
                                audio_straight_ahead = np.array([])
                                
                                # Iterate through messages
                                while reader.has_next():
                                    topic, data, timestamp = reader.read_next()
                                    msg_type = get_message(typename(topic,topic_types))
                    
                                    if topic=='/audio_data':
                                        audio_msg = deserialize_message(data,msg_type)
                                
                                        all_channels_chunk = torch.frombuffer(audio_msg.audio.data,dtype=torch.float16).view(-1,n_total_channels)
                                        selected_channels_chunk = all_channels_chunk[:,channel_indices_used]
                                    
                                        bf.record(selected_channels_chunk.T, sample_rate)
                                        signal = bf.process(FD=False).astype(np.float16)
                                                                    
                                        audio_straight_ahead_chunk = signal[excess_front:-excess_back] #.tobytes()
                                        audio_straight_ahead = np.concatenate((audio_straight_ahead,audio_straight_ahead_chunk))
                                        
                    
                                # Run voice activity detection on frame if it isn't empty
                                if (audio_straight_ahead.size > 0):
                                    audio_straight_ahead_tensor = torch.from_numpy(audio_straight_ahead).half()
                                    audio_straight_ahead_tensor.to(torch_device)
                                    voice_tensor = vad.forward(audio_straight_ahead_tensor).to(torch_device)
                                    
                                    # run ASR
                                    voice_tensor_resampled = resampler(voice_tensor).float()
                            
                                    with torch.inference_mode():
                                        emission, _ = asr_model(voice_tensor_resampled.view(1,-1))
        
                                        if (torch.isnan(emission)==False).all().item():
                                            # CPU CTC beam search decoder
                                            beam_search_result = beam_search_decoder(emission.cpu())
                                            scores = [hyp.score for hyp in beam_search_result[0]]
                                            beam_search_transcript = " ".join(beam_search_result[0][0].words).strip()
                                        
                                            # compare result to truth value, and scene
                                            # actual_transcript = cmd_transcripts[command_actual].split()
                                            incorrect_words = torchaudio.functional.edit_distance([command_actual], beam_search_result[0][0].words)
                                
                                            # Increment results dictionary
                                            # print("Actual command: %s" % command_actual)
                                            # print("Transcript: %s" % beam_search_transcript)
                                            # print("Incorrect words: %s" % incorrect_words)
                                            # print()

                                            word_error_results[model_str][scene_actual]['total_words'] += 1
                                            word_error_results[model_str]['overall']['total_words'] += 1
                                
                                            word_error_results[model_str][scene_actual]['incorrect_words'] += incorrect_words
                                            word_error_results[model_str]['overall']['incorrect_words'] += incorrect_words
        
                                        else:
                                            print("Problem with %s/%s - emission nan" % (root, file))
                                else:
                                    print("Problem with %s/%s - no audio data" % (root, file))

# analyze word error results
results_columns = ['Model',
                   'Scene',
                   'Total Commands',
                   'Incorrect Commands',
                   'Command Error Rate']
results_df = pd.DataFrame(columns=results_columns)

idx=0
for model_key in word_error_results.keys():
    for scene_key in word_error_results[model_key].keys():
    
        results_df.loc[idx,'Model'] = model_key
        results_df.loc[idx,'Scene'] = scene_key
        results_df.loc[idx,'Total Commands'] = word_error_results[model_key][scene_key]['total_words']
        results_df.loc[idx,'Incorrect Commands'] = word_error_results[model_key][scene_key]['incorrect_words']
        results_df.loc[idx,'Command Error Rate'] = results_df.loc[idx,'Incorrect Commands']/results_df.loc[idx,'Total Commands']

        idx+=1


results_df.to_csv("../../results/speech_eval.csv")