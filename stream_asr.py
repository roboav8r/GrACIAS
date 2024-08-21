import torch
import torchaudio
import torch.multiprocessing as mp

def stream(q, format, src, segment_length, sample_rate, options):
    
    print("Building StreamReader...")
    streamer = torchaudio.io.StreamReader(src=src, format=format, option=options)
    streamer.add_basic_audio_stream(frames_per_chunk=segment_length, sample_rate=sample_rate, num_channels=6)

    print(streamer.get_src_stream_info(0))
    print("Streaming...")
    print()
    for (chunk_a) in streamer.stream(timeout=-1, backoff=1.0):
        q.put([chunk_a])


class ContextCacher:
    """Cache the end of input data and prepend the next input data with it.

    Args:
        segment_length (int): The size of main segment.
            If the incoming segment is shorter, then the segment is padded.
        context_length (int): The size of the context, cached and appended.
    """

    def __init__(self, segment_length: int, context_length: int, n_channels: int):
        self.segment_length = segment_length
        self.context_length = context_length
        self.context = torch.zeros([context_length, n_channels])

    def __call__(self, chunk: torch.Tensor):
        # print("chunk size: %s" % str(chunk.size()) )
        # print("context size: %s" % str(self.context.size()) )
        # print("segment length: %s" % str(self.segment_length))
        if chunk.size(0) < self.segment_length:
            chunk = torch.nn.functional.pad(chunk, (0, self.segment_length - chunk.size(0)))
        chunk_with_context = torch.cat((self.context, chunk))
        # print("chunk with context size: %s" % str(chunk_with_context.size()) )
        self.context = chunk[-self.context_length :,:]
        # print("updated context size: %s" % str(self.context.size()) )
        return chunk_with_context

class Pipeline:
    """Build inference pipeline from RNNTBundle.

    Args:
        bundle (torchaudio.pipelines.RNNTBundle): Bundle object
        beam_width (int): Beam size of beam search decoder.
    """

    def __init__(self, bundle: torchaudio.pipelines.RNNTBundle, beam_width: int = 10):
        self.bundle = bundle
        self.feature_extractor = bundle.get_streaming_feature_extractor()
        self.decoder = bundle.get_decoder()
        self.token_processor = bundle.get_token_processor()

        self.beam_width = beam_width

        self.state = None
        self.hypotheses = None

    def infer(self, segment: torch.Tensor) -> str:
        """Perform streaming inference"""
        features, length = self.feature_extractor(segment)
        self.hypotheses, self.state = self.decoder.infer(
            features, length, self.beam_width, state=self.state, hypothesis=self.hypotheses
        )
        transcript = self.token_processor(self.hypotheses[0][0], lstrip=False)
        return transcript
    
def main():
    # Parameters
    device = "alsa"
    src = "hw:4"
    n_channels = 6
    options = {"sample_rate":"16000","channels":"6"}
    
    # Model info
    bundle=torchaudio.pipelines.EMFORMER_RNNT_BASE_LIBRISPEECH
    sample_rate = bundle.sample_rate
    segment_length = bundle.segment_length * bundle.hop_length
    context_length = bundle.right_context_length * bundle.hop_length
    pipeline = Pipeline(bundle)
    
    # Cache stream
    cacher = ContextCacher(segment_length, context_length, n_channels)
    
    # Inference
    
    ctx = mp.get_context("spawn")
    
    @torch.inference_mode()
    def infer():
        # while True:
        for _ in range(200):
            chunk = q.get()      
            # print(chunk[0][0])
            segment = cacher(chunk[0][0])
            print("segment size before entering pipeline: %s" % str(segment.size()))
            # TODO - get actual audio at location instead of channel 0
            transcript = pipeline.infer(segment[:,0])
            print(transcript, end="\r", flush=True)
    
    q = ctx.Queue()
    p = ctx.Process(target=stream, args=(q, device, src, segment_length, sample_rate, options))
    p.start()
    infer()
    p.join()


if __name__ == "__main__":
    main()
