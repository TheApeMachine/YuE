import json
import numpy as np
import einops


class CodecManipulator(object):
    r"""
    **mm tokenizer v0.1**
    see codeclm/hf/mm_tokenizer_v0.1_hf/id2vocab.json

    text tokens: 
        llama tokenizer 0~31999
    
    special tokens: "32000": "<EOD>", "32001": "<SOA>", "32002": "<EOA>", "32003": "<SOI>", "32004": "<EOI>", "32005": "<SOV>", "32006": "<EOV>", "32007": "<s_local>", "32008": "<e_local>", "32009": "<s_global>", "32010": "<e_global>", "32011": "<semantic>", "32012": "<acoustic>", "32013": "<low_level>", "32014": "<dac_16k>", "32015": "<dac_44k>", "32016": "<xcodec>", "32017": "<placeholder>", "32018": "<semantic_mert>", "32019": "<semantic_hubert>", "32020": "<visual>", "32021": "<semanticodec>"

    mm tokens:
        dac_16k: 4 codebook, 1024 vocab, 32022 - 36117
        dac_44k: 9 codebook, 1024 vocab, 36118 - 45333
        xcodec: 12 codebook, 1024 vocab, 45334 - 57621
        semantic mert: 1024, 57622 - 58645
        semantic hubert: 512, 58646 - 59157
        visual: 64000, not included in v0.1
        semanticodec 100tps 16384: semantic=16384, 59158 - 75541, acoustic=8192, 75542 - 83733
    """
    def __init__(self, codec_type, quantizer_begin=None, n_quantizer=None, teacher_forcing=False, data_feature="codec"):
        self.codec_type = codec_type
        self.mm_v0_2_cfg = {
            "dac16k": {"codebook_size": 1024, "num_codebooks": 4, "global_offset": 32022, "sep": ["<dac_16k>"], "fps": 50},
            "dac44k": {"codebook_size": 1024, "num_codebooks": 9, "global_offset": 36118, "sep": ["<dac_44k>"]},
            "xcodec": {"codebook_size": 1024, "num_codebooks": 12, "global_offset": 45334, "sep": ["<xcodec>"], "fps": 50},
            "mert": {"codebook_size": 1024, "global_offset": 57622, "sep": ["<semantic_mert>"]},
            "hubert": {"codebook_size": 512, "global_offset": 58646, "sep": ["<semantic_hubert>"]},
            "semantic/s": {"codebook_size": 16384, "num_codebooks": 1, "global_offset": 59158, "sep": ["<semanticodec>", "<semantic>"]},
            "semantic/a": {"codebook_size": 8192, "num_codebooks": 1, "global_offset": 75542, "sep": ["<semanticodec>", "<acoustic>"]},
            "semanticodec": {"codebook_size": [16384, 8192], "num_codebooks": 2, "global_offset": 59158, "sep": ["<semanticodec>"], "fps": 50},
            "special_tokens": {
                '<EOD>': 32000, '<SOA>': 32001, '<EOA>': 32002, '<SOI>': 32003, '<EOI>': 32004, '<SOV>': 32005, '<EOV>': 32006, '<s_local>': 32007, '<e_local>': 32008, '<s_global>': 32009, '<e_global>': 32010, '<semantic>': 32011, '<acoustic>': 32012, '<stage_1>': 32013, '<dac_16k>': 32014, '<dac_44k>': 32015, '<xcodec>': 32016, '<stage_2>': 32017, '<semantic_mert>': 32018, '<semantic_hubert>': 32019, '<visual>': 32020, '<semanticodec>': 32021
            },
            "metadata": {
                "len": 83734,
                "text_range": [0, 31999],
                "special_range": [32000, 32021],
                "mm_range": [32022, 83733]
            },
            "codec_range": {
                "dac16k": [32022, 36117],
                "dac44k": [36118, 45333],
                "xcodec": [45334, 57621],
                # "hifi16k": [53526, 57621],
                "mert": [57622, 58645],
                "hubert": [58646, 59157],
                "semantic/s": [59158, 75541],
                "semantic/a": [75542, 83733],
                "semanticodec": [59158, 83733]
            }
        }
        self.sep = self.mm_v0_2_cfg[self.codec_type]["sep"]
        self.sep_ids = [self.mm_v0_2_cfg["special_tokens"][s] for s in self.sep]
        self.codebook_size = self.mm_v0_2_cfg[self.codec_type]["codebook_size"]
        self.num_codebooks = self.mm_v0_2_cfg[self.codec_type]["num_codebooks"]
        self.global_offset = self.mm_v0_2_cfg[self.codec_type]["global_offset"]
        self.fps = self.mm_v0_2_cfg[self.codec_type]["fps"] if "fps" in self.mm_v0_2_cfg[self.codec_type] else None

        self.quantizer_begin = quantizer_begin if quantizer_begin is not None else 0
        self.n_quantizer = n_quantizer if n_quantizer is not None else self.num_codebooks  
        self.teacher_forcing = teacher_forcing 
        self.data_feature = data_feature


    def offset_tok_ids(self, x, global_offset=0, codebook_size=2048, num_codebooks=4):
        """
        x: (K, T)
        """
        if isinstance(codebook_size, int):
            assert x.max() < codebook_size, f"max(x)={x.max()}, codebook_size={codebook_size}"
        elif isinstance(codebook_size, list):
            for i, cs in enumerate(codebook_size):
                assert x[i].max() < cs, f"max(x)={x[i].max()}, codebook_size={cs}, layer_id={i}"
        else:
            raise ValueError(f"codebook_size={codebook_size}")
        assert x.min() >= 0, f"min(x)={x.min()}"
        assert x.shape[0] == num_codebooks or x.shape[0] == self.n_quantizer, \
            f"x.shape[0]={x.shape[0]}, num_codebooks={num_codebooks}, n_quantizer={self.n_quantizer}"

        _x = x.copy()
        _x = _x.astype(np.uint32)
        cum_offset = 0
        quantizer_begin = self.quantizer_begin
        quantizer_end = quantizer_begin+self.n_quantizer
        for k in range(self.quantizer_begin, quantizer_end): # k: quantizer_begin to quantizer_end - 1
            if isinstance(codebook_size, int):
                _x[k] += global_offset + k * codebook_size
            elif isinstance(codebook_size, list):
                _x[k] += global_offset + cum_offset
                cum_offset += codebook_size[k]
            else:
                raise ValueError(f"codebook_size={codebook_size}")
        return _x[quantizer_begin:quantizer_end]

    def unoffset_tok_ids(self, x, global_offset=0, codebook_size=2048, num_codebooks=4):
        """
        x: (K, T)
        """
        if isinstance(codebook_size, int):
            assert x.max() < global_offset + codebook_size * num_codebooks, f"max(x)={x.max()}, codebook_size={codebook_size}"
        elif isinstance(codebook_size, list):
            assert x.max() < global_offset + sum(codebook_size), f"max(x)={x.max()}, codebook_size={codebook_size}"
        assert x.min() >= global_offset, f"min(x)={x.min()}, global_offset={global_offset}"
        assert x.shape[0] == num_codebooks or x.shape[0] == self.n_quantizer, \
            f"x.shape[0]={x.shape[0]}, num_codebooks={num_codebooks}, n_quantizer={self.n_quantizer}"
        
        _x = x.copy()
        _x = _x.astype(np.uint32)
        cum_offset = 0
        quantizer_begin = self.quantizer_begin
        quantizer_end = quantizer_begin+self.n_quantizer
        for k in range(quantizer_begin, quantizer_end):
            if isinstance(codebook_size, int):
                _x[k-quantizer_begin] -= global_offset + k * codebook_size
            elif isinstance(codebook_size, list):
                _x[k-quantizer_begin] -= global_offset + cum_offset
                cum_offset += codebook_size[k]
            else:
                raise ValueError(f"codebook_size={codebook_size}")
        return _x

    def flatten(self, x):
        if len(x.shape) > 2:
            x = x.squeeze()
        assert x.shape[0] == self.num_codebooks or x.shape[0] == self.n_quantizer, \
            f"x.shape[0]={x.shape[0]}, num_codebooks={self.num_codebooks}, n_quantizer={self.n_quantizer}"
        return einops.rearrange(x, 'K T -> (T K)')

    def unflatten(self, x, n_quantizer=None):
        if x.ndim > 1 and x.shape[0] == 1:
            x = x.squeeze(0)
        assert len(x.shape) == 1
        assert x.shape[0] % self.num_codebooks == 0 or x.shape[0] % self.n_quantizer == 0, \
            f"x.shape[0]={x.shape[0]}, num_codebooks={self.num_codebooks}, n_quantizer={self.n_quantizer}"
        if n_quantizer!=self.num_codebooks:
            return einops.rearrange(x, '(T K) -> K T', K=n_quantizer)
        return einops.rearrange(x, '(T K) -> K T', K=self.num_codebooks)
        
    def get_codec_type_from_range(self, ids):
        ids_range = [ids.min(), ids.max()]
        codec_range = self.mm_v0_2_cfg["codec_range"]
        for codec_type, r in codec_range.items():
            if ids_range[0] >= r[0] and ids_range[1] <= r[1]:
                return codec_type
        raise ValueError(f"ids_range={ids_range}, codec_range={codec_range}")

    def npy2ids(self, npy):
        if isinstance(npy, str):
            data = np.load(npy)
        elif isinstance(npy, np.ndarray):
            data = npy
        else:
            raise ValueError(f"not supported type: {type(npy)}")

        assert len(data.shape)==2,  f'data shape: {data.shape} is not (n_codebook, seq_len)'
        data = self.offset_tok_ids(
            data, 
            global_offset=self.global_offset, 
            codebook_size=self.codebook_size, 
            num_codebooks=self.num_codebooks, 
        )
        data = self.flatten(data)
        codec_range = self.get_codec_type_from_range(data)
        assert codec_range == self.codec_type, f"get_codec_type_from_range(data)={codec_range}, self.codec_type={self.codec_type}"
        data = data.tolist()
        return data
    
    def ids2npy(self, token_ids):
        # make sure token_ids starts with codebook 0
        if isinstance(self.codebook_size, int):
            codebook_0_range = (self.global_offset + self.quantizer_begin*self.codebook_size, self.global_offset + (self.quantizer_begin+1)*self.codebook_size)
        elif isinstance(self.codebook_size, list):
            codebook_0_range = (self.global_offset, self.global_offset + self.codebook_size[0])
        assert token_ids[0] >= codebook_0_range[0] \
            and token_ids[0] < codebook_0_range[1], f"token_ids[0]={token_ids[self.quantizer_begin]}, codebook_0_range={codebook_0_range}"
        data = np.array(token_ids)
        data = self.unflatten(data, n_quantizer=self.n_quantizer)
        data = self.unoffset_tok_ids(
            data, 
            global_offset=self.global_offset, 
            codebook_size=self.codebook_size, 
            num_codebooks=self.num_codebooks, 
        )
        return data

    def npy_to_json_str(self, npy_path):
        data = self.npy2ids(npy_path)
        return json.dumps({"text": data, "src": npy_path, "codec": self.codec_type})
    
    def sep(self):
        return ''.join(self.sep)
    
    def sep_ids(self):
        return self.sep_ids

class StereoCodecManipulator(CodecManipulator):
    """Extension of CodecManipulator to handle stereo audio tokens"""
    
    def __init__(self, codec_type, quantizer_begin=None, n_quantizer=None, teacher_forcing=False, data_feature="codec"):
        """
        Initialize the StereoCodecManipulator
        
        Args:
            codec_type: Type of codec to use
            quantizer_begin: Start quantizer index
            n_quantizer: Number of quantizers to use
            teacher_forcing: Whether to use teacher forcing
            data_feature: Data feature type
        """
        super().__init__(codec_type, quantizer_begin, n_quantizer, teacher_forcing, data_feature)
        # Define dedicated stereo channel markers
        self.special_tokens = self.mm_v0_2_cfg["special_tokens"]
        self.left_marker = self.special_tokens["<s_local>"]
        self.right_marker = self.special_tokens["<s_global>"]
    
    def process_stereo(self, left_codes, right_codes):
        """
        Process left and right channel codes and interleave them
        
        Args:
            left_codes: Encoded tokens for left channel
            right_codes: Encoded tokens for right channel
            
        Returns:
            Combined stereo tokens with channel markers
        """
        # Add channel markers to distinguish left/right
        left_tokens = self.npy2ids(left_codes[0])
        right_tokens = self.npy2ids(right_codes[0])
        
        # Interleave with special channel markers
        # Format as: [LEFT_MARKER] [left_tokens] [RIGHT_MARKER] [right_tokens]
        stereo_tokens = [self.left_marker] + left_tokens + [self.right_marker] + right_tokens
        
        return stereo_tokens
    
    def deprocess_stereo(self, stereo_tokens):
        """
        Split interleaved stereo tokens back into separate channels
        
        Args:
            stereo_tokens: Combined stereo tokens
            
        Returns:
            Left and right channel tokens
        """
        # Find marker positions
        try:
            left_start = stereo_tokens.index(self.left_marker) + 1
        except ValueError:
            print("Warning: Left channel marker not found, using position 0")
            left_start = 0
            
        try:
            right_start = stereo_tokens.index(self.right_marker) + 1
        except ValueError:
            print("Warning: Right channel marker not found, using midpoint")
            right_start = len(stereo_tokens) // 2
        
        # Extract tokens for each channel
        if self.right_marker in stereo_tokens[left_start:]:
            right_idx = stereo_tokens[left_start:].index(self.right_marker) + left_start
            left_tokens = stereo_tokens[left_start:right_idx]
        else:
            # If right marker not found after left marker, use the first half for left
            mid_point = (len(stereo_tokens) - left_start) // 2 + left_start
            left_tokens = stereo_tokens[left_start:mid_point]
            
        right_tokens = stereo_tokens[right_start:]
        
        # Validate token types and ranges
        if left_tokens and right_tokens:
            # Convert back to numpy arrays
            left_data = self.ids2npy(left_tokens)
            right_data = self.ids2npy(right_tokens)
            
            return left_data, right_data
        else:
            # Create default empty arrays if either channel is missing
            empty_array = np.zeros((self.num_codebooks, 1), dtype=np.int32)
            if not left_tokens:
                print("Warning: No left channel tokens found, using empty array")
                left_data = empty_array 
            else:
                left_data = self.ids2npy(left_tokens)
                
            if not right_tokens:
                print("Warning: No right channel tokens found, using empty array")
                right_data = empty_array
            else:
                right_data = self.ids2npy(right_tokens)
                
            return left_data, right_data
