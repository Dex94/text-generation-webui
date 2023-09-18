import re
import numpy as np
import torch
from functools import partial
from modules import RoPE, shared
from modules.callbacks import Iteratorize
from modules.logging_colors import logger
from modules.text_generation import get_max_prompt_length
import llama_cpp

# Importa llama_cpp_cuda solo se è disponibile CUDA
if torch.cuda.is_available() and not torch.version.hip:
    try:
        import llama_cpp_cuda
    except ImportError:
        llama_cpp_cuda = None
else:
    llama_cpp_cuda = None

def llama_cpp_lib():
    return llama_cpp_cuda if not shared.args.cpu or llama_cpp_cuda is not None else llama_cpp

def ban_eos_logits_processor(eos_token, input_ids, logits):
    logits[eos_token] = -float('inf')
    return logits

def custom_token_ban_logits_processor(token_ids, input_ids, logits):
    for token_id in token_ids:
        logits[token_id] = -float('inf')
    return logits

class LlamaCppModel:
    def __init__(self):
        self.initialized = False

    def __del__(self):
        self.model.__del__()

    @classmethod
    def from_pretrained(cls, path):
        Llama = llama_cpp_lib().Llama
        LlamaCache = llama_cpp_lib().LlamaCache

        result = cls()
        cache_capacity = get_cache_capacity(shared.args.cache_capacity)
        logger.info(f"Cache capacity is {cache_capacity} bytes")

        tensor_split_list = get_tensor_split_list(shared.args.tensor_split)

        params = {
            'model_path': str(path),
            'n_ctx': shared.args.n_ctx,
            'seed': int(shared.args.llama_cpp_seed),
            'n_threads': shared.args.threads or None,
            'n_batch': shared.args.n_batch,
            'use_mmap': not shared.args.no_mmap,
            'use_mlock': shared.args.mlock,
            'mul_mat_q': shared.args.mul_mat_q,
            'low_vram': shared.args.low_vram,
            'n_gpu_layers': shared.args.n_gpu_layers,
            'rope_freq_base': RoPE.get_rope_freq_base(shared.args.alpha_value, shared.args.rope_freq_base),
            'tensor_split': tensor_split_list,
            'rope_freq_scale': 1.0 / shared.args.compress_pos_emb,
        }

        result.model = Llama(**params)
        if cache_capacity > 0:
            result.model.set_cache(LlamaCache(capacity_bytes=cache_capacity))
		# This is ugly, but the model and the tokenizer are the same object in this library.
        return result, result

    def encode(self, string):
        if isinstance(string, str):
            string = string.encode()
        return self.model.tokenize(string)

    def decode(self, ids):
        return self.model.detokenize(ids).decode('utf-8')

    def get_logits(self, tokens):
        self.model.eval(tokens)
        logits = self.model._scores
        logits = np.expand_dims(logits, 0)  # batch dim is expected
        return torch.tensor(logits, dtype=torch.float32)

    def generate(self, prompt, state, callback=None):
        LogitsProcessorList = llama_cpp_lib().LogitsProcessorList
        prompt = prompt if isinstance(prompt, str) else prompt.decode()

        # Handle truncation
        prompt = self.encode(prompt)
        prompt = prompt[-get_max_prompt_length(state):]
        prompt = self.decode(prompt)
        logit_processors = LogitsProcessorList()
        if state['ban_eos_token']:
            logit_processors.append(partial(ban_eos_logits_processor, self.model.tokenizer.()))

        if state['custom_token_bans']:
            to_ban = [int(x) for x in state['custom_token_bans'].split(',')]
            if to_ban:
                logit_processors.append(partial(custom_token_ban_logits_processor, to_ban))

        completion_chunks = self.model.create_completion(
            prompt=prompt,
            max_tokens=state['max_new_tokens'],
            temperature=state['temperature'],
            top_p=state['top_p'],
            top_k=state['top_k'],
            repeat_penalty=state['repetition_penalty'],
            tfs_z=state['tfs'],
            mirostat_mode=int(state['mirostat_mode']),
            mirostat_tau=state['mirostat_tau'],
            mirostat_eta=state['mirostat_eta'],
            stream=True,
            logits_processor=logit_processors,
        )

        output = ""
        for completion_chunk in completion_chunks:
            if shared.stop_everything:
                break
            text = completion_chunk['choices'][0]['text']
            output += text
            if callback:
                callback(text)

        return output

    def generate_with_streaming(self, *args, **kwargs):
        with Iteratorize(self.generate, args, kwargs, callback=None) as generator:
            reply = ''
            for token in generator:
                reply += token
                yield reply

def get_cache_capacity(cache_capacity):
    if not cache_capacity:
        return 0
    if 'GiB' in cache_capacity:
        return int(re.sub('[a-zA-Z]', '', cache_capacity)) * 1000 * 1000 * 1000
    elif 'MiB' in cache_capacity:
        return int(re.sub('[a-zA-Z]', '', cache_capacity)) * 1000 * 1000
    else:
        return int(cache_capacity)

def get_tensor_split_list(tensor_split):
    if not tensor_split or tensor_split.strip() == '':
        return None
    return [float(x) for x in tensor_split.strip().split(",")]

