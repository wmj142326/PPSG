##############################################################
## text2pose                                                ##
## Copyright (c) 2022-present                               ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## Naver Corporation                                        ##
## CC BY-NC-SA 4.0                                          ##
##############################################################

import os
import random
import pickle
import json
import hashlib
import re
from copy import deepcopy
from tqdm import tqdm 
import torch

import text2pose.config as config
import text2pose.utils as utils


################################################################################
## TOKENIZERS
################################################################################

import nltk

def collate_fn_general(batch):

    batch_out = {}

    batch_out['pose'] = torch.stack([item['pose'] for item in batch], dim=0)

    first_token = batch[0]['caption_tokens']
    if isinstance(first_token, torch.Tensor):
        batch_out['caption_tokens'] = torch.stack([item['caption_tokens'] for item in batch], dim=0)
    elif isinstance(first_token, str):
        batch_out['caption_tokens'] = [item['caption_tokens'] for item in batch]
    else:
        raise TypeError(f"Unsupported caption_tokens type: {type(first_token)}")

    lengths = [item['caption_lengths'] for item in batch]
    if isinstance(lengths[0], torch.Tensor):
        batch_out['caption_lengths'] = torch.stack(lengths, dim=0)
    else:
        batch_out['caption_lengths'] = torch.tensor(lengths, dtype=torch.long)
    batch_out['pose_class'] = torch.stack([item['pose_class'] for item in batch], dim=0)
    return batch_out


def Tokenizer(text_encoder_name):
    if text_encoder_name.split("_")[0] in ["glovebigru"]:
        return BaseTokenizer(text_encoder_name)
    elif text_encoder_name.split("_")[0] in ["Qwen", "qwen"]:
        return QwenTokenizer()
    else:
        raise NotImplementedError


class BaseTokenizer():
    def __init__(self, text_encoder_name):

        vocab_ref = text_encoder_name.split("_")[1]
        vocab_file = os.path.join(config.POSESCRIPT_LOCATION, config.vocab_files[vocab_ref])
        assert os.path.isfile(vocab_file), f"Vocab file not found ({vocab_file})."
        with open(vocab_file, 'rb') as f:
            self.vocab = pickle.load(f)
        
        self.pad_token_id = 0
        self.max_tokens = config.MAX_TOKENS

    def __call__(self, text):
        tokens = nltk.tokenize.word_tokenize(text.lower()) # keeps punctuations
        tokens = [self.vocab('<start>')] + [self.vocab(token) for token in tokens] + [self.vocab('<end>')]
        tokens = torch.tensor(tokens).long()
        return tokens

    def token2id(self, token):
        return self.vocab(token)

class QwenTokenizer:
    def __init__(self):
        self.pad_token_id = None      
        self.max_tokens = None        

    def __call__(self, text):
        return text

    def decode(self, tokens):
        return tokens if isinstance(tokens, str) else " ".join(tokens)


################################################################################
## POSESCRIPT DATASET
################################################################################

class PoseScript(object):

    def __init__(self, version="posescript-A1", split='train',
                    text_encoder_name='glovebigru_vocA1H1', caption_index='rand',
                    cache=True, generated_pose_samples_path=None):

        # NOTE: generated_pose_samples_path should be None if training on
        # original poses, otherwise it should be a path to a .pth file
        # containing the generated poses for each data point of a given split.
        # In the filepath, there must be a '{data_version}' and a '{split}'
        # fields to substitute.
        
        self.version = version
        self.split = split
        assert type(caption_index) is int or caption_index in ['deterministic-mix', 'rand']
        self.caption_index = caption_index
        self.use_generated_pose_samples = True if generated_pose_samples_path else False
        
        if text_encoder_name.split("_")[0] in ["Qwen", "qwen"]:
            cache=False
            
        self.cache = cache
        if cache:
            cache_file = config.cache_file_path.format(data_version=version, split=split, tokenizer=text_encoder_name)
            # create the cache file if it does not exist
            if not os.path.isfile(cache_file):
                print(f'Caching data [{version} version][{text_encoder_name} tokenization][{split} split]')
                d = PoseScript(version=version, split=split, text_encoder_name=text_encoder_name, cache=False)
                self._data_cache = []
                for index in tqdm(range(len(d))):
                    pose = d.get_pose(index)
                    caption_list = d.get_all_captions(index)
                    caption_tokens_list = [d.tokenizer(caption) for caption in caption_list]
                    caption_length_list = [len(caption_tokens) for caption_tokens in caption_tokens_list]
                    # padd tokenized captions
                    caption_tokens_list = [torch.cat( (caption_tokens, d.tokenizer.pad_token_id * torch.ones( d.tokenizer.max_tokens-len(caption_tokens), dtype=caption_tokens.dtype) ), dim=0) for caption_tokens in caption_tokens_list]
                    self._data_cache.append( (pose, caption_tokens_list, caption_length_list, d.dataIDs[index]) )
                # save data
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                with open(cache_file, 'wb') as f:
                    pickle.dump(self._data_cache, f)
                print('done')
            else:
                # load data from cache
                with open(cache_file, 'rb') as f:
                    self._data_cache = pickle.load(f)
        else:
            # Load data
            self._load_data()
            # Define tokenizer
            self.tokenizer = Tokenizer(text_encoder_name)

        # load generated pose samples
        if self.use_generated_pose_samples:
            print('Using generated poses samples.')
            self.pose_samples = torch.load(generated_pose_samples_path.format(data_version=self.version, split=self.split)) # tensor, size (dataset_size, ncaptions, nsamples, njoints, 3)

    def _load_data(self):

        self.dataIDs = utils.read_posescript_json(f"{self.split}_ids.json") # split dependent
        self.dataID_2_pose_info = utils.read_posescript_json("ids_2_dataset_sequence_and_frame_index.json")  # 20000
        
        # load automatic captions
        if "posescript-A" in self.version:
            self.captions = {data_id: [] for data_id in self.dataIDs}
            for caption_file in config.caption_files[self.version]:
                capts = utils.read_posescript_json(caption_file)
                for data_id in self.dataIDs:
                    self.captions[data_id].append(capts[str(data_id)])
        
        # load human-written captions
        elif "posescript-H" in self.version:
            capts = utils.read_posescript_json(config.caption_files[self.version]) # expecting only one file
            dataIDs = [] # to be updated with data actually available in PoseScript-H
            self.captions = {}
            for data_id in self.dataIDs:
                if str(data_id) in capts:
                    self.captions[data_id] = [capts[str(data_id)]]
                    dataIDs.append(data_id)
            self.dataIDs = dataIDs

    def get_pose(self, index):
        # load pose data
        pose_info = self.dataID_2_pose_info[str(self.dataIDs[index])]
        pose = utils.get_pose_data_from_file(pose_info)
        pose = pose.reshape(-1, 3) # (njoints, 3)
        return pose

    def get_generated_pose(self, index, cidx=None):
        # self.pose_samples is of size (dataset_size, ncaptions, nsamples, njoints, 3)
        if cidx is None: cidx = self.get_caption_index(self.pose_samples.shape[1], index)
        s = random.randint(0, self.pose_samples.shape[2]-1)
        return self.pose_samples[index][cidx][s] # (njoints, 3)
        
    def get_all_captions(self, index):
        data_id = self.dataIDs[index] if not self.cache else self._data_cache[index][-1]
        return self.captions[data_id]
        
    def get_caption(self, index, cidx=None):
        caption_list = self.get_all_captions(index)
        if cidx is None: cidx = self.get_caption_index(len(caption_list), index)
        return caption_list[cidx]
        
    def get_caption_index(self, n, index):
        if self.caption_index=='deterministic-mix':
            return index % n
        elif self.caption_index=='rand':
            return random.randint(0, n-1)
        elif self.caption_index < n:
            return self.caption_index
        raise ValueError
       
    def __len__(self):
        return len(self.dataIDs) if not self.cache else len(self._data_cache)
        
    def __getitem__(self, index, cidx=None):

        if self.cache: 
            pose, caption_tokens_list, caption_length_list, _ = deepcopy(self._data_cache[index])
            cidx = cidx if cidx else self.get_caption_index(len(caption_tokens_list), index)
            caption_tokens = caption_tokens_list[cidx]
            caption_lengths = caption_length_list[cidx]
        else:
            pose = None if self.use_generated_pose_samples else self.get_pose(index)
            caption = self.get_caption(index, cidx=cidx)
            caption_tokens = self.tokenizer(caption)
            caption_lengths = len(caption_tokens)
            
            if isinstance(self.tokenizer, QwenTokenizer):
                caption = self.get_caption(index, cidx)
                caption_tokens = caption 
                caption_lengths = torch.tensor([len(caption.split())])

        if self.use_generated_pose_samples:
            # overwrite variable content
            pose = self.get_generated_pose(index) # let cidx at None, to still select among generated samples for every automatic caption when training on human-written captions

        item = dict(pose=pose, caption_tokens=caption_tokens, caption_lengths=caption_lengths)
        return item

PVCP2_BODY_PARTS = ["Head", "Torso", "Left Arm", "Right Arm", "Left Leg", "Right Leg"]

PVCP2_ATTRIBUTE_NOISE_PAIRS = [
    ("not touching the ground", "fully touching the ground"),
    ("not fully touching the ground", "fully touching the ground"),
    ("fully touching the ground", "not touching the ground"),
    ("touching the ground", "not touching the ground"),
    ("contacting the ground", "not touching the ground"),
    ("slightly left", "slightly right"),
    ("slightly right", "slightly left"),
    ("slightly forward", "slightly backward"),
    ("slightly backward", "slightly forward"),
    ("slightly upward", "slightly downward"),
    ("slightly downward", "slightly upward"),
    ("forward", "backward"),
    ("backward", "forward"),
    ("upward", "downward"),
    ("downward", "upward"),
    ("inward", "outward"),
    ("outward", "inward"),
    ("above", "below"),
    ("below", "above"),
    ("behind", "in front of"),
    ("in front of", "behind"),
    ("left", "right"),
    ("right", "left"),
    ("not", "slightly"),
    ("slightly", "significantly"),
    ("moderately", "slightly"),
    ("significantly", "slightly"),
    ("completely", "slightly"),
    ("right-angle", "slightly"),
    ("straight", "bent"),
    ("bent", "straight"),
    ("relaxed", "lifted"),
    ("lifted", "relaxed"),
    ("standing", "stepping"),
    ("stepping", "standing"),
    ("supporting stance", "stepping"),
]

PVCP2_ATTRIBUTE_NOISE_FALLBACKS = [
    "slightly left", "slightly right", "forward", "backward", "upward",
    "downward", "inward", "outward", "slightly", "significantly",
    "straight", "slightly bent", "right-angle", "relaxed", "lifted",
    "fully touching the ground", "not touching the ground",
]


def _stable_int_seed(*parts):
    seed_text = "::".join(str(p) for p in parts)
    return int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:16], 16)


def _ratio_to_percent(mask_ratio):
    return int(round(float(mask_ratio) * 100))


def _split_sentences(text):
    text = str(text).strip()
    if not text:
        return []
    sentences = [m.group(0).strip() for m in re.finditer(r"[^.!?]+(?:[.!?]+(?=\s|$)|$)", text)]
    return [s for s in sentences if s]


def _join_pose_description(pose_description):
    if isinstance(pose_description, dict):
        return " ".join(pose_description.values())
    return str(pose_description)


def _clean_caption_text(caption):
    return caption.replace("[", "").replace("]", "")


def _attribute_phrase_pattern(phrase):
    return re.compile(
        rf"(?<![A-Za-z-]){re.escape(phrase)}(?![A-Za-z-])",
        flags=re.IGNORECASE,
    )


def _replace_first_case_insensitive(text, old, new):
    pattern = _attribute_phrase_pattern(old)
    return pattern.sub(new, text, count=1)


def _noisy_attribute_value(attribute_text, rng):
    candidates = []
    for old, new in PVCP2_ATTRIBUTE_NOISE_PAIRS:
        if _attribute_phrase_pattern(old).search(attribute_text):
            candidate = _replace_first_case_insensitive(attribute_text, old, new)
            if candidate != attribute_text and candidate not in candidates:
                candidates.append(candidate)

    if candidates:
        return rng.choice(candidates)

    fallback_candidates = [
        value for value in PVCP2_ATTRIBUTE_NOISE_FALLBACKS
        if value.lower() != attribute_text.lower()
    ]
    return rng.choice(fallback_candidates) if fallback_candidates else attribute_text


def apply_attribute_noise(pose_description, noise_ratio, rng):
    pose_description = deepcopy(pose_description)

    if isinstance(pose_description, dict):
        field_texts = list(pose_description.items())
    else:
        field_texts = [(None, str(pose_description))]

    all_attributes = []
    for field_name, field_text in field_texts:
        for match_index, match in enumerate(re.finditer(r"\[([^\[\]]+)\]", str(field_text))):
            all_attributes.append((field_name, match_index, match.group(1)))

    num_attributes = len(all_attributes)
    if num_attributes == 0 or noise_ratio <= 0:
        metadata = {
            "original_attribute_count": num_attributes,
            "noisy_attribute_count": 0,
            "actual_noise_ratio": 0.0,
            "noisy_attributes": [],
        }
        return pose_description, metadata

    num_noise = round(num_attributes * noise_ratio)
    num_noise = max(1, num_noise)
    num_noise = min(num_noise, num_attributes)
    noisy_indices = set(rng.sample(range(num_attributes), num_noise))

    replacements = {}
    noisy_attributes = []
    for attr_index in sorted(noisy_indices):
        field_name, match_index, original_value = all_attributes[attr_index]
        noisy_value = _noisy_attribute_value(original_value, rng)
        replacements[(field_name, match_index)] = noisy_value
        noisy_attributes.append({
            "field": field_name,
            "original": original_value,
            "noisy": noisy_value,
        })

    def replace_field_attributes(field_name, field_text):
        attr_counter = 0

        def repl(match):
            nonlocal attr_counter
            current_index = attr_counter
            attr_counter += 1
            replacement = replacements.get((field_name, current_index))
            if replacement is None:
                return match.group(0)
            return f"[{replacement}]"

        return re.sub(r"\[([^\[\]]+)\]", repl, str(field_text))

    if isinstance(pose_description, dict):
        noisy_description = {}
        for field_name, field_text in field_texts:
            noisy_description[field_name] = replace_field_attributes(field_name, field_text)
    else:
        noisy_description = replace_field_attributes(None, str(pose_description))

    metadata = {
        "original_attribute_count": num_attributes,
        "noisy_attribute_count": num_noise,
        "actual_noise_ratio": num_noise / num_attributes,
        "noisy_attributes": noisy_attributes,
    }
    return noisy_description, metadata


def apply_sentence_mask(pose_description, mask_ratio, rng):
    pose_description = deepcopy(pose_description)

    if isinstance(pose_description, dict):
        field_sentences = []
        all_sentences = []
        for field_name, field_text in pose_description.items():
            sentences = _split_sentences(field_text)
            field_sentences.append((field_name, sentences))
            for sentence_index, sentence in enumerate(sentences):
                all_sentences.append((field_name, sentence_index, sentence))
    else:
        sentences = _split_sentences(pose_description)
        field_sentences = [(None, sentences)]
        all_sentences = [(None, sentence_index, sentence) for sentence_index, sentence in enumerate(sentences)]

    num_sentences = len(all_sentences)
    if num_sentences <= 1 or mask_ratio <= 0:
        metadata = {
            "original_sentence_count": num_sentences,
            "masked_sentence_count": 0,
            "actual_mask_ratio": 0.0,
            "masked_sentences": [],
            "fully_missing_body_parts": [],
        }
        return pose_description, metadata

    num_mask = round(num_sentences * mask_ratio)
    num_mask = max(1, num_mask)
    num_mask = min(num_mask, num_sentences - 1)

    masked_indices = set(rng.sample(range(num_sentences), num_mask))
    masked_lookup = {
        (all_sentences[i][0], all_sentences[i][1])
        for i in masked_indices
    }
    masked_sentences = [all_sentences[i][2] for i in sorted(masked_indices)]

    if isinstance(pose_description, dict):
        masked_description = {}
        body_part_lookup = {name.lower(): name for name in PVCP2_BODY_PARTS}
        fully_missing_body_parts = []
        for field_name, sentences in field_sentences:
            kept_sentences = [
                sentence for sentence_index, sentence in enumerate(sentences)
                if (field_name, sentence_index) not in masked_lookup
            ]
            if kept_sentences:
                masked_description[field_name] = " ".join(kept_sentences).strip()
            canonical_name = body_part_lookup.get(str(field_name).lower())
            if canonical_name and sentences and not kept_sentences:
                fully_missing_body_parts.append(canonical_name)
    else:
        kept_sentences = [
            sentence for sentence_index, sentence in enumerate(field_sentences[0][1])
            if (None, sentence_index) not in masked_lookup
        ]
        masked_description = " ".join(kept_sentences).strip()
        fully_missing_body_parts = []

    metadata = {
        "original_sentence_count": num_sentences,
        "masked_sentence_count": num_mask,
        "actual_mask_ratio": num_mask / num_sentences,
        "masked_sentences": masked_sentences,
        "fully_missing_body_parts": fully_missing_body_parts,
    }
    return masked_description, metadata


class PVCP2(object):
    def __init__(self, version="PVCP2", split='train', text_encoder_name='qwen',
                 text_mode="full", keep_full_prob=0.4, train_mask_ratios=None,
                 fixed_mask_ratio=None, text_mask_seed=42, mask_manifest_path=None,
                 fixed_noise_ratio=None, text_noise_seed=42, noise_manifest_path=None):
        self.version = version
        self.split = split
        assert text_mode in ["full", "aug", "mask", "noisy"]
        self.text_mode = text_mode
        self.keep_full_prob = keep_full_prob
        self.train_mask_ratios = train_mask_ratios if train_mask_ratios is not None else [0.1, 0.2, 0.3, 0.5, 0.7]
        self.fixed_mask_ratio = fixed_mask_ratio
        self.text_mask_seed = text_mask_seed
        self.mask_manifest_path = mask_manifest_path
        self.mask_manifest = None
        self.fixed_noise_ratio = fixed_noise_ratio
        self.text_noise_seed = text_noise_seed
        self.noise_manifest_path = noise_manifest_path
        self.noise_manifest = None
        self.current_epoch = 0
        # Load data
        self._load_data()
        # Define tokenizer
        self.tokenizer = Tokenizer(text_encoder_name)
        if self.text_mode == "mask":
            assert self.fixed_mask_ratio is not None, "fixed_mask_ratio must be provided when text_mode='mask'."
            self._load_or_create_mask_manifest()
        if self.text_mode == "noisy":
            assert self.fixed_noise_ratio is not None, "fixed_noise_ratio must be provided when text_mode='noisy'."
            self._load_or_create_noise_manifest()
        
    def _load_data(self):
        self.dataIDs = utils.read_pvcp2_json(f"{self.split}_ids.json")
        self.IDs2Frame = utils.read_pvcp2_json(f"ids2frame.json")
        self.dataText = utils.read_pvcp2_json("pvcp_text.json") # text
        self.dataPose = utils.read_pvcp2_json("pvcp_pose.json") # keypoint_pose
        self.dataMesh = utils.read_pvcp2_json("pvcp_mesh.json") # Mesh_pose

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def _default_mask_manifest_path(self):
        ratio_percent = _ratio_to_percent(self.fixed_mask_ratio)
        filename = f"{self.split}_sentence_mask_{ratio_percent}_seed{self.text_mask_seed}.json"
        return os.path.join(config.PVCP2_LOCATION, "masked_text", filename)

    def _default_noise_manifest_path(self):
        ratio_percent = _ratio_to_percent(self.fixed_noise_ratio)
        filename = f"{self.split}_attribute_noise_{ratio_percent}_seed{self.text_noise_seed}.json"
        return os.path.join(config.PVCP2_LOCATION, "noisy_text", filename)

    def _load_or_create_mask_manifest(self):
        manifest_path = self.mask_manifest_path or self._default_mask_manifest_path()
        self.mask_manifest_path = manifest_path
        if os.path.isfile(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
        else:
            os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
            manifest_data = self._build_mask_manifest()
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest_data, f, indent=2, ensure_ascii=False)
            print(f"Saved fixed sentence-mask manifest: {manifest_path}")

        if isinstance(manifest_data, dict):
            manifest_records = manifest_data.get("samples", manifest_data)
            if isinstance(manifest_records, dict):
                self.mask_manifest = {str(k): v for k, v in manifest_records.items()}
            else:
                self.mask_manifest = {str(record["sample_id"]): record for record in manifest_records}
        else:
            self.mask_manifest = {str(record["sample_id"]): record for record in manifest_data}
        self._print_mask_manifest_summary()

    def _load_or_create_noise_manifest(self):
        manifest_path = self.noise_manifest_path or self._default_noise_manifest_path()
        self.noise_manifest_path = manifest_path
        if os.path.isfile(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
        else:
            os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
            manifest_data = self._build_noise_manifest()
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest_data, f, indent=2, ensure_ascii=False)
            print(f"Saved fixed attribute-noise manifest: {manifest_path}")

        if isinstance(manifest_data, dict):
            manifest_records = manifest_data.get("samples", manifest_data)
            if isinstance(manifest_records, dict):
                self.noise_manifest = {str(k): v for k, v in manifest_records.items()}
            else:
                self.noise_manifest = {str(record["sample_id"]): record for record in manifest_records}
        else:
            self.noise_manifest = {str(record["sample_id"]): record for record in manifest_data}
        self._print_noise_manifest_summary()

    def _build_mask_manifest(self):
        samples = []
        for sample_id in self.dataIDs:
            sample_id_str = str(sample_id)
            frame = self.IDs2Frame[sample_id_str]
            pose_description = self.dataText[frame]["text"]["pose"] if frame in self.dataText else "No text"
            rng = random.Random(_stable_int_seed(self.text_mask_seed, self.split, sample_id, self.fixed_mask_ratio))
            masked_description, metadata = apply_sentence_mask(pose_description, self.fixed_mask_ratio, rng)
            original_text = _clean_caption_text(_join_pose_description(pose_description))
            masked_text = _clean_caption_text(_join_pose_description(masked_description))
            record = {
                "sample_id": sample_id,
                "frame": frame,
                "mask_ratio": self.fixed_mask_ratio,
                "actual_mask_ratio": metadata["actual_mask_ratio"],
                "original_sentence_count": metadata["original_sentence_count"],
                "masked_sentence_count": metadata["masked_sentence_count"],
                "original_text": original_text,
                "masked_text": masked_text,
                "masked_sentences": metadata["masked_sentences"],
                "fully_missing_body_parts": metadata["fully_missing_body_parts"],
            }
            samples.append(record)
        return samples

    def _build_noise_manifest(self):
        samples = []
        for sample_id in self.dataIDs:
            sample_id_str = str(sample_id)
            frame = self.IDs2Frame[sample_id_str]
            pose_description = self.dataText[frame]["text"]["pose"] if frame in self.dataText else "No text"
            rng = random.Random(_stable_int_seed(self.text_noise_seed, self.split, sample_id, self.fixed_noise_ratio))
            noisy_description, metadata = apply_attribute_noise(pose_description, self.fixed_noise_ratio, rng)
            original_text = _clean_caption_text(_join_pose_description(pose_description))
            noisy_text = _clean_caption_text(_join_pose_description(noisy_description))
            record = {
                "sample_id": sample_id,
                "frame": frame,
                "noise_ratio": self.fixed_noise_ratio,
                "actual_noise_ratio": metadata["actual_noise_ratio"],
                "original_attribute_count": metadata["original_attribute_count"],
                "noisy_attribute_count": metadata["noisy_attribute_count"],
                "original_text": original_text,
                "noisy_text": noisy_text,
                "noisy_attributes": metadata["noisy_attributes"],
            }
            samples.append(record)
        return samples

    def _print_mask_manifest_summary(self):
        records = list(self.mask_manifest.values())
        if not records:
            return
        missing_count = sum(1 for record in records if record.get("fully_missing_body_parts"))
        missing_ratio = missing_count / len(records)
        ratio_percent = _ratio_to_percent(self.fixed_mask_ratio)
        print(f"PVCP2 fixed sentence-mask text: split={self.split}, ratio={ratio_percent}%, seed={self.text_mask_seed}")
        print(f"Manifest: {self.mask_manifest_path}")
        print(f"Samples with at least one fully missing body part: {missing_count}/{len(records)} ({missing_ratio:.4f})")
        for record in records[:3]:
            print(f"[Mask example] sample_id={record['sample_id']} frame={record['frame']}")
            print(f"  Full: {record['original_text']}")
            print(f"  Mask: {record['masked_text']}")

    def _print_noise_manifest_summary(self):
        records = list(self.noise_manifest.values())
        if not records:
            return
        ratio_percent = _ratio_to_percent(self.fixed_noise_ratio)
        attr_counts = [record.get("original_attribute_count", 0) for record in records]
        noisy_counts = [record.get("noisy_attribute_count", 0) for record in records]
        total_attrs = sum(attr_counts)
        total_noisy = sum(noisy_counts)
        actual_ratio = total_noisy / total_attrs if total_attrs else 0.0
        print(f"PVCP2 fixed attribute-noise text: split={self.split}, ratio={ratio_percent}%, seed={self.text_noise_seed}")
        print(f"Manifest: {self.noise_manifest_path}")
        print(f"Noisy attributes: {total_noisy}/{total_attrs} ({actual_ratio:.4f})")
        for record in records[:3]:
            print(f"[Noisy example] sample_id={record['sample_id']} frame={record['frame']}")
            print(f"  Full: {record['original_text']}")
            print(f"  Noisy: {record['noisy_text']}")
            print(f"  Changes: {record['noisy_attributes'][:5]}")
    
    def get_pose(self, index):
        pose_frame = self.IDs2Frame[str(self.dataIDs[index])]
        pose = self.dataMesh[pose_frame]['pose']
        pose = torch.tensor(pose, dtype=torch.float32)
        pose = pose.reshape(-1, 3) # (njoints, 3)
        return pose
    
    def get_pose_class(self, index):
        pose_frame = self.IDs2Frame[str(self.dataIDs[index])]
        pose_class = self.dataPose[pose_frame]['pose_class']
        if pose_class==None:
            pose_class=[0,0,0,0]
        pose_class = torch.tensor(pose_class, dtype=torch.float32)
        return pose_class

    def get_caption(self, index):
        text_frame = self.IDs2Frame[str(self.dataIDs[index])]  # 1334-->1-->"frame_000012.png",

        if text_frame in list(self.dataText.keys()):
            caption = self.dataText[text_frame]['text']['pose']
        else:
            caption = 'No text'

        if self.text_mode == "full":
            if isinstance(caption, dict):
                caption = " ".join(caption.values())
            caption=caption.replace('[', '').replace(']', '')
            return caption

        if self.text_mode == "aug":
            sample_id = self.dataIDs[index]
            rng = random.Random(_stable_int_seed(self.text_mask_seed, self.current_epoch, sample_id))
            if rng.random() >= self.keep_full_prob:
                mask_ratio = rng.choice(self.train_mask_ratios)
                caption, _ = apply_sentence_mask(caption, mask_ratio, rng)
            if isinstance(caption, dict):
                caption = " ".join(caption.values())
            caption=caption.replace('[', '').replace(']', '')
            return caption

        if self.text_mode == "mask":
            sample_id = str(self.dataIDs[index])
            if self.mask_manifest is not None and sample_id in self.mask_manifest:
                return self.mask_manifest[sample_id]["masked_text"]

        if self.text_mode == "noisy":
            sample_id = str(self.dataIDs[index])
            if self.noise_manifest is not None and sample_id in self.noise_manifest:
                return self.noise_manifest[sample_id]["noisy_text"]

        if isinstance(caption, dict):
            caption = " ".join(caption.values())
        caption=caption.replace('[', '').replace(']', '')
        return caption
        
    def __len__(self):
        return len(self.dataIDs)
    
    def __getitem__(self, index):
        pose = self.get_pose(index)
        pose_class = self.get_pose_class(index)
        caption = self.get_caption(index)
        if isinstance(self.tokenizer, QwenTokenizer):
            caption_tokens = caption
            caption_lengths = torch.tensor([len(caption.split())])
        else:
            caption_tokens = self.tokenizer(caption)
            caption_lengths = len(caption_tokens)
            
            max_len = self.tokenizer.max_tokens
            pad_token_id = self.tokenizer.pad_token_id

            if caption_lengths < max_len:
                padding = torch.full((max_len - caption_lengths,), pad_token_id, dtype=caption_tokens.dtype)
                caption_tokens = torch.cat([caption_tokens, padding], dim=0)
            else:
                caption_tokens = caption_tokens[:max_len]
                caption_lengths = max_len

        item = dict(pose=pose, caption_tokens=caption_tokens, caption_lengths=caption_lengths, pose_class=pose_class)
        return item


if __name__ == "__main__":

    dataset = PVCP2(split='train', text_encoder_name='Qwen_1_8B')
    print(f"Total samples in dataset: {len(dataset)}")

    #index = 1332-->1-->"frame_000012.png",
    index = random.randint(0, len(dataset) - 1) 
    print(f"Testing sample index: {index}")

    item = dataset[index]
    print("Pose shape:", item['pose'].shape)
    print("Pose_class shape:", item['pose_class'].shape)
    print("Caption tokens:", item['caption_tokens'])
    print("Caption length:", item['caption_lengths'])


        
