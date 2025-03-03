# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# Based on https://github.com/NVIDIA/flowtron/blob/master/data.py
# Original license text:
###############################################################################
#
#  Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
###############################################################################

"""adapted from https://github.com/keithito/tacotron"""

import re
from functools import reduce
from string import punctuation

import torch
import torch.utils.data

#########
# REGEX #
#########

# Regular expression matching text enclosed in curly braces for encoding
_curly_re = re.compile(r"(.*?)\{(.+?)\}(.*)")

# Regular expression matching whitespace:
_whitespace_re = re.compile(r"\s+")

# Regular expression separating words enclosed in curly braces for cleaning
_arpa_re = re.compile(r"{[^}]+}|\S+")


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text)


def remove_space_before_punctuation(text):
    return re.sub(r"\s([{}](?:\s|$))".format(punctuation), r"\1", text)


class Cleaner:
    def __init__(self, cleaner_names, phonemedict):
        self.cleaner_names = cleaner_names
        self.phonemedict = phonemedict

    def __call__(self, text):
        for cleaner_name in self.cleaner_names:
            sequence_fns, word_fns = self.get_cleaner_fns(cleaner_name)
            for fn in sequence_fns:
                text = fn(text)

            text = [
                reduce(lambda x, y: y(x), word_fns, split) if split[0] != "{" else split
                for split in _arpa_re.findall(text)
            ]
            text = " ".join(text)

        text = remove_space_before_punctuation(text)

        return text

    def get_cleaner_fns(self, cleaner_name):
        sequence_fns = [lowercase, collapse_whitespace]
        word_fns = []

        return sequence_fns, word_fns


def get_symbols():
    _punctuation = "'.,?! "
    _special = "-+"
    _letters = "абвгґдежзийклмнопрстуфхцчшщьюяєії"

    symbols = list(_punctuation + _special + _letters)

    return symbols


class TextProcessing:
    def __init__(
        self,
        symbol_set,
        cleaner_name,
        heteronyms_path,
        phoneme_dict_path,
        p_phoneme,
        handle_phoneme,
        handle_phoneme_ambiguous,
        prepend_space_to_text=False,
        append_space_to_text=False,
        add_bos_eos_to_text=False,
        encoding="latin-1",
    ):
        self.phonemedict = {}

        self.p_phoneme = p_phoneme
        self.handle_phoneme = handle_phoneme
        self.handle_phoneme_ambiguous = handle_phoneme_ambiguous

        self.symbols = get_symbols()
        self.cleaner_names = cleaner_name
        self.cleaner = Cleaner(cleaner_name, self.phonemedict)

        self.prepend_space_to_text = prepend_space_to_text
        self.append_space_to_text = append_space_to_text
        self.add_bos_eos_to_text = add_bos_eos_to_text

        if add_bos_eos_to_text:
            self.symbols.append("<bos>")
            self.symbols.append("<eos>")

        # Mappings from symbol to numeric ID and vice versa:
        self.symbol_to_id = {s: i for i, s in enumerate(self.symbols)}
        self.id_to_symbol = {i: s for i, s in enumerate(self.symbols)}

    def text_to_sequence(self, text):
        sequence = []

        # Check for curly braces and treat their contents as phoneme:
        while len(text):
            m = _curly_re.match(text)
            if not m:
                sequence += self.symbols_to_sequence(text)
                break
            sequence += self.symbols_to_sequence(m.group(1))
            sequence += self.phoneme_to_sequence(m.group(2))
            text = m.group(3)

        return sequence

    def sequence_to_text(self, sequence):
        result = ""
        for symbol_id in sequence:
            if symbol_id in self.id_to_symbol:
                s = self.id_to_symbol[symbol_id]
                # Enclose phoneme back in curly braces:
                if len(s) > 1 and s[0] == "@":
                    s = "{%s}" % s[1:]
                result += s
        return result.replace("}{", " ")

    def clean_text(self, text):
        text = self.cleaner(text)
        return text

    def symbols_to_sequence(self, symbols):
        return [self.symbol_to_id[s] for s in symbols if s in self.symbol_to_id]

    def encode_text(self, text, return_all=False):
        text_clean = self.clean_text(text)
        text = text_clean

        text_encoded = self.text_to_sequence(text)

        if self.prepend_space_to_text:
            text_encoded.insert(0, self.symbol_to_id[" "])

        if self.append_space_to_text:
            text_encoded.append(self.symbol_to_id[" "])

        if self.add_bos_eos_to_text:
            text_encoded.insert(0, self.symbol_to_id["<bos>"])
            text_encoded.append(self.symbol_to_id["<eos>"])

        if return_all:
            return text_encoded, text_clean

        return text_encoded


class TextProcessor(torch.utils.data.Dataset):
    def __init__(
        self,
        datasets,
        filter_length,
        hop_length,
        win_length,
        sampling_rate,
        n_mel_channels,
        mel_fmin,
        mel_fmax,
        f0_min,
        f0_max,
        max_wav_value,
        use_f0,
        use_energy_avg,
        use_log_f0,
        use_scaled_energy,
        symbol_set,
        cleaner_names,
        heteronyms_path,
        phoneme_dict_path,
        p_phoneme,
        handle_phoneme="word",
        handle_phoneme_ambiguous="ignore",
        speaker_ids=None,
        include_speakers=None,
        n_frames=-1,
        use_attn_prior_masking=True,
        prepend_space_to_text=True,
        append_space_to_text=True,
        add_bos_eos_to_text=False,
        betabinom_cache_path="",
        betabinom_scaling_factor=0.05,
        lmdb_cache_path="",
        dur_min=None,
        dur_max=None,
        combine_speaker_and_emotion=False,
        **kwargs,
    ):
        self.tp = TextProcessing(
            symbol_set,
            cleaner_names,
            heteronyms_path,
            phoneme_dict_path,
            p_phoneme=p_phoneme,
            handle_phoneme=handle_phoneme,
            handle_phoneme_ambiguous=handle_phoneme_ambiguous,
            prepend_space_to_text=prepend_space_to_text,
            append_space_to_text=append_space_to_text,
            add_bos_eos_to_text=add_bos_eos_to_text,
        )
