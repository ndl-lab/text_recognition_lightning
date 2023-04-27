# Copyright (c) 2023, National Diet Library, Japan
#
# This software is released under the CC BY 4.0.
# https://creativecommons.org/licenses/by/4.0/

import re
import numpy as np
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CTCLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character, ambiguous_char=None):
        # character (str): set of the possible characters.
        if ambiguous_char is not None:
            character = ambiguous_char + character
        dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'CTCblank' token required by CTCLoss
            self.dict[char] = i + 1

        self.character = ['[CTCblank]'] + dict_character  # dummy '[CTCblank]' token for CTCLoss (index 0)
        self.character_np = np.asarray(self.character)
        self.ambiguous_char = ambiguous_char
        self.out_of_char = re.compile(f'[^{character}]')

    def encode(self, text, batch_max_length=25):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text: text index for CTCLoss. [batch_size, batch_max_length]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]

        # The index used for padding (=0) would not affect the CTC loss calculation.
        batch_text = torch.LongTensor(len(text), batch_max_length).fill_(0)
        for i, t in enumerate(text):
            if self.ambiguous_char is not None:
                t = self.out_of_char.sub(self.ambiguous_char, t)
            text = list(t)
            try:
                text = [self.dict[char] for char in text]
            except Exception as e:
                raise e
            batch_text[i][:len(text)] = torch.LongTensor(text)
        return (batch_text.to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            text = torch.unique_consecutive(text_index[index, :])
            text = text[text != 0]
            text = ''.join(self.character_np[text.cpu()])

            texts.append(text)
        return texts
