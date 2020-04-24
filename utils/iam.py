import torch
import shelve
import warnings
from torch import Tensor
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from typing import Tuple, Any

__all__ = ['IAMWords', 'CHARACTERS', 'word_to_tensor', 'tensor_to_word']


CHARACTERS = ['!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/',
                  '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?',
                  'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                  'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                  'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                  'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
                  ' ']

ID_TO_CHARACTER = {i + 1: c for i, c in enumerate(CHARACTERS)}
CHARACTER_TO_ID = {c: i + 1 for i, c in enumerate(CHARACTERS)}


def isint(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def resize(image: Image.Image, width, height):
    target_aspect = width / height
    actual_aspect = image.width / image.height

    if target_aspect > actual_aspect:
        # Match heights
        resized = image.resize((int(height * actual_aspect),
                                height))
    else:
        # Match widths
        resized = image.resize((width,
                                int(width / actual_aspect)))

    padded = Image.new('L', (width, height), 255)
    padded.paste(resized, resized.getbbox())
    return padded


class Resize:
    def __init__(self, width: int, height: int):
        self.width, self.height = width, height

    def __call__(self, image: Image.Image):
        return resize(image, self.width, self.height)


def word_to_tensor(word: str, dtype=torch.int32):
    return Tensor([CHARACTER_TO_ID[c] for c in word]).type(dtype)


def tensor_to_word(tensor: Tensor):
    word = ''
    last_char = None
    for c in tensor.cpu().numpy():
        if c == 0:
            last_char = None
            continue
        c = ID_TO_CHARACTER[int(c)]
        if c != last_char:
            word += c
        last_char = c
    return word


class IAMWords(Dataset):
    Path('cache').mkdir(exist_ok=True)
    blacklist = shelve.open('cache/blacklist', protocol=0)

    def __init__(self, root, split='train', transform=None):
        assert split in ('train', 'test', 'valid', 'valid1', 'valid2')

        self.image_files, self.labels = self._load(Path(root), split=split, blacklist=self.blacklist)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, item: int) -> Tuple[Any, str]:
        image, label = self.image_files[item], self.labels[item]

        try:
            image = Image.open(image).convert('L')
        except UnidentifiedImageError:
            warnings.warn(f'Error opening {image.name}, blacklisting')
            self.blacklist[image.name] = True
            return self[(item + 1) % len(self)]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    @staticmethod
    def _split_form_lines(root, split: str):
        FILES = {'train': ['trainset.txt'], 'test': ['testset.txt'],
                 'valid': ['validationset1.txt', 'validationset2.txt'],
                 'valid1': ['validationset1.txt'], 'valid2': ['validationset2.txt']}
        form_lines = set()
        for file in FILES[split]:
            for line in open(root / 'iam/largeWriterIndependentTextLineRecognitionTask' / file, 'r'):
                line = line[:-1]
                form_tok_1, form_tok_2, line = line.split('-')
                form_lines.add((f'{form_tok_1}-{form_tok_2}', line))
        return form_lines

    @staticmethod
    def _parse_id(id: str):
        form_tok_1, form_tok_2, line, word = id.split('-')
        return f'{form_tok_1}-{form_tok_2}', line, word

    @staticmethod
    def _build_image_filename(root: Path, id: str):
        form_tok_1, form_tok_2, line, word = id.split('-')
        form = f'{form_tok_1}-{form_tok_2}'

        return root / 'iam/data/words' / form_tok_1 / form / f'{id}.png'

    @staticmethod
    def _load(root: Path, split: str, blacklist: shelve.Shelf = None):
        split_form_lines = IAMWords._split_form_lines(root, split)

        image_files = []
        labels = []
        for line in open(root / 'iam/data/ascii/words.txt', 'r'):
            if line.startswith('#'):
                continue
            line = line[:-1]  # Remove newline

            id, result, graylevel, x, y, w, h, tag, *words = line.split()

            if len(words) != 1:
                continue  # Only use single words

            assert result in ('ok', 'err')
            assert graylevel.isnumeric()
            assert isint(x)
            assert isint(y)
            assert isint(w)
            assert isint(h)
            words = ' '.join(words)
            assert all(c in CHARACTERS for c in words)

            if result == 'err':
                continue  # Discard invalid data

            form_id, line_id, word_id = IAMWords._parse_id(id)
            if (form_id, line_id) not in split_form_lines:
                continue

            image_file = IAMWords._build_image_filename(root, id)
            if blacklist is not None and image_file.name in blacklist:
                warnings.warn(f'{image_file.name} blacklisted, skipping')
                continue

            image_files.append(image_file)
            labels.append(words)

        assert len(image_files) == len(labels)
        return image_files, labels
