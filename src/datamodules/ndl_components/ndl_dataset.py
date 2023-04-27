# Copyright (c) 2023, National Diet Library, Japan
#
# This software is released under the CC BY 4.0.
# https://creativecommons.org/licenses/by/4.0/

from collections.abc import Sequence, Mapping
from glob import glob
import pathlib
import re

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.utils.data import Dataset, IterableDataset, ConcatDataset
from torchvision.datasets.folder import default_loader
from torchvision.transforms import transforms as V
from torchvision.transforms import functional as F

import six
import xml.etree.ElementTree as ET

from ... import utils

log = utils.get_pylogger(__name__)
pt = re.compile(r'(\{(.*)\})?(.*)')


class XMLLMDBDataset(Dataset):
    def __init__(self, data_dir, transforms, batch_max_length, **kwargs):
        super().__init__()
        self.transforms = transforms
        self.batch_max_length = batch_max_length

        import lmdb
        self.env = lmdb.open(data_dir, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        if not self.env:
            print('cannot create lmdb from %s' % (data_dir))
            exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('n_line'.encode()))
            self.nSamples = nSamples

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        with self.env.begin(write=False) as txn:
            label = txn.get(f'{index:09d}-label'.encode()).decode('utf-8')
            imgbuf = txn.get(f'{index:09d}-image'.encode())
            direction = txn.get(f'{index:09d}-direction'.encode()).decode('utf-8')

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)

            try:
                img = Image.open(buf).convert('RGB')  # for color image
            except IOError:
                img = Image.new('RGB', (self.opt.imgW, self.opt.imgH))
                label = '[dummy_label]'
            if direction == '縦':
                img = img.transpose(Image.ROTATE_90)
                direction = 1
            elif direction == '横':
                direction = 0
                pass
            elif direction == '右から左':
                label = label[::-1]
                pass
                direction = 0
            else:
                print(False, f"index:{index} dire:{direction} labe:{label} {img.size}")
                img = Image.new('RGB', (32, 32))
                label = '[dummy_label]'
                # assert False, f"index:{index} dire:{direction} labe:{label} {img.size}"
                direction = 0

            return (
                self.transforms(img),
                label[:self.batch_max_length],
                {
                    "orient": direction,
                    "file_idx": 0,
                    "pid": 0
                }
            )


class XMLRawDataset(IterableDataset):
    @staticmethod
    def find_xml(input_paths):
        if isinstance(input_paths, str):
            input_paths = [input_paths]
        elif isinstance(input_paths, Sequence):
            pass
        else:
            raise AttributeError(type(input_paths))

        files = []
        for pp in input_paths:
            for p in glob(pp):
                p = pathlib.Path(p)
                if not p.exists():
                    log.error(f"{p} is not exists")
                    continue
                files.extend(p.rglob('*.xml'))
        return sorted(files)

    def __init__(self,
                 input_paths=None, xml_files=None,
                 transforms=None, batch_max_length: int = 40,
                 additional_elements=None,
                 expand_bbox=[2, 1]):
        super().__init__()
        if xml_files is not None:
            self.xml_files = xml_files
        elif input_paths is not None:
            self.xml_files = self.find_xml(input_paths)
        if transforms is None:
            self.transforms = V.Compose([])
        else:
            self.transforms = transforms
        self.batch_max_length = batch_max_length
        self.loader = default_loader
        self.additional_elements = additional_elements
        self.expand_bbox = expand_bbox

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        idx_start = 0
        idx_end = len(self.xml_files)
        if worker_info is not None:
            idx_end = -(-idx_end // worker_info.num_workers)  # ceil
            idx_start = idx_end * worker_info.id
            idx_end = idx_start + idx_end

        for i, xml_file in enumerate(self.xml_files[idx_start:idx_end]):
            img_dir = pathlib.Path(xml_file).parent.parent / "img"

            with open(xml_file, 'r') as f:
                try:
                    tree = ET.parse(f)
                    root = tree.getroot()
                except ValueError:
                    with open(xml_file, encoding="UTF-8") as file:
                        root = ET.fromstring(file.read())
                except ET.ParseError as e:
                    log.error(xml_file)
                    raise e

            groups = pt.match(root.tag).groups()
            if groups[1]:
                namespace = f"{{{groups[1]}}}"
            else:
                namespace = ''
            if groups[2] != 'OCRDATASET':
                log.error(f'skip {groups[2]} is not subject {xml_file}')
                continue

            for line in self.iter_file(root, img_dir, namespace, xml_file, tree, idx_start + i):
                yield line

    def set_data(self, img_data, xml_data, pid):
        pass

    def iter_file(self, root, img_dir, namespace, xml_file: pathlib.Path, tree, idx):
        for page in root:
            image_name = page.attrib['IMAGENAME']
            image_path = img_dir / image_name
            if not image_path.exists():
                import sys
                print(FileNotFoundError(f'{image_path} is not Found'), file=sys.stderr)
                continue
            image = self.loader(image_path)

            for line in self.iter_page(page, image, namespace, xml_file.stem, idx):
                if line is not None:
                    yield line

    def get_target_element(self, page, namespace):
        if not self.additional_elements:
            ilines = page.iterfind(f'.//{namespace}LINE')
        else:
            import itertools
            elements = ['LINE'] + self.additional_elements
            data_list = [
                page.iterfind(f'.//{namespace}{e}')
                for e in elements
            ]
            ilines = itertools.chain.from_iterable(data_list)
        return ilines

    def iter_page(self, page, image, namespace, pid: str, idx: int):
        width, height = image.size

        ilines = self.get_target_element(page, namespace)
        df = pd.DataFrame([li.attrib for li in ilines])
        df.fillna({'STRING': ''}, inplace=True)
        if df.size == 0:
            return

        df = df.astype({
            'X': int, 'Y': int,
            'WIDTH': int, 'HEIGHT': int,
        })

        for i in range(df.shape[0]):
            line = df.iloc[i]
            x, y, w, h = line['X'], line['Y'], line.get('WIDTH', 0), line.get('HEIGHT', 0)

            if w == 0 or h == 0:
                continue

            if 'DIRECTION' not in line:
                direction = int(w < h)
            else:
                direction = int(line['DIRECTION'] == '縦')

            if direction:
                ex_dx = w / 32 * self.expand_bbox[1]
                ex_dy = w / 32 * self.expand_bbox[0]
            else:
                ex_dx = h / 32 * self.expand_bbox[0]
                ex_dy = h / 32 * self.expand_bbox[1]

            if self.expand_bbox[0] > 0:
                x = max(round(x - ex_dx), 0)
                w = min(round(w + ex_dx + x), width) - x
            if self.expand_bbox[1] > 0:
                y = max(round(y - ex_dy), 0)
                h = min(round(h + ex_dy + y), height) - y

            g = F.crop(image, y, x, h, w)

            if direction:
                g = g.transpose(Image.ROTATE_90)

            yield (
                self.transforms(g),
                line.get('STRING', '')[:self.batch_max_length],
                {
                    "orient": direction,
                    "file_idx": idx,
                    "pid": pid,
                }
            )


class XMLRawAttr(XMLRawDataset):
    """
    yield LINE Element to edit Element
    """

    def __init__(self, xml_files, output_dir, additional_elements=None):
        super().__init__(xml_files=xml_files, additional_elements=additional_elements)

        self.output_dir = None
        if output_dir is not None:
            self.output_dir = pathlib.Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def iter_file(self, root, img_dir, namespace, xml_file, tree, idx):
        for page in root:
            for line in self.iter_page(page, None, namespace, xml_file.stem, idx):
                if line is not None:
                    yield line
        if self.output_dir is not None:
            ET.register_namespace('', 'NDLOCRDATASET')
            tree.write(self.output_dir / xml_file.name, encoding='UTF-8')
            log.info(f'write xml {self.output_dir / xml_file.name}')

    def iter_page(self, page, image, namespace, pid, idx):
        ilines = self.get_target_element(page, namespace)
        for line in ilines:
            if '0' in [line.attrib.get('WIDTH', '0'), line.attrib.get('HEIGHT', '0')]:
                continue
            yield line


class XMLRawDatasetWithCli(XMLRawDataset):
    def set_data(self, img_data, xml_data, pid):
        self.img_data = img_data
        self.xml_data = xml_data
        self.pid = pid

    def __iter__(self):
        root = self.xml_data.getroot()
        groups = pt.match(root.tag).groups()
        if groups[1]:
            namespace = f"{{{groups[1]}}}"
        else:
            namespace = ''

        for line in self.iter_file(root, None, namespace, xml_file=None, tree=self.xml_data, idx=0):
            yield line

    def iter_file(self, root, img_dir, namespace, xml_file: pathlib.Path, tree, idx):
        for page in root:
            for line in self.iter_page(page, self.img_data, namespace, self.pid, idx):
                if line is not None:
                    yield line


class XMLRawAttrWithCli(XMLRawAttr):
    def __init__(self, xml_files, additional_elements=None):
        super().__init__(xml_files=xml_files, output_dir=None, additional_elements=additional_elements)

    def set_data(self, xml_data, pid):
        self.xml_data = xml_data
        self.pid = pid

    def __iter__(self):
        root = self.xml_data.getroot()
        groups = pt.match(root.tag).groups()
        if groups[1]:
            namespace = f"{{{groups[1]}}}"
        else:
            namespace = ''

        for line in self.iter_file(root, None, namespace, xml_file=pathlib.Path(self.pid), tree=self.xml_data, idx=0):
            yield line


class SyntheticDataset(Dataset):
    def __init__(self, fontpath, char, transforms, batch_max_length, n_div=20, **kwargs):
        import functools
        import random

        dtmp = ImageDraw.Draw(Image.new('L', (400, 200)))
        self._font = ImageFont.truetype(str(pathlib.Path(fontpath).expanduser()), 32)
        self._textsize = functools.partial(dtmp.multiline_textsize, font=self._font)
        char = list(char)
        random.shuffle(list(char))
        char = ''.join(char)
        n = len(char) // n_div + 1
        self.dl = [char[i*n_div:(i+1)*n_div] for i in range(n)]

        self.transforms = transforms

    def __len__(self):
        return len(self.dl)

    def __getitem__(self, i):
        if i < len(self.dl):
            textsize = self._textsize
            orient = 0
        s = self.dl[i]

        w, h = textsize(s)
        g = Image.new('RGB', (w, h), (200, 200, 200))
        d = ImageDraw.Draw(g)
        d.multiline_text((0, 0), s, font=self._font, fill=(50, 50, 50))

        if orient == 1:
            g = g.transpose(Image.ROTATE_90)

        return (
            self.transforms(g),
            s,
            {
                "orient": orient,
                "file_idx": 0,
                "pid": 0,
            })


def get_dataset(args, datasets, dataset_class="raw", concat=False, char=None):
    if isinstance(datasets, str):
        pass
    elif isinstance(datasets, Sequence):
        datasets = [
            get_dataset(args=args, datasets=dataset, dataset_class=dataset_class, concat=concat, char=char)
            for dataset in datasets
        ]
        if concat:
            datasets = ConcatDataset(datasets)
        return datasets
    elif isinstance(datasets, Mapping):
        return get_dataset(
            args,
            datasets.get('datasets'),
            datasets.get('dataset_class', dataset_class),
            datasets.get('concat', concat),
            char=char,
        )

    if dataset_class == 'raw':
        return XMLRawDataset(datasets, **args)
    elif dataset_class == 'cli':
        return XMLRawDatasetWithCli(**args)
    elif dataset_class == 'lmdb':
        dataset_class = XMLLMDBDataset
        return XMLLMDBDataset(datasets, **args)
    elif dataset_class == 'synth':
        return SyntheticDataset(datasets, char=char, **args)
