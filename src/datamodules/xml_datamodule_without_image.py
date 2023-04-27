# Copyright (c) 2023, National Diet Library, Japan
#
# This software is released under the CC BY 4.0.
# https://creativecommons.org/licenses/by/4.0/

from datasets import Dataset, DatasetDict

import joblib
import pathlib
import itertools

import pandas as pd
import numpy as np

import xml.etree.ElementTree as ET

ET.register_namespace('', 'NDLOCRDATASET')


def prep_df(df):
    df['LEN'] = df['STRING'].str.len()
    df['AUTHOR'] = (df['AUTHOR'] == 'TRUE').astype(int)
    df['TITLE'] = (df['TITLE'] == 'TRUE').astype(int)
    # df['DIRECTION'] = np.where(df['DIRECTION'] == '右から左', 1, 0)

    if 0:
        dfx, dfy = df['X'].copy(), df['Y'].copy()
        w, h = df['WIDTH'].copy(), df['HEIGHT'].copy()
        aw, ah = df['AWIDTH'].copy(), df['AHEIGHT'].copy()
        df['X'] = np.where(df['WIDTH'] > df['HEIGHT'], dfx, dfy)
        df['Y'] = np.where(df['WIDTH'] > df['HEIGHT'], dfy, dfx)
        df['WIDTH'] = np.where(w > h, w, h)
        df['HEIGHT'] = np.where(w > h, h, w)
        df['AWIDTH'] = np.where(w > h, aw, ah)
        df['AHEIGHT'] = np.where(w > h, ah, aw)


def read_xml(p):
    ret = []
    try:
        tree = ET.parse(p)
    except Exception as e:
        print('==================')
        print(p)
        print('==================')
        raise e
    for page in tree.iterfind('.//{*}PAGE'):
        page_width = page.attrib['WIDTH']
        page_height = page.attrib['HEIGHT']
        lines = list(page.iterfind('.//{*}LINE'))
        d = pd.DataFrame([li.attrib for li in lines])
        if d.size == 0:
            continue
        d['AWIDTH'] = d['WIDTH'].astype(float).mean()
        d['AHEIGHT'] = d['HEIGHT'].astype(float).mean()
        d['USAGE'] = p.parents[2].name[:-1]  # gakushu or hyouka
        d['LEVEL'] = p.parents[3].name       # 1, 2, 3, 4
        d['PWIDTH'] = page_width
        d['PHEIGHT'] = page_height
        ret.append(d)
    return ret


def generate_dataframe(path):
    list_xml = itertools.chain(*[pathlib.Path(p).rglob('*.xml') for p in path])
    list_xml = sorted(list_xml)
    list_df = joblib.Parallel(n_jobs=-1, verbose=5)(joblib.delayed(read_xml)(p) for p in list_xml)
    list_df = itertools.chain.from_iterable(list_df)
    df = pd.concat(list_df)

    prep_df(df)

    return df


def from_pd(data_dirs, text, labels, target, downsampling_rate=None, random_flip=False, level_filter=None):
    df = generate_dataframe(data_dirs)
    df = df[df[text].str.len() > 1]

    if level_filter is not None:
        df = df[df['LEVEL'].astype(int) == level_filter]

    df.reset_index(inplace=True, drop=True)

    is_train = df['USAGE'] == 'gakushu'
    level = df['LEVEL'].astype(int)
    df = df[[text] + labels]

    if random_flip:
        texts = df[text]
        target = np.where(np.random.rand(len(texts)) > 0.5, True, False)
        direction = pd.Series(target.tolist(), name='DIRECTION')
        reverse_texts = texts[direction.tolist()]  # pointer
        texts.loc[direction.tolist()] = reverse_texts.apply(lambda s: s[::-1])
        df = pd.concat([df, direction], axis=1)

    train_df = df[is_train]
    if len(train_df):
        sample_target = min(int(train_df[target].sum() * 0.1), 100)
        sample_vtarget = min(int((1-train_df[target]).sum() * 0.1), 900)

        val_df = pd.concat([
            train_df[train_df[target].astype(bool)].sample(n=sample_target),
            train_df[~train_df[target].astype(bool)].sample(n=sample_vtarget),
        ])
    else:
        val_df = pd.DataFrame(columns=[text] + labels)
    train_df.drop(val_df.index)

    if downsampling_rate is not None:
        # log.info(f"dataset downsampling {cfg.downsampling_per}%")

        df1 = df[is_train & df[target]]
        df2 = df[is_train & ~df[target]]
        df2 = df2.sample(n=min(round(len(df1) * downsampling_rate), len(df2)))
        print("target size", len(df1))
        print("non target size", len(df2))
        train_df = pd.concat([df1, df2])
    test_df = pd.concat([df[~is_train], level[~is_train]], axis=1)

    train_df.reset_index(inplace=True, drop=True)
    test_df.reset_index(inplace=True, drop=True)
    val_df.reset_index(inplace=True, drop=True)
    dataset = DatasetDict(
        {
            'train': Dataset.from_pandas(train_df),
            'val': Dataset.from_pandas(val_df),
            'test': Dataset.from_pandas(test_df),
        }
    )
    return dataset


def from_csv(path, **kwargs):
    ds = Dataset.from_csv(path)
    return ds


def from_tree(tree, text, **kwargs):
    list_df = []

    for page in tree.iterfind('.//{*}PAGE'):
        lines = list(page.iterfind('.//{*}LINE'))
        d = pd.DataFrame([li.attrib for li in lines])
        if d.size == 0:
            continue
        d['AWIDTH'] = d['WIDTH'].astype(float).mean()
        d['AHEIGHT'] = d['HEIGHT'].astype(float).mean()
        list_df.append(d)

    df = d
    if list_df:
        df = pd.concat(list_df)

        df['LEN'] = df['STRING'].str.len()
        df['AUTHOR'] = 0
        df['TITLE'] = 0

        df = df[df[text].str.len() > 1]

    df.reset_index(inplace=True, drop=True)
    dataset = DatasetDict(
        {
            'test': Dataset.from_pandas(df)
        }
    )

    return dataset
