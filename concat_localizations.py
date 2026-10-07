"""Concatenate localization CSVs into one continuous frame sequence.

Each CSV produced by experiment_inference.py numbers its frames from scratch, so
stacking several acquisitions requires re-basing the frame column: the first
frame of a file becomes the last frame of the previous file + 1.

The offset is inferred from the largest frame index in each file, which is only
correct when the last frames of an acquisition contain detections. When a run
ends with empty frames (or you know the true frame count), pass --n-frames /
n_frames so the offsets come from the acquisition lengths instead.

Usage:
    python concat_localizations.py a.csv b.csv c.csv --out all.csv
    python concat_localizations.py a.csv b.csv --n-frames 9887 9887 --out all.csv
"""
import argparse

import pandas as pd

FRAME_COL = 'frame'


def concat_localizations(csv_paths, n_frames=None, frame_col=FRAME_COL,
                         frame_origin=1, add_source=False):
    """Concatenate localization CSVs, shifting each file's frames after the previous one.

    :param csv_paths: paths to the CSVs (or DataFrames), in acquisition order
    :param n_frames: frame count per file, to keep detection-free trailing frames
                     in the numbering. A scalar applies to every file. None (default)
                     infers each span from the file's largest frame index.
    :param frame_col: name of the frame column
    :param frame_origin: first frame index of each file as written (1 for these CSVs)
    :param add_source: add a 'source' column holding the originating path
    :return: (concatenated DataFrame, list of the offset added to each file)
    """
    if n_frames is not None and not hasattr(n_frames, '__len__'):
        n_frames = [n_frames] * len(csv_paths)
    if n_frames is not None and len(n_frames) != len(csv_paths):
        raise ValueError(f'n_frames has {len(n_frames)} entries for {len(csv_paths)} files')

    dfs, offsets, offset = [], [], 0
    for i, path in enumerate(csv_paths):
        df = path.copy() if isinstance(path, pd.DataFrame) else pd.read_csv(path)
        if frame_col not in df.columns:
            raise ValueError(f'{path}: no {frame_col!r} column, got {list(df.columns)}')

        if n_frames is not None:
            span = int(n_frames[i])
        elif len(df):
            span = int(df[frame_col].max()) - frame_origin + 1
        else:
            span = 0  # nothing to place, and no length given: contributes no frames

        df[frame_col] = df[frame_col] + offset
        if add_source:
            df['source'] = str(path)
        dfs.append(df)
        offsets.append(offset)
        offset += span

    return pd.concat(dfs, ignore_index=True), offsets


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('csv', nargs='+', help='localization CSVs, in acquisition order')
    ap.add_argument('--out', required=True, help='output CSV path')
    ap.add_argument('--n-frames', type=int, nargs='+', default=None,
                    help='frames per file (one value, or one per file); defaults to '
                         'the largest frame index in each file')
    ap.add_argument('--frame-origin', type=int, default=1,
                    help='first frame index as written in the CSVs (default: 1)')
    ap.add_argument('--add-source', action='store_true',
                    help="add a 'source' column with the originating path")
    args = ap.parse_args()

    n_frames = args.n_frames
    if n_frames is not None and len(n_frames) == 1:
        n_frames = n_frames[0]

    df, offsets = concat_localizations(args.csv, n_frames=n_frames,
                                       frame_origin=args.frame_origin,
                                       add_source=args.add_source)
    df.to_csv(args.out, index=False)
    for path, off in zip(args.csv, offsets):
        print(f'+{off:>8} frames  {path}')
    print(f'--- {len(df)} detections over frames '
          f'{df[FRAME_COL].min():.0f}-{df[FRAME_COL].max():.0f} -> {args.out}')


if __name__ == '__main__':
    main()
