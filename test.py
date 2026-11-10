
import numpy as np
from pathlib import Path

data_path = Path('data/raw')

train = np.load(data_path / 'islr101_train.npy')
train_label = np.load(data_path / 'islr101_train_label.npy')
test = np.load(data_path / 'islr101_test.npy')
test_label = np.load(data_path / 'islr101_test_label.npy')

print('='*60)
print('TRAIN')
print('  shape :', train.shape)
print('  dtype :', train.dtype)
print('  min   :', train.min())
print('  max   :', train.max())
print('  mean  :', train.mean())
print()
print('TRAIN LABELS')
print('  shape :', train_label.shape)
print('  dtype :', train_label.dtype)
print('  unique classes:', len(np.unique(train_label)))
print('  min/max label:', train_label.min(), train_label.max())
print()
print('TEST')
print('  shape :', test.shape)
print('  dtype :', test.dtype)
print('  min   :', test.min())
print('  max   :', test.max())
print()
print('TEST LABELS')
print('  shape :', test_label.shape)
print('  dtype :', test_label.dtype)
print('  unique classes:', len(np.unique(test_label)))
print('='*60)

# بررسی چند نمونه برای فهمیدن معنای محورها
print()
print('Sample[0] shape after squeeze:', train[0].squeeze().shape)
print('Sample[0] non-zero frames check (axis=-1 or -2):')
s = train[0].squeeze()
print('  shape of one sample:', s.shape)
if s.ndim == 3:
    # فرض‌های ممکن را چک می‌کنیم
    print('  sum over axis=0 (channels?):', s.sum(axis=0).shape)
    print('  sum over axis=1:', s.sum(axis=1).shape)
    print('  sum over axis=2:', s.sum(axis=2).shape)
    print('  number of all-zero slices along last axis:', np.all(s == 0, axis=(0,1)).sum() if s.shape[-1] < 200 else 'too many')
    print('  number of all-zero slices along second last axis:', np.all(s == 0, axis=(0,2)).sum() if s.shape[-2] < 200 else 'too many')
