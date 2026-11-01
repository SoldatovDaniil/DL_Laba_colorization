import sys
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage import color
import matplotlib
matplotlib.use('Agg')

from colorizers import eccv16, siggraph17, load_img, preprocess_img, postprocess_tens

# Фиксированный seed. Сами модели в eval() детерминированы,
# seed страхует от случайности в окружении/предобработке.
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Цветные изображения из ImageNet val: оригинал служит эталоном,
# на вход модели подаётся только канал яркости L.
TEST_IMAGES = [
    'imgs/ILSVRC2012_val_00041580.JPEG',
    'imgs/ILSVRC2012_val_00046524.JPEG',
    'imgs/ILSVRC2012_val_00046834.JPEG',
]
PSNR_THRESHOLD = 18.0        # средний PSNR (дБ) по RGB относительно цветного оригинала
SSIM_THRESHOLD = 0.75        # средний SSIM по RGB относительно цветного оригинала
AB_GAIN_THRESHOLD = 0.25     # ошибка по ab должна быть хотя бы на 25% меньше, чем у серого baseline
SATURATION_THRESHOLD = 5.0   # средний |ab| выхода: модель действительно добавляет цвет


def load_models():
    model_eccv16 = eccv16(pretrained=True).eval()
    model_siggraph17 = siggraph17(pretrained=True).eval()
    return model_eccv16, model_siggraph17


def colorize(model, tens_l_orig, tens_l_rs):
    with torch.no_grad():
        ab_pred = model(tens_l_rs).cpu()
    return postprocess_tens(tens_l_orig, ab_pred)


def gray_baseline(tens_l_orig):
    # "Модель", которая ничего не раскрашивает: ab = 0
    return postprocess_tens(tens_l_orig, torch.zeros(1, 2, *tens_l_orig.shape[2:]))


def evaluate(img_rgb, out_rgb):
    # img_rgb: uint8 оригинал, out_rgb: float [0, 1] результат колоризации
    orig = img_rgb.astype(np.float64) / 255.0
    ab_orig = color.rgb2lab(img_rgb)[..., 1:]
    ab_out = color.rgb2lab(out_rgb)[..., 1:]
    return {
        'psnr': psnr(orig, out_rgb, data_range=1.0),
        'ssim': ssim(orig, out_rgb, data_range=1.0, channel_axis=2),
        'ab_mae': np.abs(ab_out - ab_orig).mean(),
        'saturation': np.abs(ab_out).mean(),
    }


def test_models_load():
    model_eccv16, model_siggraph17 = load_models()
    assert model_eccv16 is not None
    assert model_siggraph17 is not None
    print("[PASS] Обе модели (eccv16, siggraph17) загружены успешно")
    return model_eccv16, model_siggraph17


def test_output_shape(model_eccv16, model_siggraph17):
    img = load_img(TEST_IMAGES[0])
    tens_l_orig, tens_l_rs = preprocess_img(img, HW=(256, 256))

    with torch.no_grad():
        out_eccv16 = model_eccv16(tens_l_rs)
        out_siggraph17 = model_siggraph17(tens_l_rs)

    for name, out in [('eccv16', out_eccv16), ('siggraph17', out_siggraph17)]:
        assert tuple(out.shape) == (1, 2, 256, 256), \
            f"{name}: ожидается форма (1, 2, 256, 256), получено {tuple(out.shape)}"
    print(f"[PASS] Форма выхода eccv16: {tuple(out_eccv16.shape)}, siggraph17: {tuple(out_siggraph17.shape)}")


def test_colorization_quality(model_eccv16, model_siggraph17):
    models = {'gray (baseline)': None, 'eccv16': model_eccv16, 'siggraph17': model_siggraph17}
    scores = {name: [] for name in models}

    for path in TEST_IMAGES:
        img = load_img(path)
        tens_l_orig, tens_l_rs = preprocess_img(img, HW=(256, 256))
        for name, model in models.items():
            out = gray_baseline(tens_l_orig) if model is None else colorize(model, tens_l_orig, tens_l_rs)
            scores[name].append(evaluate(img, out))

    mean = {name: {k: np.mean([s[k] for s in per_img]) for k in per_img[0]}
            for name, per_img in scores.items()}

    print(f"\n  Среднее по {len(TEST_IMAGES)} цветным изображениям:")
    print(f"  {'Модель':<17} {'PSNR (дБ)':<11} {'SSIM':<8} {'ab MAE':<9} {'|ab|'}")
    print(f"  {'-'*55}")
    for name, m in mean.items():
        print(f"  {name:<17} {m['psnr']:<11.2f} {m['ssim']:<8.3f} {m['ab_mae']:<9.2f} {m['saturation']:.2f}")

    baseline_mae = mean['gray (baseline)']['ab_mae']
    for name in ('eccv16', 'siggraph17'):
        m = mean[name]
        assert m['psnr'] >= PSNR_THRESHOLD, f"{name} PSNR {m['psnr']:.2f} < {PSNR_THRESHOLD}"
        assert m['ssim'] >= SSIM_THRESHOLD, f"{name} SSIM {m['ssim']:.3f} < {SSIM_THRESHOLD}"
        assert m['saturation'] >= SATURATION_THRESHOLD, \
            f"{name} почти не добавляет цвет: |ab| {m['saturation']:.2f} < {SATURATION_THRESHOLD}"
        gain = 1 - m['ab_mae'] / baseline_mae
        assert gain >= AB_GAIN_THRESHOLD, \
            f"{name} ab MAE {m['ab_mae']:.2f} лишь на {gain:.0%} лучше серого baseline ({baseline_mae:.2f})"
        print(f"[PASS] {name}: ab MAE на {gain:.0%} лучше серого baseline, "
              f"PSNR>{PSNR_THRESHOLD}, SSIM>{SSIM_THRESHOLD}")


def test_reproducibility(model_eccv16):
    img = load_img(TEST_IMAGES[0])

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    tens_l_orig, tens_l_rs = preprocess_img(img, HW=(256, 256))
    out1 = colorize(model_eccv16, tens_l_orig, tens_l_rs)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    tens_l_orig, tens_l_rs = preprocess_img(img, HW=(256, 256))
    out2 = colorize(model_eccv16, tens_l_orig, tens_l_rs)

    assert np.allclose(out1, out2), "Результаты двух прогонов не совпадают"
    print("[PASS] Воспроизводимость: два прогона дают идентичный результат")


def main():
    print("=" * 60)
    print("Colorful Image Colorization - RunCheckerTest")
    print(f"Изображения: {len(TEST_IMAGES)} шт. из imgs/ | Seed: {SEED}")
    print("=" * 60)

    failed = 0

    print("\n--- Загрузка моделей ---")
    try:
        model_eccv16, model_siggraph17 = test_models_load()
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    tests = [
        ("Форма выхода",           lambda: test_output_shape(model_eccv16, model_siggraph17)),
        ("Качество колоризации",   lambda: test_colorization_quality(model_eccv16, model_siggraph17)),
        ("Воспроизводимость",      lambda: test_reproducibility(model_eccv16)),
    ]

    for name, test_fn in tests:
        print(f"\n--- {name} ---")
        try:
            test_fn()
        except AssertionError as e:
            print(f"[FAIL] {e}")
            failed += 1
        except Exception as e:
            print(f"[ERROR] {e}")
            failed += 1

    total = len(tests) + 1
    print("\n" + "=" * 60)
    if failed == 0:
        print(f"Все {total} теста пройдены успешно.")
    else:
        print(f"{failed}/{total} тестов НЕ пройдено.")
        sys.exit(1)


if __name__ == "__main__":
    main()
