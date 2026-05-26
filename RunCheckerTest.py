import sys
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage import color
import matplotlib
matplotlib.use('Agg')

from colorizers import eccv16, siggraph17, load_img, preprocess_img, postprocess_tens

# Fixed seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

TEST_IMAGE = 'imgs/ansel_adams3.jpg'
PSNR_THRESHOLD = 18.0
SSIM_THRESHOLD = 0.55


def load_models():
    model_eccv16 = eccv16(pretrained=True).eval()
    model_siggraph17 = siggraph17(pretrained=True).eval()
    return model_eccv16, model_siggraph17


def colorize(model, tens_l_orig, tens_l_rs):
    with torch.no_grad():
        ab_pred = model(tens_l_rs).cpu()
    return postprocess_tens(tens_l_orig, ab_pred)


def to_grayscale_rgb(img_rgb):
    gray = color.rgb2gray(img_rgb)
    return np.stack([gray, gray, gray], axis=-1)


def test_models_load():
    model_eccv16, model_siggraph17 = load_models()
    assert model_eccv16 is not None
    assert model_siggraph17 is not None
    print("[PASS] Обе модели (eccv16, siggraph17) загружены успешно")
    return model_eccv16, model_siggraph17


def test_output_shape(model_eccv16, model_siggraph17):
    img = load_img(TEST_IMAGE)
    tens_l_orig, tens_l_rs = preprocess_img(img, HW=(256, 256))

    with torch.no_grad():
        out_eccv16 = model_eccv16(tens_l_rs)
        out_siggraph17 = model_siggraph17(tens_l_rs)

    assert out_eccv16.shape[1] == 2, f"eccv16: ожидается 2 канала ab, получено {out_eccv16.shape[1]}"
    assert out_siggraph17.shape[1] == 2, f"siggraph17: ожидается 2 канала ab, получено {out_siggraph17.shape[1]}"
    print(f"[PASS] Форма выхода eccv16: {tuple(out_eccv16.shape)}, siggraph17: {tuple(out_siggraph17.shape)}")


def test_colorization_quality(model_eccv16, model_siggraph17):
    img = load_img(TEST_IMAGE)
    tens_l_orig, tens_l_rs = preprocess_img(img, HW=(256, 256))

    out_eccv16 = colorize(model_eccv16, tens_l_orig, tens_l_rs)
    out_siggraph17 = colorize(model_siggraph17, tens_l_orig, tens_l_rs)

    # Привести оригинал к тому же размеру что и выход
    from PIL import Image
    import numpy as np
    orig_resized = np.array(Image.fromarray((img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)).resize(
        (out_eccv16.shape[1], out_eccv16.shape[0]))) / 255.0

    out_eccv16_f = out_eccv16.astype(np.float64)
    out_siggraph17_f = out_siggraph17.astype(np.float64)
    orig_f = (orig_resized * 255).astype(np.float64) if orig_resized.max() <= 1.0 else orig_resized.astype(np.float64)
    out_eccv16_255 = (out_eccv16_f * 255) if out_eccv16_f.max() <= 1.0 else out_eccv16_f
    out_siggraph17_255 = (out_siggraph17_f * 255) if out_siggraph17_f.max() <= 1.0 else out_siggraph17_f

    psnr_eccv16 = psnr(orig_f, out_eccv16_255, data_range=255)
    psnr_siggraph17 = psnr(orig_f, out_siggraph17_255, data_range=255)
    ssim_eccv16 = ssim(orig_f, out_eccv16_255, data_range=255, channel_axis=2)
    ssim_siggraph17 = ssim(orig_f, out_siggraph17_255, data_range=255, channel_axis=2)

    print(f"\n  {'Модель':<15} {'PSNR (дБ)':<12} {'SSIM':<8} {'Порог PSNR':<12} {'Порог SSIM'}")
    print(f"  {'-'*55}")
    print(f"  {'eccv16':<15} {psnr_eccv16:<12.2f} {ssim_eccv16:<8.3f} {PSNR_THRESHOLD:<12} {SSIM_THRESHOLD}")
    print(f"  {'siggraph17':<15} {psnr_siggraph17:<12.2f} {ssim_siggraph17:<8.3f} {PSNR_THRESHOLD:<12} {SSIM_THRESHOLD}")

    assert psnr_eccv16 >= PSNR_THRESHOLD, f"eccv16 PSNR {psnr_eccv16:.2f} < {PSNR_THRESHOLD}"
    assert psnr_siggraph17 >= PSNR_THRESHOLD, f"siggraph17 PSNR {psnr_siggraph17:.2f} < {PSNR_THRESHOLD}"
    assert ssim_eccv16 >= SSIM_THRESHOLD, f"eccv16 SSIM {ssim_eccv16:.3f} < {SSIM_THRESHOLD}"
    assert ssim_siggraph17 >= SSIM_THRESHOLD, f"siggraph17 SSIM {ssim_siggraph17:.3f} < {SSIM_THRESHOLD}"
    print(f"[PASS] Качество колоризации выше порогов (PSNR>{PSNR_THRESHOLD}, SSIM>{SSIM_THRESHOLD})")


def test_reproducibility(model_eccv16):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    img = load_img(TEST_IMAGE)
    tens_l_orig, tens_l_rs = preprocess_img(img, HW=(256, 256))
    out1 = colorize(model_eccv16, tens_l_orig, tens_l_rs)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    img = load_img(TEST_IMAGE)
    tens_l_orig, tens_l_rs = preprocess_img(img, HW=(256, 256))
    out2 = colorize(model_eccv16, tens_l_orig, tens_l_rs)

    assert np.allclose(out1, out2), "Результаты двух прогонов не совпадают"
    print("[PASS] Воспроизводимость: два прогона дают идентичный результат")


def main():
    print("=" * 60)
    print("Colorful Image Colorization - RunCheckerTest")
    print(f"Изображение: {TEST_IMAGE} | Seed: {SEED}")
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

    print("\n" + "=" * 60)
    if failed == 0:
        print(f"Все {len(tests) + 1} теста пройдены успешно.")
    else:
        print(f"{failed}/{len(tests)} тестов НЕ пройдено.")
        sys.exit(1)


if __name__ == "__main__":
    main()
