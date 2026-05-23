import gradio as gr
import numpy as np
from PIL import Image
import tensorflow as tf

# ── Patch tương thích: bỏ qua quantization_config ──────────────────
from keras.layers import Dense as _Dense
_orig_dense_init = _Dense.__init__

def _patched_dense_init(self, *args, quantization_config=None, **kwargs):
    _orig_dense_init(self, *args, **kwargs)

_Dense.__init__ = _patched_dense_init
# ───────────────────────────────────────────────────────────────────

# Load model
model = tf.keras.models.load_model('hinh_hoc_model.keras', compile=False)
print("✅ Model loaded!")

CLASS_NAMES = ['hinh binh hanh', 'hinh chu nhat', 'hinh thang',
               'hinh tron', 'hinh vuong', 'tam giac']

LABELS = {
    'hinh binh hanh': '▱  Hình Bình Hành',
    'hinh chu nhat':  '📱 Hình Chữ Nhật',
    'hinh thang':     '🏔️ Hình Thang',
    'hinh tron':      '⭕ Hình Tròn',
    'hinh vuong':     '🟥 Hình Vuông',
    'tam giac':       '🔺 Hình Tam Giác',
}

FUN_FACTS = {
    'hinh binh hanh': 'Hình bình hành có 2 cặp cạnh song song — giống chữ nhật bị nghiêng! 📐',
    'hinh chu nhat':  'Hình chữ nhật có 4 góc vuông. Điện thoại, sách vở là hình này! 📚',
    'hinh thang':     'Hình thang có 1 cặp cạnh song song. Hay gặp trong kiến trúc! 🏛️',
    'hinh tron':      'Hình tròn không có góc, mọi điểm cách đều tâm. Như bánh pizza! 🍕',
    'hinh vuong':     'Hình vuông có 4 cạnh bằng nhau và 4 góc vuông. Như ô bàn cờ! ♟️',
    'tam giac':       'Tam giác có 3 cạnh, 3 góc. Mái nhà và biển báo dùng hình này! 🏠',
}

IMAGE_SIZE = 64  # Thay đúng với kích thước bạn train

def nhan_dien(image):
    if image is None:
        return {}, "Hãy chọn hoặc chụp ảnh!"
    img = Image.fromarray(image).convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    pred = model.predict(np.expand_dims(arr, 0), verbose=0)[0]
    ket_qua = {LABELS[CLASS_NAMES[i]]: float(pred[i]) for i in range(len(CLASS_NAMES))}
    best_idx = int(np.argmax(pred))
    best_cls = CLASS_NAMES[best_idx]
    conf = pred[best_idx] * 100
    fact = FUN_FACTS.get(best_cls, '')
    thong_bao = f"✅ Đây là **{LABELS[best_cls]}** ({conf:.1f}%)\n\n💡 {fact}"
    return ket_qua, thong_bao

with gr.Blocks(title="Nhận Diện Hình Học", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔷 Nhận Diện Hình Học\n### App học hình học vui cho bé yêu ✨")
    with gr.Row():
        with gr.Column():
            inp = gr.Image(sources=["upload", "webcam", "clipboard"],
                           label="📸 Chụp hoặc chọn ảnh hình học", type="numpy")
            btn = gr.Button("🔍 Nhận Diện!", variant="primary", size="lg")
        with gr.Column():
            out_label = gr.Label(label="📊 Độ tin cậy từng hình")
            out_text  = gr.Markdown(label="🎯 Kết quả")
    btn.click(fn=nhan_dien, inputs=inp, outputs=[out_label, out_text])
    inp.change(fn=nhan_dien, inputs=inp, outputs=[out_label, out_text])
    gr.Markdown("---\n**Các hình nhận diện:** Bình Hành · Chữ Nhật · Thang · Tròn · Vuông · Tam Giác")

demo.launch()
