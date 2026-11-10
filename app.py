from pathlib import Path
import streamlit as st
from vihsd.inference import Predictor, configured_artifact

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="ViHSD · Phân loại bình luận", page_icon="💬")
st.title("💬 Phân loại bình luận tiếng Việt")
st.caption("Demo học tập ViHSD · CLEAN / OFFENSIVE / HATE")

# A single resource entry. Path + file signatures invalidate the previous model.
@st.cache_resource(max_entries=1)
def load_predictor(directory, signature):
    return Predictor(directory)

try:
    directory = configured_artifact(ROOT)
    signature = tuple((str(p.relative_to(directory)), p.stat().st_size, p.stat().st_mtime_ns)
                      for p in sorted(directory.rglob("*")) if p.is_file())
    predictor = load_predictor(str(directory), signature)
except Exception as exc:
    st.warning("Chưa thể nạp model để phân loại.")
    st.info(str(exc))
    st.markdown("Chạy một notebook trên Colab, giải nén artifact của nhóm vào `artifacts/`, "
                "sau đó cấu hình `configs/active_model.json` và khởi động lại app. "
                "Cài `requirements-neural.txt` nếu dùng TextCNN hoặc PhoBERT.")
    st.stop()

st.write(f"Model đang dùng: **{predictor.manifest['model_type']} · {predictor.manifest['run_id']}**")
if predictor.manifest["run_type"] == "smoke":
    st.warning("Artifact smoke: chỉ kiểm tra luồng, không dùng để báo cáo kết quả chính thức.")
with st.form("classify"):
    text = st.text_area("Nhập bình luận", max_chars=5000, height=160,
                        placeholder="Nhập một bình luận tiếng Việt…")
    submitted = st.form_submit_button("Phân loại", type="primary")
if submitted:
    try:
        result = predictor.predict(text)
        meanings = {"CLEAN": "Bình thường", "OFFENSIVE": "Xúc phạm", "HATE": "Thù ghét"}
        st.subheader(f"{result['label']} — {meanings[result['label']]}")
        st.write("Điểm dự đoán của model")
        st.bar_chart([{"Nhãn": label, "Điểm": score} for label, score in result["scores"].items()],
                     x="Nhãn", y="Điểm")
        st.table([{"Nhãn": label, "Điểm dự đoán": f"{score:.4f}"} for label, score in result["scores"].items()])
        st.caption("Các điểm này chưa được hiệu chỉnh xác suất. Kết quả có thể sai hoặc thiếu ngữ cảnh.")
    except Exception as exc:
        st.error(str(exc))
