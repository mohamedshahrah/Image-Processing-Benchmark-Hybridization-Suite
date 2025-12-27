import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from src.algorithms import ImageProcessor  # Importing from your professional structure

st.set_page_config(page_title="Image Processing Benchmark", layout="wide")

st.title("🖼️ Image Processing Expert: Benchmark & Hybrid Lab")
st.markdown("Use the sidebar to switch between **Benchmarks** (Comparing algorithms) and **Hybrid Merge** (Combining algorithms).")

# --- Sidebar ---
st.sidebar.header("Project Controls")
mode = st.sidebar.radio("Select Mode", ["Benchmark Suite", "Hybrid / Merge Studio"])
uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file)
    img_rgb = np.array(image_pil)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(img_rgb, caption="Original Image", use_container_width=True)

    # ==========================
    # MODE 1: BENCHMARK
    # ==========================
    if mode == "Benchmark Suite":
        st.header("📊 Algorithm Benchmark Reports")
        bench_type = st.selectbox("Select Scenario", 
                                  ["Interpolation", "Noise Removal", "Thresholding", "Grayscale"])
        
        if st.button("Run Benchmark"):
            results = []
            
            if bench_type == "Interpolation":
                st.subheader("Nearest vs Bilinear vs Bicubic")
                methods = ["Nearest Neighbor", "Linear", "Bicubic"]
                cols = st.columns(len(methods))
                for idx, m in enumerate(methods):
                    res, time_taken = ImageProcessor.resize_image(img_rgb, 2.0, m)
                    with cols[idx]:
                        st.image(res, caption=m, use_container_width=True)
                        st.metric("Time (ms)", f"{time_taken:.4f}")
                        results.append({"Method": m, "Time": time_taken})

            elif bench_type == "Noise Removal":
                st.subheader("Mean vs Median Filter")
                noisy = ImageProcessor.add_noise(img_gray)
                st.image(noisy, caption="Noisy Input", width=300)
                
                filters = ["Mean (Box)", "Median", "Gaussian"]
                cols = st.columns(len(filters))
                for idx, f in enumerate(filters):
                    res, time_taken = ImageProcessor.apply_filter(noisy, f, 5)
                    diff = np.mean(np.abs(img_gray - res))
                    with cols[idx]:
                        st.image(res, caption=f, use_container_width=True)
                        st.metric("Time", f"{time_taken:.4f} ms")
                        st.write(f"**Error:** {diff:.2f}")
                        results.append({"Filter": f, "Time": time_taken, "Error": diff})
            
            elif bench_type == "Thresholding":
                res_m, t_m = ImageProcessor.threshold(img_gray, "Binary (Manual)")
                res_o, t_o = ImageProcessor.threshold(img_gray, "Otsu")
                
                c1, c2 = st.columns(2)
                with c1:
                    st.image(res_m, caption="Manual", use_container_width=True)
                    st.write(f"Time: {t_m:.4f} ms")
                with c2:
                    st.image(res_o, caption="Otsu", use_container_width=True)
                    st.write(f"Time: {t_o:.4f} ms")
                results = [{"Method": "Manual", "Time": t_m}, {"Method": "Otsu", "Time": t_o}]

            elif bench_type == "Grayscale":
                methods = ["Average", "Luma (BT-709)", "Lightness"]
                cols = st.columns(3)
                for idx, m in enumerate(methods):
                    res, t = ImageProcessor.to_grayscale(img_rgb, m)
                    with cols[idx]:
                        st.image(res, caption=m, use_container_width=True)
                        st.write(f"Time: {t:.4f} ms")
                    results.append({"Method": m, "Time": t})

            if results:
                st.dataframe(pd.DataFrame(results))

    # ==========================
    # MODE 2: HYBRID MERGE
    # ==========================
    elif mode == "Hybrid / Merge Studio":
        st.header("🧪 Hybrid Algorithm Lab")
        
        # Select A
        st.sidebar.subheader("Algorithm A")
        cat_a = st.sidebar.selectbox("Category A", ["Filter (Smoothing)", "Filter (Edge)", "Morphology"])
        if cat_a == "Filter (Smoothing)":
            algo_a_res, _ = ImageProcessor.apply_filter(img_gray, st.sidebar.selectbox("Type A", ["Gaussian", "Median"]))
        elif cat_a == "Filter (Edge)":
            algo_a_res, _ = ImageProcessor.apply_filter(img_gray, st.sidebar.selectbox("Type A", ["Sobel Edge", "Laplacian"]))
        else:
            algo_a_res, _ = ImageProcessor.apply_morphology(img_gray, st.sidebar.selectbox("Type A", ["Erosion", "Dilation"]))

        # Select B
        st.sidebar.subheader("Algorithm B")
        cat_b = st.sidebar.selectbox("Category B", ["Filter (Smoothing)", "Filter (Edge)", "Morphology"], index=1)
        if cat_b == "Filter (Smoothing)":
            algo_b_res, _ = ImageProcessor.apply_filter(img_gray, st.sidebar.selectbox("Type B", ["Gaussian", "Median"]))
        elif cat_b == "Filter (Edge)":
            algo_b_res, _ = ImageProcessor.apply_filter(img_gray, st.sidebar.selectbox("Type B", ["Sobel Edge", "Laplacian"]))
        else:
            algo_b_res, _ = ImageProcessor.apply_morphology(img_gray, st.sidebar.selectbox("Type B", ["Erosion", "Dilation"]))

        # Merge
        st.sidebar.subheader("Merge Logic")
        m_type = st.sidebar.selectbox("Op", ["Weighted Blend", "Max", "Min", "Difference"])
        
        if m_type == "Weighted Blend":
            merged = cv2.addWeighted(algo_a_res, 0.5, algo_b_res, 0.5, 0)
        elif m_type == "Max":
            merged = cv2.max(algo_a_res, algo_b_res)
        elif m_type == "Min":
            merged = cv2.min(algo_a_res, algo_b_res)
        elif m_type == "Difference":
            merged = cv2.absdiff(algo_a_res, algo_b_res)

        c1, c2, c3 = st.columns(3)
        with c1: st.image(algo_a_res, caption="Result A", use_container_width=True)
        with c2: st.image(algo_b_res, caption="Result B", use_container_width=True)
        with c3: st.image(merged, caption="Hybrid Result", use_container_width=True)

else:
    st.info("Upload an image to start.")