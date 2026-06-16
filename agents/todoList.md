# Danh sách công việc cần chỉnh sửa cho Lab 03 (Customer Segmentation)

Tài liệu này ghi lại các đầu việc cần thực hiện để tái cấu trúc dự án Lab 03 tương tự như Lab 01.

---

## 1. Cấu trúc thư mục & Di chuyển file
- [ ] Tạo thư mục mã nguồn `/Users/quangmanh/Project/lab03/src`.
- [ ] Di chuyển các file Python sau từ `notebooks/` sang `src/` sử dụng `git mv`:
  - `clustering.py` -> `src/clustering.py`
  - `eda.py` -> `src/eda.py`
  - `preprocess.py` -> `src/preprocess.py`
  - `run_pipeline.py` -> `src/run_pipeline.py`
  - `utils.py` -> `src/utils.py`
- [ ] Sửa lại cell cài đặt và imports trong `notebooks/lab03.ipynb` (Cell 2):
  - Thay thế `NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"` bằng `SRC_DIR = PROJECT_ROOT / "src"`.
  - Thêm `SRC_DIR` vào `sys.path`.

---

## 2. Tiêu chí chọn K (Model Selection)
- [ ] Cập nhật lại logic tìm và chọn số cụm $K$ tối ưu:
  - **Chỉ sử dụng**: Phương pháp Khuỷu tay (**Elbow Method** / Inertia) và Chỉ số dáng điệu (**Silhouette Coefficient**).
  - Loại bỏ hoặc không phụ thuộc vào Calinski-Harabasz và Davies-Bouldin trong hàm lựa chọn mô hình tự động (`select_best_model`).
- [ ] Cập nhật bảng biểu hiển thị và đồ thị trong notebook chỉ tập trung vào Elbow và Silhouette.

---

## 3. Tạo module mô hình tự viết (From Scratch)
- [ ] Tạo file mới `/Users/quangmanh/Project/lab03/src/model_from_scratch.py`.
- [ ] Triển khai các lớp phân cụm tự viết (From Scratch):
  - **Lớp `KMeans`**: Triển khai thuật toán K-Means (khởi tạo centroid, gán nhãn cụm, cập nhật centroid dựa trên trung bình cộng, lặp cho đến khi hội tụ).
  - **Lớp `KMedoids`**: Triển khai thuật toán K-Medoids (sử dụng medoid thực tế thay vì centroid trung bình cộng, tối thiểu hóa khoảng cách Manhattan hoặc Euclidean đến medoid).
  - **Lớp `HierarchicalClustering`**: Triển khai thuật toán phân cụm phân cấp (Agglomerative, tính toán ma trận khoảng cách, gộp cụm theo linkage như Single, Complete hoặc Average).
  - **Lớp đối chiếu thư viện**: Lớp wrapper gọi thư viện gốc (`scikit-learn` cho KMeans/Agglomerative, `scikit-learn-extra` hoặc tự viết cho KMedoids) để kiểm chứng kết quả.

---

## 4. Cập nhật và kiểm chứng trên Notebook
- [ ] Thêm một cell kiểm chứng duy nhất (Double Check) trong notebook chạy so sánh kết quả giữa mô hình tự viết và mô hình thư viện trên cùng một tập dữ liệu nhỏ (đảm bảo nhãn phân cụm và giá trị Silhouette/Inertia trùng khớp).
- [ ] Thay đổi toàn bộ các cell chạy phân cụm chính trong notebook sang sử dụng các lớp mô hình tự viết từ `src/model_from_scratch.py`.
- [ ] Đảm bảo notebook chạy không lỗi từ đầu đến cuối và lưu lại toàn bộ outputs, đồ thị phân cụm mới.
