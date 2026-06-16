### 1. Tổng quan Dự án
Notebook thực hiện bài toán **Phân khúc khách hàng (Customer Segmentation)** trên tập dữ liệu mua sắm trung tâm thương mại (`Shopping Mall Customer Segmentation Data .csv`) gồm **15.079 khách hàng** với 5 trường thông tin: `Customer ID`, `Age`, `Gender`, `Annual Income`, `Spending Score`.

---

### 2. Chi tiết các bước trong Notebook

* **Step 0 — Imports And Setup (Cell 1-2)**:
  - Thiết lập thư mục gốc dự án (`PROJECT_ROOT` là `/Users/quangmanh/Project/lab03`).
  - Thêm thư mục mã nguồn chứa các module tự viết (`utils`, `preprocess`, `eda`, `clustering`) vào `sys.path` để notebook import trực tiếp.

* **Step 1 — Load Data And Raw Overview (Cell 3-4)**:
  - Tự động tìm và load file dữ liệu thô (.csv).
  - Hiển thị thông tin tổng quan: Kích thước tập dữ liệu (`15079` dòng, `5` cột), kiểu dữ liệu (Age, Income, Spending Score là số nguyên; Gender, Customer ID là chuỗi).
  - Kiểm tra nhanh giá trị thiếu (0) và trùng lặp (0), sau đó lưu thống kê thô ra file `data/processed/metrics/eda_summary.csv`.

* **Step 2 — Data Quality Checks (Cell 5-7)**:
  - Kiểm tra sâu hơn về chất lượng dữ liệu: Xác nhận không có giá trị trống hay trùng lặp.
  - Kiểm tra khoảng giá trị hợp lệ (Age từ 18-90, Income từ 20k-200k$, Spending Score từ 1-100).
  - Thống kê ngoại lai (outliers) bằng phương pháp IQR và Z-score (kết quả ghi nhận 0 ngoại lai).
  - Vẽ và lưu các biểu đồ chất lượng dữ liệu (biểu đồ thiếu, trùng lặp và boxplots) vào thư mục `reports/figures/before_process/`.

* **Step 3 — EDA Before Preprocessing (Cell 8-9)**:
  - Khảo sát phân bố đơn biến (KDE/Histograms) của tuổi, thu nhập, điểm chi tiêu và tỷ lệ giới tính (nam/nữ khá cân bằng).
  - Phân tích tương quan tuyến tính (hệ số tương quan gần như bằng 0 giữa các trường số).
  - Vẽ các biểu đồ phân tán (scatter plots) để tìm quy luật nhóm sớm. Nổi bật nhất là đồ thị phân tán giữa **Annual Income** và **Spending Score** biểu hiện rõ rệt **4 cụm/nhóm khách hàng** phân tách theo các góc phần tư.
  - Lưu toàn bộ hình ảnh vào thư mục `reports/figures/before_process/`.

* **Step 4 — Preprocessing And Feature Scaling (Cell 10-11)**:
  - Chuẩn hóa tên cột về dạng lowercase snake_case (`customer_id`, `age`, `gender`, `annual_income`, `spending_score`).
  - Áp dụng các kỹ thuật chuẩn hóa đặc trưng số (`StandardScaler`, `MinMaxScaler`, `RobustScaler`).
  - Tạo ra 4 tập đặc trưng thử nghiệm: `behavior_numeric` (3 biến số), `behavior_plus_gender` (thêm one-hot gender), `income_spending_only` (chỉ thu nhập và chi tiêu), và `age_spending_only` (tuổi và chi tiêu).
  - Lưu dữ liệu sạch ra `customers_clean.csv` và dữ liệu sau chuẩn hóa ra `customers_scaled.csv`.

* **Step 5 — Clustering Experiments (Cell 12-13)**:
  - Thực hiện chạy thử nghiệm lưới (Grid Search) cho 3 thuật toán phân cụm: **K-Means** ($K=2 \dots 10$), **Agglomerative Clustering** (phân cấp, $N=2 \dots 10$ với các linkage khác nhau) và **DBSCAN** (quét theo mật độ với các eps/min_samples khác nhau).
  - Tính toán các chỉ số đánh giá: **Silhouette coefficient**, **Calinski-Harabasz score**, **Davies-Bouldin score**, và **Inertia**.
  - Lưu bảng kết quả thử nghiệm ra file `data/processed/metrics/clustering_metrics.csv`.

* **Step 6 — Model Selection (Cell 14-15)**:
  - Lựa chọn mô hình tối ưu dựa trên quy tắc cân bằng: ưu tiên số cụm từ 3-7, chỉ số Silhouette cao, Davies-Bouldin thấp và phân cụm trực quan rõ ràng nhất.
  - **Mô hình được chọn**: **K-Means** với **4 cụm**, sử dụng đặc trưng **`income_spending_only`** (chỉ dùng thu nhập và điểm chi tiêu) chuẩn hóa bằng **`StandardScaler`** (Model ID 56).
  - Chỉ số đạt được: Silhouette = **0.408**, Calinski-Harabasz = **15,050.68**, Davies-Bouldin = **0.7687**.

* **Step 7 — Cluster Visualization (Cell 16-17)**:
  - Trực quan hóa kết quả phân cụm của mô hình được chọn trên đồ thị 2D (Income vs Spending Score) hiển thị rõ ràng 4 nhóm khách hàng cùng tọa độ tâm cụm (centroids) đưa về thang đo gốc.
  - Lưu biểu đồ phân cụm vào `reports/figures/clustering/`.

* **Step 8 — Segment Profiling (Cell 18-19)**:
  - Tính toán thống kê mô tả (mean/median của tuổi, thu nhập, điểm chi tiêu) cho từng cụm và đặt tên thương mại kèm đề xuất hành động marketing:
    1. **Cụm 0 (24.19%)**: `High income, low spending` (Thu nhập cao, chi tiêu thấp) $\rightarrow$ Gợi ý cá nhân hóa sản phẩm để kích cầu.
    2. **Cụm 1 (25.09%)**: `Low income, high spending` (Thu nhập thấp, chi tiêu cao) $\rightarrow$ Khuyến mãi combo, giảm giá sâu.
    3. **Cụm 2 (25.63%)**: `Low income, low spending` (Thu nhập thấp, chi tiêu thấp) $\rightarrow$ Chiến dịch giữ chân chi phí thấp.
    4. **Cụm 3 (25.10%)**: `High income, high spending` (Thu nhập cao, chi tiêu cao - Nhóm VIP) $\rightarrow$ Ưu đãi đặc quyền, truy cập sớm sản phẩm mới.
  - Lưu kết quả phân cụm vào `data/processed/clustering_results.csv` và hồ sơ phân cụm vào `data/processed/metrics/segment_profiles.csv`.

* **Step 9 — Evaluation Summary (Cell 20-21)**:
  - Hiển thị bảng tổng hợp ngắn gọn các siêu tham số của mô hình được chọn, điểm số đo đạc và định nghĩa các phân khúc khách hàng.

* **Step 10 — Save Artifacts (Cell 22-23)**:
  - Đóng gói và lưu mô hình tối ưu dưới định dạng joblib tại `models/best_cluster_model.joblib`.
  - In ra báo cáo kiểm tra trạng thái lưu thành công của toàn bộ artifacts (datasets, metrics, figures, model).